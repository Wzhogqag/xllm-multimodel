/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "global_xtensor.h"

#include <glog/logging.h>

#include <algorithm>

#include "common/global_flags.h"
#include "phy_page_pool.h"

namespace xllm {

void GlobalXTensor::init(const torch::Device& device) {
  if (initialized_) {
    LOG(WARNING) << "GlobalXTensor already initialized";
    return;
  }

  auto& pool = PhyPagePool::get_instance();
  CHECK(pool.is_initialized()) << "PhyPagePool must be initialized first";

  num_total_pages_ = pool.num_total();
  if (num_total_pages_ == 0) {
    LOG(ERROR) << "GlobalXTensor: PhyPagePool has no pages";
    return;
  }

  page_size_ = FLAGS_phy_page_granularity_size;
  total_size_ = 1024 * 1024 * 1024;
  // separate multiply to avoid overflow
  total_size_ *= 128;

  VirPtr global_vir_ptr = nullptr;
  // 42 x 128GB at most, leave 1 x 128GB to kvcache virtual memory
  std::vector<VirPtr> global_vir_ptrs;
  int32_t reserve_times = 1;
  if (FLAGS_enable_activation_pooling) {
    reserve_times = 38;
  }
  global_vir_ptrs.reserve(reserve_times);
  for (int i = 0; i < reserve_times; i++) {
    vmm::create_vir_ptr(global_vir_ptr, total_size_);
    global_vir_ptrs.push_back(global_vir_ptr);
  }
  LOG(INFO) << "[VMM] " << ":Reserved "
            << 128 * reserve_times << " GB at "
            << global_vir_ptr;
  total_size_ *= reserve_times;
  vaddr_ = global_vir_ptrs[0];
  if (is_null_vir_ptr(vaddr_)) {
    LOG(ERROR) << "GlobalXTensor: failed to allocate virtual memory";
    return;
  }

  int32_t rate = std::max(0, std::min(100, FLAGS_global_xtensor_map_rate));
  size_t n_map = static_cast<size_t>(pool.num_total()) * rate / 100;
  auto pages = pool.get_pages(n_map);
  num_total_pages_ = pages.size();
  if (!map_all_pages(pages)) {
    LOG(ERROR) << "Failed to map all pages for GlobalXTensor";
    return;
  }

  threadpool_ = std::make_unique<ThreadPool>(4);

  // Start unmap thread
  unmap_running_ = true;
  unmap_working_ = true;
  unmap_thread_ = std::thread(&GlobalXTensor::unmap_worker, this);

  initialized_ = true;
  LOG(INFO) << "GlobalXTensor initialized: " << num_total_pages_ << " pages, "
            << total_size_ << " bytes";
}

void GlobalXTensor::map_page(PhyPage* page) {
  CHECK(page) << "Page is null";
  CHECK(free_offset_ % page_size_ == 0) << "Offset not aligned to page size";
  CHECK(free_offset_ < total_size_) << "Offset out of bounds";

  // 检查是否已经映射了该页面
  CHECK(page_map_.find(free_offset_) == page_map_.end())
      << "page " 
      << reinterpret_cast<uintptr_t>(add_vir_ptr_offset(vaddr_, free_offset_)) / page_size_ 
      << " already mapped to a page";

  VirPtr vaddr = add_vir_ptr_offset(vaddr_, free_offset_);
  PhyMemHandle phy_handle = page->get_phy_handle();
  vmm::map(vaddr, phy_handle);
  {
    std::lock_guard<std::mutex> lock(page_map_mtx_);
    page_map_[free_offset_] = page;
  }
  free_offset_ += page_size_;
  cv_free_offset_.notify_all();
  CHECK(free_offset_ <= total_size_);
  return;
}

bool GlobalXTensor::map_all_pages(const std::vector<PhyPage*>& pages) {
  if (pages.size() != num_total_pages_) {
    LOG(ERROR) << "Page count mismatch: expected " << num_total_pages_
               << ", got " << pages.size();
    return false;
  }

  for (size_t i = 0; i < num_total_pages_; ++i) {
    map_page(pages[i]);
  }
  return true;
}

bool GlobalXTensor::move_one_page(uintptr_t src_addr) {
  const uintptr_t base = reinterpret_cast<uintptr_t>(vaddr_);
  const size_t src_offset = src_addr - base;

  if (src_offset % page_size_ != 0) {
    return false;
  }
  auto it = page_map_.find(src_offset);
  if (it == page_map_.end()) {
    return false;
  }
  PhyPage* page = it->second;

  void* src_vaddr = reinterpret_cast<VirPtr>(src_addr);
  {
    std::lock_guard<std::mutex> lock(unmap_queue_mtx_);
    unmap_queue_.push(src_vaddr);
  }

  map_page(page);

  return true;
}

void GlobalXTensor::free_to_right_async(std::vector<PhyPage*> page_ptrs) {
  if (!pending_free_to_right_tasks_) {
    unmap_working_ = false;
  }
  pending_free_to_right_tasks_++;
  threadpool_->schedule([this, page_ptrs = std::move(page_ptrs)]() mutable {
    std::lock_guard<std::mutex> lock(mtx_);
    for (size_t i = 0; i < page_ptrs.size(); i++) {
      PhyPage* page_to_map = page_ptrs[i];
      map_page(page_to_map);
      // free_offset_ increased by page_size_ in map_page; at page granularity
      // start migration when map at boundary.
      if (free_offset_ >= total_size_) {
        migration_in_flight_.store(true);
        const uintptr_t base = reinterpret_cast<uintptr_t>(vaddr_);
        LOG(INFO) << "free_to_right_async: map at boundary, starting migration";
        migration_src_next_ = total_size_;
        dst_start_page_ =
            XTensorAllocator::get_instance().migration_dst_start_page() - base;

        free_offset_ = dst_start_page_;
      
        while (migration_src_next_ >= allocate_offset_.load()) {
          migration_src_next_ -= page_size_;
          move_one_page(base + migration_src_next_);
        }
        if (!allocate_offset_migrated_) {
          allocate_offset_.store(dst_start_page_);
        }
        allocate_offset_migrated_ = false;
        migration_in_flight_.store(false);
      }
    }
    pending_free_to_right_tasks_--;
    cv_free_offset_.notify_all();
    if (!pending_free_to_right_tasks_) {
      unmap_working_ = true;
    }
  });
}

void GlobalXTensor::maybe_switch_to_migration_dst(size_t count) {
  if (migration_in_flight_.load() && !allocate_offset_migrated_) {
    if (allocate_offset_.load() + count * page_size_ > migration_src_next_) {
      allocate_offset_.store(dst_start_page_);
      allocate_offset_migrated_ = true;
    }
  }
}

void* GlobalXTensor::allocate_from_left(size_t count) {
  maybe_switch_to_migration_dst(count);
  size_t allocated = allocate_offset_.fetch_add(page_size_ * count);
  void* result = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(vaddr_) +
                                         allocated);

  size_t ma = map_miss_count;
  // 当物理页不够时，先等待已有 free_to_right_async 完成；若仍不够则从 pool
  // get_pages 获取页指针并再调用 free_to_right_async 映射。
  wait_enough_pages(allocated + page_size_ * count);
  
  if(map_miss_count > ma)
  LOG(INFO) << "GlobalXTensor: map miss time=" << map_miss_time << ", count=" << map_miss_count << ", transfer_page_count=" << transfer_page_count <<" "<<time_1;
  return result;
}

std::vector<page_id_t> GlobalXTensor::allocate_pages_from_left(size_t count) {
  std::vector<page_id_t> result;

  maybe_switch_to_migration_dst(count);
  size_t allocated = allocate_offset_.fetch_add(page_size_ * count);
  wait_enough_pages(allocated + page_size_ * count);
  for (size_t i = 0; i < count; i++) {
    void* ptr_to_unmap = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(vaddr_) + allocated + i * page_size_);
    {
      std::lock_guard<std::mutex> lock(unmap_queue_mtx_);
      unmap_queue_.push(ptr_to_unmap);
    }
    PhyPage* page;
    {
      std::lock_guard<std::mutex> lock(page_map_mtx_);
      page = page_map_[allocated + i * page_size_];
    }
    result.push_back(page->page_id());
  }
  return result;
}

void GlobalXTensor::free_one_page_async(size_t addr) {
    size_t offset = addr - reinterpret_cast<uintptr_t>(vaddr_);
    void* ptr = reinterpret_cast<void*>(addr);
    PhyPage* page;
    {
      std::lock_guard<std::mutex> lock(unmap_queue_mtx_);
      unmap_queue_.push(ptr);
    }
    {
      std::lock_guard<std::mutex> lock(page_map_mtx_);
      page = page_map_[offset];
    }
    // Queue to unmap thread
    std::vector<PhyPage*> page_ptr = {page};
    free_to_right_async(page_ptr);
}

void GlobalXTensor::wait_enough_pages(size_t allocated) {
  std::unique_lock<std::mutex> lock(wait_enough_page_mtx_);
  if(allocated <= free_offset_) {
    return;
  }
  auto start = std::chrono::high_resolution_clock::now();
  while (allocated > free_offset_) {
    if (pending_free_to_right_tasks_ == 0) {
      size_t need = (allocated - free_offset_ + page_size_ - 1) /
                    page_size_;
      std::vector<PhyPage*> more_pages = 
          PhyPagePool::get_instance().get_pages(need);
      if (more_pages.empty()) {
        CHECK(allocated <= free_offset_)
            << "GlobalXTensor: out of memory, allocate_offset_="
            << allocated << ", free_offset_=" << free_offset_
            << ", no more pages from pool";
      }
      if (!more_pages.empty()) {
        free_to_right_async(std::move(more_pages));
      }
      transfer_page_count += need;
      map_miss_count++;
      continue;
    }
    cv_free_offset_.wait(lock);
  }
  auto end = std::chrono::high_resolution_clock::now();
  map_miss_time += (end - start).count();
}

void GlobalXTensor::unmap_worker() {
  while (unmap_running_) {
    while (unmap_working_) {
      std::unique_lock<std::mutex> lock(unmap_queue_mtx_);
      if (!unmap_queue_.empty()) {
        void* ptr = unmap_queue_.front();
        unmap_queue_.pop();
        lock.unlock();
        vmm::unmap(ptr, page_size_);
        size_t offset = reinterpret_cast<size_t>(ptr) -
                          reinterpret_cast<size_t>(vaddr_);
        {
          std::lock_guard<std::mutex> lock(page_map_mtx_);
          page_map_.erase(offset);
        }
      }
    }
  }
}

GlobalXTensor::~GlobalXTensor() {
  if (unmap_running_) {
    unmap_running_ = false;
    if (unmap_thread_.joinable()) {
      unmap_thread_.join();
    }
  }
}

}  // namespace xllm
