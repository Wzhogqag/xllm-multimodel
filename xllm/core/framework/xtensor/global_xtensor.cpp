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

namespace {
constexpr size_t kInitArenaSizeBytes = 128ULL * 1024ULL * 1024ULL * 1024ULL;
}

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
  segment_size_ = total_size_;  // 128GB per reserved segment

  VirPtr global_vir_ptr = nullptr;
  // 42 x 128GB at most, leave 1 x 128GB to kvcache virtual memory
  std::vector<VirPtr> global_vir_ptrs;
  int32_t reserve_times = 2;
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

  init_arena_size_ = std::min(total_size_, kInitArenaSizeBytes);
  infer_arena_start_ = init_arena_size_;
  init_allocate_offset_ = 0;
  allocate_offset_.store(infer_arena_start_);
  free_offset_ = infer_arena_start_;

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
  unmap_working_.store(true);
  unmap_thread_ = std::thread(&GlobalXTensor::unmap_worker, this);

  initialized_ = true;
  LOG(INFO) << "GlobalXTensor initialized: " << num_total_pages_ << " pages, "
            << total_size_ << " bytes";
}

void* GlobalXTensor::allocate_init_from_left(size_t count) {
  CHECK_GT(count, 0);
  const uintptr_t base = reinterpret_cast<uintptr_t>(vaddr_);

  void* result = reinterpret_cast<void*>(base + init_allocate_offset_);

  for (size_t i = 0; i < count; ++i) {
    maybe_switch_to_migration_dst(1);
    size_t allocated = allocate_offset_.fetch_add(page_size_);
    wait_enough_pages(allocated + page_size_);
    
    move_one_page(base + allocated, init_allocate_offset_);
    init_allocate_offset_ += page_size_;
    CHECK_LT(init_allocate_offset_, init_arena_size_)
        << "Init arena exhausted: init_free_offset=" << init_allocate_offset_
        << ", init_arena_size=" << init_arena_size_;
  }

  return result;
}

void GlobalXTensor::map_page(PhyPage* page, size_t offset) {
  CHECK(page) << "Page is null";
  CHECK(offset % page_size_ == 0) << "Offset not aligned to page size";
  CHECK(offset < total_size_) << "Offset out of bounds";

  {
    std::shared_lock<std::shared_mutex> lock(page_map_mtx_);
    CHECK(page_map_.find(offset) == page_map_.end())
        << "page " << offset / page_size_
        << " already mapped to a page";
  }

  VirPtr vaddr = add_vir_ptr_offset(vaddr_, offset);
  PhyMemHandle phy_handle = page->get_phy_handle();
  vmm::map(vaddr, phy_handle);
  {
    std::unique_lock<std::shared_mutex> lock(page_map_mtx_);
    page_map_[offset] = page;
  }
  cv_free_offset_.notify_all();
  return;
}

bool GlobalXTensor::map_all_pages(const std::vector<PhyPage*>& pages) {
  if (pages.size() != num_total_pages_) {
    LOG(ERROR) << "Page count mismatch: expected " << num_total_pages_
               << ", got " << pages.size();
    return false;
  }

  for (size_t i = 0; i < num_total_pages_; ++i) {
    map_page(pages[i], free_offset_);
    free_offset_ += page_size_;
  }
  return true;
}

bool GlobalXTensor::move_one_page(uintptr_t src_addr, size_t dst_offset) {
  const uintptr_t base = reinterpret_cast<uintptr_t>(vaddr_);
  const size_t src_offset = src_addr - base;

  if (src_offset % page_size_ != 0) {
    return false;
  }
  
  PhyPage* page = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(page_map_mtx_);
    auto it = page_map_.find(src_offset);
    CHECK(it != page_map_.end()) << "Page " << src_offset / page_size_ << " not found";
    page = it->second;
  }


  void* src_vaddr = reinterpret_cast<VirPtr>(src_addr);
  {
    std::lock_guard<std::mutex> lock(unmap_queue_mtx_);
    unmap_queue_.push(src_vaddr);
  }

  map_page(page, dst_offset);

  return true;
}

void GlobalXTensor::free_to_right_async(std::vector<PhyPage*> page_ptrs) {
  if (pending_free_to_right_tasks_.load() == 0) {
    unmap_working_.store(false);
  }
  pending_free_to_right_tasks_.fetch_add(1);
  threadpool_->schedule([this, page_ptrs = std::move(page_ptrs)]() mutable {
    std::lock_guard<std::mutex> lock(mtx_);
    for (size_t i = 0; i < page_ptrs.size(); i++) {
      PhyPage* page_to_map = page_ptrs[i];
      map_page(page_to_map, free_offset_);
      free_offset_ += page_size_;
      // free_offset_ increased by page_size_ in map_page; at page granularity
      // start migration when map at boundary.
      if (free_offset_ >= total_size_) {
        migration_src_next_.store(total_size_ - page_size_);
        migration_in_flight_.store(true);
        const uintptr_t base = reinterpret_cast<uintptr_t>(vaddr_);
        LOG(INFO) << "free_to_right_async: map at boundary, starting migration";

        free_offset_ = infer_arena_start_;
      
        size_t migration_src_next = migration_src_next_.load();
        migration_src_end_.store(allocate_offset_.load());
        while (migration_src_next > migration_src_end_.load()) {
          migration_src_next = migration_src_next_.fetch_sub(page_size_);
          move_one_page(base + migration_src_next, free_offset_);
          free_offset_ += page_size_;
          if (!allocate_offset_migrated_) {
            migration_src_end_.store(allocate_offset_.load());
          }
        }
        if (!allocate_offset_migrated_) {
          allocate_offset_.store(infer_arena_start_);
        }
        allocate_offset_migrated_ = false;
        migration_in_flight_.store(false);
      }
    }
    pending_free_to_right_tasks_.fetch_sub(1);
    cv_free_offset_.notify_all();
    if (!pending_free_to_right_tasks_) {
      unmap_working_.store(true);
    }
  });
}

void GlobalXTensor::maybe_switch_to_migration_dst(size_t count) {
  if (migration_in_flight_.load() && !allocate_offset_migrated_) {
    size_t allocate_offset = allocate_offset_.load();
    if (allocate_offset + count * page_size_ > migration_src_next_.load()) {
      migration_src_end_.store(allocate_offset);
      allocate_offset_.store(infer_arena_start_);
      allocate_offset_migrated_ = true;
    }
  }
}

void* GlobalXTensor::allocate_from_left(size_t count) {
  maybe_switch_to_migration_dst(count);
  
  const size_t alloc_size = page_size_ * count;
  size_t allocated;

  // CAS loop: if the allocation would cross a 128GB segment boundary, skip
  // allocate_offset_ to the start of the next segment and free all physical
  // pages that would have been stranded at the tail of the current segment.
  size_t old_offset = allocate_offset_.load();
  while (true) {
    const size_t seg_end = (old_offset / segment_size_ + 1) * segment_size_;
    const bool crosses = (old_offset + alloc_size > seg_end);

    size_t new_offset;
    if (crosses) {
      // Start of next 128GB segment is the effective allocation point.
      allocated  = seg_end;
      new_offset = seg_end + alloc_size;
    } else {
      allocated  = old_offset;
      new_offset = old_offset + alloc_size;
    }

    if (allocate_offset_.compare_exchange_weak(old_offset, new_offset)) {
      if (crosses) {
        // Collect physical pages stranded in [old_offset, seg_end).
        const uintptr_t base = reinterpret_cast<uintptr_t>(vaddr_);
        std::vector<PhyPage*> tail_pages;
        {
          std::shared_lock<std::shared_mutex> lock(page_map_mtx_);
          for (size_t off = old_offset; off < seg_end; off += page_size_) {
            auto it = page_map_.find(off);
            if (it != page_map_.end()) {
              tail_pages.push_back(it->second);
            }
          }
        }
        {
          std::lock_guard<std::mutex> lock(unmap_queue_mtx_);
          for (size_t off = old_offset; off < seg_end; off += page_size_) {
            unmap_queue_.push(reinterpret_cast<void*>(base + off));
          }
        }
        LOG(INFO) << "GlobalXTensor: allocate_from_left crosses 128GB boundary"
                  << " at seg_end=" << seg_end
                  << ", freeing " << tail_pages.size() << " tail pages"
                  << ", skipping allocate_offset_ to " << seg_end;
        if (!tail_pages.empty()) {
          free_to_right_async(std::move(tail_pages));
        }
      }
      break;
    }
    // CAS failed: old_offset was updated by compare_exchange_weak, retry.
  }

  void* result = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(vaddr_) +
                                         allocated);

  size_t ma = map_miss_count;
  // 当物理页不够时，先等待已有 free_to_right_async 完成；若仍不够则从 pool
  // get_pages 获取页指针并再调用 free_to_right_async 映射。
  wait_enough_pages(allocated + alloc_size);

  if (map_miss_count > ma)
    LOG(INFO) << "GlobalXTensor: map miss time=" << map_miss_time
              << ", count=" << map_miss_count
              << ", transfer_page_count=" << transfer_page_count
              << " " << time_1;
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
    PhyPage* page = nullptr;
    {
      std::shared_lock<std::shared_mutex> lock(page_map_mtx_);
      auto it = page_map_.find(allocated + i * page_size_);
      CHECK(it != page_map_.end()) << "Page " 
          << allocated / page_size_ + i << " not found";
      page = it->second;
    }
    {
      std::lock_guard<std::mutex> lock(unmap_queue_mtx_);
      unmap_queue_.push(ptr_to_unmap);
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
      std::shared_lock<std::shared_mutex> lock(page_map_mtx_);
      auto it = page_map_.find(offset);
      CHECK(it != page_map_.end()) << "Page at offset " << offset << " not found";
      page = it->second;
    }
    {
      std::lock_guard<std::mutex> lock(unmap_queue_mtx_);
      unmap_queue_.push(ptr);
    }
    // Queue to unmap thread
    std::vector<PhyPage*> page_ptr = {page};
    free_to_right_async(page_ptr);
}

void GlobalXTensor::wait_enough_pages(size_t allocated) {
  std::unique_lock<std::mutex> lock(wait_enough_page_mtx_);
  if (migration_in_flight_.load() && !allocate_offset_migrated_) {
    return;
  }
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
    while (unmap_working_.load()) {
      std::unique_lock<std::mutex> lock(unmap_queue_mtx_);
      if (!unmap_queue_.empty()) {
        void* ptr = unmap_queue_.front();
        unmap_queue_.pop();
        lock.unlock();
        vmm::unmap(ptr, page_size_);
        size_t offset = reinterpret_cast<size_t>(ptr) -
                          reinterpret_cast<size_t>(vaddr_);
        {
          std::unique_lock<std::shared_mutex> lock(page_map_mtx_);
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
