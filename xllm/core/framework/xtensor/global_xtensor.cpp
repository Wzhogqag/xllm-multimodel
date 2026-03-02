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

  auto pages = pool.get_all_pages();
  if (!map_all_pages(pages)) {
    LOG(ERROR) << "Failed to map all pages for GlobalXTensor";
    return;
  }

  threadpool_ = std::make_unique<ThreadPool>(4);

  // Start unmap thread
  unmap_running_ = true;
  unmap_thread_ = std::thread(&GlobalXTensor::unmap_worker, this);

  initialized_ = true;
  LOG(INFO) << "GlobalXTensor initialized: " << num_total_pages_ << " pages, "
            << total_size_ << " bytes";
}

bool GlobalXTensor::map_page(PhyPage* page, size_t offset) {
  CHECK(page) << "Page is null";
  CHECK(offset % page_size_ == 0) << "Offset not aligned to page size";
  CHECK(offset < total_size_) << "Offset out of bounds";

  // 检查是否已经映射了该页面
  CHECK(page_map_.find(offset) == page_map_.end())
      << "page " 
      << reinterpret_cast<uintptr_t>(add_vir_ptr_offset(vaddr_, offset)) / page_size_ 
      << " already mapped to a page";

  VirPtr vaddr = add_vir_ptr_offset(vaddr_, offset);
  PhyMemHandle phy_handle = page->get_phy_handle();
  vmm::map(vaddr, phy_handle);
  page_map_[offset] = page;
  free_offset_ += page_size_;
  CHECK(free_offset_ <= total_size_);
  return true;
}

bool GlobalXTensor::map_all_pages(const std::vector<PhyPage*>& pages) {
  if (pages.size() != num_total_pages_) {
    LOG(ERROR) << "Page count mismatch: expected " << num_total_pages_
               << ", got " << pages.size();
    return false;
  }

  for (size_t i = 0; i < num_total_pages_; ++i) {
    size_t offset = i * page_size_;
    if (!map_page(pages[i], offset)) {
      LOG(ERROR) << "Failed to map page " << i << " at offset " << offset;
      return false;
    }
  }
  return true;
}

// TODO: guard multi thread access to ptr_to_unmap_queue_ and page_map_ with
// mutex
bool GlobalXTensor::move_one_page(uintptr_t src_addr, uintptr_t dst_addr) {
  const uintptr_t base = reinterpret_cast<uintptr_t>(vaddr_);
  const size_t src_offset = src_addr - base;
  const size_t dst_offset = dst_addr - base;

  if (src_offset % page_size_ != 0 || dst_offset % page_size_ != 0) {
    return false;
  }
  auto it = page_map_.find(src_offset);
  if (it == page_map_.end()) {
    return false;
  }
  PhyPage* page = it->second;

  void* src_vaddr = reinterpret_cast<VirPtr>(src_addr);
  ptr_to_unmap_queue_.push(src_vaddr);

  auto dst_it = page_map_.find(dst_offset);
  CHECK(dst_it == page_map_.end())
      << "move_one_page: dst page at offset " << dst_offset << " is not free";

  VirPtr dst_vaddr = reinterpret_cast<VirPtr>(dst_addr);
  PhyMemHandle phy_handle = page->get_phy_handle();
  vmm::map(dst_vaddr, phy_handle);
  page_map_[dst_offset] = page;
  return true;
}

void GlobalXTensor::free_to_right_async(std::vector<PhyPage*> page_ptrs) {
  threadpool_->schedule([this, page_ptrs = std::move(page_ptrs)]() mutable {
    std::lock_guard<std::mutex> lock(mtx_);
    for (size_t i = 0; i < page_ptrs.size(); i++) {
      PhyPage* page_to_map = page_ptrs[i];
      map_page(page_to_map, free_offset_);
      // free_offset_ increased by page_size_ in map_page; at page granularity
      // notify allocator to maybe start incremental migration when map at
      // boundary.
      if (free_offset_ >= total_size_ && map_at_boundary_callback_) {
        map_at_boundary_callback_(reinterpret_cast<uintptr_t>(vaddr_),
                                  total_size_);
        free_offset_ = XTensorAllocator::get_instance().free_offset() -
                       reinterpret_cast<uintptr_t>(vaddr_);
        allocate_offset_ = XTensorAllocator::get_instance().allocate_offset() -
                           reinterpret_cast<uintptr_t>(vaddr_);
      }
    }
  });
}

void* GlobalXTensor::allocate_from_left(size_t count) {
  void* result = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(vaddr_) +
                                         allocate_offset_);
  allocate_offset_ += page_size_ * count;

  // TODO: 设置等待逻辑，当物理页不够时等待unmap释放完毕
  CHECK(allocate_offset_ <= free_offset_)
      << "GlobalXTensor: out of memory, allocate_offset_=" << allocate_offset_
      << ", free_offset_=" << free_offset_;
  return result;
}

std::vector<page_id_t> GlobalXTensor::allocate_pages_from_right(size_t count) {
  std::lock_guard<std::mutex> lock(mtx_);
  std::vector<page_id_t> result;

  // TODO: async unmap
  for (size_t i = 0; i < count; i++) {
    free_offset_ -= page_size_;
    CHECK(allocate_offset_ < free_offset_);
    void* ptr_to_unmap = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(vaddr_) + free_offset_);
    ptr_to_unmap_queue_.push(ptr_to_unmap);
    // todo: solidate
    result.push_back(page_map_[free_offset_]->page_id());
  }
  return result;
}

std::vector<page_id_t> GlobalXTensor::allocate_pages_from_left(size_t count) {
  std::lock_guard<std::mutex> lock(mtx_);
  std::vector<page_id_t> result;

  // TODO: async unmap
  for (size_t i = 0; i < count; i++) {
    CHECK(allocate_offset_ < free_offset_);
    void* ptr_to_unmap = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(vaddr_) + allocate_offset_);
    ptr_to_unmap_queue_.push(ptr_to_unmap);
    // todo: solidate
    result.push_back(page_map_[allocate_offset_]->page_id());
    allocate_offset_ += page_size_;
  }
  return result;
}

void GlobalXTensor::free_one_page_async(size_t addr) {
  threadpool_->schedule([this, addr = addr]() mutable {
    std::lock_guard<std::mutex> lock(mtx_);
    size_t offset = addr - reinterpret_cast<uintptr_t>(vaddr_);
    void* ptr = reinterpret_cast<void*>(addr);
    PhyPage* page = page_map_[offset];
    // todo: consolodate this
    // LOG(INFO) << "free:" << addr / page_size_;
    ptr_to_unmap_queue_.push(ptr);
    // Queue to unmap thread
    std::vector<PhyPage*> page_ptr = {page};
    free_to_right_async(page_ptr);
  });
}

void GlobalXTensor::unmap_worker() {
  while (unmap_running_) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (!ptr_to_unmap_queue_.empty()) {
      void* ptr = ptr_to_unmap_queue_.front();
      ptr_to_unmap_queue_.pop();
      lock.unlock();
      vmm::unmap(ptr, page_size_);
      page_map_.erase(reinterpret_cast<size_t>(ptr) -
                      reinterpret_cast<size_t>(vaddr_));
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
