/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

void GlobalXtensor::init(const torch::Device& device) {
  if (initialized_) {
    LOG(WARNING) << "GlobalXtensor already initialized";
    return;
  }

  auto& pool = PhyPagePool::get_instance();
  CHECK(pool.is_initialized()) << "PhyPagePool must be initialized first";

  num_total_pages_ = pool.num_total();
  if (num_total_pages_ == 0) {
    LOG(ERROR) << "GlobalXtensor: PhyPagePool has no pages";
    return;
  }

  page_size_ = FLAGS_phy_page_granularity_size;
  total_size_ = 1024 * 1024 * 1024;
  // separate multiply to avoid overflow
  total_size_ *= 128;

  VirPtr global_vir_ptr = nullptr;
  // 42 x 128GB at most, leave 1 x 128GB to kvcache virtual memory
  std::vector<VirPtr> global_vir_ptrs;
  global_vir_ptrs.reserve(41);
  for (int i = 0; i < 41; i++) {
    vmm::create_vir_ptr(global_vir_ptr, total_size_);
    global_vir_ptrs.push_back(global_vir_ptr);
    LOG(INFO) << "[VMM] " << i << ":Reserved "
              << total_size_ / 1024 / 1024 / 1024 << " GB at "
              << global_vir_ptr;
  }
  total_size_ *= 41;
  vaddr_ = global_vir_ptrs[0];
  if (vaddr_ == nullptr) {
    LOG(ERROR) << "GlobalXtensor: failed to allocate virtual memory";
    return;
  }

  auto pages = pool.get_all_pages();
  if (!map_all_pages(pages)) {
    LOG(ERROR) << "Failed to map all pages for GlobalXtensor";
    return;
  }
  /*
    ptr1 =
        reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(vaddr_) +
    (num_total_pages_/2)*page_size_); ptr2 =
        reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(vaddr_) +
    num_total_pages_*2*page_size_);
  */

  initialized_ = true;
  LOG(INFO) << "GlobalXtensor initialized: " << num_total_pages_ << " pages, "
            << total_size_ << " bytes";
}

bool GlobalXtensor::map_page(PhyPage* page, size_t offset) {
  CHECK(page) << "Page is null";
  CHECK(offset % page_size_ == 0) << "Offset not aligned to page size";
  CHECK(offset < total_size_) << "Offset out of bounds";

  VirPtr vaddr =
      reinterpret_cast<VirPtr>(reinterpret_cast<uintptr_t>(vaddr_) + offset);
  PhyMemHandle phy_handle = page->get_phy_handle();
  vmm::map(vaddr, phy_handle);
  page_map_[offset] = page;
  free_offset_ += page_size_;
  CHECK(free_offset_ < total_size_);
  return true;
}

bool GlobalXtensor::map_all_pages(const std::vector<PhyPage*>& pages) {
  if (pages.size() != num_total_pages_) {
    LOG(ERROR) << "Page count mismatch: expected " << num_total_pages_
               << ", got " << pages.size();
    return false;
  }

  for (size_t i = 0; i < num_total_pages_; ++i) {
    /*if(i == num_total_pages_ - 1){
      size_t offset = num_total_pages_ * page_size_ * 2;
      if (!map_page(pages[i], offset)) {
        LOG(ERROR) << "Failed to map page " << i << " at offset " << offset;
        return false;
      }
      free_offset_-=page_size_;
    }*/
    size_t offset = i * page_size_;
    if (!map_page(pages[i], offset)) {
      LOG(ERROR) << "Failed to map page " << i << " at offset " << offset;
      return false;
    }
  }
  return true;
}

std::vector<page_id_t> GlobalXtensor::allocate_from_right(size_t count) {
  std::vector<page_id_t> result;

  // TODO: async unmap
  for (size_t i = 0; i < count; i++) {
    free_offset_ -= page_size_;
    CHECK(allocate_offset_ < free_offset_);
    void* ptr_to_unmap = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(vaddr_) + free_offset_);
    vmm::unmap(ptr_to_unmap, page_size_);
    // todo: solidate
    result.push_back(page_map_[free_offset_]->page_id());
    page_map_.erase(free_offset_);
  }
  return result;
}

void GlobalXtensor::free_to_right(std::vector<PhyPage*> page_ptrs) {
  // TODO: async unmap
  for (size_t i = 0; i < page_ptrs.size(); i++) {
    void* ptr_to_map = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(vaddr_) + free_offset_);
    PhyPage* page_to_map = page_ptrs[i];
    map_page(page_to_map, free_offset_);
  }
}

void* GlobalXtensor::allocate_from_left(size_t count) {
  void* result = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(vaddr_) +
                                         allocate_offset_);
  allocate_offset_ += page_size_ * count;
  CHECK(allocate_offset_ < free_offset_);
  return result;
}

void GlobalXtensor::free_contiguous(size_t addr) {
  size_t offset = addr - reinterpret_cast<uintptr_t>(vaddr_);
  void* ptr = reinterpret_cast<void*>(addr);
  PhyPage* page = page_map_[offset];
  // todo: consolodate this
  LOG(INFO) << "free:" << addr / page_size_;
  vmm::unmap(ptr, page_size_);
  page_map_.erase(offset);
  std::vector<PhyPage*> page_ptr = {page};
  free_to_right(page_ptr);
}

}  // namespace xllm
