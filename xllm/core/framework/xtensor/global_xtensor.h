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

#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <queue>

#include "phy_page.h"
#include "xtensor_allocator.h"
#include "platform/vmm_api.h"

namespace xllm {

/**
 * GlobalXtensor maps all physical pages into a single large XTensor-backed
 * virtual address space. It provides contiguous segment allocation for
 * model weights without per-page RPC mapping.
 *
 * This is a singleton (one per worker).
 */
// TODO: KV可以不用取消映射/转移所有权
// 权重激活共置场景下，GlobalXTensor全映射
// 但是GlobalXTensor可以非全映射
// 所以PhyPagePool的范围比GlobalXTensor更大
class GlobalXtensor {
 public:
  // Get the global singleton instance
  static GlobalXtensor& get_instance() {
    static GlobalXtensor instance;
    return instance;
  }

  // Initialize (must be called after PhyPagePool::init)
  void init(const torch::Device& device);

  bool is_initialized() const { return initialized_; }

  std::vector<page_id_t> allocate_pages_from_right(size_t count);

  void free_to_right_async(std::vector<PhyPage*> page_ptrs);

  void* allocate_from_left(size_t count);

  void free_one_page_async(size_t addr);
  // void* ptr1 = nullptr;
  // void* ptr2 = nullptr;

  // Get base virtual address
  void* base_vaddr() const { return vaddr_; }

  size_t total_size() const { return total_size_; }
  size_t num_total_pages() const { return num_total_pages_; }
  size_t page_size() const { return page_size_; }
  size_t allocate_offset() const { return allocate_offset_; }
  size_t free_offset() const { return free_offset_; }
  void* activation_allocate_ptr() const {
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(vaddr_) +
                                   allocate_offset_);
  }

  void change_page_allocation() {
    allocate_offset_ = XTensorAllocator::get_instance().allocate_offset() - reinterpret_cast<uintptr_t>(vaddr_);
    return;
  }

  // Move a single page from src_addr to dst_addr if mapped at src. Returns
  // true if a page was moved. Used for incremental migration (end-to-begin).
  bool move_one_page(uintptr_t src_addr, uintptr_t dst_addr);

  // Called when map reaches boundary (free_offset_ >= total_size_) inside
  // free_to_right_internal. Callback receives (base_vaddr, total_size) so
  // allocator can start migration without calling back into GlobalXtensor.
  using MapAtBoundaryCallback = std::function<void(uintptr_t base, size_t total_size)>;
  void set_map_at_boundary_callback(MapAtBoundaryCallback cb) {
    map_at_boundary_callback_ = std::move(cb);
  }

 private:
  GlobalXtensor() = default;
  ~GlobalXtensor();
  GlobalXtensor(const GlobalXtensor&) = delete;
  GlobalXtensor& operator=(const GlobalXtensor&) = delete;

  std::unique_ptr<ThreadPool> threadpool_;
  std::thread unmap_thread_;
  bool unmap_running_ = false;
  std::queue<void*> ptr_to_unmap_queue_;

  bool map_page(PhyPage* page, size_t offset);
  bool map_all_pages(const std::vector<PhyPage*>& pages);

  void unmap_worker();

  mutable std::mutex mtx_;
  bool initialized_ = false;
  VirPtr vaddr_ = nullptr;
  size_t total_size_ = 0;
  size_t page_size_ = 0;
  size_t num_total_pages_ = 0;
  size_t allocate_offset_ = 0;
  size_t free_offset_ = 0;
  MapAtBoundaryCallback map_at_boundary_callback_;
  // 记录offset和在此映射好的物理页
  std::unordered_map<size_t, PhyPage*> page_map_ = {};
};

}  // namespace xllm
