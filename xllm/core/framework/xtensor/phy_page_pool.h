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

#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "global_xtensor.h"
#include "page_allocator.h"
#include "phy_page.h"

namespace xllm {

/**
 * PhyPagePool manages a pool of pre-allocated physical pages.
 *
 * This is a singleton class that:
 * - Pre-allocates physical pages during initialization
 * - Each page has a unique page_id for tracking
 * - Provides get/put interface for XTensor to acquire/release physical pages
 * - Avoids runtime allocation overhead during map operations
 */
class PhyPagePool {
 public:
  // Get the global singleton instance
  static PhyPagePool& get_instance() {
    static PhyPagePool pool;
    return pool;
  }

  // Initialize the pool with specified number of pages
  // device: the device to allocate physical pages on
  // num_pages: number of physical pages to pre-allocate
  void init(const torch::Device& device, size_t num_pages);

  // Check if initialized
  bool is_initialized() const { return initialized_; }

  // Get a physical page from the pool
  // Returns nullptr if pool is empty
  std::unique_ptr<PhyPage> get();

  // Get multiple physical pages from the pool in one lock (left-to-right)
  // Returns empty vector if not enough pages available
  // If partial allocation fails, all acquired pages are returned to pool
  std::vector<std::unique_ptr<PhyPage>> batch_get(size_t count);

  // Allocate contiguous virtual region from activation GlobalXTensor.
  // Reserved for activation arena growth.
  void* allocate_contiguous(size_t count);

  // Free contiguous virtual region back to activation GlobalXTensor.
  // Reserved for activation arena shrink.
  void free_contiguous(size_t addr, size_t count);

  // Put a physical page back to the pool
  void put(std::unique_ptr<PhyPage> page);

  // Put multiple physical pages back to the pool in one lock
  void batch_put(std::vector<std::unique_ptr<PhyPage>>& pages);

  // Get number of available pages in the pool
  size_t num_available() const;

  // Get total number of pages (available + in use)
  size_t num_total() const { return num_total_pages_; }

  // Get the device
  const torch::Device& device() const { return device_; }

  // For TP: set callbacks so this process can report consume/release to master
  // (only used when PageAllocator is not initialized on this process)
  void set_report_to_master(
      int32_t my_worker_rank,
      std::function<void(int32_t, size_t)> report_consume,
      std::function<void(int32_t, size_t)> report_release);

  // Get the zero page (for initializing virtual memory)
  // The returned pointer is owned by PhyPagePool, do not delete it
  PhyPage* get_zero_page();

  // ============== Global XTensor Support ==============

  // Get specified number of pages as raw pointers (PhyPage*). Ownership
  // remains with pool. Fills out with up to count pointers.
  // - When local free list is empty (e.g. initial call): assigns first count
  //   pages to out and puts the rest into local free list for get/batch_get.
  // - Otherwise: takes count pages from local free list (and global_xtensor
  //   if needed), fills out with those pointers.
  std::vector<PhyPage*> get_pages(size_t count);

 private:
  PhyPagePool() = default;
  ~PhyPagePool() = default;
  PhyPagePool(const PhyPagePool&) = delete;
  PhyPagePool& operator=(const PhyPagePool&) = delete;

  bool initialized_ = false;
  torch::Device device_{torch::kCPU};
  size_t num_total_pages_ = 0;

  mutable std::mutex mtx_;
  // All pages indexed by page_id (for jumbo xtensor)
  // This owns the pages and provides O(1) lookup by page_id
  std::vector<std::unique_ptr<PhyPage>> all_pages_;

  // Raw pointers to all pages (for GlobalXTensor, filled once at init)
  std::vector<PhyPage*> all_page_ptrs_;

  // Zero page for initializing virtual memory (owned by pool)
  std::unique_ptr<PhyPage> zero_page_;

  size_t num_available_ = 0;

  // Local free list: page_ids of the (1-map_rate) portion; get/batch_get
  // allocate from here first, put/batch_put return here (global_xtensor only
  // shrinks).
  std::deque<page_id_t> local_free_page_ids_;
  size_t get_miss_time = 0;
  size_t get_miss_count = 0;
  size_t transfer_page_count = 0;

  // For TP: report consume/release to master when this process is not master
  int32_t report_my_worker_rank_ = -1;
  std::function<void(int32_t, size_t)> report_consume_cb_;
  std::function<void(int32_t, size_t)> report_release_cb_;
};

}  // namespace xllm
