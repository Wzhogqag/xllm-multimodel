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

#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <queue>

#include "phy_page.h"
#include "xtensor_allocator.h"
#include "platform/vmm_api.h"

namespace xllm {
class XTensorDistClient;

/**
 * GlobalXTensor maps all physical pages into a single large XTensor-backed
 * virtual address space. It provides contiguous segment allocation for
 * model weights without per-page RPC mapping.
 *
 * This is a singleton (one per worker).
 */
class GlobalXTensor {
 public:
  // Get the global singleton instance
  static GlobalXTensor& get_instance() {
    static GlobalXTensor instance;
    return instance;
  }

  // Initialize (must be called after PhyPagePool::init)
  void init(const torch::Device& device);

  bool is_initialized() const { return initialized_; }

  //std::vector<page_id_t> allocate_pages_from_right(size_t count);

  std::vector<page_id_t> allocate_pages_from_left(size_t count);

  void free_to_right_async(std::vector<PhyPage*> page_ptrs);

  void* allocate_from_left(size_t count);

  void free_one_page_async(size_t addr);
  
  void set_emergency_eviction_client(std::shared_ptr<XTensorDistClient> client) {
    emergency_eviction_client_ = client;
  }

  // Get base virtual address
  void* base_vaddr() const { return vir_ptr_to_void_ptr(vaddr_); }

  size_t total_size() const { return total_size_; }
  size_t num_total_pages() const { return num_total_pages_; }
  size_t page_size() const { return page_size_; }
  size_t allocate_offset() const { return allocate_offset_; }
  size_t free_offset() const { return free_offset_; }
  void* activation_allocate_ptr() const {
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(vaddr_) +
                                   allocate_offset_);
  }
  void* init_activation_allocate_ptr() const {
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(vaddr_));
  }
  void* allocate_init_from_left(size_t count);

  // Mooncake registration status (for idempotent registration)
  bool is_mooncake_registered() const { return mooncake_registered_; }
  void set_mooncake_registered(bool registered) {
    mooncake_registered_ = registered;
  }

 private:
  GlobalXTensor() = default;
  ~GlobalXTensor();
  GlobalXTensor(const GlobalXTensor&) = delete;
  GlobalXTensor& operator=(const GlobalXTensor&) = delete;

  void wait_enough_pages(size_t allocated, size_t count);

  // 若处于 migration 且本次分配会越过 migration_src_next_，则将 allocate_offset_ 切到 dst
  void maybe_switch_to_migration_dst(size_t count);

  std::unique_ptr<ThreadPool> threadpool_;
  std::thread unmap_thread_;
  bool unmap_running_ = false;
  std::atomic<bool> unmap_working_{false};
  std::queue<void*> unmap_queue_;

  void map_page(PhyPage* page, size_t offset);
  bool map_all_pages(const std::vector<PhyPage*>& pages);

  // Move a single page from src_addr to free_offset_ if mapped at src. Returns
  // true if a page was moved. Used for incremental migration (end-to-begin).
  bool move_one_page(uintptr_t src_addr, size_t dst_offset);

  void unmap_worker();

  mutable std::mutex mtx_;
  std::mutex unmap_queue_mtx_;
  std::mutex wait_enough_page_mtx_;
  mutable std::shared_mutex page_map_mtx_;

  std::condition_variable cv_free_offset_;
  std::atomic<size_t> pending_free_to_right_tasks_{0};

  bool initialized_ = false;

  VirPtr vaddr_ = {};
  size_t segment_size_ = 0;  // size of each 128GB virtual segment
  size_t total_size_ = 0;
  size_t page_size_ = 0;
  size_t num_total_pages_ = 0;

  std::atomic<size_t> allocate_offset_{0};
  size_t free_offset_ = 0;
  size_t init_arena_size_ = 0;
  size_t infer_arena_start_ = 0;
  size_t init_allocate_offset_ = 0;

  std::atomic<bool> migration_in_flight_ = false;
  bool allocate_offset_migrated_ = false;
  std::atomic<size_t> migration_src_next_ = 0;
  std::atomic<size_t> migration_src_end_ = 0;

  // 记录offset和在此映射好的物理页
  std::unordered_map<size_t, PhyPage*> page_map_ = {};
  size_t emergency_eviction_count_ = 0;
  size_t map_miss_time = 0;
  size_t time_1 = 0;
  size_t time_2 = 0;
  size_t transfer_page_count = 0;
  bool mooncake_registered_ = false;
  std::shared_ptr<XTensorDistClient> emergency_eviction_client_;
};

}  // namespace xllm
