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

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "layer_offload_manager.h"
#include "virt_page.h"
#include "xtensor.h"  // For offset_t type definition

namespace xllm {

class BlockManager;

// Configuration constants
constexpr int32_t MIN_RESERVED_PAGES = 8;
constexpr int32_t MAX_RESERVED_PAGES = 32;
constexpr bool PAGE_PREALLOC_ENABLED = true;
constexpr double PREALLOC_THREAD_TIMEOUT = 2.0;  // seconds
/**
 * PageAllocator manages virtual page allocation for KV cache.
 *
 * Key concepts:
 * - VirtPage: Logical page for KV cache indexing, based on single-layer memory
 * - PhyPage: Physical memory page (2MB), managed by PhyPagePool
 *
 * Multi-model support:
 * - Each model has its own logical page_list (virtual pages)
 * - All models share physical pages (phy_pages)
 * - Model sleep: stops prealloc thread, unmaps and releases physical pages
 * - Model wakeup: restarts prealloc thread to refill physical pages
 *
 * Memory layout:
 * - For non-contiguous: each layer has its own K and V XTensor
 *   - mem_size_per_layer = total_phy_mem / (2 * num_layers)
 *   - num_virt_pages = mem_size_per_layer / virt_page_size
 *   - Allocating 1 virt_page consumes (2 * num_layers) phy_pages
 *
 * This is a singleton class shared by all XTensorBlockManagerImpl instances.
 */
class PageAllocator {
 public:
  // Get the global singleton instance
  static PageAllocator& get_instance() {
    static PageAllocator allocator;
    return allocator;
  }

  // Initialize the allocator (basic initialization)
  // num_phy_pages: total number of physical pages from PhyPagePool
  // dp_size: number of data parallel groups
  // max_world_size: maximum number of workers (for per-worker tracking)
  // enable_page_prealloc: whether to enable background preallocation
  void init(size_t num_phy_pages,
            int32_t dp_size = 1,
            int32_t max_world_size = 1,
            bool enable_page_prealloc = PAGE_PREALLOC_ENABLED);

  // Check if initialized
  bool is_initialized() const { return initialized_; }

  // ============ Multi-Model Management ============
  // Register a model with its layer count
  // model_id: unique identifier for the model (e.g. model name from options)
  // num_layers: number of transformer layers for this model
  // priority: model priority (0-100, default 50)
  // min_reserved_pages: initial minimum reserved pages (default 8)
  // max_reserved_pages: initial maximum reserved pages (default 32)
  // Returns true if registration successful
  bool register_model(const std::string& model_id,
                      int64_t num_layers,
                      int32_t master_status,
                      int32_t priority = 50,
                      int32_t min_reserved_pages = 8,
                      int32_t max_reserved_pages = 32);

  // Put a model to sleep:
  // - Release weight pages (via free_weight_pages)
  // - Unmap all mapped KV cache virtual pages
  // - Release physical pages back to shared pool
  // - Stop preallocation for this model
  bool sleep_model(const std::string& model_id);

  // Wake up a sleeping model:
  // - Re-map all previously mapped KV cache virtual pages
  // - Re-allocate weight pages (via alloc_weight_pages)
  // - Resume preallocation for this model
  bool wakeup_model(const std::string& model_id);

  // Check if a model is sleeping
  bool is_model_sleeping(const std::string& model_id) const;

  // Set model-specific parallel strategy (for fork master with different dp/tp)
  // This affects which workers are targeted during weight allocation
  void set_model_parallel_strategy(const std::string& model_id,
                                   int32_t dp_size,
                                   int32_t tp_size,
                                   int32_t worker_rank_base = 0);

  // Get model-specific world_size (dp_size * tp_size)
  // Returns 0 if model not found or not set
  int32_t get_model_world_size(const std::string& model_id) const;

  // Get available physical pages for a specific model
  // This considers the model's world_size and returns the minimum free pages
  // among all workers that the model uses
  size_t get_free_phy_pages_for_model(const std::string& model_id) const;

    struct ModelMemoryUsage {
        size_t weight_phy_pages = 0;
        size_t kv_cache_phy_pages = 0;
    };

    // Snapshot current per-model physical page usage.
    // weight_phy_pages: currently resident weight pages.
    // kv_cache_phy_pages: currently mapped KV cache pages.
    std::unordered_map<std::string, ModelMemoryUsage> get_model_memory_usage()
            const;

  // Pick the loadable degraded replica with minimal average memory pressure.
  // Returns nullopt if no degraded replica can be loaded (any worker pressure >= 1).
  std::optional<std::string> pick_best_loadable_model_for_base(
      const std::string& base_model_id) const;

  // Start preallocation thread (called after reserving null block)
  void start_prealloc_thread();

  // ============ KV Cache Page Allocation ============
  // Allocate a virtual page for KV cache
  // model_id: which model this allocation is for
  // dp_rank: which DP group this allocation is for
  // Consumes phy_pages_per_virt_page_ physical pages
  // Returns nullptr if no physical pages available
  std::unique_ptr<VirtPage> alloc_kv_cache_page(const std::string& model_id,
                                                int32_t dp_rank);

  // Free multiple KV cache virtual pages
  void free_kv_cache_pages(const std::string& model_id,
                           int32_t dp_rank,
                           const std::vector<int64_t>& virt_page_ids);

  // Trim reserved KV cache pages (unmap physical pages)
  void trim_kv_cache(const std::string& model_id, int32_t dp_rank);

  // ============ Weight Page Allocation ============
  // Allocate physical pages for weight tensor (full map)
  // model_id: which model this allocation is for
  // num_pages: number of physical pages (aligned up from weight size)
  // All-or-nothing: returns true if all pages allocated, false otherwise
  bool alloc_weight_pages(const std::string& model_id, size_t num_pages);

  // Free physical pages from weight tensor
  // model_id: which model
  // num_pages: same count used in alloc_weight_pages
  bool free_weight_pages(const std::string& model_id, size_t num_pages);

  // Get number of weight pages allocated for a model (not cleared on free)
  size_t get_weight_pages_allocated(const std::string& model_id) const;

  // Set weight pages count (for LIGHT_SLEEP/DEEP_SLEEP mode without physical
  // allocation)
  void set_weight_pages_count(const std::string& model_id, size_t num_pages);

  // Virtual page getters (for specific model and DP group)
  size_t get_num_free_virt_pages(const std::string& model_id,
                                 int32_t dp_rank) const;
  size_t get_num_inuse_virt_pages(const std::string& model_id,
                                  int32_t dp_rank) const;
  size_t get_num_total_virt_pages(const std::string& model_id) const;
  size_t get_num_reserved_virt_pages(const std::string& model_id,
                                     int32_t dp_rank) const;
  int32_t get_model_min_reserved_pages(const std::string& model_id) const;
  void register_prefix_block_manager(const std::string& model_id,
                                     int32_t dp_rank,
                                     const BlockManager* block_manager);
  void unregister_prefix_block_manager(const std::string& model_id,
                                       int32_t dp_rank,
                                       const BlockManager* block_manager);

  // Physical page getters (shared across all models)
  size_t get_num_free_phy_pages() const;
  size_t get_num_total_phy_pages() const;

  // Get free pages for each worker (for etcd registration)
  // Returns a vector where index i = num_total_phy_pages -
  // worker_pages_used_[i]
  std::vector<size_t> get_all_worker_free_pages() const;
  // Get workers whose free pages are below low watermark ratio.
  // Caller provides a ratio in [0, 1], e.g. FLAGS_layer_offload_low_watermark_ratio.
  std::unordered_set<int32_t> get_pressure_workers_by_low_watermark_ratio(
      double low_watermark_ratio) const;
  // Get workers whose free pages are above high watermark ratio.
  // Caller provides a ratio in [0, 1], e.g. FLAGS_layer_offload_restore_watermark_ratio.
  std::unordered_set<int32_t> get_healthy_workers_by_high_watermark_ratio(
      double high_watermark_ratio) const;
  // Get awake models whose worker window overlaps the given worker set.
  std::unordered_set<std::string> get_models_by_workers(
      const std::unordered_set<int32_t>& workers) const;
  // Reclaim scoped models' reserved KV pages down to model min_reserved_pages.
  // Intended for targeted pressure relief after prefix cache eviction.
  void reclaim_excess_reserved_pages_for_models(
      const std::unordered_set<std::string>& model_ids);
  // Reclaim all scoped models' reserved KV pages down to 0.
  void release_all_reserved_pages_for_models(
      const std::unordered_set<std::string>& model_ids);

  // Convert block_id to virt_page_id
  int64_t get_virt_page_id(int64_t block_id, size_t block_mem_size) const;

  // Get offset for XTensor map/unmap (based on single-layer)
  offset_t get_offset(int64_t virt_page_id) const;

  // Get configuration
  size_t page_size() const { return page_size_; }
  int64_t num_layers(const std::string& model_id) const;

  // Get number of physical pages consumed per virtual page allocation
  size_t phy_pages_per_virt_page(const std::string& model_id) const;

  // Update model reserved pages (min and max)
  // This allows dynamic adjustment based on priority and load
  void update_model_reserved_pages(const std::string& model_id,
                                   int32_t min_pages,
                                   int32_t max_pages);

  bool consume_phy_pages_for_worker(int32_t worker_rank, size_t num_phy_pages);
  void release_phy_pages_for_worker(int32_t worker_rank, size_t num_phy_pages);

  // ============ Layer Offload State (MVP) ============
  // Update num_layers_on_device after offloading/restoring layers.
  // Called by LayerOffloadManager after a successful offload/load batch.
  void update_layers_on_device(const std::string& model_id,
                               int32_t new_num_layers_on_device);

  // Get num_layers_on_device (0 if model not found).
  int32_t get_num_layers_on_device(const std::string& model_id) const;
  // Update/get layer offload physical page stats (0 if model not found).
  void update_layer_offloaded_phy_pages_delta(const std::string& model_id,
                                              int64_t delta_phy_pages);
  size_t get_layer_offloaded_phy_pages(const std::string& model_id) const;

  // ============ Model Step/Schedule Gating ============
  // Block new request admission for a model and return immediately.
  void block_model_schedule(const std::string& model_id);
  // Unblock model request/step schedule after model is fully restored.
  void unblock_model_schedule(const std::string& model_id);
  // Check whether model request/step schedule is currently blocked.
  bool is_model_schedule_blocked(const std::string& model_id) const;
  // Mark request lifecycle for a model.
  void mark_model_request_begin(const std::string& model_id);
  void mark_model_request_end(const std::string& model_id);
  // Wait until all in-flight requests are drained for a model.
  void wait_model_requests_drained(const std::string& model_id);
  // Wait until model schedule is unblocked and no step is running, then mark
  // step as in-flight.
  void wait_and_mark_model_step_begin(const std::string& model_id);
  // Mark model step finished.
  void mark_model_step_end(const std::string& model_id);

  // Start/stop async eviction background thread.
  void start_async_eviction_thread();
  void stop_async_eviction_thread();

  bool emergency_eviction(int32_t pages_needed, int32_t worker_rank);

  // ============ Layer Offload Monitor (master-side) ============
  void start_layer_offload_monitor();
  void stop_layer_offload_monitor();
  LayerOffloadManager* get_layer_offload_manager() {
    return layer_offload_mgr_.get();
  }

  // Consume/release physical pages for a specific DP group (update tracking)
  // Returns false if not enough physical pages available
  void consume_phy_pages_for_dp(const std::string& model_id,
                                int32_t dp_rank,
                                size_t num_phy_pages);
  void release_phy_pages_for_dp(const std::string& model_id,
                                int32_t dp_rank,
                                size_t num_phy_pages);

  // Try to consume physical pages for a specific DP group without blocking
  // Returns true if enough pages available and consumption successful, false otherwise
  bool try_to_consume_phy_pages_for_dp(const std::string& model_id,
                                        int32_t dp_rank,
                                        size_t num_phy_pages);

  bool offload_model(std::string model_id);
  bool load_model(std::string model_id);

 private:
  PageAllocator() = default;
  ~PageAllocator();
  PageAllocator(const PageAllocator&) = delete;
  PageAllocator& operator=(const PageAllocator&) = delete;

  // Per-DP group virtual page tracking
  struct DpGroupPages {
    size_t num_free_virt_pages{0};            // Protected by mtx_
    std::deque<int64_t> free_virt_page_list;  // Unmapped virtual pages
    std::deque<int64_t>
        reserved_virt_page_list;  // Mapped virtual pages ready for use
    std::set<int64_t>
        allocated_virt_page_list;  // Mapped virtual pages in use by block mgr
  };

  // Per-model state
  struct ModelState {
    int64_t num_layers = 0;
    size_t num_total_virt_pages = 0;
    size_t phy_pages_per_virt_page = 0;
    size_t weight_pages_allocated = 0;  // Not cleared on free, used for wakeup
    bool is_sleeping = false;
    // Count of pending map operations (for safe sleep)
    std::atomic<int> pending_map_ops{0};
    std::vector<DpGroupPages> dp_group_pages;
    // Model-specific parallel strategy (for fork master with different dp/tp)
    int32_t model_dp_size = 0;  // 0 means use global dp_size_
    int32_t model_tp_size = 0;  // 0 means use global tp_size
    int32_t model_worker_rank_base = 0;
    int32_t model_world_size = 0;  // = dp_size * tp_size, 0 means use global
    // Priority-based reserved pages configuration
    int32_t priority =
        50;  // Model priority (25, 50, 75, 100 based on priority_level)
    int32_t min_reserved_pages = 8;
    int32_t max_reserved_pages = 32;
    int32_t base_min_reserved_pages = 8;
    int32_t base_max_reserved_pages = 32;

    // --- Layer offload / restore state (MVP) ---
    // Number of layers currently mapped on device. Initialized to num_layers
    // when alloc_weight_pages succeeds; decremented on offload, incremented
    // on restore.
    int32_t num_layers_on_device = 0;
    // Physical pages currently offloaded via layer offload manager.
    size_t layer_offloaded_phy_pages = 0;

    // If true, no new requests can be admitted for this model.
    bool request_blocked = false;
    // If true, no new step can start for this model.
    bool schedule_blocked = false;
    // At most one step in-flight per model.
    bool step_inflight = false;
    // Number of in-flight API requests routed to this model.
    int32_t inflight_requests = 0;
  };

  // Check if enough physical pages available for a specific DP group
  // model_id: which model (to get tp_size for worker range calculation)
  // dp_rank: which DP group
  bool has_enough_phy_pages_for_dp(const std::string& model_id,
                                   int32_t dp_rank,
                                   size_t num_phy_pages) const;

  // Get worker range for a DP group [start, end)
  // Returns {start_worker, end_worker}
  std::pair<int32_t, int32_t> get_dp_group_worker_range(
      const std::string& model_id,
      int32_t dp_rank) const;

  // Get minimum free pages among workers in a range [start, end)
  size_t get_min_free_pages_in_range(int32_t start_worker,
                                     int32_t end_worker) const;
  size_t get_worker_used_pages_locked(int32_t worker_rank) const;
  void sync_reported_phy_pages_from_shm_locked() const;
  void init_reported_phy_pages_shm_if_needed();

  // Preallocation worker thread function
  void prealloc_worker();

  // Start/stop preallocation thread
  void start_prealloc_thread_internal();
  void stop_prealloc_thread(double timeout = PREALLOC_THREAD_TIMEOUT);

  // Trigger preallocation
  void trigger_preallocation();

  // Map/unmap virtual pages (broadcasts to workers in dp_rank group)
  // Returns false if broadcast fails
  bool map_virt_pages(const std::string& model_id,
                      int32_t dp_rank,
                      const std::vector<int64_t>& virt_page_ids);
  bool unmap_virt_pages(const std::string& model_id,
                        int32_t dp_rank,
                        const std::vector<int64_t>& virt_page_ids);

  // Update memory usage tracking
  void update_memory_usage();

  // Get model state (throws if not found)
  ModelState& get_model_state(const std::string& model_id);
  const ModelState& get_model_state(const std::string& model_id) const;

  // Initialization state
  bool initialized_ = false;

  // Configuration
  int32_t dp_size_ = 1;
  size_t page_size_ = 0;  // Page size (from FLAGS_phy_page_granularity_size)
  bool enable_page_prealloc_ = PAGE_PREALLOC_ENABLED;

  // Physical page tracking (shared across all models)
  size_t num_total_phy_pages_ = 0;  // Total physical pages per worker

  // Per-worker physical page tracking
  // Each worker has independent PhyPagePool with the same total pages.
  // worker_pages_used_[i] = total pages used by worker i (weight + KV cache)
  // This tracks both weight allocation (by model world_size) and
  // KV cache allocation (by DP group's workers)
  std::vector<size_t> worker_pages_used_;
  // Worker reported physical pages (from non-zero workers via shm/rpc).
  mutable std::vector<size_t> worker_reported_pages_used_;
  int32_t max_world_size_ =
      0;  // Maximum number of workers (from initial nnodes)

  int reported_phy_pages_shm_fd_ = -1;
  mutable uint64_t* reported_phy_pages_shm_ptr_ = nullptr;

  // Per-model state (key is model_id from options)
  std::unordered_map<std::string, ModelState> model_states_;
  std::unordered_map<std::string, std::unordered_map<int32_t, const BlockManager*>>
      prefix_block_managers_;

  // Reserved page limits
  int32_t min_reserved_pages_ = MIN_RESERVED_PAGES;
  int32_t max_reserved_pages_ = MAX_RESERVED_PAGES;

  // Threading
  mutable std::mutex mtx_;
  std::condition_variable cond_;
  std::atomic<bool> prealloc_running_{false};
  std::atomic<bool> prealloc_needed_{false};
  std::unique_ptr<std::thread> prealloc_thd_;

  // --- Async eviction thread (independent from prealloc_worker) ---
  // CAS flag: true while async_eviction_worker is executing Phase 1.
  // prealloc_worker sets this via CAS before signaling; cleared by worker.
  std::atomic<bool> eviction_in_progress_{false};
  // Separate condition variable exclusively for async_eviction_worker.
  // prealloc_worker notifies via notify_one(); never uses mtx_ for wait.
  std::condition_variable async_evict_cond_;
  // Protected by evict_mtx_; only used for async_evict_cond_.wait().
  std::mutex evict_mtx_;
  // Signals the async eviction worker: true = work to do.
  std::atomic<bool> eviction_needed_{false};
  std::atomic<bool> eviction_thd_running_{false};
  std::unique_ptr<std::thread> async_eviction_thd_;

  // Async eviction worker loop body.
  void async_eviction_worker();

  std::unordered_set<std::string> collect_models_on_pressure_workers(
      size_t low_watermark_pages) const;
  std::unordered_set<std::string> collect_models_on_pressure_workers(
      const std::unordered_set<int32_t> pressure_workers) const;
  std::vector<size_t> build_degraded_weight_resident_phy_pages_by_worker_locked(
      const std::string& normalized_base_model_id) const;
  std::vector<size_t> build_prefix_only_phy_pages_by_worker_locked() const;

  // Master-side layer offload manager (owned by PageAllocator)
  std::unique_ptr<LayerOffloadManager> layer_offload_mgr_;
};

}  // namespace xllm
