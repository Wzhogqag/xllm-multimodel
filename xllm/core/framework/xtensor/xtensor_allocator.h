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
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "options.h"
#include "phy_page.h"
#include "xtensor.h"
#include "xtensor_dist_client.h"
#include "xtensor_dist_server.h"

namespace xllm {

/**
 * Per-model tensor storage
 */
struct ModelTensors {
  // K tensors: one tensor per layer (indexed by layer id)
  std::vector<std::unique_ptr<XTensor>> k_tensors;
  // V tensors: one tensor per layer (indexed by layer id)
  std::vector<std::unique_ptr<XTensor>> v_tensors;
  int64_t num_layers = 0;
  size_t kv_tensor_size_per_layer = 0;

  // ============== Weight Allocation (from GlobalXTensor) ==============
  size_t weight_num_pages = 0;       // Number of pages pre-allocated
  void* weight_base_ptr = nullptr;   // Base virtual address
  size_t weight_current_offset = 0;  // Current allocation offset in bytes
  size_t weight_xtensor_offset = 0;  // Base offset inside global weight_xtensor
  bool weight_pages_reclaimable = false;

  // ============== Model-specific Parallel Strategy (for fork master)
  // ============== Each model may have different dp_size/tp_size, used in
  // broadcast operations to select correct workers. 0 means use global values.
  int32_t dp_size = 0;
  int32_t tp_size = 0;
  int32_t worker_rank_base = 0;

  // ============== Weight Segments (for D2D transfer) ==============
  // Ordered list of weight segments; offset is from Mooncake-registered weight
  // buffer base (weight_xtensor_).
  // For contiguous allocation: single segment.
  // For fallback (XTensor): multiple segments from non-contiguous pages.
  std::vector<WeightSegment> weight_segments;

  // Mooncake local segment buffers[] index for this model's registered weight
  // slice (set after register_memory for [weight_base_ptr,
  // num_pages*page_size)). -1 if not yet registered.
  int32_t mooncake_weight_buffer_index = -1;

  // Per-page reference count for weight pages (layer-sharing safety).
  std::vector<int32_t> weight_page_refcount;

  // Layer offload/load callbacks (registered by WorkerImpl, worker-local).
  std::function<int64_t(int32_t)> layer_offload_fn;
  std::function<int64_t(int32_t)> layer_load_fn;
  std::function<void()> layer_npu_sync_fn;
};

/**
 * XTensorAllocator manages XTensor objects for KV cache and model weights.
 *
 * This is a singleton class that:
 * - Creates and manages XTensor objects per model (indexed by model_id)
 * - Handles distributed XTensor operations via RPC
 * - Coordinates PhyPagePool initialization across workers
 */
class XTensorAllocator {
 public:
  enum class ActivationAllocPhase : uint8_t {
    kInit = 0,
    kRuntime = 1,
  };

  static void set_alloc_phase(ActivationAllocPhase phase);
  static ActivationAllocPhase get_alloc_phase();

  // Get the global singleton instance
  static XTensorAllocator& get_instance() {
    static XTensorAllocator instance;
    return instance;
  }

  // Initialize the allocator with device configuration
  void init(const torch::Device& device);

  // Check if initialized
  bool is_initialized() const { return initialized_; }

  // ============== KV Cache Interfaces ==============

  // Create K tensors for all layers of a model
  std::vector<torch::Tensor> create_k_tensors(const std::string& model_id,
                                              const std::vector<int64_t>& dims,
                                              torch::Dtype dtype,
                                              int64_t num_layers);

  // Create V tensors for all layers of a model
  std::vector<torch::Tensor> create_v_tensors(const std::string& model_id,
                                              const std::vector<int64_t>& dims,
                                              torch::Dtype dtype,
                                              int64_t num_layers);

  // KV tensor operations (partial mapping by offsets)
  bool map_to_kv_tensors(const std::string& model_id,
                         const std::vector<offset_t>& offsets);
  bool unmap_from_kv_tensors(const std::string& model_id,
                             const std::vector<offset_t>& offsets);

  // ============== Weight Allocation Interfaces ==============

  // Create the global weight XTensor (pool.num_total() * page_size) after
  // PhyPagePool is initialized. Idempotent. Required for Mooncake D2D on the
  // weight buffer.
  bool ensure_weight_xtensor_created();

  // Mooncake-registered weight region (same as weight_xtensor_ when present)
  void* weight_region_base_vaddr() const;
  size_t weight_region_total_size() const;
  size_t weight_region_page_size() const;
  bool is_weight_mooncake_registered() const;
  void set_weight_mooncake_registered(bool registered);

  int32_t get_model_mooncake_weight_buffer_index(
      const std::string& model_id) const;
  void set_model_mooncake_weight_buffer_index(const std::string& model_id,
                                              int32_t idx);

  // Called from WorkerImpl after MooncakeWeightTransfer is constructed; invoked
  // at the end of alloc_weight_pages_local when FLAGS_enable_xtensor is set.
  void set_mooncake_weight_register_fn(
      std::function<bool(const std::string&)> fn);

  // Allocate from pre-allocated weight region (called by model loader)
  // Increments offset within the pre-allocated region
  bool allocate_weight(const std::string& model_id, void*& ptr, size_t size);

  // Free weight allocation (called by sleep)
  // Returns the number of pages freed
  size_t free_weight_from_global_xtensor(const std::string& model_id);

  // Local helpers for RPC path
  bool alloc_weight_pages_local(const std::string& model_id, size_t num_pages);
  size_t mark_weight_pages_reclaimable(const std::string& model_id);
  // Reclaim weight pages if GlobalXTensor is short of pages
  size_t reclaim_weight_pages_if_needed(size_t target_pages = 0);

  // Unmap a contiguous weight region [ptr, ptr+size) for the model (no D2H).
  // Only unmap pages that are currently mapped; skip pages already reclaimed.
  // Returns the number of pages actually unmapped.
  size_t unmap_weight_region(const std::string& model_id,
                             void* ptr,
                             size_t size);

  // Ensure pages covering [ptr, ptr+size) are mapped (from pool), then return.
  // Only maps pages where reclaimed[page_idx]==true.
  // Returns: number of pages newly mapped; 0 if none needed; -1 on error.
  int64_t ensure_weight_pages_mapped_region(const std::string& model_id,
                                            void* ptr,
                                            size_t size);

  // ============== Multi-node Setup ==============

  // Multi-node XTensor dist setup (called by rank0 to connect to other workers)
  void setup_multi_node_xtensor_dist(const xtensor::Options& options,
                                     const std::string& master_node_addr,
                                     int32_t dp_size);

  // Initialize PhyPagePool on all workers
  int64_t init_phy_page_pools(double max_memory_utilization = 0.9,
                              int64_t max_cache_size = 0);

  // ============== Model Parallel Strategy ==============

  // Set model-specific parallel strategy (for fork master with different dp/tp)
  // This should be called before broadcast operations for the model
  void set_model_parallel_strategy(const std::string& model_id,
                                   int32_t dp_size,
                                   int32_t tp_size,
                                   int32_t worker_rank_base = 0);

  // Get model-specific parallel strategy (returns global values if not set)
  // Returns {dp_size, tp_size}
  std::pair<int32_t, int32_t> get_model_parallel_strategy(
      const std::string& model_id);
  int32_t get_model_worker_rank_base(const std::string& model_id);

  // ============== Broadcast Operations ==============

  // Broadcast KV tensor map/unmap to workers in a specific DP group
  bool broadcast_map_to_kv_tensors(const std::string& model_id,
                                   int32_t dp_rank,
                                   const std::vector<offset_t>& offsets);
  bool broadcast_unmap_from_kv_tensors(const std::string& model_id,
                                       int32_t dp_rank,
                                       const std::vector<offset_t>& offsets);

  // Broadcast weight pages allocation/free to all workers
  bool broadcast_alloc_weight_pages(const std::string& model_id,
                                    size_t num_pages);
  bool broadcast_free_weight_pages(const std::string& model_id);

  // Broadcast layer weight offload/load to all workers for a model.
  // Returns vector[worker_rank] = pages_changed (negative on failure).
  std::vector<int64_t> broadcast_offload_layer_weights(
      const std::string& model_id, int32_t layer_id);
  std::vector<int64_t> broadcast_load_layer_weights(
      const std::string& model_id, int32_t layer_id);

  // Register per-model layer offload/load callbacks (called by WorkerImpl).
  // offload_fn(layer_id) -> pages_freed; load_fn(layer_id) -> pages_allocated.
  // npu_sync_fn() is called before offload/load to flush the NPU stream.
  void register_layer_offload_callbacks(
      const std::string& model_id,
      std::function<int64_t(int32_t)> offload_fn,
      std::function<int64_t(int32_t)> load_fn,
      std::function<void()> npu_sync_fn);

  // Execute a local offload/load via registered callbacks (called by RPC service).
  int64_t local_offload_layer_weights(const std::string& model_id,
                                      int32_t layer_id);
  int64_t local_load_layer_weights(const std::string& model_id,
                                   int32_t layer_id);

  // Get XTensor dist clients (for distributed operations)
  const std::vector<std::shared_ptr<XTensorDistClient>>&
  get_xtensor_dist_clients() const {
    return xtensor_dist_clients_;
  }

  bool allocate_activation(void*& ptr, size_t size);
  bool deallocate_activation(void*& ptr);

  // Get device
  const torch::Device& device() const { return dev_; }

  // Get XTensor offsets for blocks via RPC (used by Engine in PD
  // disaggregation) Calls worker in the specified DP group to compute offsets
  // Parameters:
  //   dp_rank: Target DP rank (which DP group to query)
  //   model_id: Model identifier
  //   block_ids: Block IDs to get offsets for
  //   block_size_bytes: Size of each block in bytes
  //   layer_offsets: Output, layer_offsets[layer_id] = {k_offsets, v_offsets}
  // Returns: true on success
  bool get_xtensor_offsets(
      int32_t dp_rank,
      const std::string& model_id,
      const std::vector<int32_t>& block_ids,
      uint64_t block_size_bytes,
      std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>&
          layer_offsets);

  void enter_init_stage();

  bool exit_init_stage();
  // ============== PD Disaggregation Support (XTensor Mode) ==============

  // Convert a block_id to GlobalXTensor offsets for KV cache transfer.
  // This is only used when FLAGS_enable_xtensor is true for PD disaggregation.
  //
  // Parameters:
  //   model_id: Model identifier
  //   layer_id: Layer index
  //   block_id: Block ID within the KV cache
  //   block_size: Size of each block in bytes
  //
  // Returns: {k_offset, v_offset} relative to GlobalXTensor base address,
  //          or {UINT64_MAX, UINT64_MAX} on error.
  std::pair<uint64_t, uint64_t> get_global_offsets_for_block(
      const std::string& model_id,
      int64_t layer_id,
      int64_t block_id,
      size_t block_size);

  // Get model tensors (returns nullptr if not found)
  // Public for XTensorDistService to access num_layers
  ModelTensors* get_model_tensors(const std::string& model_id);
  // ============== ETCD Registration Support ==============
  // Get weight segments for a model (supports non-contiguous allocation)
  // Returns ordered list of {offset, size} with offset from Mooncake-registered
  // weight buffer base (weight_xtensor_).
  std::vector<WeightSegment> get_model_weight_segments(
      const std::string& model_id) const;

  // Get all model weight segments
  std::unordered_map<std::string, std::vector<WeightSegment>>
  get_all_model_weight_segments() const;

 private:
  XTensorAllocator() = default;
  ~XTensorAllocator();
  XTensorAllocator(const XTensorAllocator&) = delete;
  XTensorAllocator& operator=(const XTensorAllocator&) = delete;

  static uintptr_t align_up(uintptr_t addr, size_t page_size) {
    return ((addr + page_size - 1) / page_size) * page_size;
  }

  // Get or create model tensors (auto-creates if not exists)
  ModelTensors& get_or_create_model_tensors(const std::string& model_id);

  // Create K/V tensors implementation (handles lock and validation)
  // name: "K", "V", or future types like "index"
  std::vector<torch::Tensor> create_kv_tensors_impl_(
      const std::string& model_id,
      const std::vector<int64_t>& dims,
      torch::Dtype dtype,
      int64_t num_layers,
      const char* name);

  // Create tensors internal (must call with lock held)
  std::vector<torch::Tensor> create_tensors_internal_(
      size_t size,
      const std::vector<int64_t>& dims,
      torch::Dtype dtype,
      int64_t num_layers,
      std::vector<std::unique_ptr<XTensor>>& tensors_out);

  // Device initialization (platform-agnostic)
  void init_device_();

  // Caller must hold mtx_. Creates weight_xtensor_ if missing.
  bool ensure_weight_xtensor_created_locked();

  // Cleanup resources
  void destroy();

  void lazy_unmap_worker_loop_();

  bool initialized_ = false;
  torch::Device dev_{torch::kCPU};

  mutable std::mutex mtx_;
  // Protects activation-specific state. Lock order: activation_mtx_ before mtx_.
  mutable std::mutex activation_mtx_;
  // Serializes forward GlobalXTensor page-growth paths between activation
  // allocate_contiguous and KV map page-fault allocations.
  mutable std::mutex forward_alloc_mtx_;

  std::thread lazy_unmap_worker_;
  std::atomic<bool> lazy_unmap_stop_{false};

  // Per-model tensors storage (key: model_id)
  std::unordered_map<std::string, ModelTensors> model_tensors_;

  // Global weight virtual space (all model weights)
  std::unique_ptr<XTensor> weight_xtensor_;
  size_t weight_xtensor_next_free_offset_ = 0;
  bool weight_mooncake_registered_ = false;
  struct WeightReclaimItem {
    std::string model_id;
    size_t page_idx = 0;
  };
  std::deque<WeightReclaimItem> weight_reclaim_queue_;
  std::unordered_map<std::string, std::vector<bool>> weight_page_reclaimed_;
  static constexpr size_t kWeightReclaimWatermarkPages = 5000;
  double total_time = 0;

  std::function<bool(const std::string&)> mooncake_weight_register_fn_;

  // Zero page pointer (owned by PhyPagePool, not this class)
  PhyPage* zero_page_ = nullptr;

  // Multi-node XTensor dist members
  int32_t world_size_ = 0;  // total workers = dp_size * tp_size
  int32_t dp_size_ = 1;
  int32_t tp_size_ = 1;
  // DP group to worker clients mapping: dp_group_clients_[dp_rank][tp_rank]
  std::vector<std::vector<std::shared_ptr<XTensorDistClient>>>
      dp_group_clients_;
  // Flat list for backward compatibility and weight tensor broadcast
  std::vector<std::shared_ptr<XTensorDistClient>> xtensor_dist_clients_;
  // Master's XTensorDist server address (rank 0), for workers to report
  // consume/release
  std::string master_xtensor_dist_addr_;
  std::vector<std::unique_ptr<XTensorDistServer>> xtensor_dist_servers_;
  std::string collective_server_name_{"XTensorAllocatorCollectiveServer"};

  // Aligned size in bytes
  // 512B is the best practice to balance memory utilization and performance
  static constexpr size_t align_size = 512;

  size_t page_size_ = 0;

  void* activation_allocate_ptr = nullptr;
  void* init_activation_allocate_ptr_ = nullptr;

  size_t activation_allocated_pages = 0;  // Number of allocated pages
  // Track allocations for deallocation: ptr -> size
  std::unordered_map<void*, size_t> activation_allocated_ptrs_;
  // Track page reference counts: page_id -> allocation count
  std::unordered_map<size_t, size_t> page_refcount_;

  size_t wasted_space_ = 0;
  std::unordered_map<size_t, size_t> wasted_pages_;  // page_id -> wasted bytes
};

}  // namespace xllm
