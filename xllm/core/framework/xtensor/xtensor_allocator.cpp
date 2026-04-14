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

#include "xtensor_allocator.h"

#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "common/global_flags.h"
#include "common/macros.h"
#include "distributed_runtime/collective_service.h"
#include "global_xtensor.h"
#include "phy_page.h"
#include "phy_page_pool.h"
#include "platform/device.h"
#include "platform/vmm_api.h"
#include "server/xllm_server_registry.h"
#include "xtensor.h"

namespace xllm {

namespace {
thread_local XTensorAllocator::ActivationAllocPhase g_activation_alloc_phase =
    XTensorAllocator::ActivationAllocPhase::kRuntime;
}  // namespace

void XTensorAllocator::set_alloc_phase(ActivationAllocPhase phase) {
  g_activation_alloc_phase = phase;
}

XTensorAllocator::ActivationAllocPhase XTensorAllocator::get_alloc_phase() {
  return g_activation_alloc_phase;
}

XTensorAllocator::~XTensorAllocator() {
  if (!initialized_) {
    return;
  }

  // Stop collective server if running
  XllmServer* collective_server =
      ServerRegistry::get_instance().register_server(collective_server_name_);
  if (collective_server != nullptr) {
    collective_server->stop();
    ServerRegistry::get_instance().unregister_server(collective_server_name_);
  }

  destroy();
}

void XTensorAllocator::lazy_unmap_worker_loop_() {
  using namespace std::chrono_literals;
  while (!lazy_unmap_stop_.load(std::memory_order_relaxed)) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      reclaim_weight_pages_if_needed();
    }
    std::this_thread::sleep_for(100ms);
  }
}

void XTensorAllocator::destroy() {
  lazy_unmap_stop_.store(true, std::memory_order_relaxed);
  lazy_unmap_worker_.join();
  std::lock_guard<std::mutex> lock(mtx_);
  model_tensors_.clear();
  zero_page_ = nullptr;  // Not owned, just clear pointer
  weight_xtensor_.reset();
  weight_xtensor_next_free_offset_ = 0;
  weight_mooncake_registered_ = false;
  weight_reclaim_queue_.clear();
  weight_page_reclaimed_.clear();
  xtensor_dist_clients_.clear();
  xtensor_dist_servers_.clear();
  initialized_ = false;
}

void XTensorAllocator::init(const torch::Device& device) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (initialized_) {
    LOG(WARNING) << "XTensorAllocator already initialized, ignoring re-init";
    return;
  }

  dev_ = device;
  init_device_();
  page_size_ = FLAGS_phy_page_granularity_size;
  initialized_ = true;
  lazy_unmap_worker_ =
      std::thread(&XTensorAllocator::lazy_unmap_worker_loop_, this);
}

ModelTensors& XTensorAllocator::get_or_create_model_tensors(
    const std::string& model_id) {
  // Note: caller must hold mtx_
  auto it = model_tensors_.find(model_id);
  if (it == model_tensors_.end()) {
    model_tensors_[model_id] = ModelTensors{};
    VLOG(1) << "Auto-created model tensors entry for: " << model_id;
  }
  return model_tensors_[model_id];
}

ModelTensors* XTensorAllocator::get_model_tensors(const std::string& model_id) {
  // Note: caller must hold mtx_
  auto it = model_tensors_.find(model_id);
  if (it == model_tensors_.end()) {
    return nullptr;
  }
  return &it->second;
}

// ============== Multi-node Setup ==============

void XTensorAllocator::setup_multi_node_xtensor_dist(
    const xtensor::Options& options,
    const std::string& master_node_addr,
    int32_t dp_size) {
  const auto& devices = options.devices();
  const int32_t each_node_ranks = static_cast<int32_t>(devices.size());
  world_size_ = each_node_ranks * FLAGS_nnodes;
  dp_size_ = dp_size;
  tp_size_ = world_size_ / dp_size_;

  CHECK_EQ(world_size_ % dp_size_, 0)
      << "world_size must be divisible by dp_size";

  std::vector<std::atomic<bool>> dones(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    dones[i].store(false, std::memory_order_relaxed);
  }

  // Update collective server name with server index
  collective_server_name_ = "XTensorDistCollectiveServer";

  for (size_t i = 0; i < devices.size(); ++i) {
    // Create XTensor dist server for each device
    xtensor_dist_servers_.emplace_back(std::make_unique<XTensorDistServer>(
        i, master_node_addr, dones[i], devices[i], options));

    // Only rank0 connects to other workers
    if (FLAGS_node_rank == 0) {
      std::shared_ptr<CollectiveService> collective_service =
          std::make_shared<CollectiveService>(
              0, world_size_, devices[0].index());
      XllmServer* collective_server =
          ServerRegistry::get_instance().register_server(
              collective_server_name_);
      if (!collective_server->start(
              collective_service, master_node_addr, collective_server_name_)) {
        LOG(ERROR) << "failed to start collective server on address: "
                   << master_node_addr;
        return;
      }

      auto xtensor_dist_addrs_map = collective_service->wait();
      master_xtensor_dist_addr_ = xtensor_dist_addrs_map[0];

      // Initialize DP group clients mapping
      dp_group_clients_.resize(dp_size_);
      for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
        dp_group_clients_[dp_rank].reserve(tp_size_);
      }

      for (int32_t r = 0; r < world_size_; ++r) {
        if (xtensor_dist_addrs_map.find(r) == xtensor_dist_addrs_map.end()) {
          LOG(FATAL) << "Not all xtensor dist servers connect to master node. "
                        "Miss rank is "
                     << r;
          return;
        }
        auto client = std::make_shared<XTensorDistClient>(
            r, xtensor_dist_addrs_map[r], devices[r % each_node_ranks]);

        // Add to flat list
        xtensor_dist_clients_.emplace_back(client);

        // Add to DP group mapping
        // Workers are organized as: [dp0_tp0, dp0_tp1, ..., dp1_tp0, dp1_tp1,
        // ...]
        int32_t dp_rank = r / tp_size_;
        dp_group_clients_[dp_rank].emplace_back(client);
      }

      LOG(INFO) << "XTensor dist setup: world_size=" << world_size_
                << ", dp_size=" << dp_size_ << ", tp_size=" << tp_size_;
    }

    // Wait for all servers to be ready
    for (size_t idx = 0; idx < dones.size(); ++idx) {
      while (!dones[idx].load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
  }
}

int64_t XTensorAllocator::init_phy_page_pools(double max_memory_utilization,
                                              int64_t max_cache_size) {
  if (world_size_ <= 1) {
    // Single process single GPU, initialize locally
    Device device(dev_);
    device.set_device();

    const auto available_memory = device.free_memory();
    const auto total_memory = device.total_memory();

    int64_t cache_size = available_memory;
    if (max_memory_utilization < 1.0) {
      const int64_t buffer_memory =
          total_memory * (1.0 - max_memory_utilization);
      cache_size -= buffer_memory;
    }
    if (max_cache_size > 0) {
      cache_size = std::min(cache_size, max_cache_size);
    }

    int64_t num_pages = cache_size / FLAGS_phy_page_granularity_size;
    LOG(INFO) << "init_phy_page_pools (local): available_memory="
              << available_memory << ", total_memory=" << total_memory
              << ", cache_size=" << cache_size << ", num_pages=" << num_pages;

    PhyPagePool::get_instance().init(dev_, num_pages);

    // Initialize GlobalXTensor after PhyPagePool
    GlobalXTensor::get_instance().init(dev_);
    LOG(INFO) << "GlobalXTensor initialized (local)";

    if (!ensure_weight_xtensor_created()) {
      LOG(ERROR) << "ensure_weight_xtensor_created failed after pool init";
      return 0;
    }

    return num_pages;
  }

  // Step 1: Query available memory from all workers via RPC
  std::vector<folly::SemiFuture<MemoryInfo>> memory_futures;
  memory_futures.reserve(xtensor_dist_clients_.size());
  for (auto& client : xtensor_dist_clients_) {
    memory_futures.push_back(client->get_memory_info_async());
  }

  // Wait for all memory info responses
  auto memory_results = folly::collectAll(memory_futures).get();

  int64_t min_available_memory = std::numeric_limits<int64_t>::max();
  int64_t min_total_memory = std::numeric_limits<int64_t>::max();

  for (size_t i = 0; i < memory_results.size(); ++i) {
    if (!memory_results[i].hasValue()) {
      LOG(ERROR) << "Failed to get memory info from worker: " << i;
      return 0;
    }
    auto& info = memory_results[i].value();
    if (info.available_memory == 0 && info.total_memory == 0) {
      LOG(ERROR) << "Worker " << i << " returned invalid memory info";
      return 0;
    }

    LOG(INFO) << "Worker #" << i
              << ": available_memory=" << info.available_memory
              << ", total_memory=" << info.total_memory;

    min_available_memory =
        std::min(min_available_memory, info.available_memory);
    min_total_memory = std::min(min_total_memory, info.total_memory);
  }

  // Step 2: Calculate num_pages based on min available memory
  int64_t cache_size = min_available_memory;
  if (max_memory_utilization < 1.0) {
    const int64_t buffer_memory =
        min_total_memory * (1.0 - max_memory_utilization);
    cache_size -= buffer_memory;
  }
  if (max_cache_size > 0) {
    cache_size = std::min(cache_size, max_cache_size);
  }

  int64_t num_pages = cache_size / FLAGS_phy_page_granularity_size;
  LOG(INFO) << "init_phy_page_pools: min_available_memory="
            << min_available_memory << ", min_total_memory=" << min_total_memory
            << ", cache_size=" << cache_size << ", num_pages=" << num_pages;

  if (num_pages <= 0) {
    LOG(ERROR) << "Insufficient memory for PhyPagePool";
    return 0;
  }

  // Step 3: Broadcast InitPhyPagePool to all workers (pass master addr so
  // workers can report consume/release to master for PageAllocator)
  std::vector<folly::SemiFuture<bool>> init_futures;
  init_futures.reserve(xtensor_dist_clients_.size());
  const std::string* master_addr =
      (world_size_ > 1 && !master_xtensor_dist_addr_.empty())
          ? &master_xtensor_dist_addr_
          : nullptr;
  for (size_t i = 0; i < xtensor_dist_clients_.size(); ++i) {
    init_futures.push_back(xtensor_dist_clients_[i]->init_phy_page_pool_async(
        num_pages, master_addr, static_cast<int32_t>(i)));
  }

  // Wait for all init responses
  auto init_results = folly::collectAll(init_futures).get();
  for (size_t i = 0; i < init_results.size(); ++i) {
    if (!init_results[i].hasValue() || !init_results[i].value()) {
      LOG(ERROR) << "Failed to init PhyPagePool on worker: " << i;
      return 0;
    }
  }

  LOG(INFO) << "Successfully initialized PhyPagePool on all " << world_size_
            << " workers with " << num_pages << " pages each";
  return num_pages;
}

// ============== Model Parallel Strategy ==============

void XTensorAllocator::set_model_parallel_strategy(const std::string& model_id,
                                                   int32_t dp_size,
                                                   int32_t tp_size,
                                                   int32_t worker_rank_base) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto& tensors = get_or_create_model_tensors(model_id);
  tensors.dp_size = dp_size;
  tensors.tp_size = tp_size;
  tensors.worker_rank_base = worker_rank_base;
  if (world_size_ > 1) {
    CHECK_LE(worker_rank_base + dp_size * tp_size, world_size_)
        << "Model worker window out of range for model " << model_id;
  }
  LOG(INFO) << "Set model parallel strategy for " << model_id
            << ": dp_size=" << dp_size << ", tp_size=" << tp_size
            << ", worker_rank_base=" << worker_rank_base;
}

std::pair<int32_t, int32_t> XTensorAllocator::get_model_parallel_strategy(
    const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto* tensors = get_model_tensors(model_id);
  if (tensors && tensors->dp_size > 0 && tensors->tp_size > 0) {
    return {tensors->dp_size, tensors->tp_size};
  }
  // Fallback to global values
  return {dp_size_, tp_size_};
}

int32_t XTensorAllocator::get_model_worker_rank_base(
    const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto* tensors = get_model_tensors(model_id);
  if (tensors) {
    return std::max(0, tensors->worker_rank_base);
  }
  return 0;
}

// ============== Broadcast Operations ==============

bool XTensorAllocator::broadcast_map_to_kv_tensors(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<offset_t>& offsets) {
  if (world_size_ <= 1) {
    // Single process single GPU, just map locally
    return map_to_kv_tensors(model_id, offsets);
  }

  // Get model-specific parallel strategy
  auto [model_dp_size, model_tp_size] = get_model_parallel_strategy(model_id);

  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, model_dp_size) << "dp_rank must be < model_dp_size";

  // Calculate worker range for this DP group based on model's parallel strategy
  // Workers are organized as: [dp0_tp0, dp0_tp1, ..., dp1_tp0, dp1_tp1, ...]
  int32_t worker_rank_base = get_model_worker_rank_base(model_id);
  int32_t start_rank = worker_rank_base + dp_rank * model_tp_size;
  int32_t end_rank = start_rank + model_tp_size;

  // Broadcast to workers in this DP group via RPC asynchronously
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(model_tp_size);
  for (int32_t r = start_rank;
       r < end_rank && r < static_cast<int32_t>(xtensor_dist_clients_.size());
       ++r) {
    futures.push_back(
        xtensor_dist_clients_[r]->map_to_kv_tensors_async(model_id, offsets));
  }

  // Wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.hasValue() || !result.value()) {
      return false;
    }
  }
  return true;
}

bool XTensorAllocator::broadcast_unmap_from_kv_tensors(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<offset_t>& offsets) {
  if (world_size_ <= 1) {
    // Single process single GPU, just unmap locally
    return unmap_from_kv_tensors(model_id, offsets);
  }

  // Get model-specific parallel strategy
  auto [model_dp_size, model_tp_size] = get_model_parallel_strategy(model_id);

  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, model_dp_size) << "dp_rank must be < model_dp_size";

  // Calculate worker range for this DP group based on model's parallel strategy
  // Workers are organized as: [dp0_tp0, dp0_tp1, ..., dp1_tp0, dp1_tp1, ...]
  int32_t worker_rank_base = get_model_worker_rank_base(model_id);
  int32_t start_rank = worker_rank_base + dp_rank * model_tp_size;
  int32_t end_rank = start_rank + model_tp_size;

  // Broadcast to workers in this DP group via RPC asynchronously
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(model_tp_size);
  for (int32_t r = start_rank;
       r < end_rank && r < static_cast<int32_t>(xtensor_dist_clients_.size());
       ++r) {
    futures.push_back(xtensor_dist_clients_[r]->unmap_from_kv_tensors_async(
        model_id, offsets));
  }

  // Wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.hasValue() || !result.value()) {
      return false;
    }
  }
  return true;
}

bool XTensorAllocator::broadcast_alloc_weight_pages(const std::string& model_id,
                                                    size_t num_pages) {
  // Get model-specific parallel strategy
  auto [model_dp_size, model_tp_size] = get_model_parallel_strategy(model_id);
  int32_t worker_rank_base = get_model_worker_rank_base(model_id);
  int32_t model_world_size = model_dp_size * model_tp_size;

  if (world_size_ <= 1) {
    if (!alloc_weight_pages_local(model_id, num_pages)) {
      LOG(ERROR) << "Failed to allocate " << num_pages
                 << " weight pages locally";
      return false;
    }
    return true;
  }

  // Broadcast to all workers for this model
  std::vector<folly::SemiFuture<bool>> futures;
  int32_t num_workers = std::min(
      model_world_size, static_cast<int32_t>(xtensor_dist_clients_.size()));
  futures.reserve(num_workers);
  for (int32_t i = 0; i < num_workers; ++i) {
    int32_t actual_rank = worker_rank_base + i;
    futures.push_back(
        xtensor_dist_clients_[actual_rank]->alloc_weight_pages_async(
            model_id, num_pages));
  }

  // Wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.hasValue() || !result.value()) {
      LOG(ERROR) << "broadcast_alloc_weight_pages failed for model "
                 << model_id;
      return false;
    }
  }

  LOG(INFO) << "broadcast_alloc_weight_pages success: model=" << model_id
            << ", num_pages=" << num_pages << ", num_workers=" << num_workers;
  return true;
}

bool XTensorAllocator::broadcast_free_weight_pages(
    const std::string& model_id) {
  // Get model-specific parallel strategy
  auto [model_dp_size, model_tp_size] = get_model_parallel_strategy(model_id);
  int32_t worker_rank_base = get_model_worker_rank_base(model_id);
  int32_t model_world_size = model_dp_size * model_tp_size;

  if (world_size_ <= 1) {
    // Single process: free locally
    free_weight_from_global_xtensor(model_id);
    return true;
  }

  // Broadcast to all workers for this model
  std::vector<folly::SemiFuture<bool>> futures;
  int32_t num_workers = std::min(
      model_world_size, static_cast<int32_t>(xtensor_dist_clients_.size()));
  futures.reserve(num_workers);
  for (int32_t i = 0; i < num_workers; ++i) {
    int32_t actual_rank = worker_rank_base + i;
    futures.push_back(
        xtensor_dist_clients_[actual_rank]->free_weight_pages_async(model_id));
  }

  // Wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.hasValue() || !result.value()) {
      LOG(ERROR) << "broadcast_free_weight_pages failed for model " << model_id;
      return false;
    }
  }

  LOG(INFO) << "broadcast_free_weight_pages success: model=" << model_id
            << ", num_workers=" << num_workers;
  return true;
}

// ============== Layer Offload/Load Broadcast ==============

std::vector<int64_t> XTensorAllocator::broadcast_offload_layer_weights(
    const std::string& model_id, int32_t layer_id) {
  auto [model_dp_size, model_tp_size] = get_model_parallel_strategy(model_id);
  int32_t worker_rank_base = get_model_worker_rank_base(model_id);
  int32_t model_world_size = model_dp_size * model_tp_size;

  if (world_size_ <= 1) {
    int64_t pages = local_offload_layer_weights(model_id, layer_id);
    return {pages};
  }

  int32_t num_workers = std::min(
      model_world_size, static_cast<int32_t>(xtensor_dist_clients_.size()));
  std::vector<folly::SemiFuture<int64_t>> futures;
  futures.reserve(num_workers);
  for (int32_t i = 0; i < num_workers; ++i) {
    int32_t actual_rank = worker_rank_base + i;
    LOG(INFO) << "broadcast_offload_layer_weights: " << "layer_id=" << layer_id << ", worker_rank=" << actual_rank;
    futures.push_back(
        xtensor_dist_clients_[actual_rank]->offload_layer_weights_async(
            model_id, layer_id));
  }

  auto results = folly::collectAll(futures).get();
  std::vector<int64_t> pages_per_worker(num_workers, -1);
  for (int32_t i = 0; i < num_workers; ++i) {
    if (results[i].hasValue()) {
      pages_per_worker[i] = results[i].value();
    }
  }
  return pages_per_worker;
}

std::vector<int64_t> XTensorAllocator::broadcast_load_layer_weights(
    const std::string& model_id, int32_t layer_id) {
  auto [model_dp_size, model_tp_size] = get_model_parallel_strategy(model_id);
  int32_t worker_rank_base = get_model_worker_rank_base(model_id);
  int32_t model_world_size = model_dp_size * model_tp_size;

  if (world_size_ <= 1) {
    int64_t pages = local_load_layer_weights(model_id, layer_id);
    return {pages};
  }

  int32_t num_workers = std::min(
      model_world_size, static_cast<int32_t>(xtensor_dist_clients_.size()));
  std::vector<folly::SemiFuture<int64_t>> futures;
  futures.reserve(num_workers);
  for (int32_t i = 0; i < num_workers; ++i) {
    int32_t actual_rank = worker_rank_base + i;
    futures.push_back(
        xtensor_dist_clients_[actual_rank]->load_layer_weights_async(
            model_id, layer_id));
  }

  auto results = folly::collectAll(futures).get();
  std::vector<int64_t> pages_per_worker(num_workers, -1);
  for (int32_t i = 0; i < num_workers; ++i) {
    if (results[i].hasValue()) {
      pages_per_worker[i] = results[i].value();
    }
  }
  return pages_per_worker;
}

void XTensorAllocator::register_layer_offload_callbacks(
    const std::string& model_id,
    std::function<int64_t(int32_t)> offload_fn,
    std::function<int64_t(int32_t)> load_fn,
    std::function<void()> npu_sync_fn) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto& tensors = get_or_create_model_tensors(model_id);
  tensors.layer_offload_fn = std::move(offload_fn);
  tensors.layer_load_fn = std::move(load_fn);
  tensors.layer_npu_sync_fn = std::move(npu_sync_fn);
  LOG(INFO) << "[XTensorAllocator] Registered layer offload callbacks for "
            << model_id;
}

int64_t XTensorAllocator::local_offload_layer_weights(
    const std::string& model_id, int32_t layer_id) {
  std::function<int64_t(int32_t)> offload_fn;
  std::function<void()> sync_fn;
  {
    std::lock_guard<std::mutex> lock(mtx_);
    auto* tensors = get_model_tensors(model_id);
    if (!tensors || !tensors->layer_offload_fn) {
      LOG(ERROR) << "[XTensorAllocator] No offload callback for " << model_id;
      return -1;
    }
    offload_fn = tensors->layer_offload_fn;
    sync_fn = tensors->layer_npu_sync_fn;
  }
  if (sync_fn) sync_fn();
  return offload_fn(layer_id);
}

int64_t XTensorAllocator::local_load_layer_weights(
    const std::string& model_id, int32_t layer_id) {
  std::function<int64_t(int32_t)> load_fn;
  std::function<void()> sync_fn;
  {
    std::lock_guard<std::mutex> lock(mtx_);
    auto* tensors = get_model_tensors(model_id);
    if (!tensors || !tensors->layer_load_fn) {
      LOG(ERROR) << "[XTensorAllocator] No load callback for " << model_id;
      return -1;
    }
    load_fn = tensors->layer_load_fn;
    sync_fn = tensors->layer_npu_sync_fn;
  }
  if (sync_fn) sync_fn();
  return load_fn(layer_id);
}

// ============== KV Cache Interfaces ==============

std::vector<torch::Tensor> XTensorAllocator::create_k_tensors(
    const std::string& model_id,
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers) {
  return create_kv_tensors_impl_(model_id, dims, dtype, num_layers, "K");
}

std::vector<torch::Tensor> XTensorAllocator::create_v_tensors(
    const std::string& model_id,
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers) {
  return create_kv_tensors_impl_(model_id, dims, dtype, num_layers, "V");
}

std::vector<torch::Tensor> XTensorAllocator::create_kv_tensors_impl_(
    const std::string& model_id,
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers,
    const char* name) {
  std::lock_guard<std::mutex> lock(mtx_);

  // Get or create model tensors entry
  auto& model = get_or_create_model_tensors(model_id);

  // Select target tensors based on name
  std::vector<std::unique_ptr<XTensor>>* target_tensors = nullptr;
  if (strcmp(name, "K") == 0) {
    target_tensors = &model.k_tensors;
  } else if (strcmp(name, "V") == 0) {
    target_tensors = &model.v_tensors;
  } else {
    LOG(FATAL) << "Unknown tensor name: " << name;
  }

  CHECK(model.num_layers == 0 || model.num_layers == num_layers)
      << "Number of layers mismatch for model " << model_id;
  CHECK(target_tensors->empty())
      << name << " tensors already created for model " << model_id;
  CHECK(!dims.empty()) << name << " tensor dims cannot be empty";

  // Calculate size from dims and dtype
  size_t size = torch::scalarTypeToTypeMeta(dtype).itemsize();
  for (auto dim : dims) {
    size *= dim;
  }

  size_t page_size = FLAGS_phy_page_granularity_size;
  // Align size to page size (round up)
  if (size % page_size != 0) {
    size_t aligned_size = ((size + page_size - 1) / page_size) * page_size;
    LOG(WARNING) << name << " tensor size " << size
                 << " is not aligned to page size " << page_size
                 << ", aligning to " << aligned_size;
    size = aligned_size;
  }

  model.num_layers = num_layers;
  model.kv_tensor_size_per_layer = size;

  if (!zero_page_) {
    zero_page_ = PhyPagePool::get_instance().get_zero_page();
  }

  return create_tensors_internal_(
      size, dims, dtype, num_layers, *target_tensors);
}

bool XTensorAllocator::map_to_kv_tensors(const std::string& model_id,
                                         const std::vector<offset_t>& offsets) {
  std::lock_guard<std::mutex> lock(mtx_);

  auto* tensors = get_model_tensors(model_id);
  if (!tensors) {
    LOG(ERROR) << "Model " << model_id << " not found";
    return false;
  }

  if (tensors->k_tensors.empty() || tensors->v_tensors.empty()) {
    LOG(ERROR) << "KV tensors not created for model " << model_id;
    return false;
  }

  // Per-layer mapping for K and V tensors separately.
  // Keep original mtx_ protection for tensor container lifetime and ordering,
  // and only serialize around map() itself for GlobalXTensor forward growth.
  std::vector<std::unique_ptr<PhyPage>> pages;
  {
    std::lock_guard<std::mutex> alloc_lock(forward_alloc_mtx_);
    pages = PhyPagePool::get_instance().batch_get(offsets.size() * tensors->num_layers * 2);
  }
  for (int64_t i = 0; i < tensors->num_layers; i++) {
    auto k_xtensor = tensors->k_tensors[i].get();
    auto v_xtensor = tensors->v_tensors[i].get();
    for (int64_t j = 0; j < offsets.size(); j++) {
      k_xtensor->map_external_page(offsets[j], std::move(pages[i * 2 * offsets.size() + j * 2]));
      v_xtensor->map_external_page(offsets[j], std::move(pages[i * 2 * offsets.size() + j * 2 + 1]));
    }
  }

  return true;
}

bool XTensorAllocator::unmap_from_kv_tensors(
    const std::string& model_id,
    const std::vector<offset_t>& offsets) {
  std::lock_guard<std::mutex> lock(mtx_);

  auto* tensors = get_model_tensors(model_id);
  if (!tensors) {
    LOG(ERROR) << "Model " << model_id << " not found";
    return false;
  }

  if (tensors->k_tensors.empty() || tensors->v_tensors.empty()) {
    LOG(ERROR) << "try to unmap from KV tensors when KV tensors are not created"
               << " for model " << model_id;
    return false;
  }

  // Per-layer unmapping for K and V tensors separately
  for (int64_t i = 0; i < tensors->num_layers; i++) {
    auto k_xtensor = tensors->k_tensors[i].get();
    auto v_xtensor = tensors->v_tensors[i].get();
    for (auto offset : offsets) {
      k_xtensor->unmap(offset);
      v_xtensor->unmap(offset);
    }
  }

  return true;
}

// 这个版本是中间激活独占GlobalXTensor
bool XTensorAllocator::allocate_activation(void*& ptr, size_t size) {
  std::lock_guard<std::mutex> lock(activation_mtx_);

  CHECK(size > 0);
  size = (size + align_size - 1) & ~(align_size - 1);

  //LOG(INFO) << "[Activation]: allocate " << size << "Byte";

  auto& pool = PhyPagePool::get_instance();
  auto& global_xtensor = GlobalXTensor::get_instance();
  ActivationAllocPhase phase = get_alloc_phase();

  size_t activation_allocate_offset = 0;

  if (phase == ActivationAllocPhase::kInit) {
    if (init_activation_allocate_ptr_ == nullptr) {
      init_activation_allocate_ptr_ = global_xtensor.init_activation_allocate_ptr();
    }

    activation_allocate_offset =
        reinterpret_cast<uintptr_t>(init_activation_allocate_ptr_);
    size_t init_current_offset = activation_allocate_offset % page_size_;
    size_t size_remaining = 0;
    if (init_current_offset != 0) {
      size_remaining = page_size_ - init_current_offset;
    }

    size_t num_extra = 0;
    if (size_remaining < size) {
      num_extra = (size - size_remaining - 1) / page_size_ + 1;
    }

    if (num_extra > 0) {
      std::lock_guard<std::mutex> alloc_lock(forward_alloc_mtx_);
      void* mapped_ptr = pool.allocate_contiguous(num_extra, true, true);
      size_t expected_page_aligned =
          (activation_allocate_offset + page_size_ - 1) / page_size_ * page_size_;
      CHECK_EQ(reinterpret_cast<uintptr_t>(mapped_ptr), expected_page_aligned)
          << "Init arena mapping is not contiguous, expected="
          << reinterpret_cast<void*>(expected_page_aligned)
          << ", actual=" << mapped_ptr;
      activation_allocated_pages += num_extra;
    }

    ptr = init_activation_allocate_ptr_;
    init_activation_allocate_ptr_ =
        reinterpret_cast<void*>(activation_allocate_offset + size);
  } else {
    if (activation_allocate_ptr == nullptr) {
      activation_allocate_ptr = global_xtensor.activation_allocate_ptr();
    }
  
    activation_allocate_offset =
        reinterpret_cast<uintptr_t>(activation_allocate_ptr);
  
    size_t activation_current_offset = activation_allocate_offset % page_size_;
    size_t size_remaining;
    if (activation_current_offset == 0) {
      size_remaining = 0;
    } else {
      size_remaining = page_size_ - activation_current_offset;
    }
  
    size_t num_extra;
    if (size_remaining >= size) {
      num_extra = 0;
    } else {
      num_extra = (size - size_remaining - 1) / page_size_ + 1;
    }
  
    if (num_extra > 0) {
      // Call allocate_contiguous to inform pool to allocate additional physical
      // pages
      {
        std::lock_guard<std::mutex> alloc_lock(forward_alloc_mtx_);
        void* allocated_ptr = pool.allocate_contiguous(num_extra, true, false);
        // 分配的指针不连续，导致了页内尾部碎片
        if (reinterpret_cast<uintptr_t>(allocated_ptr) !=
            (activation_allocate_offset + page_size_ - 1) / page_size_ *
                page_size_) {
          /*
          if(activation_current_offset > 0){
            wasted_space_ += page_size_ - activation_current_offset;
            wasted_pages_[activation_allocate_offset / page_size_] = page_size_ - activation_current_offset;
            LOG(INFO) << wasted_space_ << " bytes wasted due to fragmentation";
          }
          */
          // 存在因为尾部碎片未利用，分配的物理页少了一页的风险，我们需要检查是否需要追加分配
          // 该追加分配与上一段分配必须连续，因此两次分配在同一临界区内完成
          if (num_extra * page_size_ < size) {
            pool.allocate_contiguous(1, true, false);
            num_extra++;
          }
          activation_allocate_ptr = allocated_ptr;
          activation_allocate_offset =
              reinterpret_cast<uintptr_t>(activation_allocate_ptr);
        }
      }
      activation_allocated_pages += num_extra;
    }
  
    // Allocate from current offset
    ptr = activation_allocate_ptr;
    activation_allocate_ptr =
        reinterpret_cast<void*>(activation_allocate_offset + size);
  }

  

  activation_allocated_ptrs_[ptr] = size;

  // Update page reference counts
  size_t start_page = reinterpret_cast<uintptr_t>(ptr) / page_size_;
  size_t end_page = (activation_allocate_offset + size - 1) / page_size_;
  page_refcount_[start_page]++;
  if (start_page != end_page) {
    page_refcount_[end_page]++;
  }
  return true;
}

bool XTensorAllocator::deallocate_activation(void*& ptr) {
  std::lock_guard<std::mutex> lock(activation_mtx_);

  // Find the allocation record
  auto it = activation_allocated_ptrs_.find(ptr);
  if (it == activation_allocated_ptrs_.end()) {
    LOG(ERROR) << "deallocate_activation: ptr not found in allocated_ptrs_";
    return false;
  }

  size_t alloc_size = it->second;
  activation_allocated_ptrs_.erase(it);

  //LOG(INFO) << "[Activation]: free " << alloc_size << "Byte";

  // Calculate pages this allocation spans
  size_t start_page = reinterpret_cast<uintptr_t>(ptr) / page_size_;
  size_t end_page =
      (reinterpret_cast<uintptr_t>(ptr) + alloc_size - 1) / page_size_;
  size_t page_num = end_page - start_page + 1;
  std::vector<size_t> pages_to_check;
  if (page_num == 1) {
    pages_to_check = {start_page};
  } else {
    pages_to_check = {start_page, end_page};
  }
  auto& pool = PhyPagePool::get_instance();
  if (page_num > 2) {
    pool.free_contiguous((start_page + 1) * page_size_, page_num - 2);
    activation_allocated_pages -= page_num - 2;
  }
  //  Decrease reference count and check for full deallocation
  for (auto page : pages_to_check) {
    page_refcount_[page]--;
    if (page_refcount_[page] == 0 &&
        page !=
            reinterpret_cast<uintptr_t>(activation_allocate_ptr) / page_size_ &&
        page !=
            reinterpret_cast<uintptr_t>(init_activation_allocate_ptr_) / page_size_) {
      if (wasted_pages_.find(page) != wasted_pages_.end()) {
        wasted_space_ -= wasted_pages_[page];
        wasted_pages_.erase(page);
      }
      size_t page_addr = page * page_size_;
      pool.free_contiguous(page_addr, 1);
      activation_allocated_pages--;
    }
  }

  return true;
}

void XTensorAllocator::enter_init_stage() { 
  set_alloc_phase(ActivationAllocPhase::kInit);
}

// TODO: correct this for multimodel scenario,
// currently the virtual address between models is not reused when loop back
bool XTensorAllocator::exit_init_stage() {
  set_alloc_phase(ActivationAllocPhase::kRuntime);
  LOG(INFO) << activation_allocated_pages;
  LOG(INFO) << align_up(
      reinterpret_cast<uintptr_t>(init_activation_allocate_ptr_), page_size_) 
      / page_size_;
  return true;
}

bool XTensorAllocator::allocate_weight(const std::string& model_id,
                                       void*& ptr,
                                       size_t size) {
  std::lock_guard<std::mutex> lock(mtx_);

  auto* tensors = get_model_tensors(model_id);
  if (!tensors || tensors->weight_base_ptr == nullptr) {
    LOG(ERROR) << "No pre-allocated weight region for model " << model_id;
    return false;
  }

  // Bump within pre-mapped weight pages (weight_xtensor_ slice for this model)
  size_t region_size = tensors->weight_num_pages * page_size_;

  // Check if there's enough space in pre-allocated region
  if (tensors->weight_current_offset + size > region_size) {
    LOG(ERROR) << "Not enough space in weight region for model " << model_id
               << ": requested " << size << ", available "
               << (region_size - tensors->weight_current_offset);
    return false;
  }

  // Allocate from base + current offset
  ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(tensors->weight_base_ptr) +
      tensors->weight_current_offset);

  tensors->weight_current_offset += size;

  // Increment per-page refcount for all pages covered by this allocation.
  if (!tensors->weight_page_refcount.empty()) {
    size_t alloc_start = tensors->weight_current_offset - size;
    size_t first_page = alloc_start / page_size_;
    size_t last_page = (alloc_start + size - 1) / page_size_;
    for (size_t p = first_page;
         p <= last_page && p < tensors->weight_page_refcount.size(); ++p) {
      tensors->weight_page_refcount[p]++;
    }
  }

  VLOG(1) << "XTensorAllocator: allocated " << size << " bytes for model "
          << model_id << ", ptr=" << ptr;

  return true;
}

// ============== Internal Helpers ==============

std::vector<torch::Tensor> XTensorAllocator::create_tensors_internal_(
    size_t size,
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers,
    std::vector<std::unique_ptr<XTensor>>& tensors_out) {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(num_layers);
  tensors_out.reserve(num_layers);

  for (int64_t i = 0; i < num_layers; i++) {
    auto xtensor = std::make_unique<XTensor>(size, dtype, dev_, zero_page_);
    tensors.push_back(xtensor->to_torch_tensor(0, dims));
    tensors_out.push_back(std::move(xtensor));
  }
  return tensors;
}

void XTensorAllocator::init_device_() {
  Device device(dev_);
  device.set_device();
  device.init_device_context();

  // Create a dummy PhyPage to initialize the granularity size
  // This will set FLAGS_phy_page_granularity_size
  auto dummy_page = std::make_shared<PhyPage>(dev_);

  size_t chunk_sz = FLAGS_phy_page_granularity_size;
  LOG(INFO) << "Device initialized with granularity size: " << chunk_sz
            << " bytes";
}

bool XTensorAllocator::ensure_weight_xtensor_created() {
  std::lock_guard<std::mutex> lock(mtx_);
  return ensure_weight_xtensor_created_locked();
}

bool XTensorAllocator::ensure_weight_xtensor_created_locked() {
  auto& pool = PhyPagePool::get_instance();
  if (!pool.is_initialized()) {
    LOG(ERROR) << "ensure_weight_xtensor_created: PhyPagePool not initialized";
    return false;
  }
  if (weight_xtensor_) {
    const size_t expected = pool.num_total() * page_size_;
    CHECK_EQ(weight_xtensor_->size(), expected)
        << "weight_xtensor size mismatch vs PhyPagePool";
    return true;
  }
  const size_t weight_vsize = pool.num_total() * page_size_;
  weight_xtensor_ = std::make_unique<XTensor>(
      weight_vsize, torch::kUInt8, dev_, pool.get_zero_page());
  weight_xtensor_next_free_offset_ = 0;
  LOG(INFO) << "[XTensorAllocator] Weight xtensor created (pool init), size="
            << weight_vsize << ", vaddr=" << weight_xtensor_->vaddr();
  return true;
}

void* XTensorAllocator::weight_region_base_vaddr() const {
  std::lock_guard<std::mutex> lock(mtx_);
  if (!weight_xtensor_) {
    return nullptr;
  }
  return vir_ptr_to_void_ptr(weight_xtensor_->vaddr());
}

size_t XTensorAllocator::weight_region_total_size() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return weight_xtensor_ ? weight_xtensor_->size() : 0;
}

size_t XTensorAllocator::weight_region_page_size() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return weight_xtensor_ ? weight_xtensor_->page_size() : page_size_;
}

bool XTensorAllocator::is_weight_mooncake_registered() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return weight_mooncake_registered_;
}

void XTensorAllocator::set_weight_mooncake_registered(bool registered) {
  std::lock_guard<std::mutex> lock(mtx_);
  weight_mooncake_registered_ = registered;
}

int32_t XTensorAllocator::get_model_mooncake_weight_buffer_index(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_tensors_.find(model_id);
  if (it == model_tensors_.end()) {
    return -1;
  }
  return it->second.mooncake_weight_buffer_index;
}

void XTensorAllocator::set_model_mooncake_weight_buffer_index(
    const std::string& model_id,
    int32_t idx) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_tensors_.find(model_id);
  if (it == model_tensors_.end()) {
    LOG(ERROR) << "set_model_mooncake_weight_buffer_index: unknown model "
               << model_id;
    return;
  }
  it->second.mooncake_weight_buffer_index = idx;
}

void XTensorAllocator::set_mooncake_weight_register_fn(
    std::function<bool(const std::string&)> fn) {
  std::lock_guard<std::mutex> lock(mtx_);
  mooncake_weight_register_fn_ = std::move(fn);
}

bool XTensorAllocator::alloc_weight_pages_local(const std::string& model_id,
                                                size_t num_pages) {
  std::unique_lock<std::mutex> lock(mtx_);

  auto& tensors = get_or_create_model_tensors(model_id);
  tensors.weight_pages_reclaimable = false;

  CHECK_GT(num_pages, 0);

  auto& pool = PhyPagePool::get_instance();
  CHECK(pool.is_initialized()) << "PhyPagePool must be initialized";

  if (!weight_xtensor_) {
    size_t weight_vsize = pool.num_total() * page_size_;
    weight_xtensor_ = std::make_unique<XTensor>(
        weight_vsize, torch::kUInt8, dev_, pool.get_zero_page());
    weight_xtensor_next_free_offset_ = 0;
  }

  if (!ensure_weight_xtensor_created_locked()) {
    LOG(ERROR) << "alloc_weight_pages_local: weight xtensor not available";
    return false;
  }
  if (tensors.weight_num_pages > 0 && tensors.weight_num_pages != num_pages) {
    LOG(ERROR) << "weight page count mismatch for model " << model_id
               << ", existing=" << tensors.weight_num_pages
               << ", requested=" << num_pages;
    return false;
  }
  size_t bytes = num_pages * page_size_;
  if (tensors.weight_num_pages == 0) {
    if (weight_xtensor_next_free_offset_ + bytes > weight_xtensor_->size()) {
      LOG(ERROR)
          << "[XTensorAllocator] Not enough space in weight region for model "
          << model_id << ", requested=" << num_pages << ", available="
          << weight_xtensor_->size() - weight_xtensor_next_free_offset_;
      return false;
    }
    tensors.weight_num_pages = num_pages;
    tensors.weight_xtensor_offset = weight_xtensor_next_free_offset_;
    tensors.weight_base_ptr = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(weight_xtensor_->vaddr()) +
        tensors.weight_xtensor_offset);
    tensors.weight_current_offset = 0;
    weight_xtensor_next_free_offset_ += bytes;
    weight_page_reclaimed_[model_id] = std::vector<bool>(num_pages, true);
    tensors.weight_page_refcount.assign(num_pages, 0);

    tensors.weight_segments.clear();
    tensors.weight_segments.push_back(
        {static_cast<uint64_t>(tensors.weight_xtensor_offset),
         static_cast<uint64_t>(num_pages) * page_size_});
  }
  LOG(INFO) << "[XTensorAllocator] Allocated weight pages for model "
            << model_id << ", num_pages=" << num_pages
            << ", offset=" << tensors.weight_xtensor_offset
            << ", base_ptr=" << tensors.weight_base_ptr << ", bytes=" << bytes;

  auto& reclaimed = weight_page_reclaimed_[model_id];
  if (reclaimed.size() != tensors.weight_num_pages) {
    reclaimed.assign(tensors.weight_num_pages, true);
  }
  std::vector<size_t> need_map_idx;
  need_map_idx.reserve(tensors.weight_num_pages);
  for (size_t i = 0; i < tensors.weight_num_pages; ++i) {
    if (reclaimed[i]) {
      need_map_idx.push_back(i);
    }
  }
  std::vector<std::unique_ptr<PhyPage>> pages;
  {
    std::lock_guard<std::mutex> alloc_lock(forward_alloc_mtx_);
    pages = pool.batch_get(need_map_idx.size());
    if (pages.size() != need_map_idx.size()) {
      LOG(ERROR) << "Failed to batch_get weight pages, requested="
                 << need_map_idx.size() << ", got=" << pages.size();
      return false;
    }
  }
  for (size_t i = 0; i < need_map_idx.size(); ++i) {
    size_t page_idx = need_map_idx[i];
    size_t off = tensors.weight_xtensor_offset + page_idx * page_size_;
    if (!weight_xtensor_->map_external_page(static_cast<offset_t>(off),
                                            std::move(pages[i]))) {
      LOG(ERROR) << "Failed to map weight page at offset " << off;
      return false;
    }
    reclaimed[page_idx] = false;
  }
  tensors.weight_current_offset = 0;

  // Populate weight_segments for D2D transfer support
  tensors.weight_segments.clear();
  tensors.weight_segments.push_back ( {vir_ptr_to_uintptr(tensors.weight_base_ptr),
      static_cast<uint64_t>(num_pages) * page_size_});
  LOG(INFO) << "XTensorAllocator: populated weight_segments for model "
            << model_id << ", num_pages=" << num_pages
            << ", base_ptr=" << tensors.weight_base_ptr;

#if defined(USE_NPU)
  if (FLAGS_enable_xtensor && mooncake_weight_register_fn_) {
    lock.unlock();
    if (!mooncake_weight_register_fn_(model_id)) {
      LOG(ERROR) << "Mooncake register_model_weight_slice failed for model "
                 << model_id;
      return false;
    }
    lock.lock();
  }
#endif

  return true;
}

size_t XTensorAllocator::reclaim_weight_pages_if_needed(size_t target_pages) {
  if (!weight_xtensor_) {
    return 0;
  }
  auto& global = GlobalXTensor::get_instance();
  if (!global.is_initialized()) {
    return 0;
  }

  size_t reclaimed_pages = 0;
  auto need_reclaim = [&]() {
    if (target_pages > 0) {
      return reclaimed_pages < target_pages;
    }
    size_t free_bytes = 0;
    if (global.free_offset() > global.allocate_offset()) {
      free_bytes = global.free_offset() - global.allocate_offset();
    }
    return free_bytes <= (kWeightReclaimWatermarkPages * page_size_);
  };

  auto& pool = PhyPagePool::get_instance();
  while (need_reclaim() && !weight_reclaim_queue_.empty()) {
    auto item = weight_reclaim_queue_.front();
    weight_reclaim_queue_.pop_front();

    auto* tensors = get_model_tensors(item.model_id);
    if (!tensors || !tensors->weight_pages_reclaimable ||
        item.page_idx >= tensors->weight_num_pages) {
      continue;
    }
    auto it = weight_page_reclaimed_.find(item.model_id);
    if (it == weight_page_reclaimed_.end() ||
        item.page_idx >= it->second.size() || it->second[item.page_idx]) {
      continue;
    }

    size_t page_offset =
        tensors->weight_xtensor_offset + item.page_idx * page_size_;
    auto page =
        weight_xtensor_->unmap_and_take(static_cast<offset_t>(page_offset));
    if (!page) {
      it->second[item.page_idx] = true;
      continue;
    }

    std::vector<std::unique_ptr<PhyPage>> one;
    one.push_back(std::move(page));
    pool.batch_put(one);
    if (!weight_xtensor_) {
      break;
    }
    auto it2 = weight_page_reclaimed_.find(item.model_id);
    if (it2 == weight_page_reclaimed_.end() ||
        item.page_idx >= it2->second.size()) {
      continue;
    }
    it2->second[item.page_idx] = true;
    reclaimed_pages++;
  }

  return reclaimed_pages;
}

size_t XTensorAllocator::unmap_weight_region(const std::string& model_id,
                                             void* ptr,
                                             size_t size) {
  if (ptr == nullptr || size == 0) {
    return 0;
  }
  LOG(INFO) << "unmap_weight_region: model=" << model_id << ", size=" << size;
  std::unique_lock<std::mutex> lock(mtx_);
  ModelTensors* tensors = get_model_tensors(model_id);
  if (!tensors || !weight_xtensor_ || tensors->weight_base_ptr == nullptr ||
      tensors->weight_num_pages == 0) {
    return 0;
  }
  const uintptr_t base = reinterpret_cast<uintptr_t>(tensors->weight_base_ptr);
  const uintptr_t region_end = base + tensors->weight_num_pages * page_size_;
  const uintptr_t ptr_val = reinterpret_cast<uintptr_t>(ptr);
  if (ptr_val < base || ptr_val >= region_end) {
    LOG(ERROR) << "unmap_weight_region: ptr not in model weight region";
    return 0;
  }
  size_t offset_in_region = ptr_val - base;
  size_t start_page_idx = offset_in_region / page_size_;
  size_t end_page_idx =
      (offset_in_region + size + page_size_ - 1) / page_size_;
  size_t num_pages = end_page_idx - start_page_idx;
  if (end_page_idx > tensors->weight_num_pages) {
    LOG(ERROR) << "unmap_weight_region: [ptr,ptr+size) exceeds weight region";
    return 0;
  }
  auto it = weight_page_reclaimed_.find(model_id);
  if (it == weight_page_reclaimed_.end() ||
      it->second.size() != tensors->weight_num_pages) {
    LOG(ERROR)
        << "unmap_weight_region: reclaimed bitmap missing or size mismatch";
    return 0;
  }
  std::vector<bool>& reclaimed = it->second;
  std::vector<std::unique_ptr<PhyPage>> pages_to_put;
  size_t unmapped = 0;
  for (size_t i = 0; i < num_pages; ++i) {
    size_t page_idx = start_page_idx + i;
    if (reclaimed[page_idx]) {
      continue;
    }
    if (page_idx < tensors->weight_page_refcount.size()) {
      int32_t& ref = tensors->weight_page_refcount[page_idx];
      if (ref > 0) {
        --ref;
      }
      if (ref > 0) {
        continue;
      }
    }
    size_t page_offset = tensors->weight_xtensor_offset + page_idx * page_size_;
    std::unique_ptr<PhyPage> page =
        weight_xtensor_->unmap_and_take(static_cast<offset_t>(page_offset));
    if (!page) {
      continue;
    }
    pages_to_put.push_back(std::move(page));
    reclaimed[page_idx] = true;
    unmapped++;
  }
  if (!pages_to_put.empty()) {
    auto& pool = PhyPagePool::get_instance();
    lock.unlock();
    pool.batch_put(pages_to_put);
    lock.lock();
  }
  return unmapped;
}

size_t XTensorAllocator::reclaim_mapped_zero_ref_weight_pages(
    const std::string& model_id) {
  std::unique_lock<std::mutex> lock(mtx_);
  ModelTensors* tensors = get_model_tensors(model_id);
  if (!tensors || !weight_xtensor_ || tensors->weight_num_pages == 0) {
    return 0;
  }
  auto it = weight_page_reclaimed_.find(model_id);
  if (it == weight_page_reclaimed_.end() ||
      it->second.size() != tensors->weight_num_pages) {
    LOG(ERROR) << "reclaim_mapped_zero_ref_weight_pages: reclaimed bitmap "
                  "missing or size mismatch";
    return 0;
  }

  std::vector<bool>& reclaimed = it->second;
  std::vector<std::unique_ptr<PhyPage>> pages_to_put;
  pages_to_put.reserve(tensors->weight_num_pages);

  size_t unmapped = 0;
  for (size_t page_idx = 0; page_idx < tensors->weight_num_pages; ++page_idx) {
    if (reclaimed[page_idx]) {
      continue;
    }
    if (page_idx < tensors->weight_page_refcount.size() &&
        tensors->weight_page_refcount[page_idx] > 0) {
      continue;
    }

    size_t page_offset = tensors->weight_xtensor_offset + page_idx * page_size_;
    std::unique_ptr<PhyPage> page =
        weight_xtensor_->unmap_and_take(static_cast<offset_t>(page_offset));
    if (!page) {
      continue;
    }
    pages_to_put.push_back(std::move(page));
    reclaimed[page_idx] = true;
    unmapped++;
  }

  if (!pages_to_put.empty()) {
    auto& pool = PhyPagePool::get_instance();
    lock.unlock();
    pool.batch_put(pages_to_put);
    lock.lock();
  }
  return unmapped;
}

int64_t XTensorAllocator::ensure_weight_pages_mapped_region(
    const std::string& model_id,
    void* ptr,
    size_t size) {
  if (ptr == nullptr || size == 0) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(mtx_);
  ModelTensors* tensors = get_model_tensors(model_id);
  if (!tensors || !weight_xtensor_ || tensors->weight_base_ptr == nullptr ||
      tensors->weight_num_pages == 0) {
    return -1;
  }
  const uintptr_t base = reinterpret_cast<uintptr_t>(tensors->weight_base_ptr);
  const uintptr_t region_end = base + tensors->weight_num_pages * page_size_;
  const uintptr_t ptr_val = reinterpret_cast<uintptr_t>(ptr);
  if (ptr_val < base || ptr_val >= region_end) {
    LOG(ERROR)
        << "ensure_weight_pages_mapped_region: ptr not in model weight region";
    return -1;
  }
  size_t offset_in_region = ptr_val - base;
  size_t start_page_idx = offset_in_region / page_size_;
  size_t end_page_idx =
      (offset_in_region + size + page_size_ - 1) / page_size_;
  size_t num_pages = end_page_idx - start_page_idx;
  if (end_page_idx > tensors->weight_num_pages) {
    LOG(ERROR) << "ensure_weight_pages_mapped_region: [ptr,ptr+size) exceeds "
                  "weight region";
    return -1;
  }
  auto it = weight_page_reclaimed_.find(model_id);
  if (it == weight_page_reclaimed_.end()) {
    LOG(ERROR) << "ensure_weight_pages_mapped_region: reclaimed bitmap missing";
    return -1;
  }
  std::vector<bool>& reclaimed = it->second;
  if (reclaimed.size() != tensors->weight_num_pages) {
    LOG(ERROR)
        << "ensure_weight_pages_mapped_region: reclaimed bitmap size mismatch";
    return -1;
  }
  std::vector<size_t> need_map_idx;
  for (size_t i = 0; i < num_pages; ++i) {
    size_t page_idx = start_page_idx + i;
    if (page_idx < tensors->weight_page_refcount.size()) {
      tensors->weight_page_refcount[page_idx]++;
    }
    if (reclaimed[page_idx]) {
      need_map_idx.push_back(page_idx);
    }
  }
  if (need_map_idx.empty()) {
    return 0;
  }
  auto& pool = PhyPagePool::get_instance();
  std::vector<std::unique_ptr<PhyPage>> pages;
  {
    std::lock_guard<std::mutex> alloc_lock(forward_alloc_mtx_);
    pages = pool.batch_get(need_map_idx.size());
    if (pages.size() != need_map_idx.size()) {
      LOG(ERROR) << "ensure_weight_pages_mapped_region: batch_get failed";
      return -1;
    }
  }
  for (size_t i = 0; i < need_map_idx.size(); ++i) {
    size_t page_idx = need_map_idx[i];
    size_t off = tensors->weight_xtensor_offset + page_idx * page_size_;
    if (!weight_xtensor_->map_external_page(static_cast<offset_t>(off),
                                            std::move(pages[i]))) {
      LOG(ERROR) 
          << "ensure_weight_pages_mapped_region: map_external_page failed at " 
          << off;
      return -1;
    }
    reclaimed[page_idx] = false;
  }
  return static_cast<int64_t>(need_map_idx.size());
}

size_t XTensorAllocator::mark_weight_pages_reclaimable(
    const std::string& model_id) {
  auto* tensors = get_model_tensors(model_id);
  if (!tensors || tensors->weight_num_pages == 0) {
    LOG(WARNING) << "No weight allocation found for model " << model_id;
    return 0;
  }
  if (tensors->weight_pages_reclaimable) {
    return tensors->weight_num_pages;
  }

  tensors->weight_pages_reclaimable = true;
  auto& reclaimed = weight_page_reclaimed_[model_id];
  if (reclaimed.empty()) {
    reclaimed.assign(tensors->weight_num_pages, false);
  }
  for (size_t i = 0; i < tensors->weight_num_pages; ++i) {
    if (!reclaimed[i]) {
      weight_reclaim_queue_.push_back({model_id, i});
    }
  }
  return tensors->weight_num_pages;
}

size_t XTensorAllocator::free_weight_from_global_xtensor(
    const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);

  auto* tensors = get_model_tensors(model_id);
  if (!tensors || tensors->weight_num_pages == 0) {
    LOG(WARNING) << "No weight allocation found for model " << model_id;
    return 0;
  }

  size_t num_pages = mark_weight_pages_reclaimable(model_id);
  if (num_pages == 0) {
    return 0;
  }
  LOG(INFO) << "Marked " << num_pages << " weight pages reclaimable for model "
            << model_id;
  return num_pages;
}

// ============== PD Disaggregation Support (XTensor Mode) ==============

std::pair<uint64_t, uint64_t> XTensorAllocator::get_global_offsets_for_block(
    const std::string& model_id,
    int64_t layer_id,
    int64_t block_id,
    size_t block_size) {
  constexpr uint64_t INVALID_OFFSET = UINT64_MAX;

  std::lock_guard<std::mutex> lock(mtx_);

  auto* tensors = get_model_tensors(model_id);
  if (!tensors) {
    LOG(ERROR) << "Model " << model_id << " not found for offset calculation";
    return {INVALID_OFFSET, INVALID_OFFSET};
  }

  if (layer_id < 0 || layer_id >= tensors->num_layers) {
    LOG(ERROR) << "Invalid layer_id " << layer_id << " for model " << model_id
               << " (num_layers=" << tensors->num_layers << ")";
    return {INVALID_OFFSET, INVALID_OFFSET};
  }

  if (tensors->k_tensors.empty() || tensors->v_tensors.empty()) {
    LOG(ERROR) << "KV tensors not created for model " << model_id;
    return {INVALID_OFFSET, INVALID_OFFSET};
  }

  auto& global_xtensor = GlobalXTensor::get_instance();
  if (!global_xtensor.is_initialized()) {
    LOG(ERROR) << "GlobalXTensor not initialized";
    return {INVALID_OFFSET, INVALID_OFFSET};
  }

  // Calculate the offset within the XTensor for this block
  // The offset must be aligned to page_size
  size_t page_size = FLAGS_phy_page_granularity_size;
  offset_t local_offset =
      static_cast<offset_t>((block_id * block_size / page_size) * page_size);

  // Get K tensor's physical page_id at this offset
  auto* k_xtensor = tensors->k_tensors[layer_id].get();
  page_id_t k_page_id = k_xtensor->get_phy_page_id(local_offset);
  if (k_page_id < 0) {
    LOG(ERROR) << "K cache block " << block_id << " at layer " << layer_id
               << " is not mapped (local_offset=" << local_offset << ")";
    return {INVALID_OFFSET, INVALID_OFFSET};
  }

  // Get V tensor's physical page_id at this offset
  auto* v_xtensor = tensors->v_tensors[layer_id].get();
  page_id_t v_page_id = v_xtensor->get_phy_page_id(local_offset);
  if (v_page_id < 0) {
    LOG(ERROR) << "V cache block " << block_id << " at layer " << layer_id
               << " is not mapped (local_offset=" << local_offset << ")";
    return {INVALID_OFFSET, INVALID_OFFSET};
  }

  // Calculate GlobalXTensor offsets using page_id
  // GlobalXTensor offset = page_id * page_size + (block offset within page)
  size_t offset_within_page = (block_id * block_size) % page_size;

  uint64_t k_global_offset =
      static_cast<uint64_t>(k_page_id) * page_size + offset_within_page;
  uint64_t v_global_offset =
      static_cast<uint64_t>(v_page_id) * page_size + offset_within_page;

  VLOG(2) << "get_global_offsets_for_block: model=" << model_id
          << ", layer=" << layer_id << ", block=" << block_id
          << ", block_size=" << block_size << ", k_page_id=" << k_page_id
          << ", v_page_id=" << v_page_id << ", k_offset=" << k_global_offset
          << ", v_offset=" << v_global_offset;

  return {k_global_offset, v_global_offset};
}

// TODO: refactor this function to use the new parallel strategy
bool XTensorAllocator::get_xtensor_offsets(
    int32_t dp_rank,
    const std::string& model_id,
    const std::vector<int32_t>& block_ids,
    uint64_t block_size_bytes,
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>&
        layer_offsets) {
  auto [model_dp_size, model_tp_size] = get_model_parallel_strategy(model_id);
  int32_t worker_rank_base = get_model_worker_rank_base(model_id);

  // The offsets of the xtensor in the same DP group as the master worker are
  // identical, so there is no need to fetch them via RPC.
  if (worker_rank_base == 0 && dp_rank == 0) {
    // Get model tensors to determine num_layers
    auto* tensors = get_model_tensors(model_id);
    if (!tensors) {
      LOG(ERROR) << "Model " << model_id << " not found for local calculation";
      return false;
    }

    int64_t num_layers = tensors->num_layers;
    layer_offsets.resize(num_layers);

    for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
      std::vector<uint64_t> k_offsets;
      std::vector<uint64_t> v_offsets;
      k_offsets.reserve(block_ids.size());
      v_offsets.reserve(block_ids.size());

      for (int32_t block_id : block_ids) {
        auto [k_offset, v_offset] = get_global_offsets_for_block(
            model_id, layer_id, block_id, block_size_bytes);
        if (k_offset == UINT64_MAX || v_offset == UINT64_MAX) {
          LOG(ERROR) << "Failed to get local offsets for block " << block_id
                     << " at layer " << layer_id;
          return false;
        }
        k_offsets.push_back(k_offset);
        v_offsets.push_back(v_offset);
      }
      layer_offsets[layer_id] = {std::move(k_offsets), std::move(v_offsets)};
    }

    VLOG(1) << "get_xtensor_offsets (local): model_id=" << model_id
            << ", num_blocks=" << block_ids.size()
            << ", num_layers=" << num_layers;
    return true;
  }

  if (dp_rank < 0 || dp_rank >= model_dp_size) {
    LOG(ERROR) << "Invalid dp_rank: " << dp_rank
               << ", model_dp_size=" << model_dp_size;
    return false;
  }

  int32_t actual_rank = worker_rank_base + dp_rank * model_tp_size;

  if (actual_rank >= static_cast<int32_t>(xtensor_dist_clients_.size())) {
    LOG(ERROR) << "Invalid actual_rank: " << actual_rank
               << ", xtensor_dist_clients_.size()="
               << xtensor_dist_clients_.size();
    return false;
  }

  // Call the first worker in the DP group (all workers in the same DP group
  // should have the same physical page mapping)
  auto& client = xtensor_dist_clients_[actual_rank];
  auto future =
      client->get_xtensor_offsets_async(model_id, block_ids, block_size_bytes);

  layer_offsets = std::move(future).get();
  if (layer_offsets.empty()) {
    LOG(ERROR) << "get_xtensor_offsets failed for dp_rank=" << dp_rank
               << ", model_id=" << model_id;
    return false;
  }

  VLOG(1) << "get_xtensor_offsets: dp_rank=" << dp_rank
          << ", model_id=" << model_id << ", num_blocks=" << block_ids.size()
          << ", num_layers=" << layer_offsets.size();

  return true;
}

// ============== ETCD Information Support ==============

std::vector<WeightSegment> XTensorAllocator::get_model_weight_segments(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_tensors_.find(model_id);
  if (it == model_tensors_.end()) {
    return {};
  }
  return it->second.weight_segments;
}

std::unordered_map<std::string, std::vector<WeightSegment>>
XTensorAllocator::get_all_model_weight_segments() const {
  std::lock_guard<std::mutex> lock(mtx_);
  std::unordered_map<std::string, std::vector<WeightSegment>> result;

  for (const auto& [model_id, tensors] : model_tensors_) {
    if (!tensors.weight_segments.empty()) {
      result[model_id] = tensors.weight_segments;
    }
  }

  return result;
}

}  // namespace xllm
