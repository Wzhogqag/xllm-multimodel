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

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace xllm {

/**
 * LayerOffloadManager – master-side watermark-driven layer weight offload/restore.
 *
 * Owned by PageAllocator on the master node. Coordinates offload/load across
 * all workers via XTensorAllocator broadcast RPCs. After each broadcast the
 * returned per-worker page counts adjust PageAllocator::worker_pages_used_
 * (same accounting as alloc_weight_pages), not worker_reported / SHM.
 *
 * Responsibilities:
 *   1. Monitor PageAllocator free_pages_ratio every poll_interval ms.
 *   2. When ratio < low_watermark: pick lowest-priority AWAKE model,
 *      broadcast offload to all workers, record page changes.
 *   3. When ratio >= restore_watermark: pick highest-priority DEGRADED model,
 *      broadcast load to all workers, record page changes.
 */
class LayerOffloadManager {
 public:
  LayerOffloadManager() = default;
  ~LayerOffloadManager() { stop(); }
  LayerOffloadManager(const LayerOffloadManager&) = delete;
  LayerOffloadManager& operator=(const LayerOffloadManager&) = delete;

  struct PerModelState {
    std::string model_id;
    int32_t priority = 50;
    int32_t num_layers = 0;
    bool is_degraded = false;
    int32_t num_layers_on_device = 0;  // updated after each offload/load

    // Callbacks – executed on the Worker's own threadpool thread.
    // offload_fn(layer_id): unmap physical pages for that layer; blocking.
    std::function<bool(int32_t)> offload_fn;
    // load_fn(layer_id): re-map + copy weights from host; blocking.
    std::function<bool(int32_t)> load_fn;
    // npu_sync_fn(): synchronize the NPU stream on Worker thread; blocking.
    std::function<void()> npu_sync_fn;
  };

  /**
   * Register a model for offload/load management (called by master).
   *
   * @param model_id   Model identifier (same as used in PageAllocator).
   * @param num_layers Total transformer layers in the model.
   * @param priority   Priority (higher = restored first / degraded last).
   */
  void register_model(const std::string& model_id,
                      int32_t num_layers,
                      int32_t priority);


  void start();
  void stop();

  int32_t offload_layers(PerModelState& state, int32_t count);
  int32_t load_layers(PerModelState& state, int32_t count);

  enum class OffloadContext { kMonitorDegrade, kEmergencyEviction };

  /** Pick lowest-priority awake model, offload one chunk; serialized across callers.
   *  @param free_ratio_for_log used when ctx==kMonitorDegrade and no candidate.
   *  @param allowed_models optional model whitelist for scoped offload.
   *  @return -1 if no offloadable model; otherwise layers offloaded (0 means no progress). */
  int32_t offload_internal(OffloadContext ctx,
                           double free_ratio_for_log = 0.0,
                           const std::unordered_set<std::string>* allowed_models =
                               nullptr);

 private:
  void monitor_loop();

  PerModelState* pick_lowest_priority_awake(
      const std::unordered_set<std::string>* allowed_models = nullptr);
  PerModelState* pick_highest_priority_degraded(
      const std::unordered_set<std::string>* allowed_models = nullptr);
  void update_model_copies_delta_locked(const std::string& any_model_id,
                                        int delta);


  mutable std::mutex mtx_;

  std::unordered_map<std::string, std::unique_ptr<PerModelState>> per_model_;
  std::unordered_map<std::string, int64_t> base_model_valid_copies_;

  std::atomic<bool> running_{false};
  std::unique_ptr<std::thread> monitor_thd_;
};

}  // namespace xllm
