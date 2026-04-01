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
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace xllm {

/**
 * LayerOffloadManager – MVP watermark-driven layer weight offload/restore.
 *
 * Responsibilities:
 *   1. Monitor PageAllocator free_pages_ratio every poll_interval ms.
 *   2. When ratio < low_watermark (0.10): pick lowest-priority AWAKE model,
 *      drain its in-flight batches, then offload `offload_chunk_layers` layers.
 *      Loop until ratio >= high_watermark (0.15) or no layers left.
 *   3. When ratio >= restore_watermark (0.25): pick highest-priority DEGRADED
 *      model, load `load_chunk_layers` layers. Stop if ratio drops < 0.15.
 *
 * Safety gate (hard constraint):
 *   Before any offload, LayerOffloadManager verifies:
 *     (a) in_flight_batches[model_id] == 0  (drain complete)
 *     (b) npu_sync_fn() has been called     (NPU stream flushed)
 *   If drain timeout expires, the offload is SKIPPED with LOG(ERROR).
 *
 * Thread model:
 *   - One singleton background monitor thread.
 *   - Per-model PerModelState (heap-allocated for atomic members).
 *   - WorkerImpl calls enter_step/exit_step around every step() call.
 *   - offload_fn / load_fn / npu_sync_fn are dispatched on WorkerImpl's
 *     threadpool via blocking SemiFuture (registered by WorkerImpl).
 */
class LayerOffloadManager {
 public:
  static LayerOffloadManager& get_instance() {
    static LayerOffloadManager instance;
    return instance;
  }

  // Per-model registration state (heap-allocated for std::atomic members).
  struct PerModelState {
    std::string model_id;
    int32_t priority = 50;
    int32_t num_layers = 0;

    // Tracks in-flight step() calls for this model.
    std::atomic<int> in_flight_batches{0};

    // Is the model currently degraded (some layers offloaded)?
    // num_layers_on_device < num_layers  ⟺  is_degraded == true.
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
   * Register a model with offload/load callbacks.
   * Must be called before start() or the model will be ignored until the
   * next monitor wakeup.
   *
   * @param model_id     Model identifier (same as used in PageAllocator).
   * @param num_layers   Total transformer layers in the model.
   * @param priority     Priority (higher = restored first / degraded last).
   * @param offload_fn   Blocking: offloads weights for layer_id.
   * @param load_fn      Blocking: loads weights for layer_id from host.
   * @param npu_sync_fn  Blocking: synchronizes NPU stream on worker thread.
   */
  void register_model(const std::string& model_id,
                      int32_t num_layers,
                      int32_t priority,
                      std::function<bool(int32_t)> offload_fn,
                      std::function<bool(int32_t)> load_fn,
                      std::function<void()> npu_sync_fn);

  void unregister_model(const std::string& model_id);

  // Called by WorkerImpl::step_async() BEFORE step().
  void enter_step(const std::string& model_id);
  // Called by WorkerImpl::step_async() AFTER step() (even on interrupted).
  void exit_step(const std::string& model_id);

  // Returns true if the model has some layers offloaded (is_degraded == true).
  // Returns false if the model is not registered or fully on device.
  bool is_model_degraded(const std::string& model_id) const;

  // Start the background monitor thread (idempotent).
  void start();
  // Stop the background monitor thread and join (idempotent).
  void stop();

 private:
  LayerOffloadManager() = default;
  ~LayerOffloadManager() { stop(); }
  LayerOffloadManager(const LayerOffloadManager&) = delete;
  LayerOffloadManager& operator=(const LayerOffloadManager&) = delete;

  void monitor_loop();

  // Drain a model: interrupt forward, wait for in_flight == 0, sync NPU.
  // Returns true if drain succeeded within timeout.
  bool drain_model(PerModelState& state);

  // Offload `count` tail-layers from the given model.
  // Returns the number actually offloaded (may be < count if fewer remain).
  int32_t offload_layers(PerModelState& state, int32_t count);

  // Load `count` layers (from current tail) back for the given model.
  // Returns the number actually loaded.
  int32_t load_layers(PerModelState& state, int32_t count);

  // Pick the lowest-priority AWAKE model that still has layers on device.
  // Returns nullptr if none found.
  PerModelState* pick_lowest_priority_awake();

  // Pick the highest-priority DEGRADED model.
  // Returns nullptr if none found.
  PerModelState* pick_highest_priority_degraded();

  // Calc layers to offload using phy_pages_per_layer_list (tail-accumulate).
  // Falls back to uniform estimate if list is empty.
  static int32_t calc_layers_to_offload_exact(const std::string& model_id,
                                              int32_t layers_on_device,
                                              size_t need_pages);

  mutable std::mutex mtx_;
  std::condition_variable drain_cv_;  // notified on exit_step

  std::unordered_map<std::string, std::unique_ptr<PerModelState>> per_model_;

  std::atomic<bool> running_{false};
  std::unique_ptr<std::thread> monitor_thd_;
};

}  // namespace xllm
