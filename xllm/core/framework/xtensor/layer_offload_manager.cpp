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

#include "layer_offload_manager.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>

#include "common/global_flags.h"
#include "core/common/interruption_bus.h"
#include "framework/xtensor/page_allocator.h"

namespace xllm {

// ============================================================
//  Registration
// ============================================================

void LayerOffloadManager::register_model(
    const std::string& model_id,
    int32_t num_layers,
    int32_t priority,
    std::function<bool(int32_t)> offload_fn,
    std::function<bool(int32_t)> load_fn,
    std::function<void()> npu_sync_fn) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (per_model_.count(model_id)) {
    LOG(WARNING) << "[LayerOffloadManager] model " << model_id
                 << " already registered, overwriting.";
  }
  auto state = std::make_unique<PerModelState>();
  state->model_id = model_id;
  state->num_layers = num_layers;
  state->num_layers_on_device = num_layers;  // initially all layers on device
  state->priority = priority;
  state->is_degraded = false;
  state->offload_fn = std::move(offload_fn);
  state->load_fn = std::move(load_fn);
  state->npu_sync_fn = std::move(npu_sync_fn);
  per_model_[model_id] = std::move(state);
  LOG(INFO) << "[LayerOffloadManager] Registered model=" << model_id
            << " num_layers=" << num_layers << " priority=" << priority;
}

void LayerOffloadManager::unregister_model(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  per_model_.erase(model_id);
  LOG(INFO) << "[LayerOffloadManager] Unregistered model=" << model_id;
}

// ============================================================
//  Step tracking (called from WorkerImpl::step_async)
// ============================================================

void LayerOffloadManager::enter_step(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = per_model_.find(model_id);
  if (it != per_model_.end()) {
    it->second->in_flight_batches.fetch_add(1, std::memory_order_relaxed);
  }
}

void LayerOffloadManager::exit_step(const std::string& model_id) {
  {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = per_model_.find(model_id);
    if (it != per_model_.end()) {
      it->second->in_flight_batches.fetch_sub(1, std::memory_order_release);
    }
  }
  // Notify drain_model() which may be waiting for in_flight == 0.
  drain_cv_.notify_all();
}

// ============================================================
//  Lifecycle
// ============================================================

void LayerOffloadManager::start() {
  if (running_.exchange(true)) return;  // already started
  monitor_thd_ =
      std::make_unique<std::thread>(&LayerOffloadManager::monitor_loop, this);
  LOG(INFO) << "[LayerOffloadManager] Monitor thread started.";
}

void LayerOffloadManager::stop() {
  if (!running_.exchange(false)) return;  // already stopped or never started
  drain_cv_.notify_all();
  if (monitor_thd_ && monitor_thd_->joinable()) {
    monitor_thd_->join();
  }
  monitor_thd_.reset();
  LOG(INFO) << "[LayerOffloadManager] Monitor thread stopped.";
}

// ============================================================
//  Monitor loop
// ============================================================

void LayerOffloadManager::monitor_loop() {
  while (running_.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(FLAGS_layer_offload_poll_interval_ms));

    if (!FLAGS_enable_watermark_degrade_restore_mvp) continue;

    auto& pa = PageAllocator::get_instance();
    if (!pa.is_initialized()) continue;

    size_t free_pages = pa.get_num_free_phy_pages();
    size_t total_pages = pa.get_num_total_phy_pages();
    if (total_pages == 0) continue;

    double free_ratio = static_cast<double>(free_pages) / total_pages;

    if (free_ratio < 0.5 * FLAGS_layer_offload_low_watermark_ratio) {
      // ---- DEGRADE path ----
      LOG(INFO) << "[LayerOffloadManager] Degrade triggered: free_ratio="
                << free_ratio << " free=" << free_pages << "/" << total_pages;

      int total_offloaded = 0;
      int max_rounds = 50;  // guard against infinite loop
      int round = 0;

      while (round++ < max_rounds) {
        free_pages = pa.get_num_free_phy_pages();
        free_ratio = static_cast<double>(free_pages) / total_pages;
        if (free_ratio >= FLAGS_layer_offload_low_watermark_ratio) break;

        PerModelState* target = pick_lowest_priority_awake();
        if (target == nullptr) {
          LOG(WARNING)
              << "[LayerOffloadManager] No awake model with offloadable"
              << " layers. free_ratio=" << free_ratio;
          break;
        }

        // --- Drain: interrupt forward, wait, sync NPU ---
        auto t0 = std::chrono::steady_clock::now();
        if (!drain_model(*target)) {
          LOG(ERROR) << "[LayerOffloadManager] Drain timeout for model="
                     << target->model_id << ", skipping offload this round.";
          // Resume: clear interruption flag so requests can proceed.
          InterruptionBus::get_instance().publish(false);
          break;
        }

        int32_t chunk =
            std::min(static_cast<int32_t>(FLAGS_offload_chunk_layers),
                     target->num_layers_on_device);
        if (chunk <= 0) {
          // All layers offloaded for this model; resume and try next.
          InterruptionBus::get_instance().publish(false);
          break;
        }

        int32_t offloaded = offload_layers(*target, chunk);
        total_offloaded += offloaded;

        auto t1 = std::chrono::steady_clock::now();
        double elapsed_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        LOG(INFO) << "[LayerOffloadManager] Offloaded " << offloaded
                  << " layers from model=" << target->model_id
                  << " num_layers_on_device=" << target->num_layers_on_device
                  << " free_ratio="
                  << static_cast<double>(pa.get_num_free_phy_pages()) /
                         total_pages
                  << " elapsed_ms=" << elapsed_ms;

        // Resume forward after successful offload.
        InterruptionBus::get_instance().publish(false);

        if (offloaded == 0) break;
      }

      if (total_offloaded > 0) {
        LOG(INFO) << "[LayerOffloadManager] Degrade complete, total_offloaded="
                  << total_offloaded << " final_free_ratio="
                  << static_cast<double>(pa.get_num_free_phy_pages()) /
                         total_pages;
      }

    } else if (free_ratio >= FLAGS_layer_offload_restore_watermark_ratio) {
      // ---- RESTORE path ----
      PerModelState* target = pick_highest_priority_degraded();
      if (target == nullptr) continue;

      int32_t chunk =
          std::min(static_cast<int32_t>(FLAGS_load_chunk_layers),
                   target->num_layers - target->num_layers_on_device);
      if (chunk <= 0) continue;

      auto t0 = std::chrono::steady_clock::now();
      int32_t loaded = load_layers(*target, chunk);
      auto t1 = std::chrono::steady_clock::now();
      double elapsed_ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();

      LOG(INFO) << "[LayerOffloadManager] Restored " << loaded
                << " layers to model=" << target->model_id
                << " num_layers_on_device=" << target->num_layers_on_device
                << " free_ratio="
                << static_cast<double>(pa.get_num_free_phy_pages()) /
                       total_pages
                << " elapsed_ms=" << elapsed_ms;

      // If free_ratio dropped below high_watermark while loading, stop.
      free_pages = pa.get_num_free_phy_pages();
      free_ratio = static_cast<double>(free_pages) / total_pages;
      if (free_ratio < FLAGS_layer_offload_high_watermark_ratio) {
        LOG(WARNING) << "[LayerOffloadManager] free_ratio dropped to "
                     << free_ratio << " during restore, pausing.";
      }
    }
  }
}

// ============================================================
//  Drain
// ============================================================

bool LayerOffloadManager::drain_model(PerModelState& state) {
  // Step 1: Signal all in-flight forwards to stop at the next layer boundary.
  InterruptionBus::get_instance().publish(true);

  // Step 2: Wait for in_flight_batches to reach 0.
  auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(FLAGS_layer_offload_drain_timeout_ms);

  std::unique_lock<std::mutex> lock(mtx_);
  bool drained = drain_cv_.wait_until(lock, deadline, [&state] {
    return state.in_flight_batches.load(std::memory_order_acquire) == 0;
  });

  if (!drained) {
    LOG(ERROR) << "[LayerOffloadManager] Drain timeout ("
               << FLAGS_layer_offload_drain_timeout_ms
               << "ms) for model=" << state.model_id
               << " in_flight_batches=" << state.in_flight_batches.load();
    return false;
  }

  lock.unlock();

  // Step 3: Synchronize NPU stream on Worker thread (hard constraint).
  // This ensures no ATB/CANN kernel is still accessing weight memory.
  if (state.npu_sync_fn) {
    state.npu_sync_fn();
  }

  return true;
}

// ============================================================
//  Offload layers (tail-first)
// ============================================================

int32_t LayerOffloadManager::offload_layers(PerModelState& state,
                                            int32_t count) {
  if (count <= 0 || state.num_layers_on_device <= 0) return 0;

  int32_t actual = std::min(count, state.num_layers_on_device);
  int32_t offloaded = 0;

  for (int32_t i = 0; i < actual; ++i) {
    // Offload the last layer currently on device (tail-first).
    int32_t layer_id = state.num_layers_on_device - 1 - i;
    if (layer_id < 0) break;

    bool ok = state.offload_fn ? state.offload_fn(layer_id) : false;
    if (!ok) {
      LOG(ERROR) << "[LayerOffloadManager] offload_fn failed for model="
                 << state.model_id << " layer_id=" << layer_id;
      break;
    }
    ++offloaded;
  }

  if (offloaded > 0) {
    state.num_layers_on_device -= offloaded;
    state.is_degraded = (state.num_layers_on_device < state.num_layers);
    // Reflect in PageAllocator for external visibility.
    PageAllocator::get_instance().update_layers_on_device(
        state.model_id, state.num_layers_on_device);
  }
  return offloaded;
}

// ============================================================
//  Load layers (tail-first restore)
// ============================================================

int32_t LayerOffloadManager::load_layers(PerModelState& state, int32_t count) {
  if (count <= 0) return 0;
  int32_t offloaded_count = state.num_layers - state.num_layers_on_device;
  int32_t actual = std::min(count, offloaded_count);
  if (actual <= 0) return 0;

  int32_t loaded = 0;
  for (int32_t i = 0; i < actual; ++i) {
    // Restore the layer just above current top (reverse of offload order).
    int32_t layer_id = state.num_layers_on_device + i;
    if (layer_id >= state.num_layers) break;

    bool ok = state.load_fn ? state.load_fn(layer_id) : false;
    if (!ok) {
      LOG(ERROR) << "[LayerOffloadManager] load_fn failed for model="
                 << state.model_id << " layer_id=" << layer_id;
      break;
    }
    ++loaded;
  }

  if (loaded > 0) {
    state.num_layers_on_device += loaded;
    state.is_degraded = (state.num_layers_on_device < state.num_layers);
    PageAllocator::get_instance().update_layers_on_device(
        state.model_id, state.num_layers_on_device);
  }
  return loaded;
}

// ============================================================
//  Model selection
// ============================================================

LayerOffloadManager::PerModelState*
LayerOffloadManager::pick_lowest_priority_awake() {
  // Note: caller does NOT hold mtx_; safe since we only read per_model_.
  // WorkerImpl may concurrently enter/exit step, but model map itself is
  // only modified by register/unregister (also under mtx_).
  std::lock_guard<std::mutex> lock(mtx_);
  PerModelState* candidate = nullptr;
  for (auto& [id, state] : per_model_) {
    if (state->num_layers_on_device <= 0) continue;  // nothing to offload
    if (PageAllocator::get_instance().is_model_sleeping(id)) continue;
    if (candidate == nullptr || state->priority < candidate->priority) {
      candidate = state.get();
    }
  }
  return candidate;
}

LayerOffloadManager::PerModelState*
LayerOffloadManager::pick_highest_priority_degraded() {
  std::lock_guard<std::mutex> lock(mtx_);
  PerModelState* candidate = nullptr;
  for (auto& [id, state] : per_model_) {
    if (!state->is_degraded) continue;
    if (state->num_layers_on_device >= state->num_layers) continue;
    if (candidate == nullptr || state->priority > candidate->priority) {
      candidate = state.get();
    }
  }
  return candidate;
}

// ============================================================
//  Exact layers-to-offload calculation (uses phy_pages_per_layer_list)
// ============================================================

int32_t LayerOffloadManager::calc_layers_to_offload_exact(
    const std::string& model_id,
    int32_t layers_on_device,
    size_t need_pages) {
  auto list =
      PageAllocator::get_instance().get_phy_pages_per_layer_list(model_id);
  if (list.empty() || layers_on_device <= 0) {
    return FLAGS_offload_chunk_layers;
  }

  // Accumulate from tail (last layer first).
  size_t accumulated = 0;
  int32_t n = 0;
  for (int32_t i = layers_on_device - 1; i >= 0 && accumulated < need_pages;
       --i) {
    if (static_cast<size_t>(i) < list.size()) {
      accumulated += list[static_cast<size_t>(i)];
    }
    ++n;
  }
  return std::max(n, 1);
}

}  // namespace xllm
