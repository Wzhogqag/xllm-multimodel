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
#include "framework/xtensor/page_allocator.h"
#include "framework/xtensor/xtensor_allocator.h"

namespace xllm {

// ============================================================
//  Registration
// ============================================================

void LayerOffloadManager::register_model(
    const std::string& model_id,
    int32_t num_layers,
    int32_t priority) {
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
//  Lifecycle
// ============================================================

void LayerOffloadManager::start() {
  if (running_.exchange(true)) return;  // already started
  CHECK(FLAGS_offload_chunk_layers > 0) << "offload_chunk_layers must be greater than 0";
  CHECK(FLAGS_load_chunk_layers > 0) << "load_chunk_layers must be greater than 0";
  monitor_thd_ =
      std::make_unique<std::thread>(&LayerOffloadManager::monitor_loop, this);
  LOG(INFO) << "[LayerOffloadManager] Monitor thread started (master).";
}

void LayerOffloadManager::stop() {
  if (!running_.exchange(false)) return;  // already stopped or never started
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
      int max_rounds = 50;
      int round = 0;

      while (round++ < max_rounds) {
        free_pages = pa.get_num_free_phy_pages();
        free_ratio = static_cast<double>(free_pages) / total_pages;
        if (free_ratio >= FLAGS_layer_offload_low_watermark_ratio) break;

        int32_t offloaded = offload_internal(OffloadContext::kMonitorDegrade,
                                            free_ratio);
        if (offloaded == 0) break;
        total_offloaded += offloaded;
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
          std::min(FLAGS_load_chunk_layers,
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
//  One-shot offload chunk (pick + broadcast), serialized
// ============================================================

int32_t LayerOffloadManager::offload_internal(OffloadContext ctx,
                                              double free_ratio_for_log) {
  std::lock_guard<std::mutex> lock(mtx_);

  PerModelState* target = pick_lowest_priority_awake();
  if (target == nullptr) {
    if (ctx == OffloadContext::kMonitorDegrade) {
      LOG(WARNING)
          << "[LayerOffloadManager] No awake model with offloadable"
          << " layers. free_ratio=" << free_ratio_for_log;
    } else {
      LOG(WARNING) << "[EmergencyEviction] No awake model with offloadable "
                      "layers.";
    }
    return -1;
  }

  int32_t chunk =
      std::min(FLAGS_offload_chunk_layers,
               target->num_layers_on_device);
  if (chunk <= 0) {
    return 0;
  }

  auto t0 = std::chrono::steady_clock::now();
  int32_t offloaded = offload_layers(*target, chunk);
  auto t1 = std::chrono::steady_clock::now();
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  if (ctx == OffloadContext::kMonitorDegrade) {
    auto& pa = PageAllocator::get_instance();
    size_t tp = pa.get_num_total_phy_pages();
    double denom = tp ? static_cast<double>(tp) : 1.0;
    LOG(INFO) << "[LayerOffloadManager] Offloaded " << offloaded
              << " layers from model=" << target->model_id
              << " num_layers_on_device=" << target->num_layers_on_device
              << " free_ratio="
              << static_cast<double>(pa.get_num_free_phy_pages()) / denom
              << " elapsed_ms=" << elapsed_ms;
  } else {
    LOG(INFO) << "EmergencyEviction offloaded " << offloaded
              << " layers from model=" << target->model_id;
  }

  return offloaded;
}

// ============================================================
//  Offload layers (tail-first) via broadcast
// ============================================================

int32_t LayerOffloadManager::offload_layers(PerModelState& state,
                                            int32_t count) {
  if (count <= 0 || state.num_layers_on_device <= 0) return 0;

  int32_t offloaded = 0;
  auto& allocator = XTensorAllocator::get_instance();
  auto& pa = PageAllocator::get_instance();

  for (int32_t i = 0; i < count; ++i) {
    // Offload the last layer currently on device (tail-first).
    int32_t layer_id = state.num_layers_on_device - 1 - i;
    if (layer_id < 0) break;

    auto pages_per_worker =
        allocator.broadcast_offload_layer_weights(state.model_id, layer_id);

    bool any_failed = false;
    for (size_t w = 0; w < pages_per_worker.size(); ++w) {
      if (pages_per_worker[w] < 0) {
        any_failed = true;
        LOG(ERROR) << "[LayerOffloadManager] offload failed on worker=" << w
                   << " model=" << state.model_id << " layer_id=" << layer_id;
        break;
      }
    }
    if (any_failed) break;

    //TODO: support dp
    pa.release_phy_pages_for_dp(state.model_id, 0, pages_per_worker[0]);

    ++offloaded;
  }

  if (offloaded > 0) {
    state.num_layers_on_device -= offloaded;
    state.is_degraded = (state.num_layers_on_device < state.num_layers);
    pa.update_layers_on_device(state.model_id, state.num_layers_on_device);
  }
  return offloaded;
}

// ============================================================
//  Load layers (tail-first restore) via broadcast
// ============================================================

int32_t LayerOffloadManager::load_layers(PerModelState& state, int32_t count) {
  int32_t loaded = 0;
  auto& allocator = XTensorAllocator::get_instance();
  auto& pa = PageAllocator::get_instance();

  for (int32_t i = 0; i < count; ++i) {
    int32_t layer_id = state.num_layers_on_device + i;
    if (layer_id >= state.num_layers) break;

    auto pages_per_worker =
        allocator.broadcast_load_layer_weights(state.model_id, layer_id);

    bool any_failed = false;
    for (size_t w = 0; w < pages_per_worker.size(); ++w) {
      if (pages_per_worker[w] < 0) {
        any_failed = true;
        LOG(ERROR) << "[LayerOffloadManager] load failed on worker=" << w
                   << " model=" << state.model_id << " layer_id=" << layer_id;
        break;
      }
    }
    if (any_failed) break;

    //TODO: support dp
    pa.consume_phy_pages_for_dp(state.model_id, 0, pages_per_worker[0]);

    ++loaded;
  }

  if (loaded > 0) {
    state.num_layers_on_device += loaded;
    state.is_degraded = (state.num_layers_on_device < state.num_layers);
    pa.update_layers_on_device(state.model_id, state.num_layers_on_device);
  }
  return loaded;
}

// ============================================================
//  Model selection
// ============================================================

LayerOffloadManager::PerModelState*
LayerOffloadManager::pick_lowest_priority_awake() {
  PerModelState* candidate = nullptr;
  for (auto& [id, state] : per_model_) {
    LOG(INFO) << "pick_lowest_priority_awake model=" << id
        << " num_layers_on_device=" << state->num_layers_on_device;
    if (state->num_layers_on_device <= 0) continue;
    if (PageAllocator::get_instance().is_model_sleeping(id)) continue;
    if (candidate == nullptr || state->priority < candidate->priority) {
      candidate = state.get();
    }
  }
  return candidate;
}

LayerOffloadManager::PerModelState*
LayerOffloadManager::pick_highest_priority_degraded() {
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

}  // namespace xllm
