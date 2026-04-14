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
#include <cctype>
#include <chrono>

#include "common/global_flags.h"
#include "core/framework/request/request_metric_aggregator.h"
#include "framework/prefix_cache/global_prefix_cache_manager.h"
#include "framework/xtensor/page_allocator.h"
#include "framework/xtensor/xtensor_allocator.h"

namespace xllm {

namespace {
std::string normalize_base_model_id(const std::string& model_id) {
  const size_t hash_pos = model_id.rfind('#');
  if (hash_pos == std::string::npos || hash_pos == 0 ||
      hash_pos == model_id.size() - 1) {
    return model_id;
  }
  for (size_t i = hash_pos + 1; i < model_id.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(model_id[i]))) {
      return model_id;
    }
  }
  return model_id.substr(0, hash_pos);
}
}  // namespace

void LayerOffloadManager::update_model_copies_delta_locked(
    const std::string& any_model_id,
    int delta) {
  const std::string base_model_id = normalize_base_model_id(any_model_id);
  int64_t& valid_copies = base_model_valid_copies_[base_model_id];
  valid_copies += delta;
  if (valid_copies < 0) {
    LOG(ERROR) << "Invalid model copies: " << base_model_id << " " << valid_copies;
    valid_copies = 0;
  }
  RequestMetricAggregator::instance().update_model_copies(base_model_id,
                                                          valid_copies);
}

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
  update_model_copies_delta_locked(model_id, +1);
  LOG(INFO) << "[LayerOffloadManager] Registered model=" << model_id
            << " num_layers=" << num_layers << " priority=" << priority;
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
        if (free_ratio >= FLAGS_layer_offload_low_watermark_ratio * 0.5) break;

        const auto pressure_workers = pa.get_pressure_workers_by_low_watermark_ratio(
            FLAGS_layer_offload_low_watermark_ratio);
        const auto pressure_models = pa.get_models_by_workers(pressure_workers);
        const bool scoped_offload = !pressure_models.empty();
        if (scoped_offload) {
          VLOG(1) << "[LayerOffloadManager] Scoped degrade round=" << round
                  << " pressure_workers=" << pressure_workers.size()
                  << " pressure_models=" << pressure_models.size();
        }

        int32_t offloaded =
            offload_internal(OffloadContext::kMonitorDegrade,
                             free_ratio,
                             scoped_offload ? &pressure_models : nullptr);
        if (offloaded < 0 && scoped_offload) {
          LOG(ERROR) << "[LayerOffloadManager] Scoped candidate is empty, "
                       << " free_ratio=" << free_ratio
                       << " pressure_workers=" << pressure_workers.size()
                       << " pressure_models=" << pressure_models.size();
        }
        if (offloaded <= 0) break;
        total_offloaded += offloaded;
      }

      if (total_offloaded > 0) {
        LOG(INFO) << "[LayerOffloadManager] Degrade complete, total_offloaded="
                  << total_offloaded << " final_free_ratio="
                  << static_cast<double>(pa.get_num_free_phy_pages()) /
                         total_pages;
      }

    } /*else if (free_ratio >= FLAGS_layer_offload_restore_watermark_ratio) {
      // ---- RESTORE path ----
      const auto healthy_workers =
          pa.get_healthy_workers_by_high_watermark_ratio(
              FLAGS_layer_offload_restore_watermark_ratio);
      const auto healthy_models = pa.get_models_by_workers(healthy_workers);
      const bool scoped_restore = !healthy_models.empty();
      if (scoped_restore) {
        VLOG(1) << "[LayerOffloadManager] Scoped restore "
                << "healthy_workers=" << healthy_workers.size()
                << " scoped_restore_models=" << healthy_models.size();
      }

      PerModelState* target = pick_highest_priority_degraded(
          scoped_restore ? &healthy_models : nullptr);
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
    }*/
  }
}

// ============================================================
//  One-shot offload chunk (pick + broadcast), serialized
// ============================================================

int32_t LayerOffloadManager::offload_internal(OffloadContext ctx,
                                              double free_ratio_for_log,
                                              const std::unordered_set<std::string>*
                                                  allowed_models) {
  std::lock_guard<std::mutex> lock(mtx_);

  PerModelState* target = pick_lowest_priority_awake(allowed_models);
  if (target == nullptr) {
    const bool has_filter = (allowed_models != nullptr);
    if (ctx == OffloadContext::kMonitorDegrade) {
      LOG(WARNING) << "[LayerOffloadManager] No awake model with offloadable "
                   << "layers. free_ratio=" << free_ratio_for_log
                   << " scoped_models="
                   << (has_filter ? allowed_models->size() : 0);
    } else {
      LOG(WARNING) << "[EmergencyEviction] No awake model with offloadable "
                   << "layers. scoped_models="
                   << (has_filter ? allowed_models->size() : 0);
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

std::optional<std::string> LayerOffloadManager::trigger_load_for_base_model(
    const std::string& base_model_id,
    const std::string& reason) {
  const std::string normalized_base = normalize_base_model_id(base_model_id);
  auto& pa = PageAllocator::get_instance();
  auto selected_model_id = pa.pick_best_loadable_model_for_base(normalized_base);
  if (selected_model_id.has_value()) {
    const size_t total_weight_pages =
        pa.get_weight_pages_allocated(*selected_model_id);
    const size_t offloaded_pages =
        pa.get_layer_offloaded_phy_pages(*selected_model_id);
    LOG(INFO) << "[LayerOffloadManager] trigger_load_for_base_model reason="
              << reason << " base_model_id=" << normalized_base
              << " selected_model_id=" << *selected_model_id
              << " total_weight_pages=" << total_weight_pages
              << " offloaded_weight_pages=" << offloaded_pages;

    auto& allocator = XTensorAllocator::get_instance();
    const auto [model_dp_size, model_tp_size] =
        allocator.get_model_parallel_strategy(*selected_model_id);
    const int32_t worker_rank_base =
        allocator.get_model_worker_rank_base(*selected_model_id);
    int32_t selected_world_size = pa.get_model_world_size(*selected_model_id);
    if (selected_world_size <= 0) {
      selected_world_size = model_dp_size * model_tp_size;
    }
    if (selected_world_size <= 0) {
      LOG(WARNING) << "[LayerOffloadManager] trigger_load_for_base_model "
                   << "cannot infer worker range for selected_model_id="
                   << *selected_model_id;
      return selected_model_id;
    }

    std::unordered_set<int32_t> selected_workers;
    selected_workers.reserve(static_cast<size_t>(selected_world_size));
    for (int32_t w = worker_rank_base;
         w < worker_rank_base + selected_world_size;
         ++w) {
      selected_workers.insert(w);
    }

    const auto scoped_models = pa.get_models_by_workers(selected_workers);

    //offload model weights
    int32_t degraded_layers_offloaded = 0;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      for (auto& [id, state] : per_model_) {
        if (scoped_models.count(id) == 0) continue;
        if (normalize_base_model_id(id) == normalized_base) continue;
        if (!state->is_degraded || state->num_layers_on_device <= 0) continue;
        const int32_t offloaded_layers =
            offload_layers(*state, state->num_layers_on_device);
        if (offloaded_layers > 0) {
          degraded_layers_offloaded += offloaded_layers;
          LOG(INFO) << "[LayerOffloadManager] trigger_load_for_base_model "
                    << "offloaded_degraded_model=" << id
                    << " layers=" << offloaded_layers
                    << " remaining_on_device=" << state->num_layers_on_device;
        }
      }
    }

    // evict prefix cache
    const size_t total_cached_blocks =
        GlobalPrefixCacheManager::instance().get_total_cached_blocks();
    size_t evicted_prefix_blocks = 0;
    if (total_cached_blocks > 0 && !scoped_models.empty()) {
      evicted_prefix_blocks = GlobalPrefixCacheManager::instance()
                                  .evict_global_pure_lru(total_cached_blocks,
                                                         scoped_models);
    }
    if (!scoped_models.empty()) {
      pa.reclaim_excess_reserved_pages_for_models(scoped_models);
    }

    int32_t selected_layers_loaded = 0;
    int32_t selected_layers_to_load = 0;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      auto it = per_model_.find(*selected_model_id);
      if (it == per_model_.end()) {
        LOG(WARNING) << "[LayerOffloadManager] trigger_load_for_base_model "
                     << "selected model not registered in offload manager: "
                     << *selected_model_id;
      } else {
        PerModelState& selected_state = *it->second;
        selected_layers_to_load =
            std::max(0, selected_state.num_layers - selected_state.num_layers_on_device);
        if (selected_layers_to_load > 0) {
          selected_layers_loaded = load_layers(selected_state, selected_layers_to_load);
        }
      }
    }
    LOG(INFO) << "[LayerOffloadManager] trigger_load_for_base_model selected_model_id="
              << *selected_model_id
              << " selected_layers_loaded=" << selected_layers_loaded
              << " selected_layers_to_load=" << selected_layers_to_load;

    LOG(INFO) << "[LayerOffloadManager] trigger_load_for_base_model reason="
              << reason << " base_model_id=" << normalized_base
              << " selected_model_id=" << *selected_model_id
              << " scoped_workers=" << selected_workers.size()
              << " scoped_models=" << scoped_models.size()
              << " degraded_layers_offloaded=" << degraded_layers_offloaded
              << " evicted_prefix_blocks=" << evicted_prefix_blocks
              << " selected_layers_loaded=" << selected_layers_loaded;
    return selected_model_id;
  }

  LOG(INFO) << "[LayerOffloadManager] trigger_load_for_base_model reason="
            << reason << " base_model_id=" << normalized_base
            << " no loadable degraded replica found";
  return std::nullopt;
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
  const bool was_full_before_offload = !state.is_degraded;

  // First transition from fully loaded -> degraded:
  // block new request admission, wait for in-flight requests to drain, then
  // enter step safety point before offloading.
  if (was_full_before_offload) {
    pa.block_model_schedule(state.model_id);
    pa.wait_model_requests_drained(state.model_id);
  }

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

    // DP not supported
    // if tp workers operates different number of pages, 
    // we need to apply per-worker page count
    pa.release_phy_pages_for_dp(state.model_id, 0, pages_per_worker[0]);
    pa.update_layer_offloaded_phy_pages_delta(
        state.model_id, static_cast<int64_t>(pages_per_worker[0]));

    ++offloaded;
  }

  if (offloaded > 0) {
    state.num_layers_on_device -= offloaded;
    state.is_degraded = (state.num_layers_on_device < state.num_layers);
    pa.update_layers_on_device(state.model_id, state.num_layers_on_device);
    if (was_full_before_offload) {
      pa.release_all_reserved_pages_for_models({state.model_id});
      update_model_copies_delta_locked(state.model_id, -1);
    }
  } else if (was_full_before_offload) {
    // Offload made no progress, release schedule block.
    pa.unblock_model_schedule(state.model_id);
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

    //DP not supported
    pa.consume_phy_pages_for_dp(state.model_id, 0, pages_per_worker[0]);
    pa.update_layer_offloaded_phy_pages_delta(
        state.model_id, -static_cast<int64_t>(pages_per_worker[0]));

    ++loaded;
  }

  if (loaded > 0) {
    const bool was_degraded = state.is_degraded;
    state.num_layers_on_device += loaded;
    state.is_degraded = (state.num_layers_on_device < state.num_layers);
    pa.update_layers_on_device(state.model_id, state.num_layers_on_device);
    if (was_degraded && !state.is_degraded) {
      pa.allocate_all_reserved_pages_for_models({state.model_id});
      update_model_copies_delta_locked(state.model_id, +1);
      // Fully restored, allow step scheduling again.
      pa.unblock_model_schedule(state.model_id);
    }
  }
  return loaded;
}

// ============================================================
//  Model selection
// ============================================================

LayerOffloadManager::PerModelState*
LayerOffloadManager::pick_lowest_priority_awake(
    const std::unordered_set<std::string>* allowed_models) {
  PerModelState* candidate = nullptr;
  double candidate_priority = 0.0;
  bool candidate_is_partially_offloaded = false;
  const bool scoped = (allowed_models != nullptr);
  for (auto& [id, state] : per_model_) {
    if (scoped && allowed_models->count(id) == 0) continue;
    if (state->num_layers_on_device <= 0) continue;
    if (PageAllocator::get_instance().is_model_sleeping(id)) continue;
    const bool is_partially_offloaded =
        state->num_layers_on_device > 0 &&
        state->num_layers_on_device < state->num_layers;
    const double runtime_priority =
        RequestMetricAggregator::instance().get_model_priority(normalize_base_model_id(id));
    LOG(INFO) << "pick_lowest_priority_awake model=" << id
              << " num_layers_on_device=" << state->num_layers_on_device
              << " runtime_priority=" << runtime_priority;
    if (candidate == nullptr ||
        (!candidate_is_partially_offloaded && is_partially_offloaded) ||
        (is_partially_offloaded == candidate_is_partially_offloaded &&
         runtime_priority < candidate_priority)) {
      candidate = state.get();
      candidate_priority = runtime_priority;
      candidate_is_partially_offloaded = is_partially_offloaded;
    }
  }
  return candidate;
}

LayerOffloadManager::PerModelState*
LayerOffloadManager::pick_highest_priority_degraded(
    const std::unordered_set<std::string>* allowed_models) {
  PerModelState* candidate = nullptr;
  double candidate_priority = 0.0;
  bool candidate_is_partially_loaded = false;
  const bool scoped = (allowed_models != nullptr);
  for (auto& [id, state] : per_model_) {
    if (scoped && allowed_models->count(id) == 0) continue;
    if (!state->is_degraded) continue;
    if (state->num_layers_on_device >= state->num_layers) continue;
    const bool is_partially_loaded = state->num_layers_on_device > 0;
    const double runtime_priority =
        RequestMetricAggregator::instance().get_model_priority(normalize_base_model_id(id));
    LOG(INFO) << "pick_highest_priority_degraded model=" << id
              << " num_layers_on_device=" << state->num_layers_on_device
              << " runtime_priority=" << runtime_priority;
    if (candidate == nullptr ||
        (!candidate_is_partially_loaded && is_partially_loaded) ||
        (is_partially_loaded == candidate_is_partially_loaded &&
         runtime_priority > candidate_priority)) {
      candidate = state.get();
      candidate_priority = runtime_priority;
      candidate_is_partially_loaded = is_partially_loaded;
    }
  }
  return candidate;
}

}  // namespace xllm
