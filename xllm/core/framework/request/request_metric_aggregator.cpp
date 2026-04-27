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

#include "request_metric_aggregator.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/xtensor/page_allocator.h"
#include "core/framework/xtensor/xtensor_allocator.h"

namespace xllm {
namespace {

constexpr int32_t kLoadTriggerCooldownWindows = 1;

struct PriorityScoreComponents {
  double quality_score = 0.0;
  double slo_violation_score = 0.0;
  double model_priority = 0.0;
};

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

PriorityScoreComponents compute_priority_score_components(
  double avg_ttft_ms,
  double avg_tpot_ms,
  int32_t ttft_slo_ms,
  int32_t tpot_slo_ms,
  size_t model_copies) {
  const double safe_ttft_slo_ms = std::max(1.0, static_cast<double>(ttft_slo_ms));
  const double safe_tpot_slo_ms = std::max(1.0, static_cast<double>(tpot_slo_ms));
  const double safe_model_copies =
      std::max(1.0, static_cast<double>(model_copies));

  const double ttft_ratio = avg_ttft_ms / safe_ttft_slo_ms;
  const double tpot_ratio = avg_tpot_ms / safe_tpot_slo_ms;

  // Higher latency-SLO ratio means worse service quality in current window.
  const double quality_score = 0.5 * (ttft_ratio + tpot_ratio);

  // Extra penalty when violating SLO to boost urgency.
  const double slo_violation_score =
      0.5 * std::max(0.0, ttft_ratio - 1.0) +
      0.5 * std::max(0.0, tpot_ratio - 1.0);

  // More model copies should reduce priority.
  const double model_priority =
      (quality_score + 2.0 * slo_violation_score) / safe_model_copies;
  return {quality_score, slo_violation_score, model_priority};
}

double compute_model_priority(double avg_ttft_ms,
                              double avg_tpot_ms,
                              int32_t ttft_slo_ms,
                              int32_t tpot_slo_ms,
                              size_t model_copies) {
  return compute_priority_score_components(
             avg_ttft_ms, avg_tpot_ms, ttft_slo_ms, tpot_slo_ms, model_copies)
      .model_priority;
}

}  // namespace

RequestMetricAggregator& RequestMetricAggregator::instance() {
  static RequestMetricAggregator aggregator;
  return aggregator;
}

RequestMetricAggregator::RequestMetricAggregator() {
  window_size_ms_ = FLAGS_priority_window_size;
  enabled_ = window_size_ms_ > 0;
  if (!enabled_) {
    return;
  }
  worker_ = std::thread(&RequestMetricAggregator::worker_loop, this);
}

RequestMetricAggregator::~RequestMetricAggregator() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void RequestMetricAggregator::add_sample(const std::string& model_id,
                                         double ttft_ms,
                                         double tpot_ms,
                                         bool has_ttft,
                                         bool has_tpot) {
  if (!enabled_ || model_id.empty()) {
    return;
  }
  if (!has_ttft && !has_tpot) {
    return;
  }
  std::lock_guard<std::mutex> lock(mu_);
  samples_.push_back(Sample{model_id, ttft_ms, tpot_ms, has_ttft, has_tpot});
}

void RequestMetricAggregator::update_model_slo(const std::string& model_id,
                                                int32_t ttft_slo_ms,
                                                int32_t tpot_slo_ms) {
  if (!enabled_ || model_id.empty()) {
    return;
  }
  const std::string base_model_id = normalize_base_model_id(model_id);
  std::lock_guard<std::mutex> lock(mu_);
  auto& meta = model_meta_[base_model_id];
  if (!meta.has_slo) {
    meta.ttft_slo_ms = ttft_slo_ms;
    meta.tpot_slo_ms = tpot_slo_ms;
    meta.has_slo = true;
  }
}

void RequestMetricAggregator::update_model_copies(const std::string& model_id,
                                                  size_t model_copies) {
  if (!enabled_ || model_id.empty()) {
    return;
  }
  const std::string base_model_id = normalize_base_model_id(model_id);
  std::lock_guard<std::mutex> lock(mu_);
  auto& meta = model_meta_[base_model_id];
  meta.model_copies = model_copies;
}

double RequestMetricAggregator::get_model_priority(const std::string& model_id) {
  if (!enabled_ || model_id.empty()) {
    return 0.0;
  }
  const std::string base_model_id = normalize_base_model_id(model_id);
  std::lock_guard<std::mutex> lock(mu_);
  auto it = model_meta_.find(base_model_id);
  if (it == model_meta_.end()) {
    return 0.0;
  }
  const auto& meta = it->second;
  return compute_model_priority(meta.avg_ttft_ms, meta.avg_tpot_ms,
                                meta.ttft_slo_ms, meta.tpot_slo_ms,
                                meta.model_copies);
}

std::vector<int32_t> RequestMetricAggregator::get_replica_dispatch_weights(
    const std::string& model_id,
    const std::vector<std::string>& replica_model_ids) {
  std::vector<int32_t> default_weights(replica_model_ids.size(), 1);
  if (replica_model_ids.empty()) {
    return default_weights;
  }

  const std::string base_model_id = normalize_base_model_id(model_id);
  std::lock_guard<std::mutex> lock(mu_);
  auto& cache = dispatch_weight_cache_[base_model_id];
  if (cache.replica_model_ids != replica_model_ids) {
    cache.replica_model_ids = replica_model_ids;
    cache.weights = default_weights;
  }
  if (cache.weights.size() != replica_model_ids.size()) {
    cache.weights = default_weights;
  }
  return cache.weights;
}

std::vector<int32_t> RequestMetricAggregator::compute_replica_dispatch_weights_impl(
    const std::string& model_id,
    const std::vector<std::string>& replica_model_ids) {
  std::vector<int32_t> weights(replica_model_ids.size(), 1);
  if (replica_model_ids.empty()) {
    return weights;
  }

  auto& page_allocator = PageAllocator::get_instance();
  if (!FLAGS_enable_xtensor || !page_allocator.is_initialized()) {
    return weights;
  }

  const size_t total_pages = page_allocator.get_num_total_phy_pages();
  if (total_pages == 0) {
    return weights;
  }

  constexpr int32_t kWeightScale = 100;
  std::vector<size_t> used_pages;
  used_pages.reserve(replica_model_ids.size());
  std::vector<double> raw_scores;
  raw_scores.reserve(replica_model_ids.size());
  std::vector<bool> degraded_flags(replica_model_ids.size(), false);
  double total_raw_score = 0.0;

  for (size_t i = 0; i < replica_model_ids.size(); ++i) {
    const std::string& replica_model_id = replica_model_ids[i];
    if (page_allocator.is_model_schedule_blocked(replica_model_id)) {
      degraded_flags[i] = true;
      used_pages.push_back(0);
      raw_scores.push_back(0.0);
      continue;
    }

    const size_t free =
        page_allocator.get_free_phy_pages_for_model(replica_model_id);
    const size_t used = (free >= total_pages) ? 0 : (total_pages - free);
    const size_t safe_used = std::max<size_t>(used, 1);
    const double raw_score = 1.0 / static_cast<double>(safe_used);

    used_pages.push_back(used);
    raw_scores.push_back(raw_score);
    total_raw_score += raw_score;
  }

  if (total_raw_score <= std::numeric_limits<double>::epsilon()) {
    return weights;
  }

  int32_t total_weight = 0;
  for (size_t i = 0; i < raw_scores.size(); ++i) {
    const int32_t weight = std::max<int32_t>(
        1,
        static_cast<int32_t>(std::lround(raw_scores[i] / total_raw_score *
                                         static_cast<double>(kWeightScale))));
    weights[i] = weight;
    total_weight += weight;
  }

  std::ostringstream oss;
  oss << "[dispatch weight]: base_model_id="
      << normalize_base_model_id(model_id)
      << ", total_pages=" << total_pages
      << ", total_weight=" << total_weight
      << ", replicas=[";
  bool has_logged_replica = false;
  for (size_t i = 0; i < replica_model_ids.size(); ++i) {
    if (degraded_flags[i]) {
      continue;
    }
    if (has_logged_replica) {
      oss << "; ";
    }
    has_logged_replica = true;
    oss << "{model_id=" << replica_model_ids[i]
        << ", used_pages=" << used_pages[i]
        << ", weight=" << weights[i] << "}";
  }
  oss << "]";
  LOG(INFO) << oss.str();

  return weights;
}

void RequestMetricAggregator::refresh_dispatch_weights_cache() {
  std::vector<std::pair<std::string, std::vector<std::string>>> snapshots;
  {
    std::lock_guard<std::mutex> lock(mu_);
    snapshots.reserve(dispatch_weight_cache_.size());
    for (const auto& [base_model_id, cache] : dispatch_weight_cache_) {
      if (cache.replica_model_ids.empty()) {
        continue;
      }
      snapshots.emplace_back(base_model_id, cache.replica_model_ids);
    }
  }

  for (const auto& [base_model_id, replica_model_ids] : snapshots) {
    std::vector<int32_t> updated_weights =
        compute_replica_dispatch_weights_impl(base_model_id, replica_model_ids);
    std::lock_guard<std::mutex> lock(mu_);
    auto it = dispatch_weight_cache_.find(base_model_id);
    if (it == dispatch_weight_cache_.end()) {
      continue;
    }
    if (it->second.replica_model_ids != replica_model_ids) {
      continue;
    }
    it->second.weights = std::move(updated_weights);
  }
}

void RequestMetricAggregator::worker_loop() {
  while (true) {
    std::vector<Sample> window_samples;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait_for(lock,
                   std::chrono::milliseconds(window_size_ms_),
                   [this] { return stop_; });
      if (stop_) {
        break;
      }
      if (!samples_.empty()) {
        window_samples.swap(samples_);
      }
    }

    // Refresh cached dispatch weights once per worker loop window.
    refresh_dispatch_weights_cache();

    if (FLAGS_node_rank == 0) {
      auto& page_allocator = PageAllocator::get_instance();
      if (FLAGS_enable_xtensor && page_allocator.is_initialized()) {
        const auto model_memory_usage = page_allocator.get_model_memory_usage();
        for (const auto& [model_id, usage] : model_memory_usage) {
          if (usage.weight_phy_pages == 0 && usage.kv_cache_phy_pages <= 128) {
            continue;
          }
          LOG(INFO) << "[memory sample]: model_id=" << model_id
                    << ", weight_phy_pages=" << usage.weight_phy_pages
                    << ", kv_cache_phy_pages=" << usage.kv_cache_phy_pages;
        }
      }
    }
    if (FLAGS_enable_xtensor) {
      const size_t activation_pages =
          XTensorAllocator::get_instance().get_activation_allocated_pages();
      LOG(INFO) << "[memory sample]: activation_allocated_pages="
                << activation_pages;
    }
    if (window_samples.empty()) {
      continue;
    }

    struct Aggregated {
      size_t count = 0;
      size_t ttft_count = 0;
      size_t tpot_count = 0;
      double ttft_sum_ms = 0.0;
      double tpot_sum_ms = 0.0;
      size_t violation_count = 0;
    };
    std::unordered_map<std::string, Aggregated> grouped;
    grouped.reserve(window_samples.size());
    for (const auto& sample : window_samples) {
      const std::string base_model_id = normalize_base_model_id(sample.model_id);
      int32_t ttft_slo_ms = FLAGS_priority_ttft_slo_ms;
      int32_t tpot_slo_ms = FLAGS_priority_tpot_slo_ms;
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = model_meta_.find(base_model_id);
        if (it != model_meta_.end() && it->second.has_slo) {
          ttft_slo_ms = it->second.ttft_slo_ms;
          tpot_slo_ms = it->second.tpot_slo_ms;
        }
      }
      auto& item = grouped[base_model_id];
      ++item.count;
      if (sample.has_ttft) {
        ++item.ttft_count;
        item.ttft_sum_ms += sample.ttft_ms;
      }
      if (sample.has_tpot) {
        ++item.tpot_count;
        item.tpot_sum_ms += sample.tpot_ms;
      }
      const bool violated =
          (sample.has_ttft && sample.ttft_ms > std::max(1, ttft_slo_ms)) ||
          (sample.has_tpot && sample.tpot_ms > std::max(1, tpot_slo_ms));
      if (violated) {
        ++item.violation_count;
      }
    }

    for (const auto& [base_model_id, metric] : grouped) {
      const bool has_ttft_sample = metric.ttft_count > 0;
      const bool has_tpot_sample = metric.tpot_count > 0;
      const double avg_ttft_ms =
          has_ttft_sample ? (metric.ttft_sum_ms / metric.ttft_count) : 0.0;
      const double avg_tpot_ms =
          has_tpot_sample ? (metric.tpot_sum_ms / metric.tpot_count) : 0.0;
      const double violation_rate =
          static_cast<double>(metric.violation_count) * 100.0 / metric.count;
      ModelMeta meta_snapshot;
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto& meta = model_meta_[base_model_id];
        if (!meta.has_slo) {
          meta.ttft_slo_ms = FLAGS_priority_ttft_slo_ms;
          meta.tpot_slo_ms = FLAGS_priority_tpot_slo_ms;
          meta.has_slo = true;
        }
        if (meta.model_copies == 0) {
          meta.model_copies = 1;
        }
        if (has_ttft_sample) {
          meta.avg_ttft_ms = avg_ttft_ms;
        }
        if (has_tpot_sample) {
          meta.avg_tpot_ms = avg_tpot_ms;
        }
        if (meta.load_cooldown_windows > 0) {
          --meta.load_cooldown_windows;
        }
        meta_snapshot = meta;
      }
        const PriorityScoreComponents score_components =
          compute_priority_score_components(meta_snapshot.avg_ttft_ms,
                          meta_snapshot.avg_tpot_ms,
                          meta_snapshot.ttft_slo_ms,
                          meta_snapshot.tpot_slo_ms,
                          meta_snapshot.model_copies);
      LOG(INFO) << "[priority window metric]: model_id=" << base_model_id
                << ", window_size_ms=" << window_size_ms_
                << ", ttft_samples=" << metric.ttft_count
                << ", tpot_samples=" << metric.tpot_count
                << ", violation_count=" << metric.violation_count
                << ", violation_rate=" << violation_rate
                << ", avg_ttft_ms=" << meta_snapshot.avg_ttft_ms
                << ", avg_tpot_ms=" << meta_snapshot.avg_tpot_ms
                << ", ttft_slo_ms=" << meta_snapshot.ttft_slo_ms
                << ", tpot_slo_ms=" << meta_snapshot.tpot_slo_ms
            << ", quality_score=" << score_components.quality_score
            << ", slo_violation_score="
            << score_components.slo_violation_score
                << ", model_copies=" << meta_snapshot.model_copies
            << ", model_priority=" << score_components.model_priority;

      const int32_t trigger_threshold =
          std::clamp(FLAGS_load_model_slo_violation_rate, 0, 100);
      if (violation_rate > static_cast<double>(trigger_threshold) &&
          metric.violation_count >= window_size_ms_ / 10 &&
          meta_snapshot.load_cooldown_windows == 0) {
        auto& page_allocator = PageAllocator::get_instance();
        if (FLAGS_enable_xtensor && page_allocator.is_initialized()) {
          if (auto* mgr = page_allocator.get_layer_offload_manager();
              mgr != nullptr) {
            std::optional<std::string> model_id = mgr->trigger_load_for_base_model(base_model_id,
                                             "slo_violation_rate_exceeded");
            if (model_id.has_value()) {
              std::lock_guard<std::mutex> lock(mu_);
              model_meta_[base_model_id].load_cooldown_windows =
                  kLoadTriggerCooldownWindows;
            }
          }
        }
      } else if (meta_snapshot.load_cooldown_windows > 0) {
        LOG(INFO) << "[load trigger skipped by cooldown]: model_id="
                  << base_model_id
                  << ", remaining_windows="
                  << meta_snapshot.load_cooldown_windows;
      }
    }
  }
}

}  // namespace xllm
