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
#include <cstddef>
#include <unordered_map>
#include <utility>

#include "core/common/global_flags.h"
#include "core/framework/xtensor/page_allocator.h"

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

double compute_model_priority(double avg_ttft_ms,
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
  return (quality_score + 2.0 * slo_violation_score) / safe_model_copies;
}

}  // namespace

RequestMetricAggregator& RequestMetricAggregator::instance() {
  static RequestMetricAggregator aggregator;
  return aggregator;
}

RequestMetricAggregator::RequestMetricAggregator() {
  window_size_seconds_ = FLAGS_priority_window_size;
  enabled_ = window_size_seconds_ > 0;
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
                                         double tpot_ms) {
  if (!enabled_ || model_id.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(mu_);
  samples_.push_back(Sample{model_id, ttft_ms, tpot_ms});
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

void RequestMetricAggregator::worker_loop() {
  while (true) {
    std::vector<Sample> window_samples;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait_for(lock,
                   std::chrono::seconds(window_size_seconds_),
                   [this] { return stop_; });
      if (stop_) {
        break;
      }
      if (samples_.empty()) {
        continue;
      }
      window_samples.swap(samples_);
    }

    struct Aggregated {
      size_t count = 0;
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
      item.ttft_sum_ms += sample.ttft_ms;
      item.tpot_sum_ms += sample.tpot_ms;
      const bool violated = sample.ttft_ms > std::max(1, ttft_slo_ms) ||
                            sample.tpot_ms > std::max(1, tpot_slo_ms);
      if (violated) {
        ++item.violation_count;
      }
    }

    for (const auto& [base_model_id, metric] : grouped) {
      const double avg_ttft_ms = metric.ttft_sum_ms / metric.count;
      const double avg_tpot_ms = metric.tpot_sum_ms / metric.count;
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
        meta.avg_ttft_ms = avg_ttft_ms;
        meta.avg_tpot_ms = avg_tpot_ms;
        meta_snapshot = meta;
      }
      LOG(INFO) << "priority window metric: model_id=" << base_model_id
                << ", window_size_s=" << window_size_seconds_
                << ", samples=" << metric.count
                << ", violation_count=" << metric.violation_count
                << ", violation_rate=" << violation_rate
                << ", avg_ttft_ms=" << avg_ttft_ms
                << ", avg_tpot_ms=" << avg_tpot_ms
                << ", ttft_slo_ms=" << meta_snapshot.ttft_slo_ms
                << ", tpot_slo_ms=" << meta_snapshot.tpot_slo_ms
                << ", model_copies=" << meta_snapshot.model_copies
                << ", model_priority="
                << compute_model_priority(meta_snapshot.avg_ttft_ms,
                                          meta_snapshot.avg_tpot_ms,
                                          meta_snapshot.ttft_slo_ms,
                                          meta_snapshot.tpot_slo_ms,
                                          meta_snapshot.model_copies);

      const int32_t trigger_threshold =
          std::clamp(FLAGS_load_model_slo_violation_rate, 0, 100);
      if (violation_rate > static_cast<double>(trigger_threshold)) {
        auto& page_allocator = PageAllocator::get_instance();
        if (FLAGS_enable_xtensor && page_allocator.is_initialized()) {
          if (auto* mgr = page_allocator.get_layer_offload_manager();
              mgr != nullptr) {
            mgr->trigger_load_for_base_model(base_model_id,
                                             "slo_violation_rate_exceeded");
          }
        }
      }
    }
  }
}

}  // namespace xllm
