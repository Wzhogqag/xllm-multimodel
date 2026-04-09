#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace xllm {

class RequestMetricAggregator {
 public:
  static RequestMetricAggregator& instance();

  RequestMetricAggregator(const RequestMetricAggregator&) = delete;
  RequestMetricAggregator& operator=(const RequestMetricAggregator&) = delete;

  void add_sample(const std::string& model_id, double ttft_ms, double tpot_ms);

  // Register/update per-model static parameters.
  // SLO values are only set on first registration for a base model.
  // model_copies is always refreshed with latest value.
  void update_model_slo(const std::string& model_id,
                         int32_t ttft_slo_ms,
                         int32_t tpot_slo_ms);

  void update_model_copies(const std::string& model_id, size_t model_copies);

  // Priority score based on latency/SLO quality and model copies.
  // Worse current-window quality gives higher score; more copies lower score.
  double get_model_priority(const std::string& model_id);

 private:
  struct Sample {
    std::string model_id;
    double ttft_ms = 0.0;
    double tpot_ms = 0.0;
  };

  struct ModelMeta {
    double avg_ttft_ms = 0.0;
    double avg_tpot_ms = 0.0;
    int32_t ttft_slo_ms = 0;
    int32_t tpot_slo_ms = 0;
    size_t model_copies = 0;
    bool has_slo = false;
  };

  RequestMetricAggregator();
  ~RequestMetricAggregator();

  void worker_loop();

  int window_size_seconds_ = 0;
  bool enabled_ = false;

  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::vector<Sample> samples_;
  std::unordered_map<std::string, ModelMeta> model_meta_;
  std::thread worker_;
};

}  // namespace xllm
