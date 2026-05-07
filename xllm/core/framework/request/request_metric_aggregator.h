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

  void add_sample(const std::string& model_id,
                  double ttft_ms,
                  double tpot_ms,
                  bool has_ttft = true,
                  bool has_tpot = true);

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

  // Get model TPOT SLO (ms). Falls back to global default when model meta is
  // absent.
  int32_t get_model_tpot_slo_ms(const std::string& model_id);

  // Get model TTFT SLO (ms). Falls back to global default when model meta is
  // absent.
  int32_t get_model_ttft_slo_ms(const std::string& model_id);

  // Compute memory-aware replica dispatch weights for one base model.
  // Weight is inversely proportional to per-replica max(worker_used).
  // Returned vector size equals replica_model_ids size, each weight >= 1.
  std::vector<int32_t> get_replica_dispatch_weights(
      const std::string& model_id,
      const std::vector<std::string>& replica_model_ids);

 private:
  struct Sample {
    std::string model_id;
    double ttft_ms = 0.0;
    double tpot_ms = 0.0;
    bool has_ttft = false;
    bool has_tpot = false;
  };

  struct ModelMeta {
    double avg_ttft_ms = 0.0;
    double avg_tpot_ms = 0.0;
    int32_t ttft_slo_ms = 0;
    int32_t tpot_slo_ms = 0;
    size_t model_copies = 0;
    bool has_slo = false;
    int32_t load_cooldown_windows = 0;
  };

  struct DispatchWeightCache {
    std::vector<std::string> replica_model_ids;
    std::vector<int32_t> weights;
  };

  RequestMetricAggregator();
  ~RequestMetricAggregator();

  std::vector<int32_t> compute_replica_dispatch_weights_impl(
      const std::string& model_id,
      const std::vector<std::string>& replica_model_ids);
  void refresh_dispatch_weights_cache();
  void worker_loop();

  int window_size_ms_ = 0;
  bool enabled_ = false;

  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::vector<Sample> samples_;
  std::unordered_map<std::string, ModelMeta> model_meta_;
  std::unordered_map<std::string, DispatchWeightCache> dispatch_weight_cache_;
  std::thread worker_;
};

}  // namespace xllm
