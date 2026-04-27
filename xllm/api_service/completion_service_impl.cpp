/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "completion_service_impl.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <random>
#include <string>

#include "common/instance_name.h"
#include "common/global_flags.h"
#include "completion.pb.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/framework/request/request_metric_aggregator.h"
#include "core/framework/request/request_output.h"
#include "core/framework/xtensor/page_allocator.h"
#include "core/util/utils.h"

#ifdef likely
#undef likely
#endif
#define likely(x) __builtin_expect(!!(x), 1)

#ifdef unlikely
#undef unlikely
#endif
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace xllm {
namespace {
void set_logprobs(proto::Choice* choice,
                  const std::optional<std::vector<LogProb>>& logprobs) {
  if (!logprobs.has_value() || logprobs.value().empty()) {
    return;
  }

  auto* proto_logprobs = choice->mutable_logprobs();
  for (const auto& logprob : logprobs.value()) {
    proto_logprobs->add_tokens(logprob.token);
    proto_logprobs->add_token_ids(logprob.token_id);
    proto_logprobs->add_token_logprobs(logprob.logprob);
  }
}

bool send_delta_to_client_brpc(std::shared_ptr<CompletionCall> call,
                               bool include_usage,
                               const std::string& request_id,
                               int64_t created_time,
                               const std::string& model,
                               const RequestOutput& output) {
  auto& response = call->response();

  for (const auto& seq_output : output.outputs) {
    if (!seq_output.text.empty()) {
      response.Clear();
      response.set_object("text_completion");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(seq_output.index);
      choice->set_text(seq_output.text);
      set_logprobs(choice, seq_output.logprobs);
      if (!call->write(response)) {
        return false;
      }
    }

    if (seq_output.finish_reason.has_value()) {
      response.Clear();
      response.set_object("text_completion");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(seq_output.index);
      choice->set_text("");
      choice->set_finish_reason(seq_output.finish_reason.value());
      if (!call->write(response)) {
        return false;
      }
    }
  }

  if (include_usage && output.usage.has_value()) {
    const auto& usage = output.usage.value();
    response.Clear();
    response.set_object("text_completion");
    response.set_id(request_id);
    response.set_created(created_time);
    response.set_model(model);
    response.mutable_choices();
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(
        static_cast<int32_t>(usage.num_prompt_tokens));
    proto_usage->set_completion_tokens(
        static_cast<int32_t>(usage.num_generated_tokens));
    proto_usage->set_total_tokens(static_cast<int32_t>(usage.num_total_tokens));
    if (!call->write(response)) {
      return false;
    }
  }

  if (output.finished || output.cancelled) {
    response.Clear();
    return call->finish();
  }
  return true;
}

bool send_result_to_client_brpc(std::shared_ptr<CompletionCall> call,
                                const std::string& request_id,
                                int64_t created_time,
                                const std::string& model,
                                const RequestOutput& req_output) {
  auto& response = call->response();
  response.set_object("text_completion");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  response.mutable_choices()->Reserve(req_output.outputs.size());
  for (const auto& output : req_output.outputs) {
    auto* choice = response.add_choices();
    choice->set_index(output.index);
    choice->set_text(output.text);
    set_logprobs(choice, output.logprobs);
    if (output.finish_reason.has_value()) {
      choice->set_finish_reason(output.finish_reason.value());
    }
  }

  if (req_output.usage.has_value()) {
    const auto& usage = req_output.usage.value();
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(
        static_cast<int32_t>(usage.num_prompt_tokens));
    proto_usage->set_completion_tokens(
        static_cast<int32_t>(usage.num_generated_tokens));
    proto_usage->set_total_tokens(static_cast<int32_t>(usage.num_total_tokens));
  }

  return call->write_and_finish(response);
}

}  // namespace

CompletionServiceImpl::CompletionServiceImpl(
    LLMMaster* master,
    const std::vector<std::string>& models)
    : APIServiceImpl(models), master_(master) {
  CHECK(master_ != nullptr);
  llm_model_masters_[models[0]] = std::make_unique<LLMModelMasters>();
  llm_model_masters_[models[0]]->masters.push_back(master);
}

// complete_async for brpc
void CompletionServiceImpl::process_async_impl(
    std::shared_ptr<CompletionCall> call) {
  const auto& rpc_request = call->request();
  // check if model is supported
  const auto& model = rpc_request.model();
  if (unlikely(!models_.contains(model))) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }
  auto it_mm = llm_model_masters_.find(model);
  if (it_mm == llm_model_masters_.end() || it_mm->second->masters.empty()) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }
  auto& mm = *it_mm->second;
  LLMMaster* master = nullptr;
  auto& pa = PageAllocator::get_instance();

  std::vector<LLMMaster*> non_null_candidates;
  std::vector<std::string> replica_model_ids;
  non_null_candidates.reserve(mm.masters.size());
  replica_model_ids.reserve(mm.masters.size());
  for (LLMMaster* candidate : mm.masters) {
    if (candidate == nullptr) {
      continue;
    }
    non_null_candidates.push_back(candidate);
    replica_model_ids.push_back(candidate->options().model_id());
  }

  const std::vector<int32_t> replica_weights =
      RequestMetricAggregator::instance().get_replica_dispatch_weights(
          model, replica_model_ids);

  std::vector<LLMMaster*> available_candidates;
  std::vector<int32_t> available_weights;
  available_candidates.reserve(non_null_candidates.size());
  available_weights.reserve(non_null_candidates.size());
  for (size_t i = 0; i < non_null_candidates.size(); ++i) {
    LLMMaster* candidate = non_null_candidates[i];
    if (FLAGS_enable_xtensor && pa.is_initialized() &&
        (pa.is_model_schedule_blocked(candidate->options().model_id()) ||
          candidate->get_rate_limiter()->is_sleeping())) {
      continue;
    }
    available_candidates.push_back(candidate);
    available_weights.push_back(
        i < replica_weights.size() ? std::max(1, replica_weights[i]) : 1);
  }

  if (!available_candidates.empty()) {
    size_t total_weight = 0;
    for (const int32_t weight : available_weights) {
      total_weight += static_cast<size_t>(std::max(1, weight));
    }
    if (total_weight > 0) {
      // Sample based on current available weights to better track
      // time-varying dispatch weights.
      thread_local std::mt19937_64 rng(std::random_device{}());
      std::uniform_int_distribution<size_t> dist(0, total_weight - 1);
      const size_t target = dist(rng);
      size_t accum = 0;
      for (size_t i = 0; i < available_candidates.size(); ++i) {
        accum += static_cast<size_t>(std::max(1, available_weights[i]));
        if (target < accum) {
          master = available_candidates[i];
          break;
        }
      }
    }

    if (master == nullptr) {
      master = available_candidates.back();
    }
  }
  if (unlikely(master == nullptr)) {
    if (FLAGS_enable_xtensor && pa.is_initialized()) {
      if (auto* mgr = pa.get_layer_offload_manager(); mgr != nullptr) {
        auto selected_model_id =
            mgr->trigger_load_for_base_model(model, "no_available_replica");
        if (selected_model_id.has_value()) {
          for (LLMMaster* candidate : mm.masters) {
            if (candidate != nullptr &&
                candidate->options().model_id() == selected_model_id.value()) {
              master = candidate;
              break;
            }
          }
        }
      }
    }
    if (master != nullptr) {
      LOG(INFO) << "Route request to selected degraded replica model_id="
                << master->options().model_id();
    } else {
      call->finish_with_error(
          StatusCode::UNAVAILABLE,
          "No available model replica (all replicas are degraded).");
      return;
    }
  }

  // Check if the request is being rate-limited or model is sleeping.
  // is_limited() returns true if sleeping or rate-limited.
  if (unlikely(master->get_rate_limiter()->is_limited())) {
    if (master->get_rate_limiter()->is_sleeping()) {
      call->finish_with_error(StatusCode::UNAVAILABLE,
                              "Model is currently in sleep state.");
    } else {
      call->finish_with_error(
          StatusCode::RESOURCE_EXHAUSTED,
          "The number of concurrent requests has reached the limit.");
    }
    return;
  }
  const std::string master_model_id = master->options().model_id();
  if (FLAGS_enable_xtensor && pa.is_initialized()) {
    pa.mark_model_request_begin(master_model_id);
  }

  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());
  // Tag request with the actual selected replica id for downstream scheduler
  // metrics and per-replica aggregation.
  request_params.model_id = master_model_id;
  bool include_usage = false;
  if (rpc_request.has_stream_options()) {
    include_usage = rpc_request.stream_options().include_usage();
  }

  std::optional<std::vector<int>> prompt_tokens = std::nullopt;
  if (rpc_request.has_routing()) {
    prompt_tokens = std::vector<int>{};
    prompt_tokens->reserve(rpc_request.token_ids_size());
    for (int i = 0; i < rpc_request.token_ids_size(); i++) {
      prompt_tokens->emplace_back(rpc_request.token_ids(i));
    }

    request_params.decode_address = rpc_request.routing().decode_name();
  }

  auto saved_streaming = request_params.streaming;
  auto saved_request_id = request_params.request_id;
  // schedule the request
  master->handle_request(
      std::move(rpc_request.prompt()),
      std::move(prompt_tokens),
      std::move(request_params),
      call.get(),
      [call,
       model,
       master = master,
       master_model_id,
       stream = std::move(saved_streaming),
       include_usage = include_usage,
       request_id = std::move(saved_request_id),
       created_time = absl::ToUnixSeconds(absl::Now())](
          const RequestOutput& req_output) -> bool {
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            // Reduce the number of concurrent requests when a request is
            // finished with error.
            master->get_rate_limiter()->decrease_one_request();
            if (FLAGS_enable_xtensor &&
                PageAllocator::get_instance().is_initialized()) {
              PageAllocator::get_instance().mark_model_request_end(
                  master_model_id);
            }

            return call->finish_with_error(status.code(), status.message());
          }
        }

        // Reduce the number of concurrent requests when a request is finished
        // or canceled.
        if (req_output.finished || req_output.cancelled ||
            req_output.finished_on_prefill_instance) {
          master->get_rate_limiter()->decrease_one_request();
          if (FLAGS_enable_xtensor &&
              PageAllocator::get_instance().is_initialized()) {
            PageAllocator::get_instance().mark_model_request_end(
                master_model_id);
          }
        }

        if (stream) {
          return send_delta_to_client_brpc(
              call, include_usage, request_id, created_time, model, req_output);
        }
        // NOTE: maybe need to refactor along with service, currently for
        // non-stream request in prefill instance, we send a virtual response
        return send_result_to_client_brpc(
            call, request_id, created_time, model, req_output);
      });
}

}  // namespace xllm
