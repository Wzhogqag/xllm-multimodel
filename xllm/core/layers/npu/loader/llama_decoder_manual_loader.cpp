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

#include "llama_decoder_manual_loader.h"

#include <unordered_map>

namespace xllm {
namespace layer {

namespace {
enum DecoderLayerTensorId : int {
  IN_NORM_WEIGHT = 0,
  IN_NORM_BIAS,
  IN_NORM_NEW_WEIGHT,
  IN_NORM_NEW_BIAS,

  IN_Q_WEIGHT,
  IN_Q_BIAS,
  IN_Q_DEQSCALE,
  IN_Q_OFFSET,
  IN_Q_SCALE,
  IN_Q_COMPRESS_IDX,

  IN_K_WEIGHT,
  IN_K_BIAS,
  IN_K_DEQSCALE,
  IN_K_OFFSET,
  IN_K_SCALE,
  IN_K_COMPRESS_IDX,

  IN_V_WEIGHT,
  IN_V_BIAS,
  IN_V_DEQSCALE,
  IN_V_OFFSET,
  IN_V_SCALE,
  IN_V_COMPRESS_IDX,

  IN_ATTENTION_OUT_WEIGHT,
  IN_ATTENTION_OUT_BIAS,
  IN_ATTENTION_OUT_DEQSCALE,
  IN_ATTENTION_OUT_OFFSET,
  IN_ATTENTION_OUT_SCALE,
  IN_ATTENTION_OUT_COMPRESS_IDX,

  IN_SELFOUT_NORM_WEIGHT,
  IN_SELFOUT_NORM_BIAS,
  IN_SELFOUT_NORM_NEW_WEIGHT,
  IN_SELFOUT_NORM_NEW_BIAS,

  IN_MLP_W2_WEIGHT,
  IN_MLP_W2_BIAS,
  IN_MLP_W2_DEQSCALE,
  IN_MLP_W2_OFFSET,
  IN_MLP_W2_SCALE,
  IN_MLP_W2_COMPRESS_IDX,

  IN_MLP_W1_WEIGHT,
  IN_MLP_W1_BIAS,
  IN_MLP_W1_DEQSCALE,
  IN_MLP_W1_OFFSET,
  IN_MLP_W1_SCALE,
  IN_MLP_W1_COMPRESS_IDX,

  IN_MLP_CPROJ_WEIGHT,
  IN_MLP_CPROJ_BIAS,
  IN_MLP_CPROJ_DEQSCALE,
  IN_MLP_CPROJ_OFFSET,
  IN_MLP_CPROJ_SCALE,
  IN_MLP_CPROJ_COMPRESS_IDX,
};

const std::unordered_map<std::string, int> WEIGHT_MAPPING = {
    {"input_layernorm.weight", IN_NORM_WEIGHT},
    {"self_attn.q_proj.weight", IN_Q_WEIGHT},
    {"self_attn.k_proj.weight", IN_K_WEIGHT},
    {"self_attn.v_proj.weight", IN_V_WEIGHT},
    {"self_attn.o_proj.weight", IN_ATTENTION_OUT_WEIGHT},
    {"post_attention_layernorm.weight", IN_SELFOUT_NORM_WEIGHT},
    {"mlp.gate_proj.weight", IN_MLP_W2_WEIGHT},
    {"mlp.up_proj.weight", IN_MLP_W1_WEIGHT},
    {"mlp.down_proj.weight", IN_MLP_CPROJ_WEIGHT},
};

const std::map<int, int> WEIGHT_SHARD = {{IN_Q_WEIGHT, 0},
                                         {IN_K_WEIGHT, 0},
                                         {IN_V_WEIGHT, 0},
                                         {IN_ATTENTION_OUT_WEIGHT, 1},
                                         {IN_MLP_W2_WEIGHT, 0},
                                         {IN_MLP_W1_WEIGHT, 0},
                                         {IN_MLP_CPROJ_WEIGHT, 1}};
}  // namespace

LlamaDecoderManualLoader::LlamaDecoderManualLoader(uint64_t weight_count,
                                                   const ModelContext& context)
    : BaseManualLoader(weight_count, context) {
  auto options = context.get_tensor_options();
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  for (int i = 0; i < weight_count; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
  at_placeholder_ = torch::zeros({1}).to(torch::kCPU).to(dtype_);
}

void LlamaDecoderManualLoader::load_state_dict(const StateDict& state_dict) {
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    auto it = WEIGHT_SHARD.find(index);
    if (it != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, it->second, true);
    } else {
      set_weight(state_dict, name, index, true);
    }
  }
}

void LlamaDecoderManualLoader::verify_loaded_weights() const {
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    CHECK(at_host_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void LlamaDecoderManualLoader::merge_host_at_weights() {
  at_host_weight_tensors_[IN_Q_WEIGHT] =
      torch::cat({at_host_weight_tensors_[IN_Q_WEIGHT],
                  at_host_weight_tensors_[IN_K_WEIGHT],
                  at_host_weight_tensors_[IN_V_WEIGHT]},
                 0);

  at_host_weight_tensors_[IN_K_WEIGHT] = at_placeholder_;
  at_host_weight_tensors_[IN_V_WEIGHT] = at_placeholder_;

  at_host_weight_tensors_[IN_MLP_W2_WEIGHT] =
      torch::cat({at_host_weight_tensors_[IN_MLP_W2_WEIGHT],
                  at_host_weight_tensors_[IN_MLP_W1_WEIGHT]},
                 0);

  at_host_weight_tensors_[IN_MLP_W1_WEIGHT] = at_placeholder_;
}

}  // namespace layer
}  // namespace xllm
