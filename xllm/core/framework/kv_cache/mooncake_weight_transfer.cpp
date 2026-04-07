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

#include "mooncake_weight_transfer.h"

#include <glog/logging.h>

#include <chrono>

#include "common/global_flags.h"
#include "framework/xtensor/xtensor_allocator.h"
#include "util/net.h"

namespace xllm {

MooncakeWeightTransfer::MooncakeWeightTransfer(int16_t listen_port,
                                               const torch::Device& device)
    : listen_port_(listen_port), device_id_(device.index()) {
  std::string instance_ip = net::get_local_ip_addr();
  cluster_id_ = net::convert_ip_port_to_uint64(instance_ip, listen_port_);
  mooncake_te_ = std::make_unique<MooncakeTransferEngine>(listen_port_, device);
}

bool MooncakeWeightTransfer::initialize() {
  if (initialized_) {
    return true;
  }
  addr_ = mooncake_te_->initialize();
  initialized_ = !addr_.empty();
  return initialized_;
}

bool MooncakeWeightTransfer::register_weight_xtensor() {
  auto& allocator = XTensorAllocator::get_instance();
  LOG(INFO) << "[MooncakeWeightTransfer] FLAGS_phy_page_granularity_size: "
            << FLAGS_phy_page_granularity_size;
  if (!allocator.ensure_weight_xtensor_created()) {
    LOG(ERROR) << "Weight xtensor not ready for Mooncake";
    return false;
  }
  LOG(INFO) << "MooncakeWeightTransfer: weight pool ready (per-model "
               "register_model_weight_slice after map)";
  return true;
}

bool MooncakeWeightTransfer::register_model_weight_slice(
    const std::string& model_id) {
  if (!initialized_) {
    LOG(ERROR) << "MooncakeWeightTransfer not initialized";
    return false;
  }
  auto& allocator = XTensorAllocator::get_instance();
  ModelTensors* t = allocator.get_model_tensors(model_id);
  if (!t || t->weight_base_ptr == nullptr || t->weight_num_pages == 0) {
    LOG(ERROR) << "No weight slice for model " << model_id;
    return false;
  }
  if (t->mooncake_weight_buffer_index >= 0) {
    return true;
  }
  const size_t pgsz = allocator.weight_region_page_size();
  const size_t len = t->weight_num_pages * pgsz;
  std::vector<void*> addrs = {t->weight_base_ptr};
  std::vector<size_t> lens = {len};
  if (!mooncake_te_->register_memory(addrs, lens, static_cast<int64_t>(pgsz))) {
    LOG(ERROR) << "register_model_weight_slice failed for model " << model_id;
    return false;
  }
  const size_t idx = mooncake_te_->local_segment_buffer_count() - 1;
  allocator.set_model_mooncake_weight_buffer_index(model_id,
                                                   static_cast<int32_t>(idx));
  allocator.set_weight_mooncake_registered(true);
  LOG(INFO) << "MooncakeWeightTransfer: registered weight slice model="
            << model_id << ", bytes=" << len
            << ", mooncake_buffer_index=" << idx;
  return true;
}

bool MooncakeWeightTransfer::link_d2d(const std::string& remote_addr) {
  std::string host;
  int port = 0;
  net::parse_host_port_from_addr(remote_addr, host, port);
  auto remote_cluster_id =
      net::convert_ip_port_to_uint64(host, static_cast<uint16_t>(port));

  LOG(INFO) << "MooncakeWeightTransfer::link_d2d, remote_addr=" << remote_addr
            << ", remote_cluster_id=" << remote_cluster_id;

  return mooncake_te_->open_session(remote_cluster_id, remote_addr);
}

bool MooncakeWeightTransfer::link_d2d(
    const std::vector<std::string>& remote_addrs) {
  for (const auto& remote_addr : remote_addrs) {
    if (!link_d2d(remote_addr)) {
      return false;
    }
  }
  return true;
}

bool MooncakeWeightTransfer::unlink_d2d(const std::string& remote_addr) {
  std::string host;
  int port = 0;
  net::parse_host_port_from_addr(remote_addr, host, port);
  auto remote_cluster_id =
      net::convert_ip_port_to_uint64(host, static_cast<uint16_t>(port));

  LOG(INFO) << "MooncakeWeightTransfer::unlink_d2d, remote_addr=" << remote_addr
            << ", remote_cluster_id=" << remote_cluster_id;

  return mooncake_te_->close_session(remote_cluster_id, remote_addr);
}

bool MooncakeWeightTransfer::unlink_d2d(
    const std::vector<std::string>& remote_addrs) {
  for (const auto& remote_addr : remote_addrs) {
    if (!unlink_d2d(remote_addr)) {
      return false;
    }
  }
  return true;
}

bool MooncakeWeightTransfer::pull_weights(const std::string& remote_addr,
                                          uint64_t src_offset,
                                          uint64_t dst_offset,
                                          size_t size,
                                          const std::string& model_id,
                                          int32_t remote_buf_idx) {
  auto& allocator = XTensorAllocator::get_instance();
  const int32_t buf_idx =
      allocator.get_model_mooncake_weight_buffer_index(model_id);
  if (buf_idx < 0) {
    LOG(ERROR) << "pull_weights: model " << model_id
               << " has no mooncake_weight_buffer_index; register slice first";
    return false;
  }
  const size_t local_i = static_cast<size_t>(buf_idx);
  const size_t remote_i =
      (remote_buf_idx >= 0) ? static_cast<size_t>(remote_buf_idx) : local_i;
  std::vector<uint64_t> src_offsets = {dst_offset};
  std::vector<uint64_t> dst_offsets = {src_offset};
  const auto t0 = std::chrono::high_resolution_clock::now();
  const bool ok = mooncake_te_->move_memory_by_global_offsets(
      remote_addr,
      src_offsets,
      dst_offsets,
      size,
      MooncakeTransferEngine::MoveOpcode::READ,
      local_i,
      remote_i);
  const auto t1 = std::chrono::high_resolution_clock::now();
  if (FLAGS_log_mooncake_weight_pull_timing) {
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double gib_per_s =
        sec > 0.0
            ? (static_cast<double>(size) / (1024.0 * 1024.0 * 1024.0) / sec)
            : 0.0;
    LOG(INFO) << "[MooncakeWeightTransfer] pull_weights xfer model=" << model_id
              << " bytes=" << size << " remote=" << remote_addr
              << " sec=" << sec << " GiB/s=" << gib_per_s << " ok=" << ok;
  }
  return ok;
}

bool MooncakeWeightTransfer::push_weights(const std::string& remote_addr,
                                          uint64_t src_offset,
                                          uint64_t dst_offset,
                                          size_t size,
                                          const std::string& model_id) {
  auto& allocator = XTensorAllocator::get_instance();
  const int32_t buf_idx =
      allocator.get_model_mooncake_weight_buffer_index(model_id);
  if (buf_idx < 0) {
    LOG(ERROR) << "push_weights: model " << model_id
               << " has no mooncake_weight_buffer_index; register slice first";
    return false;
  }
  const size_t local_i = static_cast<size_t>(buf_idx);
  const size_t remote_i = local_i;
  std::vector<uint64_t> src_offsets = {src_offset};
  std::vector<uint64_t> dst_offsets = {dst_offset};
  return mooncake_te_->move_memory_by_global_offsets(
      remote_addr,
      src_offsets,
      dst_offsets,
      size,
      MooncakeTransferEngine::MoveOpcode::WRITE,
      local_i,
      remote_i);
}

}  // namespace xllm
