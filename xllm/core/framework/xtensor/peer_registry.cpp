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

#include "peer_registry.h"

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <nlohmann/json.hpp>

namespace xllm {

// ============================================================
//  Public API
// ============================================================

void PeerRegistry::set_identity(const std::string& instance_id,
                                const std::string& json_path) {
  std::lock_guard<std::mutex> lock(init_mutex_);
  instance_id_ = instance_id;
  json_path_ = json_path;
  if (instance_id_.empty() || json_path_.empty()) {
    LOG(INFO) << "[PeerRegistry] D2D disabled: XLLM_INSTANCE_ID or "
                 "XLLM_PEER_TABLE_PATH not set.";
  }
}

std::vector<PeerRegistry::PeerInfo> PeerRegistry::find_peers(
    const std::string& model_id,
    int32_t layer_id) {
  if (!initialized_) {
    try_init();
    if (!initialized_) return {};
  }

  auto mit = model_to_slot_.find(model_id);
  if (mit == model_to_slot_.end()) return {};
  int model_slot = mit->second;

  auto sit = model_to_sender_slots_.find(model_id);
  if (sit == model_to_sender_slots_.end()) return {};

  std::vector<PeerInfo> result;
  result.reserve(sit->second.size());
  for (int sender_slot : sit->second) {
    int32_t loaded = shm_->loaded_layers[sender_slot][model_slot].load(
        std::memory_order_relaxed);
    if (loaded < layer_id) continue;

    int key = link_key(sender_slot, my_inst_slot_);
    auto lit = link_map_.find(key);
    if (lit == link_map_.end()) continue;

    int32_t sbuf_idx = -1;
    auto iit = inst_model_buf_idx_.find(sender_slot);
    if (iit != inst_model_buf_idx_.end()) {
      auto bit = iit->second.find(model_id);
      if (bit != iit->second.end()) sbuf_idx = bit->second;
    }
    result.emplace_back(PeerInfo{sender_slot, lit->second, sbuf_idx});
  }
  return result;
}

void PeerRegistry::update_loaded_layers(const std::string& model_id,
                                        int32_t val) {
  if (!initialized_) {
    try_init();
  }
  auto mit = model_to_slot_.find(model_id);
  if (mit == model_to_slot_.end()) return;
  shm_->loaded_layers[my_inst_slot_][mit->second].store(
      val, std::memory_order_relaxed);
}

// ============================================================
//  Lazy init
// ============================================================

void PeerRegistry::try_init() {
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (initialized_) return;
  if (instance_id_.empty() || json_path_.empty()) return;

  struct stat st{};
  if (stat(json_path_.c_str(), &st) != 0) return;  // file not ready yet

  nlohmann::json root;
  {
    std::ifstream f(json_path_);
    if (!f.is_open()) {
      LOG(ERROR) << "[PeerRegistry] Cannot open " << json_path_;
      return;
    }
    try {
      f >> root;
    } catch (const std::exception& e) {
      LOG(ERROR) << "[PeerRegistry] JSON parse error: " << e.what();
      return;
    }
  }

  // ── model_slots ──────────────────────────────────────────
  for (auto& [k, v] : root["model_slots"].items()) {
    int slot = v.get<int>();
    if (slot >= kMaxModels) {
      LOG(ERROR) << "[PeerRegistry] model slot " << slot
                 << " >= kMaxModels=" << kMaxModels;
      return;
    }
    model_to_slot_[k] = slot;
  }

  // ── instances ────────────────────────────────────────────
  for (auto& inst_json : root["instances"]) {
    int slot = inst_json["inst_slot"].get<int>();
    if (slot < 0 || slot >= kMaxInstances) {
      LOG(ERROR) << "[PeerRegistry] Invalid inst_slot=" << slot;
      return;
    }
    if (inst_json["instance_id"].get<std::string>() == instance_id_) {
      my_inst_slot_ = slot;
    }
  }
  if (my_inst_slot_ < 0) {
    LOG(ERROR) << "[PeerRegistry] instance_id=" << instance_id_
               << " not found in JSON.";
    return;
  }

  // Build model_to_sender_slots_ and inst_model_buf_idx_ from instances[].
  for (auto& inst_json : root["instances"]) {
    int slot = inst_json["inst_slot"].get<int>();
    if (slot == my_inst_slot_) continue;
    for (auto& mid : inst_json["model_ids"]) {
      model_to_sender_slots_[mid.get<std::string>()].push_back(slot);
    }
    if (inst_json.contains("model_buf_indices")) {
      for (auto& [mid, bidx] : inst_json["model_buf_indices"].items()) {
        inst_model_buf_idx_[slot][mid] = bidx.get<int32_t>();
      }
    }
  }

  // ── links ────────────────────────────────────────────────
  // Only load entries where receiver_slot == my_inst_slot_.
  for (auto& link : root["links"]) {
    int receiver_slot = link["receiver_slot"].get<int>();
    if (receiver_slot != my_inst_slot_) continue;

    int sender_slot = link["sender_slot"].get<int>();
    std::vector<std::string> addrs;
    for (auto& a : link["remote_addrs"]) {
      addrs.push_back(a.get<std::string>());
    }
    link_map_[link_key(sender_slot, my_inst_slot_)] = std::move(addrs);
  }

  if (link_map_.empty()) {
    LOG(WARNING) << "[PeerRegistry] No links found for receiver="
                 << instance_id_ << ". D2D pull will not be available.";
  }

  // ── attach shm ───────────────────────────────────────────
  std::string shm_name = root.value("shm_name", kShmName);
  shm_fd_ = shm_open(shm_name.c_str(), O_RDWR, 0666);
  if (shm_fd_ < 0) {
    LOG(ERROR) << "[PeerRegistry] shm_open(" << shm_name
               << ") failed: " << strerror(errno);
    return;
  }
  void* ptr = mmap(nullptr,
                   sizeof(PeerShmState),
                   PROT_READ | PROT_WRITE,
                   MAP_SHARED,
                   shm_fd_,
                   0);
  if (ptr == MAP_FAILED) {
    LOG(ERROR) << "[PeerRegistry] mmap failed: " << strerror(errno);
    close(shm_fd_);
    shm_fd_ = -1;
    return;
  }
  shm_ = reinterpret_cast<PeerShmState*>(ptr);

  initialized_ = true;
  LOG(INFO) << "[PeerRegistry] Initialized: instance=" << instance_id_
            << " slot=" << my_inst_slot_ << " models=" << model_to_slot_.size()
            << " links_as_receiver=" << link_map_.size();
}

void PeerRegistry::detach_shm() {
  if (shm_) {
    munmap(shm_, sizeof(PeerShmState));
    shm_ = nullptr;
  }
  if (shm_fd_ >= 0) {
    close(shm_fd_);
    shm_fd_ = -1;
  }
}

PeerRegistry::~PeerRegistry() { detach_shm(); }

}  // namespace xllm
