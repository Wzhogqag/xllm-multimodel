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

#pragma once

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {

/**
 * PeerRegistry – cross-instance D2D weight pull registry.
 *
 * Static info (JSON file, written once by management script after link_d2d):
 *   model_slots:  model_id → int slot index (global, across all instances)
 *   instances[]:  inst_slot, instance_id, model_ids[]
 *   links[]:      sender_slot, receiver_slot, remote_addrs[TP]
 *                 remote_addrs are per (sender, receiver) pair because
 *                 Mooncake assigns session-specific endpoints.
 *
 * Dynamic info (POSIX shm /xllm_peer_state):
 *   loaded_layers[MAX_INSTANCES][MAX_MODELS]: atomic<int32_t>
 *     -1  = no layer on device (or model not on this instance)
 *     k≥0 = layers 0..k are on device
 *
 * Limits:
 *   MAX_INSTANCES: max number of running xllm instances on the machine.
 *   MAX_MODELS:    max number of distinct model types across all instances.
 *                  Each instance may hold any subset of these model types.
 */
class PeerRegistry {
 public:
  static constexpr int kMaxInstances = 16;
  static constexpr int kMaxModels =
      16;  // distinct model types, not per-instance count
  static constexpr const char* kShmName = "/xllm_peer_state";

  struct PeerInfo {
    int inst_slot = -1;
    std::vector<std::string>
        remote_addrs;             // one per TP rank, specific to this receiver
    int32_t sender_buf_idx = -1;  // Mooncake buffer ordinal on the sender side
  };

  static PeerRegistry& get_instance() {
    static PeerRegistry inst;
    return inst;
  }

  // Called once at startup.
  // instance_id: value of XLLM_INSTANCE_ID env var.
  // json_path:   value of XLLM_PEER_TABLE_PATH env var.
  // Both may be empty → D2D disabled silently.
  void set_identity(const std::string& instance_id,
                    const std::string& json_path);

  // Find all peer instances (excluding self) that have layers 0..layer_id on
  // device for model_id, and for which a D2D link to this instance exists.
  // Returns empty vector if D2D unavailable. Thread-safe; triggers lazy init.
  // Caller should filter by remote_addrs.size() to match its own TP size.
  std::vector<PeerInfo> find_peers(const std::string& model_id,
                                   int32_t layer_id);

  // Update self's loaded_layers for model_id in shm.
  // val = index of last layer on device (-1 if none).
  // No-op if not initialized.
  void update_loaded_layers(const std::string& model_id, int32_t val);

  bool is_initialized() const { return initialized_; }

 private:
  PeerRegistry() = default;
  ~PeerRegistry();
  PeerRegistry(const PeerRegistry&) = delete;
  PeerRegistry& operator=(const PeerRegistry&) = delete;

  void try_init();
  void detach_shm();

  // Shared memory layout.
  struct PeerShmState {
    // [inst_slot][model_slot] → last layer on device, -1 = unavailable.
    std::atomic<int32_t> loaded_layers[kMaxInstances][kMaxModels];
  };

  // Compact key: sender_slot * kMaxInstances + receiver_slot.
  static int link_key(int sender_slot, int receiver_slot) {
    return sender_slot * kMaxInstances + receiver_slot;
  }

  std::string instance_id_;
  std::string json_path_;

  bool initialized_ = false;
  std::mutex init_mutex_;

  int my_inst_slot_ = -1;

  // model_id → model slot index (global)
  std::unordered_map<std::string, int> model_to_slot_;

  // model_id → sender inst_slots that have this model (excluding self)
  std::unordered_map<std::string, std::vector<int>> model_to_sender_slots_;

  // link_key(sender_slot, my_inst_slot_) → remote_addrs[TP]
  // Only entries where receiver == self are populated.
  std::unordered_map<int, std::vector<std::string>> link_map_;

  // inst_slot → (model_id → mooncake buf_idx on that sender instance)
  std::unordered_map<int, std::unordered_map<std::string, int32_t>>
      inst_model_buf_idx_;

  PeerShmState* shm_ = nullptr;
  int shm_fd_ = -1;
};

}  // namespace xllm
