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

#pragma once

#include <glog/logging.h>

#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/block/block.h"
#include "prefix_cache.h"

namespace xllm {

// Global manager for multi-model prefix cache with global LRU eviction
// Only activated when FLAGS_enable_xtensor && FLAGS_enable_prefix_cache
class GlobalPrefixCacheManager {
 public:
  using Node = PrefixCache::Node;

  // Singleton instance
  static GlobalPrefixCacheManager& instance() {
    static GlobalPrefixCacheManager instance;
    return instance;
  }

  // Delete copy and move
  GlobalPrefixCacheManager(const GlobalPrefixCacheManager&) = delete;
  GlobalPrefixCacheManager& operator=(const GlobalPrefixCacheManager&) = delete;

  // Register/unregister a model's PrefixCache
  void register_cache(const std::string& model_id, PrefixCache* cache);
  void unregister_cache(const std::string& model_id);

  // Global LRU operations (called by PrefixCache when global mode is enabled)
  void on_node_accessed(Node* node);
  void on_node_created(Node* node);
  void remove_node(Node* node);

  // Global LRU eviction for a specific model
  // Uses global LRU order but only evicts blocks from the specified model
  // This avoids cross-model memory pool issues while still leveraging global
  // heat info
  size_t evict_for_model(size_t n_blocks,
                         const std::string& model_id,
                         std::vector<Node*>* evicted_nodes = nullptr);

  // Get a model's PrefixCache
  PrefixCache* get_cache(const std::string& model_id);

  // Statistics
  size_t get_total_cached_blocks() const;

  // Emergency eviction: evict blocks from global LRU regardless of model
  // This is used when global physical memory is tight
  // Returns the number of blocks actually evicted
  size_t evict_global_pure_lru(size_t n_blocks);

 private:
  GlobalPrefixCacheManager() = default;

  // Global LRU list (front = MRU, back = LRU)
  std::list<Node*> global_lru_list_;

  // Model ID to PrefixCache mapping
  std::unordered_map<std::string, PrefixCache*> model_caches_;

  // Mutex for thread safety
  mutable std::mutex mutex_;
};

}  // namespace xllm