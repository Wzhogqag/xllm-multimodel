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

#include "global_prefix_cache_manager.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <algorithm>

namespace xllm {

void GlobalPrefixCacheManager::register_cache(const std::string& model_id,
                                              PrefixCache* cache) {
  std::lock_guard<std::mutex> lock(mutex_);
  model_caches_[model_id] = cache;
  LOG(INFO) << "Registered prefix cache for model: " << model_id
            << ", total models: " << model_caches_.size();
}

void GlobalPrefixCacheManager::unregister_cache(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Remove all nodes belonging to this model from global LRU
  auto it = global_lru_list_.begin();
  while (it != global_lru_list_.end()) {
    if ((*it)->model_id == model_id) {
      it = global_lru_list_.erase(it);
    } else {
      ++it;
    }
  }

  model_caches_.erase(model_id);
  LOG(INFO) << "Unregistered prefix cache for model: " << model_id;
}

void GlobalPrefixCacheManager::on_node_accessed(Node* node) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Remove from current position and move to front (MRU)
  global_lru_list_.remove(node);
  global_lru_list_.push_front(node);
  node->last_access_time = absl::ToUnixMicros(absl::Now());
}

void GlobalPrefixCacheManager::on_node_created(Node* node) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Insert new node at front (MRU)
  global_lru_list_.push_front(node);
  node->last_access_time = absl::ToUnixMicros(absl::Now());
}

void GlobalPrefixCacheManager::remove_node(Node* node) {
  std::lock_guard<std::mutex> lock(mutex_);
  global_lru_list_.remove(node);
}

size_t GlobalPrefixCacheManager::evict_for_model(
    size_t n_blocks,
    const std::string& model_id,
    std::vector<Node*>* evicted_nodes) {
  std::lock_guard<std::mutex> lock(mutex_);

  size_t evicted_count = 0;

  LOG(INFO) << "Global LRU eviction for model: " << model_id
            << ", need to evict " << n_blocks << " blocks"
            << ", total cached blocks: " << global_lru_list_.size();

  // Evict from the back (LRU) of global list
  // BUT only evict blocks from the specified model
  // This uses global heat info while avoiding cross-model memory pool issues
  auto it = global_lru_list_.rbegin();
  while (it != global_lru_list_.rend() && evicted_count < n_blocks) {
    Node* node = *it;

    // Only evict blocks from this model AND not in use
    if (node->model_id == model_id && !node->block.is_shared()) {
      if (evicted_nodes) {
        evicted_nodes->push_back(node);
      }

      // Erase from global LRU list
      auto forward_it = std::next(it).base();
      it = decltype(it)(global_lru_list_.erase(forward_it));
      evicted_count++;

      LOG(INFO) << "Evicted block: block_id=" << node->block.id()
                << ", last_access_time=" << node->last_access_time;
    } else {
      ++it;
    }
  }

  LOG(INFO) << "Global LRU eviction completed for model: " << model_id
            << ", evicted=" << evicted_count
            << ", remaining cached blocks: " << global_lru_list_.size();

  return evicted_count;
}

PrefixCache* GlobalPrefixCacheManager::get_cache(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = model_caches_.find(model_id);
  return (it != model_caches_.end()) ? it->second : nullptr;
}

size_t GlobalPrefixCacheManager::get_total_cached_blocks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return global_lru_list_.size();
}

}  // namespace xllm