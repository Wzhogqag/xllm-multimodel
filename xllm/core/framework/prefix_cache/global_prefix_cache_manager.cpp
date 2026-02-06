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
#include <unordered_map>

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
  std::vector<Murmur3Key> evicted_keys;
  PrefixCache* source_cache_for_notify = nullptr;

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
    if (node->model_id == model_id && node->block.ref_count() <= 2) {
      if (source_cache_for_notify == nullptr) {
        source_cache_for_notify = get_cache(model_id);
      }
      if (source_cache_for_notify) {
        Murmur3Key key(node->block.get_immutable_hash_value());
        evicted_keys.push_back(key);
        if (evicted_nodes == nullptr) {
          source_cache_for_notify->remove_from_hash_table(key);
        }
      }

      // Erase from global LRU list
      auto forward_it = std::next(it).base();
      it = decltype(it)(global_lru_list_.erase(forward_it));
      evicted_count++;
      if (evicted_nodes) {
        evicted_nodes->push_back(node);
      } else {
        // Complete eviction here (see NOTE above).
        delete node;
      }

      LOG(INFO) << "Evicted block: block_id=" << node->block.id()
                << ", last_access_time=" << node->last_access_time;
    } else {
      ++it;
    }
  }

  LOG(INFO) << "Global LRU eviction completed for model: " << model_id
            << ", evicted=" << evicted_count
            << ", remaining cached blocks: " << global_lru_list_.size();

  if (evicted_nodes == nullptr && source_cache_for_notify != nullptr &&
      !evicted_keys.empty()) {
    source_cache_for_notify->on_global_evicted(evicted_keys);
  }

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

size_t GlobalPrefixCacheManager::evict_global_pure_lru(size_t n_blocks) {
  std::lock_guard<std::mutex> lock(mutex_);
  const size_t total_cached_before = global_lru_list_.size();
  LOG(INFO) << "[GlobalEvict] before: total_cached=" << total_cached_before
            << " n_blocks=" << n_blocks;
  size_t evicted_count = 0;
  std::unordered_map<PrefixCache*, std::vector<Murmur3Key>>
      evicted_keys_by_cache;

  // Evict from the back (LRU) of global list, regardless of model
  auto it = global_lru_list_.rbegin();
  while (it != global_lru_list_.rend() && evicted_count < n_blocks) {
    Node* node = *it;

    // Only evict blocks that are not in use (ref_count <= 2)
    if (node->block.ref_count() <= 2) {
      // Get the model's PrefixCache to remove from hash table
      auto* source_cache = model_caches_[node->model_id];
      if (source_cache) {
        // Remove from the model's hash table
        Murmur3Key key(node->block.get_immutable_hash_value());
        source_cache->remove_from_hash_table(key);
        evicted_keys_by_cache[source_cache].push_back(key);
      }

      // Remove from global LRU list
      auto forward_it = std::next(it).base();
      it = decltype(it)(global_lru_list_.erase(forward_it));

      // Delete node will trigger Block destructor, which will free the block
      // back to its original BlockManager, potentially releasing physical pages
      delete node;
      evicted_count++;
    } else {
      ++it;  // Skip blocks that are actively in use
    }
  }

  for (auto& [cache, keys] : evicted_keys_by_cache) {
    if (cache) {
      cache->on_global_evicted(keys);
    }
  }
  LOG(INFO) << "[GlobalEvict] after: evicted=" << evicted_count
            << " remaining=" << global_lru_list_.size();
  if (evicted_count > 0) {
    VLOG(1) << "Emergency global eviction: evicted " << evicted_count
            << " blocks, remaining: " << global_lru_list_.size();
  }

  return evicted_count;
}

}  // namespace xllm