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
  size_t evicted_count = 0;
  std::vector<Murmur3Key> evicted_keys;
  PrefixCache* source_cache_for_notify = nullptr;
  // Defer hash removal and delete until after releasing mutex_: insert() takes
  // cache_mutex_ then global mutex_; remove_from_hash_table must not run under
  // mutex_ or we deadlock.
  std::vector<Node*> pending_delete;

  {
    std::lock_guard<std::mutex> lock(mutex_);

    LOG(INFO) << "Global LRU eviction for model: " << model_id
              << ", need to evict " << n_blocks << " blocks"
              << ", total cached blocks: " << global_lru_list_.size();

    auto it = global_lru_list_.rbegin();
    while (it != global_lru_list_.rend() && evicted_count < n_blocks) {
      Node* node = *it;
      if (node->model_id == model_id && !node->block.is_shared()) {
        if (source_cache_for_notify == nullptr) {
          source_cache_for_notify = get_cache(model_id);
        }
        if (source_cache_for_notify) {
          evicted_keys.emplace_back(node->block.get_immutable_hash_value());
        }

        LOG(INFO) << "Evicted block: block_id=" << node->block.id()
                  << ", last_access_time=" << node->last_access_time;

        auto forward_it = std::next(it).base();
        it = decltype(it)(global_lru_list_.erase(forward_it));
        evicted_count++;
        if (evicted_nodes) {
          evicted_nodes->push_back(node);
        } else {
          pending_delete.push_back(node);
        }
      } else {
        ++it;
      }
    }

    LOG(INFO) << "Global LRU eviction completed for model: " << model_id
              << ", evicted=" << evicted_count
              << ", remaining cached blocks: " << global_lru_list_.size();
  }

  if (evicted_nodes == nullptr) {
    for (Node* node : pending_delete) {
      if (source_cache_for_notify) {
        Murmur3Key key(node->block.get_immutable_hash_value());
        source_cache_for_notify->remove_from_hash_table(key);
      }
      delete node;
    }
    if (source_cache_for_notify != nullptr && !evicted_keys.empty()) {
      source_cache_for_notify->on_global_evicted(evicted_keys);
    }
  }

  return evicted_count;
}
PrefixCache* GlobalPrefixCacheManager::get_cache(const std::string& model_id) {
  auto it = model_caches_.find(model_id);
  return (it != model_caches_.end()) ? it->second : nullptr;
}

size_t GlobalPrefixCacheManager::get_total_cached_blocks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return global_lru_list_.size();
}

size_t GlobalPrefixCacheManager::evict_global_pure_lru(
    size_t n_blocks,
    const std::unordered_set<std::string>& model_filter) {
  size_t evicted_count = 0;
  std::unordered_map<PrefixCache*, std::vector<Murmur3Key>>
      evicted_keys_by_cache;
  struct Pending {
    PrefixCache* cache;
    Murmur3Key key;
    Node* node;
  };
  std::vector<Pending> pending;
  const bool has_model_filter = !model_filter.empty();

  {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = global_lru_list_.rbegin();
    while (it != global_lru_list_.rend() && evicted_count < n_blocks) {
      Node* node = *it;

      if (has_model_filter && model_filter.count(node->model_id) == 0) {
        ++it;
        continue;
      }

      if (node->block.ref_count() <= 2) {
        auto cache_it = model_caches_.find(node->model_id);
        auto* source_cache =
            cache_it != model_caches_.end() ? cache_it->second : nullptr;
        Murmur3Key key(node->block.get_immutable_hash_value());
        if (source_cache != nullptr) {
          evicted_keys_by_cache[source_cache].push_back(key);
        }

        auto forward_it = std::next(it).base();
        it = decltype(it)(global_lru_list_.erase(forward_it));

        pending.push_back(Pending{source_cache, key, node});
        evicted_count++;
      } else {
        ++it;
      }
    }

    if (evicted_count > 0) {
      VLOG(1) << "Emergency global eviction: evicted " << evicted_count
              << " blocks, remaining: " << global_lru_list_.size()
              << ", model_filter_size=" << model_filter.size();
    }
  }

  for (const Pending& p : pending) {
    if (p.cache != nullptr) {
      p.cache->remove_from_hash_table(p.key);
    }
    delete p.node;
  }

  for (auto& [cache, keys] : evicted_keys_by_cache) {
    if (cache) {
      cache->on_global_evicted(keys);
    }
  }

  return evicted_count;
}

}  // namespace xllm