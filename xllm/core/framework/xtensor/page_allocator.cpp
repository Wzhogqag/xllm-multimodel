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

#include "page_allocator.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <optional>
#include <stdexcept>

#include "common/global_flags.h"
#include "framework/prefix_cache/global_prefix_cache_manager.h"
#include "xtensor_allocator.h"

namespace xllm {

void PageAllocator::init(size_t num_phy_pages,
                         int32_t dp_size,
                         bool enable_page_prealloc) {
  std::lock_guard<std::mutex> lock(mtx_);

  if (initialized_) {
    LOG(WARNING) << "PageAllocator already initialized, ignoring re-init";
    return;
  }

  dp_size_ = dp_size;
  page_size_ = FLAGS_phy_page_granularity_size;
  enable_page_prealloc_ = enable_page_prealloc;

  // Set total physical pages from parameter
  num_total_phy_pages_ = num_phy_pages;
  num_free_phy_pages_ = num_total_phy_pages_;

  initialized_ = true;

  LOG(INFO) << "Init PageAllocator: "
            << "dp_size=" << dp_size_
            << ", page_size=" << page_size_ / (1024 * 1024) << "MB"
            << ", num_total_phy_pages=" << num_total_phy_pages_
            << ", enable_prealloc=" << enable_page_prealloc;
}

PageAllocator::~PageAllocator() {
  try {
    if (enable_page_prealloc_ && prealloc_thd_ != nullptr) {
      stop_prealloc_thread(PREALLOC_THREAD_TIMEOUT);
    }
  } catch (...) {
    // Silently ignore exceptions during cleanup
  }
}

bool PageAllocator::register_model(const std::string& model_id,
                                   int64_t num_layers,
                                   int32_t priority,
                                   int32_t min_reserved_pages,
                                   int32_t max_reserved_pages) {
  LOG(INFO) << "[PageAllocator] Starting register_model for " << model_id
            << ", num_layers=" << num_layers << ", priority=" << priority
            << ", min_reserved_pages=" << min_reserved_pages
            << ", max_reserved_pages=" << max_reserved_pages;
  std::lock_guard<std::mutex> lock(mtx_);

  LOG(INFO) << "[PageAllocator] Acquired lock for " << model_id;

  CHECK(initialized_) << "PageAllocator not initialized";

  if (model_states_.find(model_id) != model_states_.end()) {
    LOG(WARNING) << "Model " << model_id << " already registered";
    return false;
  }

  CHECK(num_layers > 0) << "num_layers must be > 0, got " << num_layers;
  CHECK(num_total_phy_pages_ > 0)
      << "num_total_phy_pages_ must be > 0, got " << num_total_phy_pages_;

  // Create state directly in map to avoid atomic assignment issues
  auto& state = model_states_[model_id];
  state.num_layers = num_layers;
  // Calculate virtual pages based on single-layer memory size
  state.num_total_virt_pages = num_total_phy_pages_ / num_layers;
  // Each virt_page needs to map on all K and V XTensors
  state.phy_pages_per_virt_page = 2 * num_layers;
  state.is_sleeping = false;

  LOG(INFO) << "[PageAllocator] Calculated num_total_virt_pages="
            << state.num_total_virt_pages << " for " << model_id
            << " (num_total_phy_pages_=" << num_total_phy_pages_
            << ", num_layers=" << num_layers << ")";

  // Initialize priority and reserved pages configuration
  state.priority = priority;
  state.min_reserved_pages = min_reserved_pages;
  state.max_reserved_pages = max_reserved_pages;
  state.base_min_reserved_pages = min_reserved_pages;
  state.base_max_reserved_pages = max_reserved_pages;

  LOG(INFO) << "[PageAllocator] Initializing dp_group_pages for " << model_id
            << ", dp_size_=" << dp_size_
            << ", num_total_virt_pages=" << state.num_total_virt_pages;

  // Initialize per-DP group page lists
  state.dp_group_pages.resize(dp_size_);
  for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
    auto& dp_pages = state.dp_group_pages[dp_rank];
    dp_pages.num_free_virt_pages = state.num_total_virt_pages;
    LOG(INFO) << "[PageAllocator] Initializing dp_rank=" << dp_rank << " for "
              << model_id << ", pushing " << state.num_total_virt_pages
              << " pages";

    for (size_t i = 0; i < state.num_total_virt_pages; ++i) {
      dp_pages.free_virt_page_list.push_back(static_cast<int64_t>(i));
    }
    LOG(INFO) << "[PageAllocator] Completed dp_rank=" << dp_rank << " for "
              << model_id;
  }

  LOG(INFO) << "Registered model " << model_id << ": "
            << "num_layers=" << num_layers << ", num_total_virt_pages="
            << model_states_[model_id].num_total_virt_pages
            << ", phy_pages_per_virt_page="
            << model_states_[model_id].phy_pages_per_virt_page
            << ", priority=" << priority
            << ", min_reserved_pages=" << min_reserved_pages
            << ", max_reserved_pages=" << max_reserved_pages;

  return true;
}

void PageAllocator::unregister_model(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);

  auto it = model_states_.find(model_id);
  if (it == model_states_.end()) {
    LOG(WARNING) << "Model " << model_id << " not found for unregister";
    return;
  }

  // Release any weight pages
  if (it->second.weight_pages_allocated > 0) {
    num_free_phy_pages_ += it->second.weight_pages_allocated;
  }

  model_states_.erase(it);
  LOG(INFO) << "Unregistered model " << model_id;
}

void PageAllocator::sleep_model(const std::string& model_id) {
  std::vector<std::pair<int32_t, std::vector<int64_t>>> pages_to_unmap;
  size_t total_phy_pages_to_release = 0;
  size_t phy_pages_per_virt = 0;

  {
    std::unique_lock<std::mutex> lock(mtx_);

    auto it = model_states_.find(model_id);
    if (it == model_states_.end()) {
      LOG(WARNING) << "Model " << model_id << " not found for sleep";
      return;
    }

    ModelState& state = it->second;
    if (state.is_sleeping) {
      LOG(WARNING) << "Model " << model_id << " is already sleeping";
      return;
    }

    // Wait for pending map/unmap operations to complete
    while (state.pending_map_ops.load() > 0) {
      LOG(INFO) << "Waiting for " << state.pending_map_ops.load()
                << " pending map ops to complete before sleeping model "
                << model_id;
      cond_.wait(lock);
    }

    state.is_sleeping = true;
    phy_pages_per_virt = state.phy_pages_per_virt_page;

    // Collect all mapped pages (reserved + allocated) from each DP group
    for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
      auto& dp_pages = state.dp_group_pages[dp_rank];
      std::vector<int64_t> virt_page_ids;

      // Collect reserved pages
      virt_page_ids.insert(virt_page_ids.end(),
                           dp_pages.reserved_virt_page_list.begin(),
                           dp_pages.reserved_virt_page_list.end());

      // Collect allocated pages (in use by block manager)
      virt_page_ids.insert(virt_page_ids.end(),
                           dp_pages.allocated_virt_page_list.begin(),
                           dp_pages.allocated_virt_page_list.end());

      if (!virt_page_ids.empty()) {
        total_phy_pages_to_release += virt_page_ids.size() * phy_pages_per_virt;
        pages_to_unmap.emplace_back(dp_rank, std::move(virt_page_ids));
      }
    }

    LOG(INFO) << "Sleeping model " << model_id << ", will release "
              << total_phy_pages_to_release << " physical pages";
  }

  // Unmap pages outside the lock
  for (auto& [dp_rank, virt_page_ids] : pages_to_unmap) {
    if (!virt_page_ids.empty()) {
      unmap_virt_pages(model_id, dp_rank, virt_page_ids);
    }
  }

  // Update state after unmapping
  // Note: We keep reserved_virt_page_list and allocated_virt_page_list
  // unchanged so that XTensorBlockManagerImpl's state remains consistent. On
  // wakeup, we will re-map these same pages.
  {
    std::lock_guard<std::mutex> lock(mtx_);
    // Release physical pages back to shared pool
    num_free_phy_pages_ += total_phy_pages_to_release;
    update_memory_usage();
    cond_.notify_all();
  }

  LOG(INFO) << "Model " << model_id << " is now sleeping";
}

void PageAllocator::wakeup_model(const std::string& model_id) {
  std::vector<std::pair<int32_t, std::vector<int64_t>>> pages_to_map;
  size_t total_phy_pages_needed = 0;
  size_t phy_pages_per_virt = 0;

  {
    std::lock_guard<std::mutex> lock(mtx_);

    auto it = model_states_.find(model_id);
    if (it == model_states_.end()) {
      LOG(WARNING) << "Model " << model_id << " not found for wakeup";
      return;
    }

    ModelState& state = it->second;
    if (!state.is_sleeping) {
      LOG(WARNING) << "Model " << model_id << " is not sleeping";
      return;
    }

    phy_pages_per_virt = state.phy_pages_per_virt_page;

    // Collect all pages that need to be re-mapped (reserved + allocated)
    for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
      auto& dp_pages = state.dp_group_pages[dp_rank];
      std::vector<int64_t> virt_page_ids;

      // Collect reserved pages
      virt_page_ids.insert(virt_page_ids.end(),
                           dp_pages.reserved_virt_page_list.begin(),
                           dp_pages.reserved_virt_page_list.end());

      // Collect allocated pages (in use by block manager)
      virt_page_ids.insert(virt_page_ids.end(),
                           dp_pages.allocated_virt_page_list.begin(),
                           dp_pages.allocated_virt_page_list.end());

      if (!virt_page_ids.empty()) {
        total_phy_pages_needed += virt_page_ids.size() * phy_pages_per_virt;
        pages_to_map.emplace_back(dp_rank, std::move(virt_page_ids));
      }
    }

    // Check if we have enough physical pages
    if (num_free_phy_pages_ < total_phy_pages_needed) {
      LOG(ERROR) << "Not enough physical pages for wakeup: need "
                 << total_phy_pages_needed << ", available "
                 << num_free_phy_pages_;
      return;
    }

    // Consume physical pages
    num_free_phy_pages_ -= total_phy_pages_needed;
    state.is_sleeping = false;
    update_memory_usage();

    LOG(INFO) << "Waking up model " << model_id << ", will map "
              << total_phy_pages_needed << " physical pages";
  }

  // Re-map pages outside the lock
  for (auto& [dp_rank, virt_page_ids] : pages_to_map) {
    if (!virt_page_ids.empty()) {
      map_virt_pages(model_id, dp_rank, virt_page_ids);
    }
  }

  // Trigger preallocation to refill reserved pages
  if (enable_page_prealloc_) {
    trigger_preallocation();
  }

  LOG(INFO) << "Model " << model_id << " is now awake";
}

bool PageAllocator::is_model_sleeping(const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_states_.find(model_id);
  if (it == model_states_.end()) {
    return false;
  }
  return it->second.is_sleeping;
}

void PageAllocator::start_prealloc_thread() {
  if (enable_page_prealloc_) {
    start_prealloc_thread_internal();
  }
}

bool PageAllocator::has_enough_phy_pages(size_t num_phy_pages) const {
  // Note: Caller must hold mtx_
  return num_free_phy_pages_ >= num_phy_pages;
}

bool PageAllocator::consume_phy_pages(size_t num_phy_pages) {
  // Note: Caller must hold mtx_
  if (num_free_phy_pages_ < num_phy_pages) {
    LOG(WARNING) << "Not enough physical pages: need " << num_phy_pages
                 << ", available " << num_free_phy_pages_;
    return false;
  }
  num_free_phy_pages_ -= num_phy_pages;
  return true;
}

void PageAllocator::release_phy_pages(size_t num_phy_pages) {
  num_free_phy_pages_ += num_phy_pages;
}

PageAllocator::ModelState& PageAllocator::get_model_state(
    const std::string& model_id) {
  auto it = model_states_.find(model_id);
  CHECK(it != model_states_.end()) << "Model " << model_id << " not registered";
  return it->second;
}

const PageAllocator::ModelState& PageAllocator::get_model_state(
    const std::string& model_id) const {
  auto it = model_states_.find(model_id);
  CHECK(it != model_states_.end()) << "Model " << model_id << " not registered";
  return it->second;
}

std::unique_ptr<VirtPage> PageAllocator::alloc_kv_cache_page(
    const std::string& model_id,
    int32_t dp_rank) {
  std::unique_lock<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PageAllocator not initialized";
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";

  ModelState& state = get_model_state(model_id);
  CHECK(!state.is_sleeping)
      << "Cannot allocate from sleeping model " << model_id;

  auto& dp_pages = state.dp_group_pages[dp_rank];
  std::optional<int64_t> virt_page_id;
  size_t phy_pages_needed = state.phy_pages_per_virt_page;

  while (!virt_page_id.has_value()) {
    // Fast path: allocate from reserved pages (already mapped)
    if (!dp_pages.reserved_virt_page_list.empty()) {
      virt_page_id = dp_pages.reserved_virt_page_list.front();
      dp_pages.reserved_virt_page_list.pop_front();
      dp_pages.num_free_virt_pages--;
      // Physical pages already consumed when reserved

      // Trigger preallocation to refill reserved pool if getting low
      if (dp_pages.reserved_virt_page_list.size() <
          static_cast<size_t>(state.min_reserved_pages)) {
        prealloc_needed_ = true;
        cond_.notify_all();
      }

      // Track this page as allocated
      dp_pages.allocated_virt_page_list.insert(*virt_page_id);

      update_memory_usage();
      return std::make_unique<VirtPage>(*virt_page_id, page_size_);
    }

    // Slow path: allocate from free pages (need to map)
    if (!dp_pages.free_virt_page_list.empty() &&
        has_enough_phy_pages(phy_pages_needed)) {
      virt_page_id = dp_pages.free_virt_page_list.front();
      dp_pages.free_virt_page_list.pop_front();
      dp_pages.num_free_virt_pages--;
      if (!consume_phy_pages(phy_pages_needed)) {
        // Rollback: put the page back
        dp_pages.free_virt_page_list.push_front(*virt_page_id);
        dp_pages.num_free_virt_pages++;
        virt_page_id.reset();
        continue;  // Try again or wait
      }
      break;
    }

    // Check if we're out of resources
    if (dp_pages.free_virt_page_list.empty()) {
      LOG(ERROR) << "[PageAllocator] FATAL: No free virtual pages left for "
                 << "model=" << model_id << " dp_rank=" << dp_rank
                 << ". Process may exit.";
      throw std::runtime_error("No free virtual pages left for model " +
                               model_id + " dp_rank " +
                               std::to_string(dp_rank));
    }
    if (!has_enough_phy_pages(phy_pages_needed)) {
      // Physical page pool exhausted: try emergency prefix cache eviction
      // to free pages instead of failing (multi-model safety).
      LOG(INFO) << "[PageAllocator] physical pages exhausted: model="
                << model_id << " dp_rank=" << dp_rank
                << " free_phy=" << num_free_phy_pages_
                << " total_phy=" << num_total_phy_pages_
                << " phy_needed=" << phy_pages_needed;
      if (FLAGS_enable_prefix_cache && FLAGS_enable_xtensor) {
        lock.unlock();
        size_t total_cached =
            GlobalPrefixCacheManager::instance().get_total_cached_blocks();
        LOG(INFO) << "[PageAllocator] emergency eviction: total_cached="
                  << total_cached;
        if (total_cached > 0) {
          size_t target_evict = std::max(total_cached / 50, size_t(16));
          size_t evicted =
              GlobalPrefixCacheManager::instance().evict_global_pure_lru(
                  target_evict);
          LOG(INFO) << "[PageAllocator] emergency eviction done: evicted="
                    << evicted << " free_phy=" << num_free_phy_pages_;
          if (evicted > 0) {
            VLOG(1) << "alloc_kv_cache_page: evicted " << evicted
                    << " prefix cache blocks to free physical pages";
          }
        }
        lock.lock();
      }
      if (!has_enough_phy_pages(phy_pages_needed)) {
        if (!enable_page_prealloc_) {
          LOG(ERROR) << "[PageAllocator] FATAL: No free physical pages left "
                     << "(free_phy=" << num_free_phy_pages_
                     << " total_phy=" << num_total_phy_pages_
                     << "). Process may exit.";
          throw std::runtime_error("No free physical pages left");
        }
        LOG(INFO) << "[PageAllocator] waiting for physical pages or prealloc: "
                  << "model=" << model_id << " dp_rank=" << dp_rank
                  << " free_phy=" << num_free_phy_pages_;
        // Wait for background preallocation or page freeing
        cond_.wait(lock);
      }
      continue;
    }

    if (!enable_page_prealloc_) {
      LOG(ERROR) << "[PageAllocator] FATAL: Inconsistent state, no pages "
                 << "available (model=" << model_id << " dp_rank=" << dp_rank
                 << "). Process may exit.";
      throw std::runtime_error(
          "Inconsistent page allocator state: no pages available");
    }

    // Wait for background preallocation or page freeing
    cond_.wait(lock);
  }

  CHECK(virt_page_id.has_value()) << "Virtual page ID should be set";

  // Increment pending ops before releasing lock
  state.pending_map_ops.fetch_add(1);

  // Release lock before mapping (slow path)
  lock.unlock();

  bool map_success = map_virt_pages(model_id, dp_rank, {*virt_page_id});

  // Decrement pending ops and notify waiters
  {
    std::lock_guard<std::mutex> guard(mtx_);
    state.pending_map_ops.fetch_sub(1);
    cond_.notify_all();

    if (!map_success) {
      // If mapping fails, return page to free list and restore phy pages
      (void)unmap_virt_pages(model_id, dp_rank, {*virt_page_id});
      dp_pages.free_virt_page_list.push_front(*virt_page_id);
      dp_pages.num_free_virt_pages++;
      release_phy_pages(phy_pages_needed);
      LOG(ERROR) << "Failed to map virtual page " << *virt_page_id;
      return nullptr;
    }
  }

  if (enable_page_prealloc_) {
    trigger_preallocation();
  }

  // Track this page as allocated
  {
    std::lock_guard<std::mutex> guard(mtx_);
    dp_pages.allocated_virt_page_list.insert(*virt_page_id);
    update_memory_usage();
  }

  return std::make_unique<VirtPage>(*virt_page_id, page_size_);
}

void PageAllocator::free_kv_cache_page(const std::string& model_id,
                                       int32_t dp_rank,
                                       int64_t virt_page_id) {
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";

  size_t phy_pages_per_virt = 0;

  {
    std::lock_guard<std::mutex> lock(mtx_);

    ModelState& state = get_model_state(model_id);
    auto& dp_pages = state.dp_group_pages[dp_rank];
    phy_pages_per_virt = state.phy_pages_per_virt_page;

    // Remove from allocated list
    dp_pages.allocated_virt_page_list.erase(virt_page_id);

    dp_pages.num_free_virt_pages++;

    // Fast path: keep page mapped for reuse if not sleeping and pool not full
    // Use per-model max_reserved_pages instead of global max_reserved_pages_
    if (!state.is_sleeping &&
        dp_pages.reserved_virt_page_list.size() <
            static_cast<size_t>(state.max_reserved_pages)) {
      dp_pages.reserved_virt_page_list.push_back(virt_page_id);
      update_memory_usage();
      cond_.notify_all();
      return;
    }
    // Slow path: need to unmap (model sleeping or reserved pool full)
  }

  // Slow path: unmap physical pages and add to free list
  unmap_virt_pages(model_id, dp_rank, {virt_page_id});
  {
    std::lock_guard<std::mutex> lock(mtx_);
    ModelState& state = get_model_state(model_id);
    auto& dp_pages = state.dp_group_pages[dp_rank];
    dp_pages.free_virt_page_list.push_back(virt_page_id);
    release_phy_pages(phy_pages_per_virt);
    update_memory_usage();
    cond_.notify_all();
  }
}

void PageAllocator::free_kv_cache_pages(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<int64_t>& virt_page_ids) {
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";

  std::vector<int64_t> pages_to_unmap;
  size_t phy_pages_per_virt = 0;

  {
    std::lock_guard<std::mutex> lock(mtx_);

    ModelState& state = get_model_state(model_id);
    auto& dp_pages = state.dp_group_pages[dp_rank];
    phy_pages_per_virt = state.phy_pages_per_virt_page;

    // Remove all pages from allocated list
    for (int64_t virt_page_id : virt_page_ids) {
      dp_pages.allocated_virt_page_list.erase(virt_page_id);
    }

    dp_pages.num_free_virt_pages += virt_page_ids.size();

    // If model is sleeping, unmap all pages
    if (state.is_sleeping) {
      pages_to_unmap = virt_page_ids;
    } else {
      // Use per-model max_reserved_pages instead of global max_reserved_pages_
      size_t num_to_reserve =
          state.max_reserved_pages - dp_pages.reserved_virt_page_list.size();

      if (num_to_reserve > 0) {
        // Fast path: keep some pages mapped for reuse
        size_t actual_reserve = std::min(num_to_reserve, virt_page_ids.size());
        for (size_t i = 0; i < actual_reserve; ++i) {
          dp_pages.reserved_virt_page_list.push_back(virt_page_ids[i]);
        }
        cond_.notify_all();

        // Remaining pages need to be unmapped
        for (size_t i = actual_reserve; i < virt_page_ids.size(); ++i) {
          pages_to_unmap.push_back(virt_page_ids[i]);
        }
      } else {
        pages_to_unmap = virt_page_ids;
      }
    }
  }

  if (pages_to_unmap.empty()) {
    update_memory_usage();
    return;
  }

  // Slow path: unmap physical pages
  unmap_virt_pages(model_id, dp_rank, pages_to_unmap);
  {
    std::lock_guard<std::mutex> lock(mtx_);
    ModelState& state = get_model_state(model_id);
    auto& dp_pages = state.dp_group_pages[dp_rank];
    for (int64_t virt_page_id : pages_to_unmap) {
      dp_pages.free_virt_page_list.push_back(virt_page_id);
    }
    release_phy_pages(pages_to_unmap.size() * phy_pages_per_virt);
    update_memory_usage();
    cond_.notify_all();
  }
}

void PageAllocator::trim_kv_cache(const std::string& model_id,
                                  int32_t dp_rank) {
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";

  std::vector<int64_t> pages_to_unmap;
  size_t phy_pages_per_virt = 0;

  {
    std::lock_guard<std::mutex> lock(mtx_);
    ModelState& state = get_model_state(model_id);
    auto& dp_pages = state.dp_group_pages[dp_rank];
    phy_pages_per_virt = state.phy_pages_per_virt_page;

    pages_to_unmap.assign(dp_pages.reserved_virt_page_list.begin(),
                          dp_pages.reserved_virt_page_list.end());
    dp_pages.reserved_virt_page_list.clear();

    if (pages_to_unmap.empty()) {
      update_memory_usage();
      return;
    }
  }

  unmap_virt_pages(model_id, dp_rank, pages_to_unmap);

  {
    std::lock_guard<std::mutex> lock(mtx_);
    ModelState& state = get_model_state(model_id);
    auto& dp_pages = state.dp_group_pages[dp_rank];
    for (int64_t virt_page_id : pages_to_unmap) {
      dp_pages.free_virt_page_list.push_back(virt_page_id);
    }
    release_phy_pages(pages_to_unmap.size() * phy_pages_per_virt);
    update_memory_usage();
  }
}

bool PageAllocator::alloc_weight_pages(const std::string& model_id,
                                       size_t num_pages) {
  {
    std::lock_guard<std::mutex> lock(mtx_);

    CHECK(initialized_) << "PageAllocator not initialized";

    ModelState& state = get_model_state(model_id);
    num_pages += WEIGHTS_EXTRA_NUM_PAGES;
    // Allocate physical pages directly for weight tensor
    // All-or-nothing: either allocate all requested pages or fail
    if (num_free_phy_pages_ < num_pages) {
      LOG(ERROR) << "Not enough physical pages for weight allocation: "
                 << "requested " << num_pages << ", available "
                 << num_free_phy_pages_;
      return false;
    }

    num_free_phy_pages_ -= num_pages;
    state.weight_pages_allocated = num_pages;
    update_memory_usage();
  }

  // Map weight tensor (full map)
  try {
    auto& allocator = XTensorAllocator::get_instance();
    allocator.broadcast_map_weight_tensor(model_id, num_pages);
  } catch (const std::exception& e) {
    // Rollback on failure
    std::lock_guard<std::mutex> lock(mtx_);
    num_free_phy_pages_ += num_pages;
    ModelState& state = get_model_state(model_id);
    state.weight_pages_allocated = 0;
    update_memory_usage();
    LOG(ERROR) << "Failed to map weight tensor: " << e.what();
    return false;
  }

  VLOG(1) << "Allocated and mapped " << num_pages
          << " physical pages for weight tensor of model " << model_id;
  return true;
}

void PageAllocator::free_weight_pages(const std::string& model_id,
                                      size_t num_pages) {
  // Unmap weight tensor first (full unmap)
  try {
    auto& allocator = XTensorAllocator::get_instance();
    allocator.broadcast_unmap_weight_tensor(model_id);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to unmap weight tensor: " << e.what();
    // Continue to release pages anyway
  }

  {
    std::lock_guard<std::mutex> lock(mtx_);

    CHECK(initialized_) << "PageAllocator not initialized";

    // Note: weight_pages_allocated is NOT cleared here
    // It's preserved for wakeup to know how many pages to re-allocate

    num_free_phy_pages_ += num_pages;
    update_memory_usage();
    cond_.notify_all();
  }

  VLOG(1) << "Unmapped and freed " << num_pages
          << " physical pages from weight tensor of model " << model_id;
}

size_t PageAllocator::get_num_free_virt_pages(const std::string& model_id,
                                              int32_t dp_rank) const {
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";
  std::lock_guard<std::mutex> lock(mtx_);
  const ModelState& state = get_model_state(model_id);
  return state.dp_group_pages[dp_rank].num_free_virt_pages;
}

size_t PageAllocator::get_num_inuse_virt_pages(const std::string& model_id,
                                               int32_t dp_rank) const {
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";
  std::lock_guard<std::mutex> lock(mtx_);
  const ModelState& state = get_model_state(model_id);
  return state.num_total_virt_pages -
         state.dp_group_pages[dp_rank].num_free_virt_pages;
}

size_t PageAllocator::get_num_total_virt_pages(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  const ModelState& state = get_model_state(model_id);
  return state.num_total_virt_pages;
}

size_t PageAllocator::get_weight_pages_allocated(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  const ModelState& state = get_model_state(model_id);
  return state.weight_pages_allocated;
}

size_t PageAllocator::get_num_reserved_virt_pages(const std::string& model_id,
                                                  int32_t dp_rank) const {
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";
  std::lock_guard<std::mutex> lock(mtx_);
  const ModelState& state = get_model_state(model_id);
  return state.dp_group_pages[dp_rank].reserved_virt_page_list.size();
}

size_t PageAllocator::get_num_free_phy_pages() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return num_free_phy_pages_;
}

size_t PageAllocator::get_num_total_phy_pages() const {
  return num_total_phy_pages_;
}

void PageAllocator::log_memory_state(const std::string& tag) const {
  std::lock_guard<std::mutex> lock(mtx_);
  LOG(INFO) << tag << " phy_pool: free=" << num_free_phy_pages_
            << " total=" << num_total_phy_pages_;
  for (const auto& [model_id, state] : model_states_) {
    for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
      const auto& dp = state.dp_group_pages[dp_rank];
      size_t reserved = dp.reserved_virt_page_list.size();
      size_t allocated = dp.allocated_virt_page_list.size();
      LOG(INFO) << tag << " model=" << model_id << " dp_rank=" << dp_rank
                << " reserved=" << reserved << " allocated=" << allocated;
    }
  }
}

int64_t PageAllocator::get_virt_page_id(int64_t block_id,
                                        size_t block_mem_size) const {
  return block_id * block_mem_size / page_size_;
}

offset_t PageAllocator::get_offset(int64_t virt_page_id) const {
  // Offset for single-layer XTensor map/unmap
  return virt_page_id * page_size_;
}

int64_t PageAllocator::num_layers(const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  const ModelState& state = get_model_state(model_id);
  return state.num_layers;
}

size_t PageAllocator::phy_pages_per_virt_page(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  const ModelState& state = get_model_state(model_id);
  return state.phy_pages_per_virt_page;
}

void PageAllocator::update_model_reserved_pages(const std::string& model_id,
                                                int32_t min_pages,
                                                int32_t max_pages) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_states_.find(model_id);
  if (it != model_states_.end()) {
    int32_t old_min = it->second.min_reserved_pages;
    int32_t old_max = it->second.max_reserved_pages;

    // Ensure min <= max
    int32_t new_min = std::min(min_pages, max_pages);
    int32_t new_max = std::max(min_pages, max_pages);

    it->second.min_reserved_pages = new_min;
    it->second.max_reserved_pages = new_max;

    // Trigger preallocation if needed
    prealloc_needed_ = true;
    cond_.notify_all();

    // LOG INFO
    LOG(INFO) << "[PriorityAlloc] Model " << model_id
              << " reserved pages updated: "
              << "min=" << old_min << "->" << new_min << ", max=" << old_max
              << "->" << new_max << ", priority=" << it->second.priority;
  }
}

void PageAllocator::prealloc_worker() {
  while (prealloc_running_.load()) {
    // Per-model, per-DP group pages to reserve
    std::vector<std::tuple<std::string, int32_t, std::vector<int64_t>>>
        model_dp_pages_to_reserve;

    {
      std::unique_lock<std::mutex> lock(mtx_);

      // Wait until preallocation is needed or thread is stopped
      cond_.wait(lock, [this] {
        return prealloc_needed_.load() || !prealloc_running_.load();
      });

      if (!prealloc_running_.load()) {
        break;
      }

      prealloc_needed_ = false;

      // Dynamic adjustment of min/max_reserved_pages based on
      // reserved_virt_page_list size Only adjust if dynamic adjustment is
      // enabled
      // Throttle: adjust at most once per 10s to avoid frequent trim/unmap
      // (frequent unmap can leave zombie VAs and cause 507899 on next map)
      constexpr double kPriorityAllocThrottleSeconds = 10.0;
      auto now = std::chrono::steady_clock::now();
      double elapsed_s =
          std::chrono::duration<double>(now - last_priority_alloc_adjust_time_)
              .count();
      bool throttle_ok = (elapsed_s >= kPriorityAllocThrottleSeconds);

      if (FLAGS_enable_dynamic_reserved_pages && throttle_ok) {
        bool phy_tight = (num_total_phy_pages_ > 0 &&
                          num_free_phy_pages_ < num_total_phy_pages_ * 0.05);
        for (auto& [model_id, state] : model_states_) {
          if (state.is_sleeping) {
            continue;
          }

          // Calculate average reserved pages across all DP ranks
          size_t total_reserved = 0;
          size_t num_dp_groups = 0;
          for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
            const auto& dp_pages = state.dp_group_pages[dp_rank];
            total_reserved += dp_pages.reserved_virt_page_list.size();
            num_dp_groups++;
          }
          size_t avg_reserved =
              (num_dp_groups > 0) ? total_reserved / num_dp_groups : 0;

          // Get current and base values
          int32_t current_min = state.min_reserved_pages;
          int32_t current_max = state.max_reserved_pages;
          int32_t base_min = state.base_min_reserved_pages;
          int32_t base_max = state.base_max_reserved_pages;
          // Calculate thresholds (based on current min/max)
          // Low threshold: 50% of min (trigger increase)
          // High threshold: 90% of max (trigger decrease)
          int32_t low_threshold = static_cast<int32_t>(current_min * 0.5);
          int32_t high_threshold = static_cast<int32_t>(current_max * 0.9);
          int32_t new_min = current_min;
          int32_t new_max = current_max;

          bool adjusted = false;

          // Case 1: reserved_virt_page_list is too small (< 50% of min)
          // Increase both min and max to encourage more preallocation
          bool at_cap =
              (current_min >= base_min * 2 && current_max >= base_max * 2);
          bool no_pages_but_high_target =
              (avg_reserved == 0 && current_min > base_min);
          if (!phy_tight && avg_reserved > 0 && !at_cap &&
              !no_pages_but_high_target &&
              avg_reserved < static_cast<size_t>(low_threshold)) {
            // Increase by 25% or at least 4 pages, but don't exceed base * 2
            int32_t min_increase =
                std::max(4, static_cast<int32_t>(current_min * 0.25));
            int32_t max_increase =
                std::max(4, static_cast<int32_t>(current_max * 0.25));

            new_min = std::min(current_min + min_increase, base_min * 2);
            new_max = std::min(current_max + max_increase, base_max * 2);

            // Ensure min <= max and maintain reasonable ratio (max should be at
            // least 2x min)
            if (new_min > new_max) {
              new_max = new_min;
            }
            if (new_max < new_min * 2) {
              new_max = new_min * 2;
            }

            // Only treat as adjusted and log when values actually change
            adjusted = (new_min != current_min || new_max != current_max);
            if (adjusted) {
              LOG(INFO) << "[PriorityAlloc] Model " << model_id
                        << " increasing min/max: reserved=" << avg_reserved
                        << " < threshold=" << low_threshold
                        << ", min=" << current_min << "->" << new_min
                        << ", max=" << current_max << "->" << new_max;
            }
          }
          // Case 2: reserved_virt_page_list is too large (> 90% of max)
          // Decrease both min and max to reduce memory usage
          else if (avg_reserved > static_cast<size_t>(high_threshold)) {
            // Decrease by 20% or at least 4 pages, but don't go below base
            // values
            int32_t min_decrease =
                std::max(4, static_cast<int32_t>(current_min * 0.2));
            int32_t max_decrease =
                std::max(4, static_cast<int32_t>(current_max * 0.2));

            new_min = std::max(current_min - min_decrease, base_min);
            new_max = std::max(current_max - max_decrease, base_max);

            // Ensure min <= max and maintain reasonable ratio
            if (new_min > new_max) {
              new_min = new_max;
            }
            if (new_max < new_min * 2) {
              new_max = new_min * 2;
            }

            // Only treat as adjusted and log when values actually change
            adjusted = (new_min != current_min || new_max != current_max);
            if (adjusted) {
              LOG(INFO) << "[PriorityAlloc] Model " << model_id
                        << " decreasing min/max: reserved=" << avg_reserved
                        << " > threshold=" << high_threshold
                        << ", min=" << current_min << "->" << new_min
                        << ", max=" << current_max << "->" << new_max;
            }
          }

          // Update if changed
          if (adjusted && (new_min != current_min || new_max != current_max)) {
            state.min_reserved_pages = new_min;
            state.max_reserved_pages = new_max;
            last_priority_alloc_adjust_time_ = now;
            prealloc_needed_ = true;
          }
        }
      }

      if (FLAGS_enable_prefix_cache && FLAGS_enable_xtensor) {
        size_t free_phy_pages = num_free_phy_pages_;
        size_t total_phy_pages = num_total_phy_pages_;
        constexpr double kProactiveEvictionWatermark = 0.15;  // 15%
        constexpr double kEmergencyWatermark = 0.05;          // 5%
        if (total_phy_pages > 0) {
          double free_ratio =
              static_cast<double>(free_phy_pages) / total_phy_pages;
          if (free_ratio > kEmergencyWatermark &&
              free_ratio <= kProactiveEvictionWatermark) {
            double elapsed_proactive = std::chrono::duration<double>(
                                           now - last_proactive_eviction_time_)
                                           .count();
            if (elapsed_proactive >= kPriorityAllocThrottleSeconds) {
              lock.unlock();
              size_t total_cached = GlobalPrefixCacheManager::instance()
                                        .get_total_cached_blocks();
              if (total_cached > 0) {
                size_t target = std::min(
                    std::max(total_cached / 20, size_t(20)), size_t(80));
                size_t evicted =
                    GlobalPrefixCacheManager::instance().evict_global_pure_lru(
                        target);
                VLOG(1) << "[PageAllocator] proactive eviction (free "
                        << (free_ratio * 100) << "%): evicted=" << evicted;
              }
              last_proactive_eviction_time_ = now;
              lock.lock();
            }
          }
        }
      }

      // Emergency eviction: Check global physical memory pressure
      // If free physical pages are below 5% threshold, trigger emergency
      // PrefixCache eviction to free up memory for active requests
      if (FLAGS_enable_prefix_cache && FLAGS_enable_xtensor) {
        size_t free_phy_pages = num_free_phy_pages_;
        size_t total_phy_pages = num_total_phy_pages_;

        // Trigger emergency eviction if free pages < 5% of total
        if (total_phy_pages > 0 && free_phy_pages < total_phy_pages * 0.05) {
          LOG(WARNING) << "Global physical memory tight! Free pages: "
                       << free_phy_pages << "/" << total_phy_pages << " ("
                       << (free_phy_pages * 100 / total_phy_pages) << "%)"
                       << ". Triggering emergency PrefixCache eviction.";

          // Release lock before calling evict_global_pure_lru to avoid deadlock
          // (evict_global_pure_lru will acquire its own mutex)
          lock.unlock();

          // Evict 10% of cached blocks (or at least 100 blocks)
          size_t total_cached =
              GlobalPrefixCacheManager::instance().get_total_cached_blocks();
          size_t target_evict = std::max(total_cached / 10, size_t(100));

          size_t actual_evicted =
              GlobalPrefixCacheManager::instance().evict_global_pure_lru(
                  target_evict);

          VLOG(1) << "Emergency eviction finished. Evicted: " << actual_evicted
                  << " blocks out of " << total_cached
                  << " total cached blocks.";

          // Re-acquire lock for the rest of the function
          lock.lock();
          if (num_free_phy_pages_ < total_phy_pages * 0.05) {
            LOG(WARNING) << "[PageAllocator] WARN: free pages still very low "
                         << "after emergency eviction: " << num_free_phy_pages_
                         << "/" << total_phy_pages
                         << " evicted_blocks=" << actual_evicted
                         << " total_cached_before=" << total_cached
                         << ". Risk of OOM or alloc failure (cache empty).";
            // Second round: more aggressive eviction to try to free full pages
            lock.unlock();
            size_t total_cached2 =
                GlobalPrefixCacheManager::instance().get_total_cached_blocks();
            if (total_cached2 > 0) {
              size_t target2 = std::max(total_cached2 / 10, size_t(100));
              size_t evicted2 =
                  GlobalPrefixCacheManager::instance().evict_global_pure_lru(
                      target2);
              LOG(WARNING)
                  << "[PageAllocator] : second eviction evicted_blocks="
                  << evicted2 << " total_cached=" << total_cached2;
            }
            lock.lock();
          }
        }
      }

      // Check each model for preallocation needs
      for (auto& [model_id, state] : model_states_) {
        // Skip sleeping models
        if (state.is_sleeping) {
          continue;
        }

        // Check each DP group for preallocation needs
        for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
          auto& dp_pages = state.dp_group_pages[dp_rank];

          size_t current_reserved = dp_pages.reserved_virt_page_list.size();
          size_t to_reserve = 0;
          if (current_reserved <
              static_cast<size_t>(state.min_reserved_pages)) {
            to_reserve = state.min_reserved_pages - current_reserved;
          }

          // Limit by available free virtual pages
          to_reserve =
              std::min(to_reserve, dp_pages.free_virt_page_list.size());

          // Limit by available physical pages (shared resource)
          size_t max_by_phy =
              num_free_phy_pages_ / state.phy_pages_per_virt_page;
          to_reserve = std::min(to_reserve, max_by_phy);

          if (to_reserve == 0) {
            continue;
          }

          // Get pages from free list and consume physical pages
          std::vector<int64_t> pages_to_reserve;
          for (size_t i = 0;
               i < to_reserve && !dp_pages.free_virt_page_list.empty();
               ++i) {
            pages_to_reserve.push_back(dp_pages.free_virt_page_list.front());
            dp_pages.free_virt_page_list.pop_front();
          }
          if (!consume_phy_pages(pages_to_reserve.size() *
                                 state.phy_pages_per_virt_page)) {
            // Rollback: put pages back to free list
            for (auto it = pages_to_reserve.rbegin();
                 it != pages_to_reserve.rend();
                 ++it) {
              dp_pages.free_virt_page_list.push_front(*it);
            }
            continue;  // Skip this model/dp_rank
          }
          model_dp_pages_to_reserve.emplace_back(
              model_id, dp_rank, std::move(pages_to_reserve));
        }
      }
    }

    // Map pages for each model and DP group
    for (auto& [model_id, dp_rank, pages_to_reserve] :
         model_dp_pages_to_reserve) {
      if (pages_to_reserve.empty()) {
        continue;
      }

      // Check if model went to sleep before mapping, and increment pending ops
      {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = model_states_.find(model_id);
        if (it == model_states_.end() || it->second.is_sleeping) {
          // Model was unregistered or went to sleep, return pages and skip
          if (it != model_states_.end()) {
            auto& dp_pages = it->second.dp_group_pages[dp_rank];
            for (auto pg_it = pages_to_reserve.rbegin();
                 pg_it != pages_to_reserve.rend();
                 ++pg_it) {
              dp_pages.free_virt_page_list.push_front(*pg_it);
            }
            release_phy_pages(pages_to_reserve.size() *
                              it->second.phy_pages_per_virt_page);
          }
          continue;
        }
        // Increment pending ops before mapping
        it->second.pending_map_ops.fetch_add(1);
      }

      bool map_success = map_virt_pages(model_id, dp_rank, pages_to_reserve);

      // Decrement pending ops and handle result
      {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = model_states_.find(model_id);
        if (it != model_states_.end()) {
          it->second.pending_map_ops.fetch_sub(1);
          cond_.notify_all();  // Notify sleep_model if waiting
        }

        if (!map_success) {
          // If mapping fails, return pages to free list and release phy pages
          if (it != model_states_.end()) {
            auto& dp_pages = it->second.dp_group_pages[dp_rank];
            for (auto pg_it = pages_to_reserve.rbegin();
                 pg_it != pages_to_reserve.rend();
                 ++pg_it) {
              dp_pages.free_virt_page_list.push_front(*pg_it);
            }
            release_phy_pages(pages_to_reserve.size() *
                              it->second.phy_pages_per_virt_page);
          }
          LOG(ERROR) << "Failed to preallocate " << pages_to_reserve.size()
                     << " virtual pages for model=" << model_id
                     << " dp_rank=" << dp_rank;
          continue;
        }

        // Mapping succeeded, update reserved list
        if (it != model_states_.end() && !it->second.is_sleeping) {
          auto& dp_pages = it->second.dp_group_pages[dp_rank];
          for (int64_t virt_page_id : pages_to_reserve) {
            dp_pages.reserved_virt_page_list.push_back(virt_page_id);
          }
          update_memory_usage();
        } else {
          // Model went to sleep or was unregistered, release pages
          if (it != model_states_.end()) {
            release_phy_pages(pages_to_reserve.size() *
                              it->second.phy_pages_per_virt_page);
          }
        }
      }
      VLOG(1) << "Preallocated " << pages_to_reserve.size()
              << " virtual pages for model=" << model_id
              << " dp_rank=" << dp_rank;
    }
  }
}

void PageAllocator::start_prealloc_thread_internal() {
  if (prealloc_thd_ == nullptr) {
    prealloc_running_ = true;
    prealloc_thd_ =
        std::make_unique<std::thread>(&PageAllocator::prealloc_worker, this);

    // Initial preallocation trigger
    trigger_preallocation();
  }
}

void PageAllocator::stop_prealloc_thread(double timeout) {
  if (prealloc_thd_ != nullptr) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      prealloc_running_ = false;
      cond_.notify_all();
    }

    if (prealloc_thd_->joinable()) {
      auto future =
          std::async(std::launch::async, [this]() { prealloc_thd_->join(); });

      if (future.wait_for(std::chrono::duration<double>(timeout)) ==
          std::future_status::timeout) {
        LOG(WARNING) << "Preallocation thread did not stop within timeout";
      }
    }
    prealloc_thd_.reset();
    VLOG(1) << "Stopped page preallocation thread";
  }
}

void PageAllocator::trigger_preallocation() {
  std::lock_guard<std::mutex> lock(mtx_);
  prealloc_needed_ = true;
  cond_.notify_all();
}

bool PageAllocator::map_virt_pages(const std::string& model_id,
                                   int32_t dp_rank,
                                   const std::vector<int64_t>& virt_page_ids) {
  // Convert virtual page IDs to offsets (for single-layer XTensor)
  std::vector<offset_t> offsets;
  offsets.reserve(virt_page_ids.size());

  for (int64_t virt_page_id : virt_page_ids) {
    offsets.push_back(get_offset(virt_page_id));
  }

  // Broadcast to workers in this DP group
  auto& allocator = XTensorAllocator::get_instance();
  if (!allocator.broadcast_map_to_kv_tensors(model_id, dp_rank, offsets)) {
    LOG(ERROR) << "Failed to broadcast map_to_kv_tensors for model=" << model_id
               << " dp_rank=" << dp_rank;
    return false;
  }
  return true;
}

bool PageAllocator::unmap_virt_pages(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<int64_t>& virt_page_ids) {
  // Convert virtual page IDs to offsets (for single-layer XTensor)
  std::vector<offset_t> offsets;
  offsets.reserve(virt_page_ids.size());

  for (int64_t virt_page_id : virt_page_ids) {
    offsets.push_back(get_offset(virt_page_id));
  }

  // Broadcast to workers in this DP group
  auto& allocator = XTensorAllocator::get_instance();
  if (!allocator.broadcast_unmap_from_kv_tensors(model_id, dp_rank, offsets)) {
    LOG(ERROR) << "Failed to broadcast unmap_from_kv_tensors for model="
               << model_id << " dp_rank=" << dp_rank;
    return false;
  }
  return true;
}

void PageAllocator::update_memory_usage() {
  // Note: Caller must hold mtx_
  size_t free_phy_pages = num_free_phy_pages_;

  // Calculate physical memory usage
  size_t used_phy_mem = (num_total_phy_pages_ - free_phy_pages) * page_size_;

  VLOG(2) << "Memory usage: "
          << "dp_size=" << dp_size_ << ", free_phy_pages=" << free_phy_pages
          << ", used_phy_mem=" << used_phy_mem / (1024 * 1024) << "MB"
          << ", num_models=" << model_states_.size();
}

}  // namespace xllm
