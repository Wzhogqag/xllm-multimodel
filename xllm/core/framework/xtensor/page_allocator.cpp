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

#include "page_allocator.h"

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_set>

#include "common/global_flags.h"
#include "core/distributed_runtime/master.h"
#include "framework/block/block_manager.h"
#include "framework/prefix_cache/global_prefix_cache_manager.h"
#include "xtensor_allocator.h"

namespace xllm {

namespace {
constexpr int32_t kPhyPageUsedCounterMaxWorkers = 1024;

std::string normalize_base_model_id(const std::string& model_id) {
  const size_t hash_pos = model_id.rfind('#');
  if (hash_pos == std::string::npos || hash_pos == 0 ||
      hash_pos == model_id.size() - 1) {
    return model_id;
  }
  for (size_t i = hash_pos + 1; i < model_id.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(model_id[i]))) {
      return model_id;
    }
  }
  return model_id.substr(0, hash_pos);
}
}  // namespace

void PageAllocator::init(size_t num_phy_pages,
                         int32_t dp_size,
                         int32_t max_world_size,
                         bool enable_page_prealloc) {
  std::lock_guard<std::mutex> lock(mtx_);

  if (initialized_) {
    LOG(WARNING) << "PageAllocator already initialized, ignoring re-init";
    return;
  }

  dp_size_ = dp_size;
  max_world_size_ = max_world_size > 0 ? max_world_size : 1;
  page_size_ = FLAGS_phy_page_granularity_size;
  enable_page_prealloc_ = enable_page_prealloc;

  // Set total physical pages from parameter (same for all workers)
  num_total_phy_pages_ = num_phy_pages;

  // Initialize per-worker page tracking
  // All workers start with 0 pages used
  worker_pages_used_.resize(max_world_size_, 0);
  worker_reported_pages_used_.resize(max_world_size_, 0);
  init_reported_phy_pages_shm_if_needed();

  initialized_ = true;

  LOG(INFO) << "Init PageAllocator: "
            << "dp_size=" << dp_size_ << ", max_world_size=" << max_world_size_
            << ", page_size=" << page_size_ / (1024 * 1024) << "MB"
            << ", num_total_phy_pages=" << num_total_phy_pages_
            << ", enable_prealloc=" << enable_page_prealloc;
}

PageAllocator::~PageAllocator() {
  try {
    if (enable_page_prealloc_ && prealloc_thd_ != nullptr) {
      stop_prealloc_thread(PREALLOC_THREAD_TIMEOUT);
    }
    stop_async_eviction_thread();
    if (reported_phy_pages_shm_ptr_ != nullptr) {
      munmap(reported_phy_pages_shm_ptr_,
             sizeof(uint64_t) * kPhyPageUsedCounterMaxWorkers);
      reported_phy_pages_shm_ptr_ = nullptr;
    }
    if (reported_phy_pages_shm_fd_ >= 0) {
      close(reported_phy_pages_shm_fd_);
      reported_phy_pages_shm_fd_ = -1;
    }
  } catch (...) {
    // Silently ignore exceptions during cleanup
  }
}

void PageAllocator::init_reported_phy_pages_shm_if_needed() {
  if (reported_phy_pages_shm_ptr_ != nullptr) {
    return;
  }
  const size_t shm_bytes =
      sizeof(uint64_t) * static_cast<size_t>(kPhyPageUsedCounterMaxWorkers);
  
  std::string kPhyPageUsedCounterShmName =
      "/xllm_activation_phy_pages_used_" + std::to_string(FLAGS_port - FLAGS_node_rank);
  int fd = shm_open(
      kPhyPageUsedCounterShmName.c_str(), O_CREAT | O_RDWR, static_cast<mode_t>(0666));
  if (fd < 0) {
    LOG(WARNING) << "Failed to open phy-page report shm: " << strerror(errno);
    return;
  }
  if (ftruncate(fd, static_cast<off_t>(shm_bytes)) != 0) {
    LOG(WARNING) << "Failed to resize phy-page report shm: " << strerror(errno);
    close(fd);
    return;
  }
  void* addr =
      mmap(nullptr, shm_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) {
    LOG(WARNING) << "Failed to map phy-page report shm: " << strerror(errno);
    close(fd);
    return;
  }
  reported_phy_pages_shm_fd_ = fd;
  reported_phy_pages_shm_ptr_ = static_cast<uint64_t*>(addr);

  // no activation will be allocated before this function call
  // ensure the share memory is initialized before being read
  for (int32_t i = 0; i < kPhyPageUsedCounterMaxWorkers; ++i) {
    __atomic_store_n(&reported_phy_pages_shm_ptr_[i],
                    static_cast<uint64_t>(0),
                    __ATOMIC_RELAXED);
  }
}

void PageAllocator::sync_reported_phy_pages_from_shm_locked() const {
  if (reported_phy_pages_shm_ptr_ == nullptr) {
    return;
  }
  const int32_t bound =
      std::min(max_world_size_, kPhyPageUsedCounterMaxWorkers);
  for (int32_t i = 0; i < bound; ++i) {
    worker_reported_pages_used_[i] = static_cast<size_t>(
        __atomic_load_n(&reported_phy_pages_shm_ptr_[i], __ATOMIC_RELAXED));
  }
}

size_t PageAllocator::get_worker_used_pages_locked(int32_t worker_rank) const {
  // LOG(INFO) << "Worker " << worker_rank << " used pages: " <<
  // worker_pages_used_[worker_rank] << " + " <<
  // worker_reported_pages_used_[worker_rank];
  return worker_pages_used_[worker_rank] +
         worker_reported_pages_used_[worker_rank];
}

bool PageAllocator::register_model(const std::string& model_id,
                                   int64_t num_layers,
                                   int32_t master_status,
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
  state.num_total_virt_pages = num_total_phy_pages_ / num_layers / 2;
  // Each virt_page needs to map on all K and V XTensors
  state.phy_pages_per_virt_page = 2 * num_layers;
  // Always start with is_sleeping = false to allow KV cache allocation
  // If sleeping is needed, call sleep_model after initialization
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

  if (layer_offload_mgr_) {
    layer_offload_mgr_->register_model(
        model_id, static_cast<int32_t>(num_layers), priority);
  }

  return true;
}

bool PageAllocator::sleep_model(const std::string& model_id) {
  std::vector<std::pair<int32_t, std::vector<int64_t>>> pages_to_unmap;
  std::vector<bool> unmap_success;
  size_t total_phy_pages_to_release = 0;
  size_t phy_pages_per_virt = 0;
  size_t weight_pages = 0;

  {
    std::unique_lock<std::mutex> lock(mtx_);

    auto it = model_states_.find(model_id);
    if (it == model_states_.end()) {
      LOG(WARNING) << "Model " << model_id << " not found for sleep";
      return false;
    }

    ModelState& state = it->second;
    if (state.is_sleeping) {
      LOG(WARNING) << "Model " << model_id << " is already sleeping";
      return false;
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
    weight_pages = state.weight_pages_allocated;

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
              << total_phy_pages_to_release << " KV cache pages and "
              << weight_pages << " weight pages";
  }

  // Release weight pages first (reuse existing function)
  if (weight_pages > 0) {
    if (!free_weight_pages(model_id, weight_pages)) {
      LOG(ERROR) << "Failed to free weight pages during sleep for model "
                 << model_id << ", keep consumed weight page count";
      return false;
    }
  }

  bool has_unmap_failures = false;
  unmap_success.assign(pages_to_unmap.size(), true);

  // Unmap pages outside the lock
  for (size_t i = 0; i < pages_to_unmap.size(); ++i) {
    auto& [dp_rank, virt_page_ids] = pages_to_unmap[i];
    if (!virt_page_ids.empty()) {
      if (!unmap_virt_pages(model_id, dp_rank, virt_page_ids)) {
        has_unmap_failures = true;
        unmap_success[i] = false;
      }
    }
  }

  // Update state after unmapping
  // Note: We keep reserved_virt_page_list and allocated_virt_page_list
  // unchanged so that XTensorBlockManagerImpl's state remains consistent. On
  // wakeup, we will re-map these same pages.
  {
    std::lock_guard<std::mutex> lock(mtx_);
    // Release physical pages from per-worker tracking for each DP group
    for (size_t i = 0; i < pages_to_unmap.size(); ++i) {
      if (!unmap_success[i]) {
        continue;
      }
      auto& [dp_rank, virt_page_ids] = pages_to_unmap[i];
      size_t pages_released = virt_page_ids.size() * phy_pages_per_virt;
      auto [start_w, end_w] = get_dp_group_worker_range(model_id, dp_rank);
      for (int32_t w = start_w; w < end_w && w < max_world_size_; ++w) {
        if (worker_pages_used_[w] >= pages_released) {
          worker_pages_used_[w] -= pages_released;
        } else {
          worker_pages_used_[w] = 0;
        }
      }
    }
    update_memory_usage();
    cond_.notify_all();
  }

  if (has_unmap_failures) {
    LOG(ERROR) << "Model " << model_id
               << " sleep encountered KV unmap failures, page accounting was "
                  "not released for failed groups";
    return false;
  }
  LOG(INFO) << "Model " << model_id << " is now sleeping";
  return true;
}

bool PageAllocator::wakeup_model(const std::string& model_id) {
  std::vector<std::pair<int32_t, std::vector<int64_t>>> groups_to_map;
  std::vector<size_t> pages_to_consume_per_worker;
  size_t total_phy_pages_needed = 0;
  size_t phy_pages_per_virt = 0;
  size_t weight_pages = 0;
  int32_t weight_end_worker = 0;

  {
    std::lock_guard<std::mutex> lock(mtx_);

    auto it = model_states_.find(model_id);
    if (it == model_states_.end()) {
      LOG(WARNING) << "Model " << model_id << " not found for wakeup";
      return false;
    }

    ModelState& state = it->second;
    if (!state.is_sleeping) {
      LOG(WARNING) << "Model " << model_id << " is not sleeping";
      return false;
    }

    phy_pages_per_virt = state.phy_pages_per_virt_page;
    weight_pages = state.weight_pages_allocated;
    int32_t model_world_size =
        state.model_world_size > 0 ? state.model_world_size : max_world_size_;
    weight_end_worker = std::min(model_world_size, max_world_size_);
    pages_to_consume_per_worker.assign(max_world_size_, 0);

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
        size_t pages_needed = virt_page_ids.size() * phy_pages_per_virt;
        auto [start_w, end_w] = get_dp_group_worker_range(model_id, dp_rank);
        total_phy_pages_needed += pages_needed;
        groups_to_map.push_back({dp_rank, std::move(virt_page_ids)});
        for (int32_t w = start_w; w < end_w && w < max_world_size_; ++w) {
          pages_to_consume_per_worker[w] += pages_needed;
        }
      }
    }

    // Weight pages are allocated on all workers used by this model.
    for (int32_t w = state.model_worker_rank_base;
         w < state.model_worker_rank_base + model_world_size;
         ++w) {
      pages_to_consume_per_worker[w] += weight_pages;
    }

    // Phase 1 (lock): check KV + weight requirements together.
    sync_reported_phy_pages_from_shm_locked();
    for (int32_t w = 0; w < max_world_size_; ++w) {
      if (pages_to_consume_per_worker[w] == 0) {
        continue;
      }
      size_t worker_free =
          num_total_phy_pages_ - get_worker_used_pages_locked(w);
      if (worker_free < pages_to_consume_per_worker[w]) {
        LOG(ERROR) << "Not enough physical pages for wakeup worker=" << w
                   << ": need " << pages_to_consume_per_worker[w]
                   << ", available " << worker_free;
        return false;
      }
    }

    // Only consume page counts after all checks pass.
    for (int32_t w = 0; w < max_world_size_; ++w) {
      if (pages_to_consume_per_worker[w] > 0) {
        worker_pages_used_[w] += pages_to_consume_per_worker[w];
      }
    }
    update_memory_usage();

    LOG(INFO) << "Waking up model " << model_id << ", will map "
              << total_phy_pages_needed << " KV cache pages and "
              << weight_pages << " weight pages";
  }

  // Phase 2 (unlock): execute actual mapping/allocation.
  bool map_ok = true;
  for (auto& [dp_rank, virt_page_ids] : groups_to_map) {
    if (virt_page_ids.empty()) {
      continue;
    }
    if (!map_virt_pages(model_id, dp_rank, virt_page_ids)) {
      map_ok = false;
      LOG(ERROR) << "Failed to map KV cache pages for wakeup model=" << model_id
                 << " dp_rank=" << dp_rank;
      break;
    }
  }

  bool weight_ok = true;
  if (map_ok && weight_pages > 0) {
    auto& allocator = XTensorAllocator::get_instance();
    weight_ok = allocator.broadcast_alloc_weight_pages(model_id, weight_pages);
    if (!weight_ok) {
      LOG(ERROR) << "Failed to re-allocate weight pages for model " << model_id;
    }
  }

  if (!map_ok || !weight_ok) {
    LOG(ERROR) << "Failed to wake up model " << model_id
               << ", keep sleep state and keep consumed page counts "
               << "(no rollback due to unknown mapping state)";
    return false;
  }

  // Phase 3: mark awake only after all operations succeed.
  {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = model_states_.find(model_id);
    if (it != model_states_.end()) {
      it->second.is_sleeping = false;
    }
    update_memory_usage();
  }

  // Trigger preallocation to refill reserved pages
  if (enable_page_prealloc_) {
    trigger_preallocation();
  }

  LOG(INFO) << "Model " << model_id << " is now awake";
  return true;
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

std::pair<int32_t, int32_t> PageAllocator::get_dp_group_worker_range(
    const std::string& model_id,
    int32_t dp_rank) const {
  // Note: Caller must hold mtx_
  auto it = model_states_.find(model_id);
  int32_t tp_size = 1;
  int32_t worker_rank_base = 0;
  if (it != model_states_.end() && it->second.model_tp_size > 0) {
    tp_size = it->second.model_tp_size;
    worker_rank_base = std::max(0, it->second.model_worker_rank_base);
  }
  int32_t start_worker = worker_rank_base + dp_rank * tp_size;
  int32_t end_worker = start_worker + tp_size;
  return {start_worker, std::min(end_worker, max_world_size_)};
}

size_t PageAllocator::get_min_free_pages_in_range(int32_t start_worker,
                                                  int32_t end_worker) const {
  // Note: Caller must hold mtx_
  sync_reported_phy_pages_from_shm_locked();
  size_t min_free = num_total_phy_pages_;
  for (int32_t w = start_worker; w < end_worker && w < max_world_size_; ++w) {
    size_t worker_free = num_total_phy_pages_ - get_worker_used_pages_locked(w);
    min_free = std::min(min_free, worker_free);
  }
  return min_free;
}

bool PageAllocator::has_enough_phy_pages_for_dp(const std::string& model_id,
                                                int32_t dp_rank,
                                                size_t num_phy_pages) const {
  // Note: Caller must hold mtx_
  auto [start_w, end_w] = get_dp_group_worker_range(model_id, dp_rank);
  size_t min_free = get_min_free_pages_in_range(start_w, end_w);
  return min_free >= num_phy_pages;
}

void PageAllocator::consume_phy_pages_for_dp(const std::string& model_id,
                                             int32_t dp_rank,
                                             size_t num_phy_pages) {
  // Note: Caller must hold mtx_
  auto [start_w, end_w] = get_dp_group_worker_range(model_id, dp_rank);
  size_t min_free = get_min_free_pages_in_range(start_w, end_w);
  if (min_free < num_phy_pages) {
    LOG(ERROR) << "Not enough weight physical pages for dp_rank=" << dp_rank
                 << ": need " << num_phy_pages << ", available " << min_free;
    return;
  }
  for (int32_t w = start_w; w < end_w && w < max_world_size_; ++w) {
    worker_pages_used_[w] += num_phy_pages;
  }
  return;
}

bool PageAllocator::try_to_consume_phy_pages_for_dp(const std::string& model_id,
                                             int32_t dp_rank,
                                             size_t num_phy_pages) {
  // Note: Caller must hold mtx_
  auto [start_w, end_w] = get_dp_group_worker_range(model_id, dp_rank);
  size_t min_free = get_min_free_pages_in_range(start_w, end_w);
  // 此条路径是KV分配调用的，中间激活优先级大于KV（KV不足可以在调度侧请求抢占重算）
  const double min_free_ratio =
      std::clamp(FLAGS_kv_prealloc_min_free_ratio, 0.0, 1.0);
  if (min_free < num_phy_pages ||
      min_free - num_phy_pages < num_total_phy_pages_ * min_free_ratio) {
    LOG(WARNING) << "Not enough KV physical pages for dp_rank=" << dp_rank
                 << ": need " << num_phy_pages << ", available " << min_free;
    return false;
  }
  for (int32_t w = start_w; w < end_w && w < max_world_size_; ++w) {
    worker_pages_used_[w] += num_phy_pages;
  }
  return true;
}

void PageAllocator::release_phy_pages_for_dp(const std::string& model_id,
                                             int32_t dp_rank,
                                             size_t num_phy_pages) {
  // Note: Caller must hold mtx_
  auto [start_w, end_w] = get_dp_group_worker_range(model_id, dp_rank);
  for (int32_t w = start_w; w < end_w && w < max_world_size_; ++w) {
    if (worker_pages_used_[w] >= num_phy_pages) {
      worker_pages_used_[w] -= num_phy_pages;
    } else {
      LOG(WARNING) << "Worker " << w << " pages underflow during release";
      worker_pages_used_[w] = 0;
    }
  }
}

bool PageAllocator::consume_phy_pages_for_worker(int32_t worker_rank,
                                                 size_t num_phy_pages) {
  // Note: Caller must hold mtx_
  if (worker_rank < 0 || worker_rank >= max_world_size_) {
    LOG(ERROR) << "Invalid worker_rank=" << worker_rank
               << ", max_world_size=" << max_world_size_;
    return false;
  }
  if (num_total_phy_pages_ - worker_reported_pages_used_[worker_rank] <
      num_phy_pages) {
    LOG(WARNING) << "Not enough physical pages for worker_rank=" << worker_rank
                 << ": need " << num_phy_pages << ", available "
                 << num_total_phy_pages_ -
                        worker_reported_pages_used_[worker_rank];
    return false;
  }
  worker_reported_pages_used_[worker_rank] += num_phy_pages;
  if (reported_phy_pages_shm_ptr_ != nullptr &&
      worker_rank < kPhyPageUsedCounterMaxWorkers) {
    __atomic_store_n(
        &reported_phy_pages_shm_ptr_[worker_rank],
        static_cast<uint64_t>(worker_reported_pages_used_[worker_rank]),
        __ATOMIC_RELAXED);
  }
  return true;
}

void PageAllocator::release_phy_pages_for_worker(int32_t worker_rank,
                                                 size_t num_phy_pages) {
  // Note: Caller must hold mtx_
  if (worker_rank < 0 || worker_rank >= max_world_size_) {
    LOG(ERROR) << "Invalid worker_rank=" << worker_rank
               << ", max_world_size=" << max_world_size_;
    return;
  }
  if (worker_reported_pages_used_[worker_rank] >= num_phy_pages) {
    worker_reported_pages_used_[worker_rank] -= num_phy_pages;
  } else {
    LOG(WARNING) << "Worker " << worker_rank
                 << " pages underflow during release";
    worker_reported_pages_used_[worker_rank] = 0;
  }
  if (reported_phy_pages_shm_ptr_ != nullptr &&
      worker_rank < kPhyPageUsedCounterMaxWorkers) {
    __atomic_store_n(
        &reported_phy_pages_shm_ptr_[worker_rank],
        static_cast<uint64_t>(worker_reported_pages_used_[worker_rank]),
        __ATOMIC_RELAXED);
  }
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
        has_enough_phy_pages_for_dp(model_id, dp_rank, phy_pages_needed)) {
      virt_page_id = dp_pages.free_virt_page_list.front();
      dp_pages.free_virt_page_list.pop_front();
      dp_pages.num_free_virt_pages--;
      if (!try_to_consume_phy_pages_for_dp(model_id, dp_rank, phy_pages_needed)) {
        // Rollback: put the page back
        dp_pages.free_virt_page_list.push_front(*virt_page_id);
        dp_pages.num_free_virt_pages++;
        virt_page_id.reset();
        return nullptr;
      }
      break;
    }

    // Check if we're out of resources
    if (dp_pages.free_virt_page_list.empty()) {
      LOG(WARNING) << "[PageAllocator] FATAL: No free virtual pages left for "
                   << "model=" << model_id << " dp_rank=" << dp_rank
                   << ". Process may exit.";
    }
    if (!has_enough_phy_pages_for_dp(model_id, dp_rank, phy_pages_needed)) {
      // Physical page pool exhausted: try emergency prefix cache eviction
      // to free pages instead of failing (multi-model safety).
      if (FLAGS_enable_prefix_cache && FLAGS_enable_xtensor) {
        lock.unlock();
        size_t total_cached =
            GlobalPrefixCacheManager::instance().get_total_cached_blocks();
        if (total_cached > 0) {
          size_t target_evict = std::max(total_cached / 50, size_t(16));
          size_t evicted =
              GlobalPrefixCacheManager::instance().evict_global_pure_lru(
                  target_evict);
          if (evicted > 0) {
            VLOG(1) << "alloc_kv_cache_page: evicted " << evicted
                    << " prefix cache blocks to free physical pages";
          }
        }
        lock.lock();
      }
      if (!has_enough_phy_pages_for_dp(model_id, dp_rank, phy_pages_needed)) {
        if (!enable_page_prealloc_) {
          LOG(ERROR) << "[PageAllocator] FATAL: No free physical pages left "
                     << "(free_phy=" << get_num_free_phy_pages()
                     << " total_phy=" << num_total_phy_pages_
                     << "). Process may exit.";
          throw std::runtime_error("No free physical pages left");
        }
        LOG(WARNING) << "Not enough physical pages for model=" << model_id
                     << " dp_rank=" << dp_rank
                     << ", prealloc enabled, return nullptr instead of waiting";
        return nullptr;
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
      dp_pages.free_virt_page_list.push_front(*virt_page_id);
      dp_pages.num_free_virt_pages++;
      release_phy_pages_for_dp(model_id, dp_rank, phy_pages_needed);
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

    if (pages_to_unmap.empty()) {
      update_memory_usage();
      return;
    }
  }

  // Slow path: unmap physical pages
  bool unmap_success = unmap_virt_pages(model_id, dp_rank, pages_to_unmap);
  {
    std::lock_guard<std::mutex> lock(mtx_);
    ModelState& state = get_model_state(model_id);
    auto& dp_pages = state.dp_group_pages[dp_rank];
    if (unmap_success) {
      for (int64_t virt_page_id : pages_to_unmap) {
        dp_pages.free_virt_page_list.push_back(virt_page_id);
      }
      release_phy_pages_for_dp(
          model_id, dp_rank, pages_to_unmap.size() * phy_pages_per_virt);
      cond_.notify_all();
    } else {
      // Keep these pages mapped to avoid accounting drift on failed unmap.
      for (int64_t virt_page_id : pages_to_unmap) {
        dp_pages.reserved_virt_page_list.push_back(virt_page_id);
      }
      LOG(ERROR) << "Failed to unmap KV cache pages for model=" << model_id
                 << " dp_rank=" << dp_rank
                 << ", pages moved back to reserved list";
    }
    update_memory_usage();
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

  bool unmap_success = unmap_virt_pages(model_id, dp_rank, pages_to_unmap);

  {
    std::lock_guard<std::mutex> lock(mtx_);
    ModelState& state = get_model_state(model_id);
    auto& dp_pages = state.dp_group_pages[dp_rank];
    if (unmap_success) {
      for (int64_t virt_page_id : pages_to_unmap) {
        dp_pages.free_virt_page_list.push_back(virt_page_id);
      }
      release_phy_pages_for_dp(
          model_id, dp_rank, pages_to_unmap.size() * phy_pages_per_virt);
      cond_.notify_all();
    } else {
      for (int64_t virt_page_id : pages_to_unmap) {
        dp_pages.reserved_virt_page_list.push_back(virt_page_id);
      }
      LOG(ERROR) << "Failed to trim KV cache for model=" << model_id
                 << " dp_rank=" << dp_rank << ", reserved pages restored";
    }
    update_memory_usage();
  }
}

bool PageAllocator::alloc_weight_pages(const std::string& model_id,
                                       size_t num_pages) {
  int32_t model_world_size = 0;
  {
    std::lock_guard<std::mutex> lock(mtx_);

    CHECK(initialized_) << "PageAllocator not initialized";

    ModelState& state = get_model_state(model_id);

    // Get model's world_size (how many workers it uses)
    model_world_size =
        state.model_world_size > 0 ? state.model_world_size : max_world_size_;

    // Check if enough pages available for all workers this model uses
    // Find the minimum free pages among target workers
    sync_reported_phy_pages_from_shm_locked();
    size_t min_free_pages = num_total_phy_pages_;
    for (int32_t i = 0; i < model_world_size && i < max_world_size_; ++i) {
      size_t worker_free =
          num_total_phy_pages_ - get_worker_used_pages_locked(i);
      min_free_pages = std::min(min_free_pages, worker_free);
    }

    if (min_free_pages < num_pages) {
      LOG(ERROR) << "Not enough physical pages for weight allocation: "
                 << "requested " << num_pages << ", min_available "
                 << min_free_pages << " (model_world_size=" << model_world_size
                 << ")";
      return false;
    }

    // Update per-worker page usage
    for (int32_t i = state.model_worker_rank_base;
         i < state.model_worker_rank_base + model_world_size;
         ++i) {
      worker_pages_used_[i] += num_pages;
    }

    state.weight_pages_allocated = num_pages;
    state.layer_offloaded_phy_pages = 0;

    // All layers start on device.
    state.num_layers_on_device = static_cast<int32_t>(state.num_layers);

    update_memory_usage();
  }

  // Broadcast to all workers to allocate weight pages in GlobalXTensor
  auto& allocator = XTensorAllocator::get_instance();
  if (!allocator.broadcast_alloc_weight_pages(model_id, num_pages)) {
    LOG(ERROR) << "Failed to broadcast alloc_weight_pages for model "
               << model_id << ", keep consumed weight page count (no rollback)";
    return false;
  }

  LOG(INFO) << "Allocated " << num_pages
            << " physical pages for weight (weight_xtensor) of model "
            << model_id << " (model_world_size=" << model_world_size << ")";
  return true;
}

bool PageAllocator::free_weight_pages(const std::string& model_id,
                                      size_t num_pages) {
  // Broadcast to all workers to free weight pages in GlobalXTensor
  auto& allocator = XTensorAllocator::get_instance();
  if (!allocator.broadcast_free_weight_pages(model_id)) {
    LOG(ERROR) << "Failed to broadcast free_weight_pages for model " << model_id
               << ", keep consumed weight page count (no rollback)";
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(mtx_);

    CHECK(initialized_) << "PageAllocator not initialized";

    // Note: weight_pages_allocated is NOT cleared here
    // It's preserved for wakeup to know how many pages to re-allocate

    // Update per-worker page usage
    ModelState& state = get_model_state(model_id);
    state.layer_offloaded_phy_pages = num_pages;
    int32_t model_world_size =
        state.model_world_size > 0 ? state.model_world_size : max_world_size_;
    for (int32_t i = 0; i < model_world_size && i < max_world_size_; ++i) {
      if (worker_pages_used_[i] >= num_pages) {
        worker_pages_used_[i] -= num_pages;
      } else {
        LOG(WARNING) << "Worker " << i
                     << " pages underflow: used=" << worker_pages_used_[i]
                     << ", trying to free=" << num_pages;
        worker_pages_used_[i] = 0;
      }
    }

    update_memory_usage();
    cond_.notify_all();
  }

  LOG(INFO) << "Freed " << num_pages
            << " physical pages from weight (global xtensor) of model "
            << model_id;
  return true;
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

void PageAllocator::set_weight_pages_count(const std::string& model_id,
                                           size_t num_pages) {
  std::lock_guard<std::mutex> lock(mtx_);
  ModelState& state = get_model_state(model_id);
  state.weight_pages_allocated = num_pages;
  // only called when fork master with sleep mode.
  state.layer_offloaded_phy_pages = num_pages;
  LOG(INFO) << "Set weight pages count for model " << model_id << ": "
            << num_pages;
}

size_t PageAllocator::get_num_reserved_virt_pages(const std::string& model_id,
                                                  int32_t dp_rank) const {
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";
  std::lock_guard<std::mutex> lock(mtx_);
  const ModelState& state = get_model_state(model_id);
  return state.dp_group_pages[dp_rank].reserved_virt_page_list.size();
}

int32_t PageAllocator::get_model_min_reserved_pages(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  const ModelState& state = get_model_state(model_id);
  return state.min_reserved_pages;
}

void PageAllocator::register_prefix_block_manager(
    const std::string& model_id,
    int32_t dp_rank,
    const BlockManager* block_manager) {
  if (block_manager == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(mtx_);
  prefix_block_managers_[model_id][dp_rank] = block_manager;
}

void PageAllocator::unregister_prefix_block_manager(
    const std::string& model_id,
    int32_t dp_rank,
    const BlockManager* block_manager) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto model_it = prefix_block_managers_.find(model_id);
  if (model_it == prefix_block_managers_.end()) {
    return;
  }
  auto& dp_map = model_it->second;
  auto dp_it = dp_map.find(dp_rank);
  if (dp_it != dp_map.end() && dp_it->second == block_manager) {
    dp_map.erase(dp_it);
  }
  if (dp_map.empty()) {
    prefix_block_managers_.erase(model_it);
  }
}

size_t PageAllocator::get_num_free_phy_pages() const {
  // std::lock_guard<std::mutex> lock(mtx_);
  //  Return minimum free pages across all workers
  return get_min_free_pages_in_range(0, max_world_size_);
}

size_t PageAllocator::get_num_total_phy_pages() const {
  return num_total_phy_pages_;
}

std::vector<size_t> PageAllocator::get_all_worker_free_pages() const {
  std::lock_guard<std::mutex> lock(mtx_);
  sync_reported_phy_pages_from_shm_locked();
  std::vector<size_t> result;
  result.reserve(max_world_size_);
  for (int32_t i = 0; i < max_world_size_; ++i) {
    size_t free_pages = num_total_phy_pages_ - get_worker_used_pages_locked(i);
    result.push_back(free_pages);
  }
  return result;
}

std::unordered_set<int32_t>
PageAllocator::get_pressure_workers_by_low_watermark_ratio(
    double low_watermark_ratio) const {
  std::unordered_set<int32_t> pressure_workers;
  if (max_world_size_ <= 0 || num_total_phy_pages_ == 0) {
    return pressure_workers;
  }
  const double clamped_ratio = std::clamp(low_watermark_ratio, 0.0, 1.0);
  const size_t low_watermark_pages =
      static_cast<size_t>(num_total_phy_pages_ * clamped_ratio);
  if (low_watermark_pages == 0) {
    return pressure_workers;
  }

  std::lock_guard<std::mutex> lock(mtx_);
  pressure_workers.reserve(static_cast<size_t>(max_world_size_));
  sync_reported_phy_pages_from_shm_locked();
  for (int32_t worker = 0; worker < max_world_size_; ++worker) {
    const size_t worker_free =
        num_total_phy_pages_ - get_worker_used_pages_locked(worker);
    if (worker_free < low_watermark_pages) {
      pressure_workers.insert(worker);
    }
  }
  return pressure_workers;
}

std::unordered_set<int32_t>
PageAllocator::get_healthy_workers_by_high_watermark_ratio(
    double high_watermark_ratio) const {
  std::unordered_set<int32_t> healthy_workers;
  if (max_world_size_ <= 0 || num_total_phy_pages_ == 0) {
    return healthy_workers;
  }
  const double clamped_ratio = std::clamp(high_watermark_ratio, 0.0, 1.0);
  const size_t high_watermark_pages =
      static_cast<size_t>(num_total_phy_pages_ * clamped_ratio);
  if (high_watermark_pages == 0) {
    return healthy_workers;
  }

  std::lock_guard<std::mutex> lock(mtx_);
  healthy_workers.reserve(static_cast<size_t>(max_world_size_));
  sync_reported_phy_pages_from_shm_locked();
  for (int32_t worker = 0; worker < max_world_size_; ++worker) {
    const size_t worker_free =
        num_total_phy_pages_ - get_worker_used_pages_locked(worker);
    if (worker_free >= high_watermark_pages) {
      healthy_workers.insert(worker);
    }
  }
  return healthy_workers;
}

std::unordered_set<std::string> PageAllocator::get_models_by_workers(
    const std::unordered_set<int32_t>& workers) const {
  if (workers.empty()) {
    return {};
  }
  std::lock_guard<std::mutex> lock(mtx_);
  return collect_models_on_pressure_workers(workers);
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
      if (FLAGS_enable_dynamic_reserved_pages) {
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
          if (avg_reserved < static_cast<size_t>(low_threshold)) {
            // Increase by 25% or at least 4 pages, but don't exceed base * 4
            int32_t min_increase =
                std::max(4, static_cast<int32_t>(current_min * 0.25));
            int32_t max_increase =
                std::max(4, static_cast<int32_t>(current_max * 0.25));

            new_min = std::min(current_min + min_increase, base_min * 4);
            new_max = std::min(current_max + max_increase, base_max * 4);

            // Ensure min <= max and maintain reasonable ratio (max should be at
            // least 2x min)
            if (new_min > new_max) {
              new_max = new_min;
            }
            if (new_max < new_min * 2) {
              new_max = new_min * 2;
            }

            adjusted = true;
            LOG(INFO) << "[PriorityAlloc] fModel " << model_id
                      << " increasing min/max: reserved=" << avg_reserved
                      << " < threshold=" << low_threshold
                      << ", min=" << current_min << "->" << new_min
                      << ", max=" << current_max << "->" << new_max;
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

            adjusted = true;
            LOG(INFO) << "[PriorityAlloc] Model " << model_id
                      << " decreasing min/max: reserved=" << avg_reserved
                      << " > threshold=" << high_threshold
                      << ", min=" << current_min << "->" << new_min
                      << ", max=" << current_max << "->" << new_max;
          }

          // Update if changed
          if (adjusted && (new_min != current_min || new_max != current_max)) {
            state.min_reserved_pages = new_min;
            state.max_reserved_pages = new_max;
            prealloc_needed_ = true;
          }
        }
      }

      // Watermark-based async eviction signal.
      // Replaces the old inline emergency eviction (< 5% threshold).
      // async_eviction_worker handles actual PrefixCache eviction
      // independently.
      if (FLAGS_enable_prefix_cache && FLAGS_enable_xtensor) {
        size_t free_phy_pages = get_num_free_phy_pages();
        size_t total_phy_pages = num_total_phy_pages_;

        if (total_phy_pages > 0 &&
            free_phy_pages <
                static_cast<size_t>(total_phy_pages *
                                    FLAGS_layer_offload_low_watermark_ratio)) {
          // CAS: only one signal in flight at a time.
          bool expected = false;
          if (eviction_in_progress_.compare_exchange_strong(expected, true)) {
            eviction_needed_.store(true);
            async_evict_cond_.notify_one();
            VLOG(1) << "[PageAllocator] Low watermark reached ("
                    << free_phy_pages << "/" << total_phy_pages
                    << "), signaled async_eviction_worker.";
          }
        }
      }

      // Check each model for preallocation needs
      for (auto& [model_id, state] : model_states_) {
        // Skip sleeping models
        if (state.is_sleeping || state.schedule_blocked) {
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

          // Limit by available physical pages for this DP group's workers
          auto [start_w, end_w] = get_dp_group_worker_range(model_id, dp_rank);
          size_t min_free_phy = get_min_free_pages_in_range(start_w, end_w);
          size_t max_by_phy = min_free_phy / state.phy_pages_per_virt_page;
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
          if (!try_to_consume_phy_pages_for_dp(
                  model_id,
                  dp_rank,
                  pages_to_reserve.size() * state.phy_pages_per_virt_page)) {
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
            release_phy_pages_for_dp(
                model_id,
                dp_rank,
                pages_to_reserve.size() * it->second.phy_pages_per_virt_page);
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
            release_phy_pages_for_dp(
                model_id,
                dp_rank,
                pages_to_reserve.size() * it->second.phy_pages_per_virt_page);
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
            release_phy_pages_for_dp(
                model_id,
                dp_rank,
                pages_to_reserve.size() * it->second.phy_pages_per_virt_page);
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

    // Also start the independent eviction thread alongside prealloc.
    start_async_eviction_thread();
  }
  trigger_preallocation();
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
  // Calculate min free pages across all workers
  size_t min_free_phy_pages = get_min_free_pages_in_range(0, max_world_size_);

  // Calculate physical memory usage (based on max used worker)
  sync_reported_phy_pages_from_shm_locked();
  size_t max_used = 0;
  for (int32_t i = 0; i < max_world_size_; ++i) {
    max_used = std::max(max_used, get_worker_used_pages_locked(i));
  }
  size_t used_phy_mem = max_used * page_size_;

  VLOG(2) << "Memory usage: "
          << "dp_size=" << dp_size_
          << ", min_free_phy_pages=" << min_free_phy_pages
          << ", used_phy_mem=" << used_phy_mem / (1024 * 1024) << "MB"
          << ", num_models=" << model_states_.size();
}

// ============================================================
//  Layer Offload State Accessors (MVP)
// ============================================================

void PageAllocator::update_layers_on_device(const std::string& model_id,
                                            int32_t new_num_layers_on_device) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_states_.find(model_id);
  if (it == model_states_.end()) {
    LOG(WARNING) << "[PageAllocator] update_layers_on_device: model "
                 << model_id << " not found";
    return;
  }
  it->second.num_layers_on_device = new_num_layers_on_device;
}

int32_t PageAllocator::get_num_layers_on_device(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_states_.find(model_id);
  if (it == model_states_.end()) return 0;
  return it->second.num_layers_on_device;
}

void PageAllocator::update_layer_offloaded_phy_pages_delta(
    const std::string& model_id,
    int64_t delta_phy_pages) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_states_.find(model_id);
  if (it == model_states_.end()) {
    LOG(WARNING) << "[PageAllocator] update_layer_offloaded_phy_pages_delta: "
                 << "model " << model_id << " not found";
    return;
  }
  ModelState& state = it->second;
  if (delta_phy_pages >= 0) {
    state.layer_offloaded_phy_pages += static_cast<size_t>(delta_phy_pages);
    return;
  }
  const size_t delta = static_cast<size_t>(-delta_phy_pages);
  if (state.layer_offloaded_phy_pages < delta) {
    LOG(WARNING) << "[PageAllocator] layer_offloaded_phy_pages underflow for "
                 << model_id << ", current=" << state.layer_offloaded_phy_pages
                 << " delta=" << delta;
    state.layer_offloaded_phy_pages = 0;
    return;
  }
  state.layer_offloaded_phy_pages -= delta;
}

size_t PageAllocator::get_layer_offloaded_phy_pages(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_states_.find(model_id);
  if (it == model_states_.end()) return 0;
  return it->second.layer_offloaded_phy_pages;
}

void PageAllocator::block_model_schedule(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_states_.find(model_id);
  if (it == model_states_.end()) {
    LOG(WARNING) << "[PageAllocator] block_model_schedule: model " << model_id
                 << " not found";
    return;
  }
  it->second.request_blocked = true;
  cond_.notify_all();
}

void PageAllocator::unblock_model_schedule(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_states_.find(model_id);
  if (it == model_states_.end()) {
    LOG(WARNING) << "[PageAllocator] unblock_model_schedule: model "
                 << model_id << " not found";
    return;
  }
  it->second.request_blocked = false;
  it->second.schedule_blocked = false;
  cond_.notify_all();
}

bool PageAllocator::is_model_schedule_blocked(const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_states_.find(model_id);
  if (it == model_states_.end()) {
    return false;
  }
  return it->second.request_blocked || it->second.schedule_blocked;
}

void PageAllocator::mark_model_request_begin(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_states_.find(model_id);
  if (it == model_states_.end()) {
    LOG(WARNING) << "[PageAllocator] request_begin: model " << model_id
                 << " not found";
    return;
  }
  ++it->second.inflight_requests;
}

void PageAllocator::mark_model_request_end(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_states_.find(model_id);
  if (it == model_states_.end()) {
    LOG(WARNING) << "[PageAllocator] request_end: model " << model_id
                 << " not found";
    return;
  }
  if (it->second.inflight_requests <= 0) {
    LOG(WARNING) << "[PageAllocator] request_end underflow: model " << model_id
                 << " inflight_requests=" << it->second.inflight_requests;
    it->second.inflight_requests = 0;
  } else {
    --it->second.inflight_requests;
  }
  cond_.notify_all();
}

void PageAllocator::wait_model_requests_drained(const std::string& model_id) {
  std::unique_lock<std::mutex> lock(mtx_);
  while (true) {
    auto it = model_states_.find(model_id);
    if (it == model_states_.end()) {
      LOG(WARNING) << "[PageAllocator] wait_model_requests_drained: model "
                   << model_id << " not found";
      return;
    }
    if (it->second.inflight_requests == 0) {
      return;
    }
    cond_.wait(lock);
  }
}

void PageAllocator::wait_and_mark_model_step_begin(const std::string& model_id) {
  std::unique_lock<std::mutex> lock(mtx_);
  while (true) {
    auto it = model_states_.find(model_id);
    if (it == model_states_.end()) {
      LOG(WARNING) << "[PageAllocator] step_begin: model " << model_id
                   << " not found";
      return;
    }
    if (!it->second.schedule_blocked && !it->second.step_inflight) {
      it->second.step_inflight = true;
      return;
    }
    cond_.wait(lock);
  }
}

void PageAllocator::mark_model_step_end(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_states_.find(model_id);
  if (it == model_states_.end()) {
    LOG(WARNING) << "[PageAllocator] step_end: model " << model_id
                 << " not found";
    return;
  }
  it->second.step_inflight = false;
  cond_.notify_all();
}

// ============================================================
//  Async Eviction Thread
// ============================================================

void PageAllocator::start_async_eviction_thread() {
  if (async_eviction_thd_ != nullptr) return;
  eviction_thd_running_.store(true);
  async_eviction_thd_ = std::make_unique<std::thread>(
      &PageAllocator::async_eviction_worker, this);
  LOG(INFO) << "[PageAllocator] Async eviction thread started.";
}

void PageAllocator::stop_async_eviction_thread() {
  if (async_eviction_thd_ == nullptr) return;
  {
    std::lock_guard<std::mutex> lock(evict_mtx_);
    eviction_thd_running_.store(false);
    eviction_needed_.store(true);  // unblock the wait
    async_evict_cond_.notify_all();
  }
  if (async_eviction_thd_->joinable()) {
    async_eviction_thd_->join();
  }
  async_eviction_thd_.reset();
  LOG(INFO) << "[PageAllocator] Async eviction thread stopped.";
}

void PageAllocator::async_eviction_worker() {
  while (eviction_thd_running_.load()) {
    {
      std::unique_lock<std::mutex> lock(evict_mtx_);
      async_evict_cond_.wait(lock, [this] {
        return eviction_needed_.load() || !eviction_thd_running_.load();
      });
      if (!eviction_thd_running_.load()) break;
      eviction_needed_.store(false);
    }

    size_t total_pages = get_num_total_phy_pages();
    if (total_pages == 0) {
      eviction_in_progress_.store(false);
      continue;
    }

    auto high_wm = static_cast<size_t>(
        total_pages * FLAGS_layer_offload_high_watermark_ratio);

    // Phase 1: Feedback-driven eviction loop (runs without mtx_).
    // Both evict_global_pure_lru and reclaim_excess manage their own locks.
    int rounds = 0;
    const size_t low_wm_pages = static_cast<size_t>(
        total_pages * FLAGS_layer_offload_low_watermark_ratio);
    while (get_num_free_phy_pages() < high_wm) {
      size_t evicted = 0;
      std::unordered_set<std::string> pressure_models;
      if (FLAGS_enable_prefix_cache && FLAGS_enable_xtensor) {
        {
          std::lock_guard<std::mutex> lock(mtx_);
          pressure_models = collect_models_on_pressure_workers(low_wm_pages);
        }
        evicted = GlobalPrefixCacheManager::instance().evict_global_pure_lru(
            500, pressure_models);
      }
      reclaim_excess_reserved_pages_for_models(pressure_models);

      size_t free_now = get_num_free_phy_pages();
      LOG(INFO) << "[async_eviction_worker] round=" << rounds
              << " evicted=" << evicted << " free=" << free_now << "/"
              << total_pages;

      if (evicted == 0) break;  // PrefixCache empty, cannot recover further
      ++rounds;
    }

    // Phase 1 done: clear flag and wake any thread blocked in
    // alloc_kv_cache_page.
    eviction_in_progress_.store(false);
    {
      std::lock_guard<std::mutex> lock(mtx_);
      cond_.notify_all();
    }

    // Phase 2: Extreme pressure log (MVP: just log; P1 adds Master report).
    size_t free_pages = get_num_free_phy_pages();
    double free_ratio = static_cast<double>(free_pages) / total_pages;
    if (free_ratio < FLAGS_layer_offload_low_watermark_ratio * 0.5) {
      LOG(ERROR) << "[PageAllocator] CRITICAL: free_ratio=" << free_ratio
                 << " still below 50% of low_watermark after eviction. "
                 << "Layer offload may be needed.";
    } else if (free_ratio < FLAGS_layer_offload_low_watermark_ratio) {
      LOG(WARNING) << "[PageAllocator] WARNING: free_ratio=" << free_ratio
                   << " still below low_watermark after eviction.";
    }
  }
}

void PageAllocator::reclaim_excess_reserved_pages_for_models(
    const std::unordered_set<std::string>& model_ids) {
  // Step 1: Collect pages to unmap under lock.
  // Tuple: (model_id, dp_rank, virt_page_ids, phy_per_virt)
  std::vector<std::tuple<std::string, int32_t, std::vector<int64_t>, size_t>>
      to_unmap;
  {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& [model_id, state] : model_states_) {
      if (state.is_sleeping || model_ids.count(model_id) == 0) continue;
      for (int32_t dp = 0; dp < dp_size_; ++dp) {
        auto& dp_pages = state.dp_group_pages[dp];
        int32_t current =
            static_cast<int32_t>(dp_pages.reserved_virt_page_list.size());
        int32_t excess = current - state.min_reserved_pages;
        if (excess <= 0) continue;

        std::vector<int64_t> ids;
        ids.reserve(static_cast<size_t>(excess));
        for (int i = 0; i < excess; ++i) {
          ids.push_back(dp_pages.reserved_virt_page_list.back());
          dp_pages.reserved_virt_page_list.pop_back();
        }
        to_unmap.emplace_back(
            model_id, dp, std::move(ids), state.phy_pages_per_virt_page);
      }
    }
  }

  // Step 2: Unmap outside lock (VMM/RPC can take ms).
  for (auto& [model_id, dp, ids, phy_per_virt] : to_unmap) {
    if (unmap_virt_pages(model_id, dp, ids)) {
      std::lock_guard<std::mutex> lock(mtx_);
      release_phy_pages_for_dp(model_id, dp, ids.size() * phy_per_virt);
      auto it = model_states_.find(model_id);
      if (it != model_states_.end()) {
        auto& dp_pages = it->second.dp_group_pages[dp];
        for (int64_t id : ids) {
          dp_pages.free_virt_page_list.push_back(id);
        }
      }
    } else {
      // Restore on failure to avoid accounting drift.
      std::lock_guard<std::mutex> lock(mtx_);
      auto it = model_states_.find(model_id);
      if (it != model_states_.end()) {
        auto& dp_pages = it->second.dp_group_pages[dp];
        for (int64_t id : ids) {
          dp_pages.reserved_virt_page_list.push_back(id);
        }
      }
      LOG(ERROR) << "[PageAllocator] reclaim_excess: unmap failed for "
                 << model_id << " dp=" << dp;
    }
  }
}

void PageAllocator::release_all_reserved_pages_for_models(
    const std::unordered_set<std::string>& model_ids) {
  // evict prefix cache
  const size_t total_cached_blocks =
      GlobalPrefixCacheManager::instance().get_total_cached_blocks();
  size_t evicted_prefix_blocks = 0;
  if (total_cached_blocks > 0 && !model_ids.empty()) {
    evicted_prefix_blocks = GlobalPrefixCacheManager::instance()
                                .evict_global_pure_lru(total_cached_blocks,
                                                       model_ids);
  }
  // Tuple: (model_id, dp_rank, virt_page_ids, phy_per_virt)
  std::vector<std::tuple<std::string, int32_t, std::vector<int64_t>, size_t>>
      to_unmap;
  {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& [model_id, state] : model_states_) {
      if (state.is_sleeping || model_ids.count(model_id) == 0) continue;
      for (int32_t dp = 0; dp < dp_size_; ++dp) {
        auto& dp_pages = state.dp_group_pages[dp];
        if (dp_pages.reserved_virt_page_list.empty()) continue;
        std::vector<int64_t> ids;
        ids.reserve(dp_pages.reserved_virt_page_list.size());
        while (!dp_pages.reserved_virt_page_list.empty()) {
          ids.push_back(dp_pages.reserved_virt_page_list.back());
          dp_pages.reserved_virt_page_list.pop_back();
        }
        to_unmap.emplace_back(
            model_id, dp, std::move(ids), state.phy_pages_per_virt_page);
      }
    }
  }

  for (auto& [model_id, dp, ids, phy_per_virt] : to_unmap) {
    if (unmap_virt_pages(model_id, dp, ids)) {
      std::lock_guard<std::mutex> lock(mtx_);
      release_phy_pages_for_dp(model_id, dp, ids.size() * phy_per_virt);
      auto it = model_states_.find(model_id);
      if (it != model_states_.end()) {
        auto& dp_pages = it->second.dp_group_pages[dp];
        for (int64_t id : ids) {
          dp_pages.free_virt_page_list.push_back(id);
        }
      }
      update_memory_usage();
      cond_.notify_all();
    } else {
      std::lock_guard<std::mutex> lock(mtx_);
      auto it = model_states_.find(model_id);
      if (it != model_states_.end()) {
        auto& dp_pages = it->second.dp_group_pages[dp];
        for (int64_t id : ids) {
          dp_pages.reserved_virt_page_list.push_back(id);
        }
      }
      LOG(ERROR) << "[PageAllocator] release_all_reserved: unmap failed for "
                 << model_id << " dp=" << dp;
      update_memory_usage();
    }
  }
}

std::unordered_set<std::string>
PageAllocator::collect_models_on_pressure_workers(
    size_t low_watermark_pages) const {
  std::unordered_set<std::string> pressure_models;
  if (low_watermark_pages == 0) {
    return pressure_models;
  }
  std::unordered_set<int32_t> pressure_workers;
  pressure_workers.reserve(static_cast<size_t>(max_world_size_));
  sync_reported_phy_pages_from_shm_locked();
  for (int32_t worker = 0; worker < max_world_size_; ++worker) {
    const size_t worker_free =
        num_total_phy_pages_ - get_worker_used_pages_locked(worker);
    if (worker_free < low_watermark_pages) {
      pressure_workers.insert(worker);
    }
  }
  if (pressure_workers.empty()) {
    return pressure_models;
  }
  pressure_models = collect_models_on_pressure_workers(pressure_workers);
  return pressure_models;
}

std::unordered_set<std::string> 
PageAllocator::collect_models_on_pressure_workers(
    const std::unordered_set<int32_t> pressure_workers) const {
  std::unordered_set<std::string> pressure_models;
  for (const auto& [model_id, state] : model_states_) {
    if (state.is_sleeping) {
      continue;
    }
    if (state.model_world_size == 0) {
      pressure_models.insert(model_id);
      continue;
    }
    int32_t start_w = state.model_worker_rank_base;
    int32_t end_w = start_w + state.model_world_size;
    for (int32_t worker = start_w; worker < end_w; ++worker) {
      if (pressure_workers.count(worker) > 0) {
        pressure_models.insert(model_id);
        break;
      }
    }
  }
  return pressure_models;
}

bool PageAllocator::emergency_eviction(int32_t pages_needed,
    int32_t worker_rank) {
  LOG(INFO) << "EmergencyEviction called, pages_needed=" << pages_needed
      << ", worker_rank=" << worker_rank;
  std::unordered_set<int32_t> pressure_worker = {worker_rank};
  std::unordered_set<std::string> pressure_models =
      collect_models_on_pressure_workers(pressure_worker);
  GlobalPrefixCacheManager& prefix_cache_manager = GlobalPrefixCacheManager::instance();
  size_t total_cached = prefix_cache_manager.get_total_cached_blocks();
  prefix_cache_manager.evict_global_pure_lru(
      total_cached, pressure_models);
  release_all_reserved_pages_for_models(pressure_models);

  size_t total_pages = get_num_total_phy_pages();
  sync_reported_phy_pages_from_shm_locked();
  const size_t worker_used = get_worker_used_pages_locked(worker_rank);
  if (worker_used >= total_pages) {
    LOG(FATAL) << "EmergencyEviction: worker used >= total (bookkeeping "
                    "inconsistency?), worker_rank="
                 << worker_rank << " used=" << worker_used
                 << " total=" << total_pages;
  }
  size_t worker_free = total_pages - get_worker_used_pages_locked(worker_rank);
  if (worker_free >= static_cast<size_t>(pages_needed)) {
    LOG(INFO) << "EmergencyEviction success, worker_free=" << worker_free
              << " >= pages_needed=" << pages_needed;
    return true;
  }

  if (!layer_offload_mgr_) {
    LOG(WARNING) << "EmergencyEviction: no layer_offload_mgr_ available";
    return false;
  }

  while (worker_free < static_cast<size_t>(pages_needed)) {
    int32_t offloaded = layer_offload_mgr_->offload_internal(
        LayerOffloadManager::OffloadContext::kEmergencyEviction,
        0.0,
        pressure_models.empty() ? nullptr : &pressure_models);
    if (offloaded <= 0) break;
    sync_reported_phy_pages_from_shm_locked();
    {
      const size_t u = get_worker_used_pages_locked(worker_rank);
      worker_free = (u >= total_pages) ? 0 : (total_pages - u);
    }
  }
  return worker_free >= static_cast<size_t>(pages_needed);
}

void PageAllocator::start_layer_offload_monitor() {
  if (!layer_offload_mgr_) {
    layer_offload_mgr_ = std::make_unique<LayerOffloadManager>();
  }
  layer_offload_mgr_->start();
}

void PageAllocator::stop_layer_offload_monitor() {
  if (layer_offload_mgr_) {
    layer_offload_mgr_->stop();
  }
}

void PageAllocator::set_model_parallel_strategy(const std::string& model_id,
                                                int32_t dp_size,
                                                int32_t tp_size,
                                                int32_t worker_rank_base) {
  std::lock_guard<std::mutex> lock(mtx_);

  auto it = model_states_.find(model_id);
  if (it == model_states_.end()) {
    LOG(WARNING) << "Model " << model_id
                 << " not found for set_model_parallel_strategy";
    return;
  }

  ModelState& state = it->second;
  state.model_dp_size = dp_size;
  state.model_tp_size = tp_size;
  state.model_worker_rank_base = std::max(0, worker_rank_base);
  state.model_world_size = dp_size * tp_size;
  CHECK_LE(state.model_worker_rank_base + state.model_world_size,
           max_world_size_)
      << "Model worker window out of range for model " << model_id;

  LOG(INFO) << "Set model parallel strategy for " << model_id
            << ": dp_size=" << dp_size << ", tp_size=" << tp_size
            << ", worker_rank_base=" << state.model_worker_rank_base
            << ", world_size=" << state.model_world_size;
}

int32_t PageAllocator::get_model_world_size(const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);

  auto it = model_states_.find(model_id);
  if (it == model_states_.end()) {
    return 0;
  }
  return it->second.model_world_size;
}

size_t PageAllocator::get_free_phy_pages_for_model(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);

  auto it = model_states_.find(model_id);
  int32_t model_world_size = max_world_size_;
  int32_t worker_rank_base = 0;
  if (it != model_states_.end() && it->second.model_world_size > 0) {
    model_world_size = it->second.model_world_size;
    worker_rank_base = std::max(0, it->second.model_worker_rank_base);
  }

  // Find the minimum free pages among all workers this model uses
  return get_min_free_pages_in_range(worker_rank_base,
                                     worker_rank_base + model_world_size);
}

std::unordered_map<std::string, PageAllocator::ModelMemoryUsage>
PageAllocator::get_model_memory_usage() const {
  std::lock_guard<std::mutex> lock(mtx_);

  std::unordered_map<std::string, ModelMemoryUsage> usage;
  usage.reserve(model_states_.size());
  for (const auto& [model_id, state] : model_states_) {
    ModelMemoryUsage model_usage;
    if (!state.is_sleeping) {
      const size_t offloaded = state.layer_offloaded_phy_pages;
      const size_t total_weight = state.weight_pages_allocated;
      if (offloaded > total_weight) {
        LOG(ERROR) << "[PageAllocator] get_model_memory_usage: offloaded_weight_pages="
                   << offloaded << " > total_weight_pages=" << total_weight
                   << " for model " << model_id;
        model_usage.weight_phy_pages = 0;
      } else {
        model_usage.weight_phy_pages = total_weight - offloaded;
      }

      // Actual mapped KV pages include both in-use pages and preallocated
      // reserved pages.
      size_t kv_virt_pages_mapped = 0;
      for (const auto& dp_pages : state.dp_group_pages) {
        kv_virt_pages_mapped += dp_pages.allocated_virt_page_list.size();
        kv_virt_pages_mapped += dp_pages.reserved_virt_page_list.size();
      }
      model_usage.kv_cache_phy_pages =
          kv_virt_pages_mapped * state.phy_pages_per_virt_page;
      usage.emplace(model_id, model_usage);
    }
  }
  return usage;
}

std::vector<size_t> PageAllocator::build_prefix_only_phy_pages_by_worker_locked()
    const {
  std::vector<size_t> prefix_only_phy_pages_by_worker(
      static_cast<size_t>(max_world_size_), 0);

  for (const auto& [model_id, state] : model_states_) {
    auto providers_it = prefix_block_managers_.find(model_id);
    if (providers_it == prefix_block_managers_.end()) {
      continue;
    }
    const auto& providers = providers_it->second;
    for (const auto& [dp_rank, block_manager] : providers) {
      if (block_manager == nullptr ||
          dp_rank < 0 ||
          dp_rank >= static_cast<int32_t>(state.dp_group_pages.size())) {
        continue;
      }

      const size_t reclaimable_virt_pages =
          block_manager->num_prefix_only_reclaimable_virt_pages();
      if (reclaimable_virt_pages == 0) {
        continue;
      }

      const size_t current_reserved =
          state.dp_group_pages[dp_rank].reserved_virt_page_list.size();
      const size_t min_reserved =
          static_cast<size_t>(std::max(0, state.min_reserved_pages));
      const size_t reserve_capacity =
          current_reserved >= min_reserved ? 0 : (min_reserved - current_reserved);
      const size_t reclaimable_unmapped_virt_pages =
          reclaimable_virt_pages > reserve_capacity
              ? (reclaimable_virt_pages - reserve_capacity)
              : 0;
      if (reclaimable_unmapped_virt_pages == 0) {
        continue;
      }

      const size_t reclaimable_phy_pages =
          reclaimable_unmapped_virt_pages * state.phy_pages_per_virt_page;
      auto [start_w, end_w] = get_dp_group_worker_range(model_id, dp_rank);
      for (int32_t w = start_w; w < end_w; ++w) {
        prefix_only_phy_pages_by_worker[static_cast<size_t>(w)] +=
            reclaimable_phy_pages;
      }
    }
  }

  return prefix_only_phy_pages_by_worker;
}

std::vector<size_t>
PageAllocator::build_degraded_weight_resident_phy_pages_by_worker_locked(
    const std::string& normalized_base_model_id) const {
  std::vector<size_t> degraded_resident_pages_by_worker(
      static_cast<size_t>(max_world_size_), 0);

  for (const auto& [model_id, state] : model_states_) {
    if (normalize_base_model_id(model_id) == normalized_base_model_id) continue;
    if (state.is_sleeping) continue;

    const size_t offloaded = state.layer_offloaded_phy_pages;
    if (state.num_layers_on_device == state.num_layers) continue;

    const size_t total = state.weight_pages_allocated;
    if (offloaded > total) {
      LOG(ERROR)
          << "[PageAllocator] build_degraded_weight_resident_phy_pages_by_worker_locked: "
          << "offloaded_weight_pages=" << offloaded
          << " > total_weight_pages=" << total << " for model " << model_id;
      continue;
    }

    const size_t resident = total - offloaded;
    if (resident == 0) continue;

    int32_t world_size = max_world_size_;
    int32_t worker_rank_base = 0;
    if (state.model_world_size > 0) {
      world_size = state.model_world_size;
      worker_rank_base = std::max(0, state.model_worker_rank_base);
    }
    if (world_size <= 0 || worker_rank_base >= max_world_size_) {
      LOG(ERROR)
          << "[PageAllocator] build_degraded_weight_resident_phy_pages_by_worker_locked: "
          << "model " << model_id << " world_size=" << world_size
          << " worker_rank_base=" << worker_rank_base << " out of range, skip";
      continue;
    }

    const int32_t end_worker =
        std::min(worker_rank_base + world_size, max_world_size_);
    for (int32_t w = worker_rank_base; w < end_worker; ++w) {
      degraded_resident_pages_by_worker[static_cast<size_t>(w)] += resident;
    }
  }

  return degraded_resident_pages_by_worker;
}

std::optional<std::string> PageAllocator::pick_best_loadable_model_for_base(
    const std::string& base_model_id) const {
  std::unique_lock<std::mutex> lock(mtx_);
  const std::string normalized_base = normalize_base_model_id(base_model_id);

  sync_reported_phy_pages_from_shm_locked();

  const std::vector<size_t> degraded_resident_pages_by_worker =
      build_degraded_weight_resident_phy_pages_by_worker_locked(normalized_base);
  // avoid deadlock
  //const std::vector<size_t> prefix_only_phy_pages_by_worker =
  //    build_prefix_only_phy_pages_by_worker_locked();
  struct PrefixProbeTask {
    const BlockManager* block_manager = nullptr;
    size_t phy_pages_per_virt_page = 0;
    size_t reserve_capacity_virt_pages = 0;
    int32_t start_worker = 0;
    int32_t end_worker = 0;
  };

  std::vector<PrefixProbeTask> prefix_probe_tasks;
  for (const auto& [model_id, state] : model_states_) {
    auto providers_it = prefix_block_managers_.find(model_id);
    if (providers_it == prefix_block_managers_.end()) {
      continue;
    }
    const auto& providers = providers_it->second;
    for (const auto& [dp_rank, block_manager] : providers) {
      if (block_manager == nullptr ||
          dp_rank < 0 ||
          dp_rank >= static_cast<int32_t>(state.dp_group_pages.size())) {
        continue;
      }
      const size_t current_reserved =
          state.dp_group_pages[dp_rank].reserved_virt_page_list.size();
      const size_t min_reserved =
          static_cast<size_t>(std::max(0, state.min_reserved_pages));
      const size_t reserve_capacity_virt_pages =
          current_reserved >= min_reserved ? 0 : (min_reserved - current_reserved);
      auto [start_w, end_w] = get_dp_group_worker_range(model_id, dp_rank);
      if (start_w >= end_w) {
        continue;
      }
      prefix_probe_tasks.push_back(PrefixProbeTask{
          block_manager,
          state.phy_pages_per_virt_page,
          reserve_capacity_virt_pages,
          start_w,
          end_w,
      });
    }
  }

  std::vector<size_t> prefix_only_phy_pages_by_worker(
      static_cast<size_t>(max_world_size_), 0);

  // Release PageAllocator lock before calling into BlockManager to avoid
  // lock-order inversion with XTensorBlockManagerImpl -> PageAllocator path.
  lock.unlock();
  for (const auto& task : prefix_probe_tasks) {
    const size_t reclaimable_virt_pages =
        task.block_manager->num_prefix_only_reclaimable_virt_pages();
    if (reclaimable_virt_pages <= task.reserve_capacity_virt_pages) {
      continue;
    }
    const size_t reclaimable_unmapped_virt_pages =
        reclaimable_virt_pages - task.reserve_capacity_virt_pages;
    const size_t reclaimable_phy_pages =
        reclaimable_unmapped_virt_pages * task.phy_pages_per_virt_page;
    for (int32_t w = task.start_worker; w < task.end_worker; ++w) {
      prefix_only_phy_pages_by_worker[static_cast<size_t>(w)] +=
          reclaimable_phy_pages;
    }
  }
  lock.lock();

  std::optional<std::string> best_model_id;
  double best_avg_pressure = std::numeric_limits<double>::infinity();
  int candidates = 0;
  int loadable = 0;

  for (const auto& [model_id, state] : model_states_) {
    if (state.is_sleeping) continue;
    if (normalize_base_model_id(model_id) != normalized_base) continue;

    const size_t min_reserved_phy_pages =
        static_cast<size_t>(std::max(0, state.min_reserved_pages)) *
        state.phy_pages_per_virt_page;
    const size_t offloaded =
        state.layer_offloaded_phy_pages + min_reserved_phy_pages;
    if (state.num_layers < state.num_layers_on_device) {
      LOG(ERROR) << "[PageAllocator] pick_best_loadable_model_for_base: "
                 << "num_layers_on_device=" << state.num_layers_on_device
                 << " > total_layers=" << state.num_layers
                 << " for model " << model_id;
      continue;
    }
    const size_t layers_offloaded = state.num_layers - state.num_layers_on_device;
    if (layers_offloaded == 0) continue;  // not degraded

    const size_t total = state.weight_pages_allocated;

    int32_t world_size = max_world_size_;
    int32_t worker_rank_base = 0;
    if (state.model_world_size > 0) {
      world_size = state.model_world_size;
      worker_rank_base = std::max(0, state.model_worker_rank_base);
    }
    if (world_size <= 0 || worker_rank_base >= max_world_size_) {
      LOG(ERROR) << "[PageAllocator] pick_best_loadable_model_for_base: "
                 << "model " << model_id << " world_size=" << world_size
                 << " worker_rank_base=" << worker_rank_base
                 << " out of range, skip";
      continue;
    }
    const int32_t end_worker =
        std::min(worker_rank_base + world_size, max_world_size_);
    const int32_t worker_count = end_worker - worker_rank_base;

    ++candidates;
    bool is_loadable = true;
    double pressure_sum = 0.0;
    for (int32_t w = worker_rank_base; w < end_worker; ++w) {
      const size_t worker_used = get_worker_used_pages_locked(w);
      if (worker_used > num_total_phy_pages_) {
        LOG(ERROR) << "[PageAllocator] pick_best_loadable_model_for_base: "
                   << "model " << model_id << " worker " << w << " used pages=" << worker_used
                   << " > total_phy_pages=" << num_total_phy_pages_;
        is_loadable = false;
        break;
      }
      const size_t worker_free = num_total_phy_pages_ - worker_used;
      const size_t degraded_resident_on_worker =
          degraded_resident_pages_by_worker[static_cast<size_t>(w)];
      const size_t prefix_only_reclaimable_on_worker =
          prefix_only_phy_pages_by_worker[static_cast<size_t>(w)];
      const size_t denominator = worker_free + degraded_resident_on_worker +
                                 prefix_only_reclaimable_on_worker;
      LOG(INFO) << "[PageAllocator] pick_best_loadable_model_for_base: "
                << "model " << model_id << " worker " << w << " layers_offloaded=" << layers_offloaded
                << " offloaded_pages=" << offloaded << " denominator=" << denominator
                << " worker_free=" << worker_free
                << " degraded_resident_on_worker=" << degraded_resident_on_worker
                << " prefix_only_reclaimable_on_worker=" << prefix_only_reclaimable_on_worker;
      if (offloaded >= denominator) {
        // pressure >= 1 means this replica is not loadable.
        is_loadable = false;
        break;
      }
      pressure_sum += static_cast<double>(offloaded) /
                      static_cast<double>(denominator);
    }
    if (!is_loadable) continue;

    ++loadable;
    const double avg_pressure = pressure_sum / static_cast<double>(worker_count);
    if (!best_model_id.has_value() || avg_pressure < best_avg_pressure) {
      best_model_id = model_id;
      best_avg_pressure = avg_pressure;
    }
  }

  if (!best_model_id.has_value()) {
    LOG(INFO) << "[PageAllocator] pick_best_loadable_model_for_base: "
              << "base_model_id=" << normalized_base
              << " has no loadable degraded replica, candidates=" << candidates
              << " loadable=" << loadable;
    return std::nullopt;
  }

  LOG(INFO) << "[PageAllocator] pick_best_loadable_model_for_base: "
            << "base_model_id=" << normalized_base
            << " selected_model_id=" << *best_model_id
            << " avg_memory_pressure=" << best_avg_pressure
            << " candidates=" << candidates
            << " loadable=" << loadable;
  return best_model_id;
}

bool PageAllocator::offload_model(std::string model_id) {
  if (!layer_offload_mgr_) {
    LOG(WARNING) << "offload_model: no layer_offload_mgr_ available";
    return false;
  }

  constexpr auto kWaitStep = std::chrono::milliseconds(100);

  while (true) {
    int32_t min_reserved_pages = 0;
    bool ready_to_offload = false;

    {
      auto it = model_states_.find(model_id);
      if (it == model_states_.end()) {
        LOG(WARNING) << "offload_model: model " << model_id
                     << " not found in page allocator";
        return false;
      }

      ModelState& state = it->second;
      const size_t current_reserved =
          state.dp_group_pages[0].reserved_virt_page_list.size();
      if (current_reserved != 0) {
        state.schedule_blocked = true;
        break;
      }

      std::this_thread::sleep_for(kWaitStep);
    }
  }

  LOG(INFO) << "Triggering offload for model instance: "<<model_id;
  return layer_offload_mgr_->offload_model(model_id);
}

bool PageAllocator::load_model(std::string model_id) {
  if (!layer_offload_mgr_) {
    LOG(WARNING) << "load_model: no layer_offload_mgr_ available";
    return false;
  }

  return layer_offload_mgr_->load_model(model_id);
}
}  // namespace xllm
