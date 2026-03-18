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

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <future>
#include <optional>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>

#include "common/global_flags.h"
#include "core/distributed_runtime/master.h"
#include "framework/prefix_cache/global_prefix_cache_manager.h"
#include "xtensor_allocator.h"

namespace xllm {

namespace {
constexpr const char* kPhyPageUsedCounterShmName = "/xllm_phy_pages_used_counter_v1";
constexpr int32_t kPhyPageUsedCounterMaxWorkers = 1024;
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
  int fd =
      shm_open(kPhyPageUsedCounterShmName, O_CREAT | O_RDWR, static_cast<mode_t>(0666));
  if (fd < 0) {
    LOG(WARNING) << "Failed to open phy-page report shm: " << strerror(errno);
    return;
  }
  if (ftruncate(fd, static_cast<off_t>(shm_bytes)) != 0) {
    LOG(WARNING) << "Failed to resize phy-page report shm: " << strerror(errno);
    close(fd);
    return;
  }
  void* addr = mmap(nullptr, shm_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) {
    LOG(WARNING) << "Failed to map phy-page report shm: " << strerror(errno);
    close(fd);
    return;
  }
  reported_phy_pages_shm_fd_ = fd;
  reported_phy_pages_shm_ptr_ = static_cast<uint64_t*>(addr);
}

void PageAllocator::sync_reported_phy_pages_from_shm_locked() const {
  if (reported_phy_pages_shm_ptr_ == nullptr) {
    return;
  }
  const int32_t bound = std::min(max_world_size_, kPhyPageUsedCounterMaxWorkers);
  for (int32_t i = 0; i < bound; ++i) {
    worker_reported_pages_used_[i] =
        static_cast<size_t>(__atomic_load_n(&reported_phy_pages_shm_ptr_[i],
                                            __ATOMIC_RELAXED));
  }
}

size_t PageAllocator::get_worker_used_pages_locked(int32_t worker_rank) const {
  //LOG(INFO) << "Worker " << worker_rank << " used pages: " << worker_pages_used_[worker_rank] << " + " << worker_reported_pages_used_[worker_rank];
  return worker_pages_used_[worker_rank] + worker_reported_pages_used_[worker_rank];
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
        w < state.model_worker_rank_base + model_world_size; ++w) {
      pages_to_consume_per_worker[w] += weight_pages;
    }

    // Phase 1 (lock): check KV + weight requirements together.
    sync_reported_phy_pages_from_shm_locked();
    for (int32_t w = 0; w < max_world_size_; ++w) {
      if (pages_to_consume_per_worker[w] == 0) {
        continue;
      }
      size_t worker_free = num_total_phy_pages_ - get_worker_used_pages_locked(w);
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

bool PageAllocator::consume_phy_pages_for_dp(const std::string& model_id,
                                             int32_t dp_rank,
                                             size_t num_phy_pages) {
  // Note: Caller must hold mtx_
  auto [start_w, end_w] = get_dp_group_worker_range(model_id, dp_rank);
  size_t min_free = get_min_free_pages_in_range(start_w, end_w);
  if (min_free < num_phy_pages) {
    LOG(WARNING) << "Not enough physical pages for dp_rank=" << dp_rank
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
                 << num_total_phy_pages_ - worker_reported_pages_used_[worker_rank];
    return false;
  }
  worker_reported_pages_used_[worker_rank] += num_phy_pages;
  if (reported_phy_pages_shm_ptr_ != nullptr &&
      worker_rank < kPhyPageUsedCounterMaxWorkers) {
    __atomic_store_n(&reported_phy_pages_shm_ptr_[worker_rank],
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
    LOG(WARNING) << "Worker " << worker_rank << " pages underflow during release";
    worker_reported_pages_used_[worker_rank] = 0;
  }
  if (reported_phy_pages_shm_ptr_ != nullptr &&
      worker_rank < kPhyPageUsedCounterMaxWorkers) {
    __atomic_store_n(&reported_phy_pages_shm_ptr_[worker_rank],
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
      if (!consume_phy_pages_for_dp(model_id, dp_rank, phy_pages_needed)) {
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
      size_t worker_free = num_total_phy_pages_ - get_worker_used_pages_locked(i);
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
        i < state.model_worker_rank_base + model_world_size; ++i) {
      worker_pages_used_[i] += num_pages;
    }

    state.weight_pages_allocated = num_pages;
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
            << " physical pages for weight (global xtensor) of model "
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

      auto collect_models_on_pressure_workers =
          [this](size_t low_watermark_pages) -> std::unordered_set<std::string> {
        std::unordered_set<std::string> pressure_models;
        if (low_watermark_pages == 0) {
          return pressure_models;
        }

        std::unordered_set<int32_t> pressure_workers;
        pressure_workers.reserve(max_world_size_);
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

        LOG(INFO) << "Pressure workers: " << pressure_workers.size();
        for (const auto& worker : pressure_workers) {
          LOG(INFO) << "Worker " << worker << " used pages: " << get_worker_used_pages_locked(worker);
        }

        for (const auto& [model_id, state] : model_states_) {
          if (state.is_sleeping) {
            continue;
          }
          // Add models to pressure_models based on worker pressure
          if (state.model_world_size == 0) {
            pressure_models.insert(model_id);
            continue;
          }
          int32_t start_w = state.model_worker_rank_base;
          int32_t end_w = start_w + state.model_world_size;
          for (int32_t worker = start_w; worker < end_w; ++worker) {
            if (pressure_workers.find(worker) != pressure_workers.end()) {
              pressure_models.insert(model_id);
              break;
            }
          }
        }
        LOG(INFO) << "Pressure models: " << pressure_models.size();
        for (const auto& model : pressure_models) {
          LOG(INFO) << "Model " << model;
        }
        return pressure_models;
      };

      // Emergency eviction: Check global physical memory pressure
      // If free physical pages are below 5% threshold, trigger emergency
      // PrefixCache eviction to free up memory for active requests
      if (FLAGS_enable_prefix_cache && FLAGS_enable_xtensor) {
        size_t free_phy_pages = get_num_free_phy_pages();
        size_t total_phy_pages = num_total_phy_pages_;
        size_t low_watermark_pages = total_phy_pages * FLAGS_low_watermark_pages / 100;

        // Trigger emergency eviction if free pages < 5% of total

        //LOG(INFO) << "Free pages: " << free_phy_pages << "/" << total_phy_pages;
        if (total_phy_pages > 0 && free_phy_pages < low_watermark_pages) {
          LOG(WARNING) << "Global physical memory tight! Free pages: "
                       << free_phy_pages << "/" << total_phy_pages << " ("
                       << (free_phy_pages * 100 / total_phy_pages) << "%)"
                       << ". Triggering emergency PrefixCache eviction.";
          std::unordered_set<std::string> pressure_models =
              collect_models_on_pressure_workers(low_watermark_pages);

          // Release lock before calling evict_global_pure_lru to avoid deadlock
          // (evict_global_pure_lru will acquire its own mutex)
          lock.unlock();

          // Evict 10% of cached blocks (or at least 100 blocks)
          size_t total_cached =
              GlobalPrefixCacheManager::instance().get_total_cached_blocks();
          size_t target_evict = std::max(total_cached / 10, size_t(100));

          size_t actual_evicted =
              GlobalPrefixCacheManager::instance().evict_global_pure_lru(
                  target_evict, pressure_models);

          VLOG(1) << "Emergency eviction finished. Evicted: " << actual_evicted
                  << " blocks out of " << total_cached << " total cached blocks"
                  << ", pressure_models=" << pressure_models.size();

          // Re-acquire lock for the rest of the function
          lock.lock();
          if (get_num_free_phy_pages() < low_watermark_pages) {
            LOG(ERROR) << "[PageAllocator] Critical: free pages still very low "
                       << "after emergency eviction: "
                       << get_num_free_phy_pages() << "/" << total_phy_pages
                       << " evited_blocks=" << actual_evicted
                       << " total_cached_before=" << total_cached
                       << ". Risk of OOM or alloc failure." << get_num_free_phy_pages();
            // Second round: more aggressive eviction to try to free full pages
            lock.unlock();
            size_t total_cached2 =
                GlobalPrefixCacheManager::instance().get_total_cached_blocks();
            if (total_cached2 > 0) {
              size_t target2 = std::max(total_cached2 / 2, size_t(500));
              pressure_models =
                  collect_models_on_pressure_workers(low_watermark_pages);
              size_t evicted2 =
                  GlobalPrefixCacheManager::instance().evict_global_pure_lru(
                      target2, pressure_models);
              LOG(ERROR)
                  << "[PageAllocator] Critical: second eviction evicted_blocks="
                  << evicted2 << " total_cached=" << total_cached2
                  << " pressure_models=" << pressure_models.size() << " " << get_num_free_phy_pages();
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
          if (!consume_phy_pages_for_dp(
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
  CHECK_LE(state.model_worker_rank_base + state.model_world_size, max_world_size_)
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

}  // namespace xllm
