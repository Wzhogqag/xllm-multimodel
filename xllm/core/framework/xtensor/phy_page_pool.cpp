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

#include "phy_page_pool.h"

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <unordered_set>

namespace xllm {

namespace {
constexpr int32_t kPhyPageUsedCounterMaxWorkers = 16;
}  // namespace

void PhyPagePool::init(const torch::Device& device, size_t num_pages) {
  std::lock_guard<std::mutex> lock(mtx_);

  if (initialized_) {
    LOG(WARNING) << "PhyPagePool already initialized, ignoring re-init";
    return;
  }

  device_ = device;
  num_total_pages_ = num_pages;

  LOG(INFO) << "PhyPagePool: pre-allocating " << num_pages
            << " physical pages on device " << device;

  // Pre-allocate zero page first (used by all XTensors for initialization)
  // Zero page has page_id = -1
  zero_page_ = std::make_unique<PhyPage>(device_, -1);

  // Pre-allocate all physical pages for data with unique page_ids
  all_pages_.reserve(num_pages);

  num_available_.store(num_pages, std::memory_order_relaxed);

  all_page_ptrs_.reserve(num_pages);
  for (size_t i = 0; i < num_pages; ++i) {
    page_id_t page_id = static_cast<page_id_t>(i);
    all_pages_.push_back(std::make_unique<PhyPage>(device_, page_id));
    all_page_ptrs_.push_back(all_pages_.back().get());
  }

  for (size_t i = 0; i < num_total_pages_; ++i) {
    local_free_page_ids_.push_back(static_cast<page_id_t>(i));
  }

  initialized_ = true;

  LOG(INFO) << "PhyPagePool: successfully pre-allocated " << num_pages
            << " physical pages (page_id 0-" << (num_pages - 1)
            << ") + 1 zero page";
}

PhyPagePool::~PhyPagePool() {
  if (shared_report_counter_ptr_ != nullptr) {
    if (report_my_worker_rank_ >= 0 &&
        report_my_worker_rank_ < kPhyPageUsedCounterMaxWorkers) {
      __atomic_store_n(&shared_report_counter_ptr_[report_my_worker_rank_],
                       static_cast<uint64_t>(0),
                       __ATOMIC_RELAXED);
    }
    munmap(shared_report_counter_ptr_,
           sizeof(uint64_t) * kPhyPageUsedCounterMaxWorkers);
    shared_report_counter_ptr_ = nullptr;
  }
  if (shared_report_counter_fd_ >= 0) {
    close(shared_report_counter_fd_);
    shared_report_counter_fd_ = -1;
  }
}

bool PhyPagePool::init_shared_report_counter_if_needed(int32_t worker_rank) {
  if (worker_rank < 0 || worker_rank >= kPhyPageUsedCounterMaxWorkers) {
    LOG(WARNING) << "Invalid worker_rank for shared report counter: "
                 << worker_rank;
    return false;
  }

  const size_t shm_bytes =
      sizeof(uint64_t) * static_cast<size_t>(kPhyPageUsedCounterMaxWorkers);
  
  std::string kPhyPageUsedCounterShmName = 
      "/xllm_activation_phy_pages_used_" + std::to_string(FLAGS_port - FLAGS_node_rank);
  int fd = shm_open(
      kPhyPageUsedCounterShmName.c_str(), O_CREAT | O_RDWR, static_cast<mode_t>(0666));
  if (fd < 0) {
    LOG(WARNING) << "Failed to open phy-page report shm: " << strerror(errno);
    return false;
  }
  if (ftruncate(fd, static_cast<off_t>(shm_bytes)) != 0) {
    LOG(WARNING) << "Failed to resize phy-page report shm: " << strerror(errno);
    close(fd);
    return false;
  }

  void* addr =
      mmap(nullptr, shm_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) {
    LOG(WARNING) << "Failed to map phy-page report shm: " << strerror(errno);
    close(fd);
    return false;
  }

  shared_report_counter_fd_ = fd;
  shared_report_counter_ptr_ = static_cast<uint64_t*>(addr);
  shared_report_counter_local_used_ = 0;
  __atomic_store_n(&shared_report_counter_ptr_[report_my_worker_rank_],
                   static_cast<uint64_t>(0),
                   __ATOMIC_RELAXED);
  return true;
}

void PhyPagePool::report_consume_via_shared_counter(size_t count) {
  if (report_my_worker_rank_ < 0) {
    return;
  }
  shared_report_counter_local_used_ += count;
  __atomic_store_n(&shared_report_counter_ptr_[report_my_worker_rank_],
                   static_cast<uint64_t>(shared_report_counter_local_used_),
                   __ATOMIC_RELAXED);
}

void PhyPagePool::report_release_via_shared_counter(size_t count) {
  if (report_my_worker_rank_ < 0) {
    return;
  }
  if (shared_report_counter_local_used_ >= count) {
    shared_report_counter_local_used_ -= count;
  } else {
    shared_report_counter_local_used_ = 0;
  }
  __atomic_store_n(&shared_report_counter_ptr_[report_my_worker_rank_],
                   static_cast<uint64_t>(shared_report_counter_local_used_),
                   __ATOMIC_RELAXED);
}

std::unique_ptr<PhyPage> PhyPagePool::get() {
  // 第一阶段：在持锁状态下，尽量从本地空闲列表获取页面。
  {
    std::lock_guard<std::mutex> lock(mtx_);

    CHECK(initialized_) << "PhyPagePool not initialized";

    if (num_available_.load(std::memory_order_relaxed) < 1) {
      LOG(WARNING) << "PhyPagePool: no free pages available";
      return nullptr;
    }
    if (FLAGS_global_xtensor_map_rate != 100) {
      if (!local_free_page_ids_.empty()) {
        page_id_t page_id = local_free_page_ids_.front();
        local_free_page_ids_.pop_front();
        num_available_.fetch_sub(1, std::memory_order_relaxed);
        return std::move(all_pages_[page_id]);
      }
    }
  }

  // 第二阶段：本地没有空闲页，在不持有 mtx_ 的情况下向 GlobalXTensor 申请页面，
  // 避免 get_pages() 再次尝试获取同一把锁导致死锁。
  auto& global_xtensor = GlobalXTensor::get_instance();
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<page_id_t> ids = global_xtensor.allocate_pages_from_left(1);
  auto end = std::chrono::high_resolution_clock::now();
  auto delta = (end - start).count();

  CHECK(!ids.empty())
      << "GlobalXTensor::allocate_pages_from_left(1) returned no pages";
  page_id_t page_id = ids[0];

  {
    std::lock_guard<std::mutex> lock(mtx_);
    CHECK(initialized_) << "PhyPagePool not initialized";
    CHECK(page_id >= 0 && page_id < static_cast<page_id_t>(num_total_pages_))
        << "Invalid page_id from GlobalXTensor: " << page_id;
    CHECK(all_pages_[page_id] != nullptr)
        << "PhyPagePool: all_pages_ entry for page_id " << page_id
        << " is null";

    get_miss_count++;
    get_miss_time += delta;
    transfer_page_count += 1;
    LOG(INFO) << "PhyPagePool: get miss time=" << get_miss_time
              << ", count=" << get_miss_count
              << ", transfer_page_count=" << transfer_page_count;

    size_t num_available = num_available_.fetch_sub(1, std::memory_order_relaxed);
    CHECK(num_available >= 1)
        << "PhyPagePool: num_available_ underflow on get, num_available_="
        << num_available;
    // LOG(INFO) << "kvcache:" << num_available << ", count: 1";

    return std::move(all_pages_[page_id]);
  }
}

std::vector<std::unique_ptr<PhyPage>> PhyPagePool::batch_get(size_t count) {
  if (count == 0) {
    return {};
  }

  std::vector<std::unique_ptr<PhyPage>> result;
  result.reserve(count);

  size_t from_local = 0;
  size_t need_from_global = 0;

  // 第一阶段：在持锁状态下尽量从本地空闲列表获取页面，但不在持锁状态下调用
  // GlobalXTensor，避免锁重入死锁。
  {
    std::lock_guard<std::mutex> lock(mtx_);

    CHECK(initialized_) << "PhyPagePool not initialized";
    size_t num_available = num_available_.fetch_sub(count, std::memory_order_relaxed);
    CHECK (num_available >= count) 
        << "PhyPagePool: not enough free pages, requested " 
        << count << ", available " << num_available;
    // LOG(INFO) << "batch_get: " << num_available << ", count:" << count;

    if (FLAGS_global_xtensor_map_rate != 100) {
      from_local = std::min(count, local_free_page_ids_.size());
      for (size_t i = 0; i < from_local; ++i) {
        page_id_t page_id = local_free_page_ids_.front();
        local_free_page_ids_.pop_front();
        result.push_back(std::move(all_pages_[page_id]));
      }
      need_from_global = count - from_local;
    } else {
      need_from_global = count;
    }

    if (need_from_global == 0) {
      // 全部从本地获取
      return result;
    }
  }

  // 第二阶段：在不持有 mtx_ 的情况下从 GlobalXTensor 申请剩余页面。
  auto& global_xtensor = GlobalXTensor::get_instance();
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<page_id_t> page_ids =
      global_xtensor.allocate_pages_from_left(need_from_global);
  auto end = std::chrono::high_resolution_clock::now();
  auto delta = (end - start).count();

  CHECK(page_ids.size() == need_from_global)
      << "GlobalXTensor::allocate_pages_from_left(" << need_from_global
      << ") returned " << page_ids.size() << " pages";

  // 第三阶段：重新加锁，将 GlobalXTensor 返回的页面转移出 all_pages_。
  {
    std::lock_guard<std::mutex> lock(mtx_);

    CHECK(initialized_) << "PhyPagePool not initialized";

    get_miss_count++;
    get_miss_time += delta;
    transfer_page_count += need_from_global;

    for (size_t i = 0; i < need_from_global; ++i) {
      page_id_t page_id = page_ids[i];
      CHECK(page_id >= 0 && page_id < static_cast<page_id_t>(num_total_pages_))
          << "Invalid page_id from GlobalXTensor: " << page_id;
      CHECK(all_pages_[page_id] != nullptr)
          << "PhyPagePool: all_pages_ entry for page_id " << page_id
          << " is null";
      result.push_back(std::move(all_pages_[page_id]));
    }
  }

  return result;
}

void PhyPagePool::put(std::unique_ptr<PhyPage> page) {
  if (page == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  // Verify the page belongs to this pool (same device)
  CHECK(page->device() == device_) << "Page device mismatch: expected "
                                   << device_ << ", got " << page->device();

  page_id_t page_id = page->page_id();
  CHECK(page_id >= 0 && page_id < static_cast<page_id_t>(num_total_pages_))
      << "Invalid page_id: " << page_id;

  // Return ownership to pool and to local free list (global_xtensor only
  // shrinks, never grows)
  all_pages_[page_id] = std::move(page);
  if (FLAGS_global_xtensor_map_rate != 100) {
    local_free_page_ids_.push_back(page_id);
  } else {
    GlobalXTensor::get_instance().free_to_right_async({all_pages_[page_id].get()});
  }
  size_t num_free = num_available_.fetch_add(1, std::memory_order_relaxed);
  // LOG(INFO) << "kvcache: free" << num_free << ", count: 1";
}

void PhyPagePool::batch_put(std::vector<std::unique_ptr<PhyPage>>& pages) {
  if (pages.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  size_t put_count = 0;
  for (auto& page : pages) {
    if (page == nullptr) {
      continue;
    }
    // Verify the page belongs to this pool (same device)
    CHECK(page->device() == device_) << "Page device mismatch: expected "
                                     << device_ << ", got " << page->device();

    page_id_t page_id = page->page_id();
    CHECK(page_id >= 0 && page_id < static_cast<page_id_t>(num_total_pages_))
        << "Invalid page_id: " << page_id;

    // Return ownership to pool and to local free list
    all_pages_[page_id] = std::move(page);
    if (FLAGS_global_xtensor_map_rate != 100) {
      local_free_page_ids_.push_back(page_id);
    } else {
      GlobalXTensor::get_instance().free_to_right_async({all_pages_[page_id].get()});
    }
    put_count++;
  }

  size_t num_free = num_available_.fetch_add(put_count, std::memory_order_relaxed);
  // LOG(INFO) << "weight/kvcache: free" << num_free << ", count:" << pages.size();
}

void PhyPagePool::set_report_to_master(int32_t my_worker_rank) {
  report_my_worker_rank_ = my_worker_rank;
  CHECK(init_shared_report_counter_if_needed(my_worker_rank))
      << "Failed to initialize shared report counter for worker "
      << my_worker_rank;
}

void* PhyPagePool::allocate_contiguous(size_t count, bool is_activation, bool is_init) {
  auto& global_xtensor = GlobalXTensor::get_instance();
  auto& page_allocator = PageAllocator::get_instance();
  void* result = nullptr;
  if (is_init) {
    result = global_xtensor.allocate_init_from_left(count);
  } else {
    result = global_xtensor.allocate_from_left(count);
  }
  int32_t worker_rank =
      report_my_worker_rank_ >= 0 ? report_my_worker_rank_ : device_.index();
  if (is_activation) {
    if (page_allocator.is_initialized()) {
      page_allocator.consume_phy_pages_for_worker(worker_rank, count);
    } else {
      report_consume_via_shared_counter(count);
    }
  }
  size_t num_available = num_available_.fetch_sub(count, std::memory_order_relaxed);
  CHECK(num_available >= count) << "PhyPagePool contiguous alloc exceeds available pages";
  // LOG(INFO) << "activation/weight:" << num_available << ", count:" << count;
  return result;
}

void PhyPagePool::free_contiguous(size_t addr, size_t count) {
  auto& global_xtensor = GlobalXTensor::get_instance();
  auto& page_allocator = PageAllocator::get_instance();
  for (size_t i = 0; i < count; i++) {
    global_xtensor.free_one_page_async(addr);
    addr += 2 * 1024 * 1024;
  }
  int32_t worker_rank =
      report_my_worker_rank_ >= 0 ? report_my_worker_rank_ : device_.index();
  if (page_allocator.is_initialized()) {
    page_allocator.release_phy_pages_for_worker(worker_rank, count);
  } else {
    report_release_via_shared_counter(count);
  }
  size_t num_free = num_available_.fetch_add(count, std::memory_order_release);
  // LOG(INFO) << "activation: free" << num_free << ", count:" << count;
}

size_t PhyPagePool::num_available() const {
  return num_available_.load();
}

PhyPage* PhyPagePool::get_zero_page() {
  std::lock_guard<std::mutex> lock(mtx_);
  CHECK(initialized_) << "PhyPagePool not initialized";
  CHECK(zero_page_) << "Zero page not created";
  return zero_page_.get();
}

// ============== Global XTensor Support ==============

std::vector<PhyPage*> PhyPagePool::get_pages(size_t count) {
  std::lock_guard<std::mutex> lock(mtx_);
  CHECK(initialized_) << "PhyPagePool not initialized";

  std::vector<PhyPage*> result;
  if (count == 0) {
    LOG(WARNING) << "PhyPagePool: get_pages requested 0 pages";
    return result;
  }

  // if global_xtensor_map_rate is below 100, 
  // this is the place to trigger emergency eviction
  if (local_free_page_ids_.size() < count) {
    LOG(FATAL) << "PhyPagePool: not enough free pages for get_pages, requested "
               << count << ", available " << local_free_page_ids_.size();
    return result;
  }

  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    page_id_t page_id = local_free_page_ids_.front();
    local_free_page_ids_.pop_front();
    result.push_back(all_page_ptrs_[page_id]);
  }

  return result;
}

}  // namespace xllm
