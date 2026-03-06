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

#include <glog/logging.h>

#include <algorithm>
#include <unordered_set>

namespace xllm {

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

  num_available_ = num_pages;

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

std::unique_ptr<PhyPage> PhyPagePool::get() {
  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  if (num_available_ < 1) {
    LOG(WARNING) << "PhyPagePool: no free pages available";
    return nullptr;
  }

  page_id_t page_id;
  if (!local_free_page_ids_.empty()) {
    page_id = local_free_page_ids_.front();
    local_free_page_ids_.pop_front();
  } else {
    get_miss_count++;
    auto start = std::chrono::high_resolution_clock::now();
    auto& global_xtensor = GlobalXTensor::get_instance();
    std::vector<page_id_t> ids = global_xtensor.allocate_pages_from_left(1);
    auto end = std::chrono::high_resolution_clock::now();
    get_miss_time += (end - start).count();
    transfer_page_count += 1;
    LOG(INFO) << "PhyPagePool: get miss time=" << get_miss_time << ", count=" << get_miss_count << ", transfer_page_count=" << transfer_page_count;
    page_id = ids[0];
  }

  num_available_--;

  return std::move(all_pages_[page_id]);
}

std::vector<std::unique_ptr<PhyPage>> PhyPagePool::batch_get(size_t count) {
  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  if (count == 0) {
    return {};
  }

  if (num_available_ < count) {
    LOG(WARNING) << "PhyPagePool: not enough free pages, requested " << count
                 << ", available " << num_available_;
    return {};
  }

  std::vector<std::unique_ptr<PhyPage>> result;
  result.reserve(count);

  size_t from_local = std::min(count, local_free_page_ids_.size());
  for (size_t i = 0; i < from_local; ++i) {
    page_id_t page_id = local_free_page_ids_.front();
    local_free_page_ids_.pop_front();
    result.push_back(std::move(all_pages_[page_id]));
  }

  size_t need = count - from_local;
  if (need > 0) {
    get_miss_count++;
    auto start = std::chrono::high_resolution_clock::now();
    auto& global_xtensor = GlobalXTensor::get_instance();
    std::vector<page_id_t> page_ids =
        global_xtensor.allocate_pages_from_left(need);
    auto end = std::chrono::high_resolution_clock::now();
    get_miss_time += (end - start).count();
    for (size_t i = 0; i < need; ++i) {
      result.push_back(std::move(all_pages_[page_ids[i]]));
    }
    transfer_page_count += need;
    //LOG(INFO) << "PhyPagePool: get miss time=" << get_miss_time << ", count=" << get_miss_count << ", transfer_page_count=" << transfer_page_count;
  }

  num_available_ -= count;

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
  local_free_page_ids_.push_back(page_id);
  num_available_++;
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
    local_free_page_ids_.push_back(page_id);
    put_count++;
  }

  num_available_ += put_count;
}

void PhyPagePool::set_report_to_master(
    int32_t my_worker_rank,
    std::function<void(int32_t, size_t)> report_consume,
    std::function<void(int32_t, size_t)> report_release) {
  report_my_worker_rank_ = my_worker_rank;
  report_consume_cb_ = std::move(report_consume);
  report_release_cb_ = std::move(report_release);
}

void* PhyPagePool::allocate_contiguous(size_t count) {
  auto& global_xtensor = GlobalXTensor::get_instance();
  auto& page_allocator = PageAllocator::get_instance();
  if (page_allocator.is_initialized()) {
    page_allocator.consume_phy_pages_for_worker(device_.index(), count);
  } else if (report_consume_cb_) {
    report_consume_cb_(report_my_worker_rank_, count);
  }
  void* result = global_xtensor.allocate_from_left(count);
  CHECK(num_available_ >= count) << "PhyPagePool contiguous alloc exceeds available pages";
  num_available_ -= count;
  return result;
}

void PhyPagePool::free_contiguous(size_t addr, size_t count) {
  auto& global_xtensor = GlobalXTensor::get_instance();
  auto& page_allocator = PageAllocator::get_instance();
  if (page_allocator.is_initialized()) {
    page_allocator.release_phy_pages_for_worker(device_.index(), count);
  } else if (report_release_cb_) {
    report_release_cb_(report_my_worker_rank_, count);
  }
  for (size_t i = 0; i < count; i++) {
    global_xtensor.free_one_page_async(addr);
    addr += 2 * 1024 * 1024;
  }
  num_available_ += count;
}

size_t PhyPagePool::num_available() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return num_available_;
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
