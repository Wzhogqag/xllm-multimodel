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

  initialized_ = true;

  LOG(INFO) << "PhyPagePool: successfully pre-allocated " << num_pages
            << " physical pages (page_id 0-" << (num_pages - 1)
            << ") + 1 zero page";
}

std::unique_ptr<PhyPage> PhyPagePool::get() {
  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  auto& global_xtensor = GlobalXTensor::get_instance();
  if (num_available_ < 1) {
    LOG(WARNING) << "PhyPagePool: no free pages available";
    return nullptr;
  }

  // FIFO: pop from front to allocate left-to-right
  std::vector<page_id_t> page_id = global_xtensor.allocate_pages_from_right(1);

  num_available_--;

  // Move ownership to caller
  return std::move(all_pages_[page_id[0]]);
}

std::vector<std::unique_ptr<PhyPage>> PhyPagePool::batch_get(size_t count) {
  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  if (count == 0) {
    return {};
  }

  auto& global_xtensor = GlobalXTensor::get_instance();

  if (num_available_ < count) {
    LOG(WARNING) << "PhyPagePool: not enough free pages, requested " << count
                 << ", available " << num_available_;
    return {};
  }

  std::vector<std::unique_ptr<PhyPage>> result;
  result.reserve(count);

  std::vector<page_id_t> page_ids = global_xtensor.allocate_pages_from_right(count);

  num_available_ -= count;

  // FIFO: pop from front to allocate left-to-right
  for (size_t i = 0; i < count; ++i) {
    result.push_back(std::move(all_pages_[page_ids[i]]));
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

  // Return ownership to pool
  all_pages_[page_id] = std::move(page);

  num_available_++;

  auto& global_xtensor = GlobalXTensor::get_instance();
  std::vector<PhyPage*> page_ptr = {page.get()};
  global_xtensor.free_to_right_async(page_ptr);
}

void PhyPagePool::batch_put(std::vector<std::unique_ptr<PhyPage>>& pages) {
  if (pages.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  std::vector<PhyPage*> pages_ptr;

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

    // Return ownership to pool
    pages_ptr.push_back(page.get());
    all_pages_[page_id] = std::move(page);
    // Use push_front to keep smaller page_ids at front for KV cache allocation
  }

  num_available_ += pages.size();

  auto& global_xtensor = GlobalXTensor::get_instance();
  global_xtensor.free_to_right_async(pages_ptr);
}

void* PhyPagePool::allocate_contiguous(size_t count) {
  auto& global_xtensor = GlobalXTensor::get_instance();
  void* result = global_xtensor.allocate_from_left(count);
  CHECK(num_available_ >= count) << "PhyPagePool contiguous alloc exceeds available pages";
  num_available_ -= count;
  return result;
}

void PhyPagePool::free_contiguous(size_t addr, size_t count) {
  auto& global_xtensor = GlobalXTensor::get_instance();
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

const std::vector<PhyPage*>& PhyPagePool::get_all_pages() const {
  CHECK(initialized_) << "PhyPagePool not initialized";
  return all_page_ptrs_;
}

}  // namespace xllm
