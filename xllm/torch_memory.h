#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>
#include <deque>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "core/framework/xtensor/xtensor_allocator.h"
#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

using namespace xllm;

inline void process_events();
inline void insert_events(void* ptr, aclrtStream stream);

namespace torch_memory_detail {

inline uint64_t elapsed_us(std::chrono::steady_clock::time_point start,
                           std::chrono::steady_clock::time_point end) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

inline std::atomic<uint64_t>& alloc_mtx_wait_time_us_sum() {
  static std::atomic<uint64_t> v{0};
  return v;
}

inline std::atomic<uint64_t>& free_mtx_wait_time_us_sum() {
  static std::atomic<uint64_t> v{0};
  return v;
}

inline std::atomic<uint64_t>& alloc_activation_time_us_sum() {
  static std::atomic<uint64_t> v{0};
  return v;
}

inline std::atomic<uint64_t>& dealloc_activation_time_us_sum() {
  static std::atomic<uint64_t> v{0};
  return v;
}

inline std::atomic<uint64_t>& query_event_time_us_sum() {
  static std::atomic<uint64_t> v{0};
  return v;
}

inline std::atomic<uint64_t>& destroy_event_time_us_sum() {
  static std::atomic<uint64_t> v{0};
  return v;
}

inline std::atomic<uint64_t>& create_event_time_us_sum() {
  static std::atomic<uint64_t> v{0};
  return v;
}

inline std::atomic<uint64_t>& record_event_time_us_sum() {
  static std::atomic<uint64_t> v{0};
  return v;
}

inline void add_time(std::atomic<uint64_t>& dst, uint64_t delta) {
  dst.fetch_add(delta, std::memory_order_relaxed);
}

struct PendingFreeRequest {
  void* ptr;
  aclrtStream stream;
  aclrtContext context;
  c10_npu::NPUStream npu_stream;
};

class AclContextGuard {
 public:
  explicit AclContextGuard(aclrtContext target_context)
      : active_(false) {
    if (target_context == nullptr) {
      return;
    }

    aclError set_ret = aclrtSetCurrentContext(target_context);
    if (set_ret != ACL_ERROR_NONE) {
      LOG(ERROR) << "aclrtSetCurrentContext failed" << set_ret;
      return;
    }
    active_ = true;
  }

  ~AclContextGuard() = default;

  bool active() const { return active_; }

 private:
  bool active_;
};

inline aclrtContext get_current_context_or_null() {
  aclrtContext context = nullptr;
  aclError ret = aclrtGetCurrentContext(&context);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "aclrtGetCurrentContext failed" << ret;
    return nullptr;
  }
  return context;
}

struct TimingSnapshot {
  uint64_t alloc_mtx_wait_us;
  uint64_t free_mtx_wait_us;
  uint64_t alloc_activation_us;
  uint64_t dealloc_activation_us;
  uint64_t query_event_us;
  uint64_t destroy_event_us;
  uint64_t create_event_us;
  uint64_t record_event_us;
};

inline TimingSnapshot load_timing_snapshot() {
  return TimingSnapshot{
      alloc_mtx_wait_time_us_sum().load(std::memory_order_relaxed),
      free_mtx_wait_time_us_sum().load(std::memory_order_relaxed),
      alloc_activation_time_us_sum().load(std::memory_order_relaxed),
      dealloc_activation_time_us_sum().load(std::memory_order_relaxed),
      query_event_time_us_sum().load(std::memory_order_relaxed),
      destroy_event_time_us_sum().load(std::memory_order_relaxed),
      create_event_time_us_sum().load(std::memory_order_relaxed),
      record_event_time_us_sum().load(std::memory_order_relaxed)};
}

inline void maybe_log_timing_window_1s() {
  static std::mutex log_mu;
  static bool initialized = false;
  static std::chrono::steady_clock::time_point last_log_tp;
  static TimingSnapshot last_snapshot{};

  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(log_mu);
  if (!initialized) {
    initialized = true;
    last_log_tp = now;
    last_snapshot = load_timing_snapshot();
    return;
  }

  if (now - last_log_tp < std::chrono::seconds(1)) {
    return;
  }

  const TimingSnapshot current = load_timing_snapshot();
  /*LOG(INFO) << "[torch_memory timing 1s window] "
            << "window_us=" << elapsed_us(last_log_tp, now) << ", "
            << "alloc_mtx_wait_us=" << (current.alloc_mtx_wait_us - last_snapshot.alloc_mtx_wait_us) << ", "
            << "free_mtx_wait_us=" << (current.free_mtx_wait_us - last_snapshot.free_mtx_wait_us) << ", "
            << "query_event_us=" << (current.query_event_us - last_snapshot.query_event_us) << ", "
            << "destroy_event_us=" << (current.destroy_event_us - last_snapshot.destroy_event_us) << ", "
            << "create_event_us=" << (current.create_event_us - last_snapshot.create_event_us) << ", "
            << "record_event_us=" << (current.record_event_us - last_snapshot.record_event_us) << ", "
            << "alloc_activation_us="
            << (current.alloc_activation_us - last_snapshot.alloc_activation_us) << ", "
            << "dealloc_activation_us="
            << (current.dealloc_activation_us - last_snapshot.dealloc_activation_us);*/

  last_log_tp = now;
  last_snapshot = current;
}

class EventPool {
 public:
  std::shared_ptr<c10_npu::NPUEvent> acquire() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (!free_events_.empty()) {
        std::shared_ptr<c10_npu::NPUEvent> event = free_events_.back();
        free_events_.pop_back();
        return event;
      }
    }

    const auto create_start = std::chrono::steady_clock::now();
    auto event = std::make_shared<c10_npu::NPUEvent>();
    const auto create_end = std::chrono::steady_clock::now();
    add_time(create_event_time_us_sum(), elapsed_us(create_start, create_end));
    return event;
  }

  void release(const std::shared_ptr<c10_npu::NPUEvent>& event) {
    if (event == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    free_events_.push_back(event);
  }

  ~EventPool() {
    std::vector<std::shared_ptr<c10_npu::NPUEvent>> pending_destroy;
    {
      std::lock_guard<std::mutex> lock(mu_);
      pending_destroy.swap(free_events_);
    }
    for (auto& event : pending_destroy) {
      if (event == nullptr) {
        continue;
      }
      const auto destroy_start = std::chrono::steady_clock::now();
      event.reset();
      const auto destroy_end = std::chrono::steady_clock::now();
      add_time(destroy_event_time_us_sum(), elapsed_us(destroy_start, destroy_end));
    }
  }

 private:
  std::mutex mu_;
  std::vector<std::shared_ptr<c10_npu::NPUEvent>> free_events_;
};

inline EventPool& event_pool() {
  static EventPool pool;
  return pool;
}

class ContextEventPoolRegistry {
 public:
  std::shared_ptr<c10_npu::NPUEvent> acquire(aclrtContext context) {
    EventPool* pool = get_or_create_pool(context);
    if (pool == nullptr) {
      return nullptr;
    }
    return pool->acquire();
  }

  void release(aclrtContext context,
               const std::shared_ptr<c10_npu::NPUEvent>& event) {
    EventPool* pool = get_or_create_pool(context);
    if (pool == nullptr) {
      return;
    }
    pool->release(event);
  }

 private:
  EventPool* get_or_create_pool(aclrtContext context) {
    if (context == nullptr) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(mu_);
    auto it = pools_.find(context);
    if (it == pools_.end()) {
      auto inserted = pools_.emplace(context, std::make_unique<EventPool>());
      return inserted.first->second.get();
    }
    return it->second.get();
  }

  std::mutex mu_;
  std::unordered_map<aclrtContext, std::unique_ptr<EventPool>> pools_;
};

inline ContextEventPoolRegistry& context_event_pools() {
  static ContextEventPoolRegistry registry;
  return registry;
}

struct StreamBucket {
  std::mutex mu;
  std::deque<std::pair<std::shared_ptr<c10_npu::NPUEvent>, void*>> events;
  aclrtContext context = nullptr;
};

inline std::mutex& buckets_mu() {
  static std::mutex mu;
  return mu;
}

inline ska::flat_hash_map<aclrtStream, std::unique_ptr<StreamBucket>>& stream_buckets() {
  static ska::flat_hash_map<aclrtStream, std::unique_ptr<StreamBucket>> buckets;
  return buckets;
}

inline StreamBucket* get_or_create_bucket(aclrtStream stream, aclrtContext context) {
  std::lock_guard<std::mutex> lock(buckets_mu());
  auto& buckets = stream_buckets();
  auto it = buckets.find(stream);
  if (it == buckets.end()) {
    auto bucket = std::make_unique<StreamBucket>();
    bucket->context = context;
    auto inserted = buckets.emplace(stream, std::move(bucket));
    return inserted.first->second.get();
  }
  if (it->second->context == nullptr) {
    it->second->context = context;
  } else if (context != nullptr && it->second->context != context) {
    LOG(ERROR) << "stream bucket context mismatch for stream=" << stream;
  }
  return it->second.get();
}

inline std::vector<StreamBucket*> snapshot_buckets() {
  std::vector<StreamBucket*> snapshot;
  std::lock_guard<std::mutex> lock(buckets_mu());
  auto& buckets = stream_buckets();
  snapshot.reserve(buckets.size());
  for (auto& entry : buckets) {
    snapshot.push_back(entry.second.get());
  }
  return snapshot;
}

inline void process_one_bucket(StreamBucket* bucket) {
  if (bucket == nullptr) {
    return;
  }

  AclContextGuard context_guard(bucket->context);
  if (!context_guard.active()) {
    return;
  }

  while (true) {
    const auto lock_wait_start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(bucket->mu);
    const auto lock_wait_end = std::chrono::steady_clock::now();
    add_time(alloc_mtx_wait_time_us_sum(), elapsed_us(lock_wait_start, lock_wait_end));

    if (bucket->events.empty()) {
      return;
    }

    std::shared_ptr<c10_npu::NPUEvent> event = bucket->events.front().first;
    if (event == nullptr) {
      LOG(ERROR) << "null NPUEvent in delayed free queue";
      return;
    }

    const auto query_start = std::chrono::steady_clock::now();
    bool is_complete = false;
    bool query_ok = true;
    try {
      is_complete = event->query();
    } catch (const std::exception& e) {
      query_ok = false;
      LOG(ERROR) << "NPUEvent::query threw exception: " << e.what();
    } catch (...) {
      query_ok = false;
      LOG(ERROR) << "NPUEvent::query threw unknown exception";
    }
    const auto query_end = std::chrono::steady_clock::now();
    add_time(query_event_time_us_sum(), elapsed_us(query_start, query_end));

    if (!query_ok) {
      return;
    }

    if (!is_complete) {
      return;
    }

    void* ptr = bucket->events.front().second;
    bucket->events.pop_front();
    lock.unlock();

    const auto dealloc_start = std::chrono::steady_clock::now();
    bool res = XTensorAllocator::get_instance().deallocate_activation(ptr);
    if (res != true) {
      LOG(ERROR) << "[自定义分配器free] XTensorAllocator::deallocate_activation 失败";
    }
    const auto dealloc_end = std::chrono::steady_clock::now();
    add_time(dealloc_activation_time_us_sum(), elapsed_us(dealloc_start, dealloc_end));

    context_event_pools().release(bucket->context, event);
  }
}

inline void insert_events_with_context(void* ptr,
                                       aclrtStream stream,
                                       aclrtContext context,
                                       c10_npu::NPUStream npu_stream);

class AsyncFreeWorker {
 public:
  AsyncFreeWorker() : stop_(false), worker_(&AsyncFreeWorker::run, this) {}

  ~AsyncFreeWorker() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  void enqueue(void* ptr, aclrtStream stream) {
    aclrtContext context = get_current_context_or_null();
    if (context == nullptr) {
      LOG(ERROR) << "my_custom_free enqueue without valid ACL context";
      return;
    }

    c10_npu::NPUStream npu_stream = c10_npu::getCurrentNPUStream();

    const auto lock_wait_start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mu_);
      const auto lock_wait_end = std::chrono::steady_clock::now();
      add_time(free_mtx_wait_time_us_sum(), elapsed_us(lock_wait_start, lock_wait_end));
      pending_.push_back(PendingFreeRequest{ptr, stream, context, npu_stream});
    }
    cv_.notify_one();
  }

 private:
  void run() {
    std::vector<PendingFreeRequest> local_pending;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait_for(lock, std::chrono::milliseconds(1),
                     [this]() { return stop_ || !pending_.empty(); });
        if (stop_ && pending_.empty()) {
          break;
        }
        local_pending.swap(pending_);
      }

      for (const auto& req : local_pending) {
        insert_events_with_context(req.ptr, req.stream, req.context,
                                   req.npu_stream);
      }
      local_pending.clear();

      process_events();
      maybe_log_timing_window_1s();
    }
  }

  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_;
  std::vector<PendingFreeRequest> pending_;
  std::thread worker_;
};

inline AsyncFreeWorker& async_free_worker() {
  static AsyncFreeWorker worker;
  return worker;
}

}  // namespace torch_memory_detail

inline void process_events();
inline void insert_events(void* ptr, aclrtStream stream);

inline void* my_custom_alloc(size_t size, int device, aclrtStream stream) {
  (void)stream;
  void* ptr = nullptr;
  if (size <= 0 || device < 0) {
    return nullptr;
  }

  const auto alloc_start = std::chrono::steady_clock::now();
  bool res = XTensorAllocator::get_instance().allocate_activation(ptr, size);
  const auto alloc_end = std::chrono::steady_clock::now();
  torch_memory_detail::add_time(torch_memory_detail::alloc_activation_time_us_sum(),
                                torch_memory_detail::elapsed_us(alloc_start, alloc_end));
  if (res != true) {
    LOG(ERROR) << "[自定义分配器alloc] XTensorAllocator::allocate_activation 失败";
    return nullptr;
  }

  torch_memory_detail::maybe_log_timing_window_1s();
  return ptr;
}

inline void process_events() {
  // Process each stream bucket independently. For one stream, processing stops
  // at the first incomplete event to preserve FIFO delayed-free semantics.
  std::vector<torch_memory_detail::StreamBucket*> buckets =
      torch_memory_detail::snapshot_buckets();
  for (torch_memory_detail::StreamBucket* bucket : buckets) {
    torch_memory_detail::process_one_bucket(bucket);
  }
}

inline void my_custom_free(void* ptr, size_t size, int device, aclrtStream stream) {
  (void)size;
  (void)device;
  if (ptr == nullptr) {
    return;
  }
  torch_memory_detail::async_free_worker().enqueue(ptr, stream);
}

inline void insert_events(void* ptr, aclrtStream stream) {
  aclrtContext context = torch_memory_detail::get_current_context_or_null();
  if (context == nullptr) {
    LOG(ERROR) << "insert_events called without valid ACL context";
    return;
  }
  c10_npu::NPUStream npu_stream = c10_npu::getCurrentNPUStream();
  torch_memory_detail::insert_events_with_context(ptr, stream, context,
                                                  npu_stream);
}

inline void torch_memory_detail::insert_events_with_context(void* ptr, aclrtStream stream,
                                                            aclrtContext context,
                                                            c10_npu::NPUStream npu_stream) {
  AclContextGuard context_guard(context);
  if (!context_guard.active()) {
    return;
  }

  StreamBucket* bucket = get_or_create_bucket(stream, context);
  std::shared_ptr<c10_npu::NPUEvent> event = context_event_pools().acquire(context);
  if (event == nullptr) {
    return;
  }

  const auto record_start = std::chrono::steady_clock::now();
  bool record_ok = true;
  try {
    event->record(npu_stream);
  } catch (const std::exception& e) {
    record_ok = false;
    LOG(ERROR) << "NPUEvent::record threw exception: " << e.what();
  } catch (...) {
    record_ok = false;
    LOG(ERROR) << "NPUEvent::record threw unknown exception";
  }
  const auto record_end = std::chrono::steady_clock::now();
  torch_memory_detail::add_time(torch_memory_detail::record_event_time_us_sum(),
                                torch_memory_detail::elapsed_us(record_start, record_end));
  if (!record_ok) {
    context_event_pools().release(context, event);
    return;
  }

  const auto lock_wait_start = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(bucket->mu);
  const auto lock_wait_end = std::chrono::steady_clock::now();
  add_time(free_mtx_wait_time_us_sum(), elapsed_us(lock_wait_start, lock_wait_end));
  bucket->events.emplace_back(event, ptr);
}