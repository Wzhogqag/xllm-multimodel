#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <deque>
#include <memory>
#include <unordered_map>

#include "acl/acl.h"
#include "core/framework/xtensor/xtensor_allocator.h"

using namespace xllm;

std::mutex mtx;
ska::flat_hash_map<aclrtStream, std::deque<std::pair<aclrtEvent, void*>>> npu_events;

void process_events();
void insert_events(void* ptr, aclrtStream stream);
void* my_custom_alloc(size_t size, int device, aclrtStream stream) {
  void* ptr = NULL;
  if (size <= 0 || device < 0) return NULL;
  std::lock_guard<std::mutex> lock(mtx);

  // printf("[自定义分配器alloc] allocate 调用: size=%zd, device=%d\n", size,
  // device);

  // 设置设备上下文(多卡不设置可能会报错？)
/*
  aclError ret = aclrtSetDevice(device);
  if (ret != ACL_ERROR_NONE) {
      fprintf(stderr, "[自定义分配器alloc] aclrtSetDevice 失败: device=%d,
  error=%d\n", device, ret); return NULL;
  }
*/
  process_events();
  bool res = XTensorAllocator::get_instance().allocate_activation(ptr, size);
  if (res != true) {
    fprintf(stderr,
            "[自定义分配器alloc] XTensorAllocator::allocate_activation 失败\n");
    return NULL;
  }

  return ptr;
}

void process_events() {
  // Process outstanding npuEvents. Events that are completed are removed
  // from the queue, and the 'event_count' for the corresponding allocation
  // is decremented. Stops at the first event which has not been completed.
  // Since events on different devices or streams may occur out of order,
  // the processing of some events may be delayed.
  for (auto it = npu_events.begin(); it != npu_events.end();) {
    while (!it->second.empty()) {
      auto &e = it->second.front();
      aclrtEvent event = e.first;
      void *ptr = e.second;

      aclrtEventRecordedStatus status;
      aclError ret = aclrtQueryEventStatus(event, &status);
      if (ret != ACL_ERROR_NONE) {
        LOG(ERROR) << "aclrtQueryEventStatus failed" << ret;
      }

      if (status != ACL_EVENT_RECORDED_STATUS_COMPLETE) {
        break;
      }

      bool res = XTensorAllocator::get_instance().deallocate_activation(ptr);
      aclrtDestroyEvent(event);
      it->second.pop_front();
    }

    if (it->second.empty()) {
      it = npu_events.erase(it);
    } else {
      it++;
    }
  }
}

void my_custom_free(void* ptr, size_t size, int device, aclrtStream stream) {
  if (ptr == NULL) {
    return;
  }
  std::lock_guard<std::mutex> lock(mtx);

  // printf("[自定义分配器free] free 调用: ptr=%p, size=%zu, device=%d\n", ptr,
  // size, device);

  // 设置设备上下文
/*
  aclError ret = aclrtSetDevice(device);
  if (ret != ACL_ERROR_NONE) {
      fprintf(stderr, "[自定义分配器free] aclrtSetDevice 失败: device=%d,
  error=%d\n", device, ret); return;
  }
*/
  insert_events(ptr, stream);
}

void insert_events(void *ptr, aclrtStream stream) {
  //c10_npu::NPUEvent* event = new c10_npu::NPUEvent(ACL_EVENT_CAPTURE_STREAM_PROGRESS);
  aclrtEvent event;
  aclrtCreateEvent(&event);
  NPUStatus rets = c10_npu::emptyAllNPUStream(true);
  if (rets != NPU_STATUS_SUCCESS) {
    ASCEND_LOGE("MakeSureQueueEmpty fail, ret: %s", rets.c_str());
  }
  aclError ret = aclrtRecordEvent(event, stream);  
  if (ret != ACL_ERROR_NONE) {  
    LOG(ERROR) << "aclrtRecordEvent failed" << ret;
  }
  
  npu_events[stream].emplace_back(event, ptr);
}