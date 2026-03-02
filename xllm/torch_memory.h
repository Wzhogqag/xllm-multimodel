#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "acl/acl.h"
#include "core/framework/xtensor/xtensor_allocator.h"

using namespace xllm;

void* my_custom_alloc(size_t size, int device, aclrtStream stream) {
  void* ptr = NULL;
  if (size <= 0 || device < 0) return NULL;

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
  bool res = XTensorAllocator::get_instance().allocate_activation(ptr, size);
  if (res != true) {
    fprintf(stderr,
            "[自定义分配器alloc] XTensorAllocator::allocate_activation 失败\n");
    return NULL;
  }

  return ptr;
}

void my_custom_free(void* ptr, size_t size, int device, aclrtStream stream) {
  if (ptr == NULL) {
    return;
  }

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
  c10_npu::npuSynchronizeDevice(true);

  bool res = XTensorAllocator::get_instance().deallocate_activation(ptr, size);
}