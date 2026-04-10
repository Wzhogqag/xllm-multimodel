// 测试目的：
//   两进程，全量映射 VA+PA → Write #1 成功 →
//   Acceptor 调 cleanupTransportMem()（HcclMemDereg + 清 session）→
//   再调 aclrtUnmapMem → 观察是否能成功（与之前不 Dereg 直接 unmap 对比）
//
// 用法：
//   进程 0（Acceptor，先启动）:
//     ./hccl_virt_mem_test 0 <dev_phy> <dev_logic> <host_ip> <host_port>
//     <device_ip>
//   进程 1（Initiator，后启动）:
//     ./hccl_virt_mem_test 1 <dev_phy> <dev_logic> <host_ip> <host_port>
//     <device_ip> \
//       <r_dev_phy> <r_dev_logic> <r_host_ip> <r_host_port> <r_device_ip>
//
// 示例（同机 phy=2/logic=2 和 phy=3/logic=3）：
//   终端1: ./hccl_virt_mem_test 0 2 2 11.87.191.100 18132 172.16.2.10
//   终端2: ./hccl_virt_mem_test 1 3 3 11.87.191.100 18133 172.16.3.10 2
//   2 11.87.191.100 18132 172.16.2.10
//
// 进程间同步文件：
//   /tmp/hccl_test_w1done      — Initiator 完成 Write#1 后创建
//   /tmp/hccl_test_cleaned     — Acceptor  cleanup 后创建
//   /tmp/hccl_test_acceptor_pid

#include <arpa/inet.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "acl/acl.h"

enum class DevType : uint32_t { DEV_TYPE_NOSOC = 0 };

struct RankInfo {
  uint64_t rankId = 0xFFFFFFFF;
  uint64_t serverIdx = 0;
  struct in_addr hostIp = {};
  uint64_t hostPort = 0;
  uint64_t deviceLogicId = 0;
  uint64_t devicePhyId = 0;
  DevType deviceType = DevType::DEV_TYPE_NOSOC;
  struct in_addr deviceIp = {};
  uint64_t devicePort = 0;
  uint64_t pid = 0;
};

extern "C" {
int initTransportMem(RankInfo* local_rank_info);
int transportMemAccept(RankInfo* local_rank_info);
int regLocalRmaMem(void* addr, uint64_t length);
int transportMemTask(RankInfo* local_rank_info,
                     RankInfo* remote_rank_info,
                     int op_code,
                     uint64_t offset,
                     uint64_t req_len,
                     void* local_mem,
                     aclrtStream stream);
int cleanupTransportMem();  // HcclMemDereg + 清 session
}

#define LOGI(fmt, ...)                                   \
  do {                                                   \
    fprintf(stdout, "[INFO ] " fmt "\n", ##__VA_ARGS__); \
    fflush(stdout);                                      \
  } while (0)
#define LOGE(fmt, ...)                                   \
  do {                                                   \
    fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__); \
    fflush(stderr);                                      \
  } while (0)

static constexpr size_t MEM_SIZE = 64ULL * 1024 * 1024;  // 64MB

static const char* SYNC_W1_DONE = "/tmp/hccl_test_w1done";
static const char* SYNC_CLEANED = "/tmp/hccl_test_cleaned";
static const char* SYNC_ACCEPTOR_PID = "/tmp/hccl_test_acceptor_pid";

static void sync_create(const char* path, uint64_t val = 1) {
  FILE* f = fopen(path, "w");
  if (f) {
    fprintf(f, "%lu", val);
    fclose(f);
  }
  LOGI("[sync] created %s (val=%lu)", path, val);
}

static uint64_t sync_wait_val(const char* path, int timeout_sec = 60) {
  LOGI("[sync] waiting for %s ...", path);
  struct stat st{};
  for (int i = 0; i < timeout_sec * 10; ++i) {
    if (stat(path, &st) == 0) {
      FILE* f = fopen(path, "r");
      uint64_t val = 0;
      if (f) {
        fscanf(f, "%lu", &val);
        fclose(f);
      }
      LOGI("[sync] got %s (val=%lu)", path, val);
      return val;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  LOGE("[sync] timeout waiting for %s", path);
  return 0;
}

static void sync_wait(const char* path, int timeout_sec = 60) {
  sync_wait_val(path, timeout_sec);
}

static void fill_rank_info(RankInfo& ri,
                           int dev_phy,
                           int dev_logic,
                           const char* host_ip,
                           uint16_t host_port,
                           const char* device_ip,
                           int rank_id) {
  ri.rankId = rank_id;
  ri.serverIdx = 0;
  ri.devicePhyId = dev_phy;
  ri.deviceLogicId = dev_logic;
  ri.devicePort = 16666;
  ri.hostPort = host_port;
  inet_pton(AF_INET, host_ip, &ri.hostIp);
  inet_pton(AF_INET, device_ip, &ri.deviceIp);
}

// 全量映射：VA = PA = MEM_SIZE（保证 HcclMemReg 能成功）
static int alloc_full_mem(int dev_logic,
                          void*& va_ptr,
                          aclrtDrvMemHandle& pa_handle) {
  aclrtPhysicalMemProp prop{};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev_logic;
  prop.reserve = 0;

  size_t granularity = 0;
  aclError ret = aclrtMemGetAllocationGranularity(
      &prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity);
  if (ret != ACL_ERROR_NONE || granularity == 0) {
    LOGE("aclrtMemGetAllocationGranularity failed, ret=%d", ret);
    return -1;
  }
  size_t aligned = ((MEM_SIZE + granularity - 1) / granularity) * granularity;
  LOGI("granularity=%zuMB, aligned=%zuMB",
       granularity / 1024 / 1024,
       aligned / 1024 / 1024);

  ret = aclrtReserveMemAddress(&va_ptr, aligned, 0, nullptr, 1);
  if (ret != ACL_ERROR_NONE) {
    LOGE("ReserveMemAddress failed, ret=%d", ret);
    return -1;
  }
  LOGI("Reserved VA=%p size=%zuMB", va_ptr, aligned / 1024 / 1024);

  // 全量 PA
  ret = aclrtMallocPhysical(&pa_handle, aligned, &prop, 0);
  if (ret != ACL_ERROR_NONE) {
    LOGE("MallocPhysical failed, ret=%d", ret);
    aclrtReleaseMemAddress(va_ptr);
    return -1;
  }
  LOGI("Allocated PA handle=%p size=%zuMB (FULL)",
       (void*)pa_handle,
       aligned / 1024 / 1024);

  ret = aclrtMapMem(va_ptr, aligned, 0, pa_handle, 0);
  if (ret != ACL_ERROR_NONE) {
    LOGE("MapMem failed, ret=%d", ret);
    aclrtFreePhysical(pa_handle);
    aclrtReleaseMemAddress(va_ptr);
    return -1;
  }
  LOGI("Mapped FULL PA -> VA[0, %zuMB)", aligned / 1024 / 1024);
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc < 7) {
    fprintf(stderr,
            "Usage:\n"
            "  Acceptor : %s 0 <dev_phy> <dev_logic> <host_ip> <host_port> "
            "<device_ip>\n"
            "  Initiator: %s 1 <dev_phy> <dev_logic> <host_ip> <host_port> "
            "<device_ip>"
            " <r_phy> <r_logic> <r_host_ip> <r_host_port> <r_device_ip>\n",
            argv[0],
            argv[0]);
    return 1;
  }

  int role = std::atoi(argv[1]);

  if (role == 0) {
    remove(SYNC_W1_DONE);
    remove(SYNC_CLEANED);
    remove(SYNC_ACCEPTOR_PID);
  }

  int dev_phy = std::atoi(argv[2]);
  int dev_logic = std::atoi(argv[3]);
  const char* host_ip = argv[4];
  uint16_t host_port = (uint16_t)std::atoi(argv[5]);
  const char* device_ip = argv[6];

  if (aclInit(nullptr) != ACL_ERROR_NONE) {
    LOGE("aclInit failed");
    return 1;
  }
  if (aclrtSetDevice(dev_logic) != ACL_ERROR_NONE) {
    LOGE("aclrtSetDevice(%d) failed", dev_logic);
    return 1;
  }

  void* va_ptr = nullptr;
  aclrtDrvMemHandle pa_handle{};
  if (alloc_full_mem(dev_logic, va_ptr, pa_handle) != 0) return 1;

  LOGI("=== regLocalRmaMem: FULL VA+PA (%zuMB) ===", MEM_SIZE / 1024 / 1024);
  int ret = regLocalRmaMem(va_ptr, MEM_SIZE);
  if (ret != 0) {
    LOGE("regLocalRmaMem failed, ret=%d", ret);
    return 1;
  }

  RankInfo local_rank{};
  fill_rank_info(
      local_rank, dev_phy, dev_logic, host_ip, host_port, device_ip, role);

  LOGI("=== initTransportMem ===");
  ret = initTransportMem(&local_rank);
  if (ret != 0) {
    LOGE("initTransportMem failed, ret=%d", ret);
    return 1;
  }
  LOGI("initTransportMem OK, hostPort=%lu, device_pid=%lu",
       local_rank.hostPort,
       local_rank.pid);

  if (role == 0) {
    // ===================== Acceptor =====================
    sync_create(SYNC_ACCEPTOR_PID, local_rank.pid);

    LOGI("=== [Acceptor] transportMemAccept (HcclMemReg) ===");
    ret = transportMemAccept(&local_rank);
    if (ret != 0) {
      LOGE("[Acceptor] transportMemAccept failed, ret=%d", ret);
      return 1;
    }
    LOGI("[Acceptor] HcclMemReg OK, connection established");

    // 等 Initiator 完成 Write #1
    sync_wait(SYNC_W1_DONE);
    LOGI("[Acceptor] Write #1 done, start cleanup");

    // 正常清理：HcclMemDereg + 清 session
    LOGI(
        "=== [Acceptor] cleanupTransportMem (HcclMemDereg + close session) "
        "===");
    ret = cleanupTransportMem();
    LOGI("[Acceptor] cleanupTransportMem ret=%d %s",
         ret,
         ret == 0 ? "[OK]" : "[FAIL]");

    // 关键：cleanup 之后再 unmap，预期成功
    LOGI("=== [Acceptor] aclrtUnmapMem AFTER cleanupTransportMem ===");
    aclError acl_ret = aclrtUnmapMem(va_ptr);
    LOGI("[Acceptor] aclrtUnmapMem ret=%d %s",
         acl_ret,
         acl_ret == 0 ? "[PASS — unmap succeeded after proper dereg]"
                      : "[FAIL — unmap still blocked]");

    sync_create(SYNC_CLEANED);

    // 等 Initiator Write #2 结果
    LOGI("[Acceptor] waiting 30s for Initiator Write #2...");
    std::this_thread::sleep_for(std::chrono::seconds(30));
    LOGI("[Acceptor] done");

    if (acl_ret == 0) {
      // 已成功 unmap，直接释放 PA 和 VA
      aclrtFreePhysical(pa_handle);
      aclrtReleaseMemAddress(va_ptr);
    } else {
      // unmap 失败，补一次再释放
      aclrtUnmapMem(va_ptr);
      aclrtFreePhysical(pa_handle);
      aclrtReleaseMemAddress(va_ptr);
    }

  } else {
    // ===================== Initiator =====================
    if (argc < 12) {
      LOGE("Initiator needs remote params");
      return 1;
    }

    int rphy = std::atoi(argv[7]);
    int rlogic = std::atoi(argv[8]);
    const char* rip = argv[9];
    uint16_t rport = (uint16_t)std::atoi(argv[10]);
    const char* rdevip = argv[11];

    RankInfo remote_rank{};
    fill_rank_info(remote_rank, rphy, rlogic, rip, rport, rdevip, 0);

    uint64_t acceptor_pid = sync_wait_val(SYNC_ACCEPTOR_PID);
    remote_rank.pid = acceptor_pid;
    LOGI("Got Acceptor device_pid=%lu", acceptor_pid);

    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_ERROR_NONE) {
      LOGE("aclrtCreateStream failed");
      return 1;
    }

    // Write #1（全量 PA，预期成功）
    LOGI("=== [Test 1] Write #1 (%zuMB) — FULL PA, expect PASS ===",
         MEM_SIZE / 1024 / 1024);
    ret = transportMemTask(&local_rank,
                           &remote_rank,
                           1,
                           (uint64_t)va_ptr,
                           MEM_SIZE,
                           va_ptr,
                           stream);
    LOGI("[Test 1] Write #1 ret=%d %s", ret, ret == 0 ? "[PASS]" : "[FAIL]");

    sync_create(SYNC_W1_DONE);

    // 等 Acceptor cleanup + unmap
    sync_wait(SYNC_CLEANED);

    // Write #2（Acceptor 已 cleanup，session 已断）
    LOGI("=== [Test 2] Write #2 after Acceptor cleanupTransportMem ===");
    LOGI("    Hypothesis: session gone → error");
    ret = transportMemTask(&local_rank,
                           &remote_rank,
                           1,
                           (uint64_t)va_ptr,
                           MEM_SIZE,
                           va_ptr,
                           stream);
    LOGI("[Test 2] Write #2 ret=%d %s",
         ret,
         ret == 0 ? "[UNEXPECTED PASS]" : "[EXPECTED FAIL — session closed]");

    aclrtDestroyStream(stream);
    aclrtUnmapMem(va_ptr);
    aclrtFreePhysical(pa_handle);
    aclrtReleaseMemAddress(va_ptr);
  }

  aclrtResetDevice(dev_logic);
  aclFinalize();
  if (role == 0) {
    remove(SYNC_W1_DONE);
    remove(SYNC_CLEANED);
    remove(SYNC_ACCEPTOR_PID);
  }
  return 0;
}
