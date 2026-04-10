// Pinned host -> NPU device bandwidth (H2D only). Not part of Mooncake.
//
// CANN 头文件一般在 ascend-toolkit/latest/<arch>-linux/include，而不是
// ASCEND_HOME/include。 推荐：用仓库 scripts/bench_transfer_compare.sh h2d
// 自动探测路径并编译。 手动编译示例（aarch64 把 aarch64-linux 换成你机器上的
// *-linux 目录名）：
//   INC=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include
//   LIB=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64
//   g++ -O2 -std=c++17 scripts/h2d_pinned_bench.cpp -o h2d_pinned_bench
//   -I"$INC" -L"$LIB" -lascendcl
//
// Run:
//   ./h2d_pinned_bench [device_id] [size_bytes] [warmup] [iters]
// Example:
//   ./h2d_pinned_bench 0 $((256*1024*1024)) 10 50

#include <acl/acl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now_sec() {
  using Clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  int device = 0;
  size_t nbytes = 256ull * 1024 * 1024;
  int warmup = 10;
  int iters = 50;
  if (argc >= 2) device = std::atoi(argv[1]);
  if (argc >= 3) nbytes = static_cast<size_t>(std::atoll(argv[2]));
  if (argc >= 4) warmup = std::atoi(argv[3]);
  if (argc >= 5) iters = std::atoi(argv[4]);

  if (aclInit(nullptr) != ACL_SUCCESS) {
    std::fprintf(stderr, "aclInit failed\n");
    return 1;
  }
  if (aclrtSetDevice(device) != ACL_SUCCESS) {
    std::fprintf(stderr, "aclrtSetDevice failed\n");
    aclFinalize();
    return 1;
  }

  void* h = nullptr;
  void* d = nullptr;
  if (aclrtMallocHost(&h, nbytes) != ACL_SUCCESS || !h) {
    std::fprintf(stderr, "aclrtMallocHost failed\n");
    aclFinalize();
    return 1;
  }
  if (aclrtMalloc(&d, nbytes, ACL_MEM_MALLOC_HUGE_ONLY) != ACL_SUCCESS) {
    std::fprintf(stderr, "aclrtMalloc failed\n");
    aclrtFreeHost(h);
    aclFinalize();
    return 1;
  }

  std::memset(h, 0x5a, nbytes);

  aclrtStream stream = nullptr;
  if (aclrtCreateStream(&stream) != ACL_SUCCESS || !stream) {
    std::fprintf(stderr, "aclrtCreateStream failed\n");
    aclrtFree(d);
    aclrtFreeHost(h);
    aclFinalize();
    return 1;
  }

  auto h2d_async = [&]() -> aclError {
    aclError e = aclrtMemcpyAsync(
        d, nbytes, h, nbytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (e != ACL_SUCCESS) return e;
    return aclrtSynchronizeStream(stream);
  };

  for (int i = 0; i < warmup; ++i) {
    h2d_async();
  }

  std::vector<double> secs;
  secs.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    double t0 = now_sec();
    aclError e = h2d_async();
    double t1 = now_sec();
    if (e != ACL_SUCCESS) {
      std::fprintf(stderr, "aclrtMemcpyAsync/Sync H2D failed: %d\n", e);
      break;
    }
    secs.push_back(t1 - t0);
  }

  aclrtDestroyStream(stream);

  double sum = 0;
  for (double s : secs) sum += s;
  double avg = secs.empty() ? 0 : sum / secs.size();
  double gbs = (avg > 0) ? (static_cast<double>(nbytes) / 1e9 / avg) : 0;

  std::printf(
      "[H2D pinned->device] device=%d size=%zu bytes (%.2f MiB) "
      "warmup=%d iters=%d avg_s=%.9f GB/s=%.3f "
      "(aclrtMemcpyAsync + aclrtSynchronizeStream)\n",
      device,
      nbytes,
      nbytes / (1024.0 * 1024.0),
      warmup,
      static_cast<int>(secs.size()),
      avg,
      gbs);

  aclrtFree(d);
  aclrtFreeHost(h);
  aclrtResetDevice(device);
  aclFinalize();
  return 0;
}
