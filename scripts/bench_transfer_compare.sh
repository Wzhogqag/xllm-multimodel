#!/usr/bin/env bash
# 对比三项带宽（思路一致、数据量可对齐）：
#   1) Pinned host -> 本卡 device（本脚本编译 h2d_pinned_bench.cpp）
#   2) Mooncake 节点内 D2D（双进程 transfer_engine_ascend_perf）
#   3) Mooncake 节点间 D2D（同上，target/initiator 在不同机器）
#
# 用法：
#   仅跑 H2D：        ./bench_transfer_compare.sh h2d 2 $((1024*1024*1024)) 10 50
#   打印 Mooncake 模板： ./bench_transfer_compare.sh mooncake
#   全部（先 H2D，再打印 Mooncake 命令）： ./bench_transfer_compare.sh all
#
# H2D 可选参数（须紧跟子命令，字节数请用算术展开，勿写裸的 1024*1024*1024）：
#   ./bench_transfer_compare.sh h2d [device_id] [size_bytes] [warmup] [iters]
#   例：./bench_transfer_compare.sh h2d 2 $((1024*1024*1024)) 10 50
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
H2D_PINNED_BENCH="${SCRIPT_DIR}/h2d_pinned_bench"

# 解析 acl/acl.h：优先 ASCEND_INCLUDE_DIR，否则 *-linux/include（与 Mooncake common.cmake 一致）
detect_ascend_cann_paths() {
  if [[ -n "${ASCEND_INCLUDE_DIR:-}" && -f "${ASCEND_INCLUDE_DIR}/acl/acl.h" ]]; then
    ASCEND_LIB_DIR="${ASCEND_LIB_DIR:-${ASCEND_INCLUDE_DIR}/../lib64}"
    return 0
  fi
  local base="${ASCEND_HOME:-}"
  if [[ -z "${base}" ]]; then
    if [[ -d /usr/local/Ascend/latest ]]; then
      base="/usr/local/Ascend/latest"
    else
      base="/usr/local/Ascend/ascend-toolkit/latest"
    fi
  fi
  local arch=""
  shopt -s nullglob
  local candidates=("${base}"/*-linux)
  shopt -u nullglob
  if [[ ${#candidates[@]} -gt 0 ]]; then
    arch="${candidates[0]}"
  fi
  if [[ -n "${arch}" && -f "${arch}/include/acl/acl.h" ]]; then
    ASCEND_INCLUDE_DIR="${arch}/include"
    ASCEND_LIB_DIR="${arch}/lib64"
    return 0
  fi
  if [[ -n "${ASCEND_HOME:-}" && -f "${ASCEND_HOME}/include/acl/acl.h" ]]; then
    ASCEND_INCLUDE_DIR="${ASCEND_HOME}/include"
    ASCEND_LIB_DIR="${ASCEND_HOME}/lib64"
    return 0
  fi
  echo "ERROR: 找不到 acl/acl.h。请设置 ASCEND_INCLUDE_DIR 为 .../aarch64-linux/include（或 x86_64-linux）。" >&2
  return 1
}

# ---------- 可调：与 Mooncake perf 对齐的字节量 ----------
# 与 Mooncake 中 batch_size * block_size 或单块 block_size 一致便于对比
SIZE_BYTES="${SIZE_BYTES:-$((256 * 1024 * 1024))}"
# H2D 侧
DEVICE_ID="${DEVICE_ID:-0}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"

# Mooncake 二进制（编译 Mooncake 后修改）
MOONCAKE_PERF="${MOONCAKE_PERF:-${REPO_ROOT}/third_party/Mooncake/build/mooncake-transfer-engine/example/transfer_engine_ascend_perf}"

# ---------- 节点内 Mooncake 示例占位（按你环境改）----------
# 同一主机、两台进程、不同端口、不同 NPU；local_server_name 需带 :npu_<物理id>
INTRA_INIT_IP="${INTRA_INIT_IP:-127.0.0.1}"
INTRA_INIT_PORT="${INTRA_INIT_PORT:-12345}"
INTRA_TGT_PORT="${INTRA_TGT_PORT:-12346}"
# segment_id = target 日志里实际监听的 IP:端口（P2PHANDSHAKE 时常与填参不一致，以日志为准）
INTRA_SEGMENT_ID="${INTRA_SEGMENT_ID:-${INTRA_INIT_IP}:${INTRA_TGT_PORT}}"
INTRA_DEV_INIT="${INTRA_DEV_INIT:-0}"
INTRA_DEV_TGT="${INTRA_DEV_TGT:-1}"

# ---------- 节点间 Mooncake 示例占位 ----------
INTER_TARGET_IP="${INTER_TARGET_IP:-10.0.0.2}"
INTER_TARGET_PORT="${INTER_TARGET_PORT:-12346}"
INTER_SEGMENT_ID="${INTER_SEGMENT_ID:-${INTER_TARGET_IP}:${INTER_TARGET_PORT}}"

build_h2d() {
  detect_ascend_cann_paths || exit 1
  echo "Using ASCEND_INCLUDE_DIR=${ASCEND_INCLUDE_DIR}"
  echo "Using ASCEND_LIB_DIR=${ASCEND_LIB_DIR}"
  g++ -O2 -std=c++17 "${SCRIPT_DIR}/h2d_pinned_bench.cpp" -o "${SCRIPT_DIR}/h2d_pinned_bench" \
    -I"${ASCEND_INCLUDE_DIR}" -L"${ASCEND_LIB_DIR}" -lascendcl
  echo "Built: ${SCRIPT_DIR}/h2d_pinned_bench"
}

run_h2d() {
  build_h2d
  if [[ ! -x "${H2D_PINNED_BENCH}" ]]; then
    echo "ERROR: 编译失败或不可执行: ${H2D_PINNED_BENCH}" >&2
    exit 1
  fi
  echo "========== [1] pinned -> device (H2D) device=${DEVICE_ID} SIZE_BYTES=${SIZE_BYTES} =========="
  "${H2D_PINNED_BENCH}" "${DEVICE_ID}" "${SIZE_BYTES}" "${WARMUP}" "${ITERS}"
}

print_mooncake() {
  echo ""
  echo "========== [2][3] Mooncake D2D（节点内 / 节点间共用同一二进制）=========="
  echo "要求：USE_ASCEND=ON 编译 Mooncake；Ascend Transport 依赖见 third_party/Mooncake/doc/zh/ascend_transport.md"
  echo "注册内存须 2MB 对齐；local_server_name 使用 ip:port:npu_<物理卡号>。"
  echo ""
  if [[ ! -x "${MOONCAKE_PERF}" ]]; then
    echo "WARN: 未找到可执行文件: ${MOONCAKE_PERF}"
    echo "      请设置 MOONCAKE_PERF 或先编译 Mooncake example。"
  fi
  BS="${BLOCK_SIZE:-$((SIZE_BYTES / 32))}"
  # 若整除不了，用 SIZE_BYTES 作单块
  if [[ "${BS}" -le 0 ]]; then BS="${SIZE_BYTES}"; fi
  BZ="${BATCH_SIZE:-32}"
  BIT="${BLOCK_ITERATION:-1}"

  echo "--- 节点内 D2D：先 target（卡${INTRA_DEV_TGT}），再 initiator（卡${INTRA_DEV_INIT}）---"
  echo "终端 A (target):"
  cat <<EOF
${MOONCAKE_PERF} --metadata_server=P2PHANDSHAKE \\
  --local_server_name=${INTRA_INIT_IP}:${INTRA_TGT_PORT}:npu_<TARGET_PHY_ID> \\
  --protocol=hccl --operation=write --device_id=${INTRA_DEV_TGT} \\
  --mode=target --block_size=${BS} --batch_size=${BZ} --block_iteration=${BIT}
EOF
  echo "看日志里 Transfer Engine RPC listening on <IP>:<实际端口>，将 initiator 的 --segment_id 改为 该 IP:端口。"
  echo "终端 B (initiator):"
  cat <<EOF
${MOONCAKE_PERF} --metadata_server=P2PHANDSHAKE \\
  --local_server_name=${INTRA_INIT_IP}:${INTRA_INIT_PORT}:npu_<INIT_PHY_ID> \\
  --protocol=hccl --operation=write --segment_id=${INTRA_SEGMENT_ID} \\
  --device_id=${INTRA_DEV_INIT} --mode=initiator --block_size=${BS} --batch_size=${BZ} --block_iteration=${BIT}
EOF

  echo ""
  echo "--- 节点间 D2D：target 在机器 B，initiator 在机器 A（IP/端口按实际替换）---"
  echo "机器 B (target): 同上，local_server_name 用 B 的 IP:port:npu_x"
  echo "机器 A (initiator):"
  cat <<EOF
${MOONCAKE_PERF} --metadata_server=P2PHANDSHAKE \\
  --local_server_name=<A_IP>:<A_PORT>:npu_<A_PHY_ID> \\
  --protocol=hccl --operation=write --segment_id=${INTER_SEGMENT_ID} \\
  --device_id=0 --mode=initiator --block_size=${BS} --batch_size=${BZ} --block_iteration=${BIT}
EOF
  echo ""
  echo "带宽对齐建议：令 batch_size*block_size（或 block_iteration 展开后的总传输量）与 SIZE_BYTES=${SIZE_BYTES} 同一量级；"
  echo " Mooncake 侧打印的 rate 与 [1] 的 GB/s 同列对比。"
}

MODE="${1:-all}"
shift || true

case "${MODE}" in
  h2d)
    [[ $# -ge 1 ]] && DEVICE_ID="$1"
    [[ $# -ge 2 ]] && SIZE_BYTES="$2"
    [[ $# -ge 3 ]] && WARMUP="$3"
    [[ $# -ge 4 ]] && ITERS="$4"
    run_h2d
    ;;
  mooncake) print_mooncake ;;
  all) run_h2d; print_mooncake ;;
  *)
    echo "Usage: $0 [h2d|mooncake|all] [h2d: device_id size_bytes warmup iters]" >&2
    exit 1
    ;;
esac
