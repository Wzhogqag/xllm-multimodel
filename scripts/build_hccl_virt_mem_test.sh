#!/usr/bin/env bash
# 编译 hccl_virt_mem_test（无 glog 依赖，直接用 fprintf）
# 用法：bash scripts/build_hccl_virt_mem_test.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- CANN toolkit 路径探测 ----
ASCEND_BASE="${ASCEND_HOME:-}"
if [[ -z "${ASCEND_BASE}" ]]; then
    if   [[ -d /usr/local/Ascend/latest ]];                  then ASCEND_BASE="/usr/local/Ascend/latest"
    elif [[ -d /usr/local/Ascend/ascend-toolkit/latest ]];   then ASCEND_BASE="/usr/local/Ascend/ascend-toolkit/latest"
    else echo "ERROR: 找不到 Ascend toolkit，请设置 ASCEND_HOME" >&2; exit 1; fi
fi
shopt -s nullglob
ARCH_DIRS=("${ASCEND_BASE}"/*-linux)
shopt -u nullglob
[[ ${#ARCH_DIRS[@]} -eq 0 ]] && { echo "ERROR: ${ASCEND_BASE}/*-linux 不存在" >&2; exit 1; }
ASCEND_ARCH="${ARCH_DIRS[0]}"
ASCEND_INC="${ASCEND_ARCH}/include"
ASCEND_LIB="${ASCEND_ARCH}/lib64"

# HCCL 内部头文件（与 Mooncake CMakeLists.txt 保持一致）
HCCL_INC="${ASCEND_INC}/hccl"
HCCL_EXP="${ASCEND_INC}/experiment"
HCCL_EXP_INC="${ASCEND_INC}/experiment/hccl"
HCCL_EXP_SLOG="${ASCEND_INC}/experiment/slog/toolchain"
HCCL_EXP_META="${ASCEND_INC}/experiment/metadef/common/util/error_manager"
HCCL_EXP_RT="${ASCEND_INC}/experiment/runtime"
HCCL_EXP_MSPROF="${ASCEND_INC}/experiment/msprof"

# MPI（来自项目 vcpkg）
VCPKG_INC="${REPO_ROOT}/vcpkg_installed/arm64-linux/include"
VCPKG_LIB="${REPO_ROOT}/vcpkg_installed/arm64-linux/lib"

# Mooncake hccl_transport 相关（直接用已编译好的静态库，不再编译 .cpp）
MOONCAKE_SRC="${REPO_ROOT}/third_party/Mooncake/mooncake-transfer-engine"
MOONCAKE_INC="${MOONCAKE_SRC}/include"
BUILD_DIR="${REPO_ROOT}/build/cmake.linux-aarch64-cpython-311"
ASCEND_TRANSPORT_LIB="${BUILD_DIR}/third_party/Mooncake/mooncake-transfer-engine/src/transport/ascend_transport/hccl_transport/ascend_transport_c/libascend_transport_mem.a"

OUT="${SCRIPT_DIR}/hccl_virt_mem_test"

echo "=== 编译 hccl_virt_mem_test ==="
echo "  ASCEND_INC   = ${ASCEND_INC}"
echo "  HCCL_EXP_INC = ${HCCL_EXP_INC} (+ siblings)"
echo "  VCPKG_INC    = ${VCPKG_INC}"
echo "  OUT          = ${OUT}"

g++ -O0 -g -std=c++17 \
    "${SCRIPT_DIR}/hccl_virt_mem_test.cpp" \
    -o "${OUT}" \
    -I"${ASCEND_INC}" \
    -I"${HCCL_INC}" \
    -I"${HCCL_EXP}" \
    -I"${HCCL_EXP_INC}" \
    -I"${MOONCAKE_INC}" \
    -I/usr/local/include \
    -L"${ASCEND_LIB}" \
    "${ASCEND_TRANSPORT_LIB}" \
    -lascendcl \
    -lhccl \
    -lhccl_plf \
    -lc_sec \
    "${BUILD_DIR}/vcpkg_installed/arm64-linux/lib/libglog.a" \
    "${BUILD_DIR}/vcpkg_installed/arm64-linux/lib/libgflags.a" \
    -Wl,-rpath,"${ASCEND_LIB}"

echo ""
echo "=== 编译完成: ${OUT} ==="
echo ""
echo "运行（同机 卡0=Acceptor 卡1=Initiator）："
echo "  终端1（先启动）: ${OUT} 0 0 0 127.0.0.1 18000"
echo "  终端2（后启动）: ${OUT} 1 1 1 127.0.0.1 18001 0 0 127.0.0.1 18000"
