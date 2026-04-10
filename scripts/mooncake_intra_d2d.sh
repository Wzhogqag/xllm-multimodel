#!/usr/bin/env bash
# 节点内 Mooncake Ascend D2D：双进程（target + initiator），仅用环境变量填卡号与端口。
#
# 必填环境变量（物理卡号，与 npu-smi / hccn 一致）：
#   INIT_PHY   — initiator 使用的 NPU 物理 ID
#   TGT_PHY    — target 使用的 NPU 物理 ID
#
# 可选：
#   INIT_LOGIC / TGT_LOGIC — 逻辑 device id（容器内与物理不一致时必设；默认等于对应 PHY）
#   HOST_IP    — 本机「参数面/业务」IP，须与 Ascend Transport 使用的 hostIp 一致（见 hccl_transport
#                日志里 hostIp，例如 11.87.191.100）。不要用 127.0.0.1，除非本机自测且 transport 也绑在 lo。
#   PORT_INIT  — initiator 侧 RPC 端口，默认 12345
#   PORT_TGT   — target 侧 RPC 端口，默认 12346
#   SEGMENT_ID — initiator 专用：填 target 日志里的「实际监听 IP:端口」（P2PHANDSHAKE 常与 PORT_TGT 不同）
#   MOONCAKE_BIN — 可执行文件，默认 one_sided 示例（需先编译 Mooncake）
#   BLOCK_SIZE / BATCH_SIZE — 与官方 gflags 一致
# 日志：脚本会 export GLOG_logtostderr=1。若仍无终端输出，检查是否 export 了
#   MC_LOG_DIR（日志会写目录、不刷屏，见 Mooncake config.cpp），可 unset MC_LOG_DIR。
#
# 用法：
#   source 环境后：
#     ./mooncake_intra_d2d.sh print     # 打印当前参数与两条命令模板
#     ./mooncake_intra_d2d.sh target    # 仅启动 target（终端 1）
#     export SEGMENT_ID=...             # 从 target 日志抄实际 IP:端口
#     ./mooncake_intra_d2d.sh initiator # 仅启动 initiator（终端 2）
#
set -euo pipefail
export ASCEND_RT_VISIBLE_DEVICES=6,7
export HOST_IP='11.887.191.100'          # 换成你本机参数面/业务网 IP（与 hccl 日志里 hostIp 一致，勿用 127.0.0.1 除非你知道绑在 lo）
export INIT_PHY=6                  # initiator 物理卡 ID
export TGT_PHY=7    

export INIT_LOGIC=0
export TGT_LOGIC=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------- CANN：与仓库文档一致（见 docs/zh/getting_started/launch_xllm.md）----------
source_setenv() {
  # 本脚本使用 set -u；CANN 的 setenv 会引用可能未设的 $2/$3，必须先关 nounset
  set +e
  set +u
  if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck source=/dev/null
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
  fi
  if [[ -f /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash ]]; then
    # shellcheck source=/dev/null
    source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
  fi
  set -e
  set -u
}

HOST_IP="${HOST_IP:-127.0.0.1}"

warn_host_ip() {
  if [[ "${HOST_IP}" == "127.0.0.1" ]] || [[ "${HOST_IP}" == "localhost" ]]; then
    echo "WARN: HOST_IP=${HOST_IP}。Ascend Transport / Mooncake 在真机上通常要填参数面 IP（与日志里 hostIp 一致，如 11.87.191.100），否则 listening/segment 可能异常。请: export HOST_IP=<你的 hostIp>" >&2
  fi
}
PORT_INIT="${PORT_INIT:-12345}"
PORT_TGT="${PORT_TGT:-12346}"
BLOCK_SIZE="${BLOCK_SIZE:-8388608}"
BATCH_SIZE="${BATCH_SIZE:-32}"
METADATA="${METADATA:-P2PHANDSHAKE}"
OPERATION="${OPERATION:-write}"
PROTOCOL="${PROTOCOL:-hccl}"

MOONCAKE_BIN="${MOONCAKE_BIN:-${REPO_ROOT}/third_party/Mooncake/build/mooncake-transfer-engine/example/transfer_engine_ascend_one_sided}"

# device_id=65536 表示使用 device_logicid + device_phyid（与官方示例一致）
DEVICE_SENTINEL=65536

# glog 默认可能不写终端；不设则看起来像「没日志」
mooncake_log_to_terminal() {
  export GLOG_logtostderr=1
  export GLOG_minloglevel=0
  # 若曾 export MC_LOG_DIR 且目录可写，Mooncake 会把日志写文件、关 stderr（见 transfer_engine config.cpp）
  if [[ "${MC_LOG_DIR_FORCE:-}" != "1" ]] && [[ -n "${MC_LOG_DIR:-}" ]]; then
    echo "NOTE: 已设置 MC_LOG_DIR=${MC_LOG_DIR}，日志可能在目录内而非终端。取消: unset MC_LOG_DIR" >&2
  fi
}

common_flags() {
  echo "--metadata_server=${METADATA} --protocol=${PROTOCOL} --operation=${OPERATION}"
}


run_target() {
  echo "[mooncake_intra_d2d] 进入 target 模式" >&2
  if [[ -z "${INIT_PHY:-}" ]] || [[ -z "${TGT_PHY:-}" ]]; then
    echo "ERROR: 未设置 INIT_PHY 或 TGT_PHY（当前 INIT_PHY='${INIT_PHY-未设置}' TGT_PHY='${TGT_PHY-未设置}'）" >&2
    echo "请先执行，例如: export INIT_PHY=0 TGT_PHY=12   # 按 npu-smi / 你的 hostIp 侧卡号填写" >&2
    exit 1
  fi
  INIT_LOGIC="${INIT_LOGIC:-$INIT_PHY}"
  TGT_LOGIC="${TGT_LOGIC:-$TGT_PHY}"
  source_setenv
  if [[ ! -x "${MOONCAKE_BIN}" ]]; then
    echo "ERROR: 未找到可执行文件: ${MOONCAKE_BIN}" >&2
    echo "请先编译 Mooncake（USE_ASCEND=ON），生成 transfer_engine_ascend_one_sided" >&2
    exit 1
  fi
  mooncake_log_to_terminal
  warn_host_ip
  echo "========== TARGET 卡 逻辑=${TGT_LOGIC} 物理=${TGT_PHY} HOST_IP=${HOST_IP} 端口=${PORT_TGT} =========="
  echo "请在本终端观察日志，找到: Transfer Engine RPC ... listening on <IP>:<端口>"
  echo "执行: ${MOONCAKE_BIN} ... (GLOG_logtostderr=1)" >&2
  exec "${MOONCAKE_BIN}" $(common_flags) \
    --local_server_name="${HOST_IP}:${PORT_TGT}:npu_${TGT_PHY}" \
    --device_id="${DEVICE_SENTINEL}" \
    --device_logicid="${TGT_LOGIC}" \
    --device_phyid="${TGT_PHY}" \
    --mode=target \
    --block_size="${BLOCK_SIZE}" \
    --batch_size="${BATCH_SIZE}"
}

run_initiator() {
  echo "[mooncake_intra_d2d] 进入 initiator 模式" >&2
  if [[ -z "${INIT_PHY:-}" ]] || [[ -z "${TGT_PHY:-}" ]]; then
    echo "ERROR: 未设置 INIT_PHY 或 TGT_PHY" >&2
    exit 1
  fi
  INIT_LOGIC="${INIT_LOGIC:-$INIT_PHY}"
  TGT_LOGIC="${TGT_LOGIC:-$TGT_PHY}"
  source_setenv
  if [[ ! -x "${MOONCAKE_BIN}" ]]; then
    echo "ERROR: 未找到可执行文件: ${MOONCAKE_BIN}" >&2
    exit 1
  fi
  SEGMENT_ID="${SEGMENT_ID:?请先 export SEGMENT_ID=<target 日志中的 IP:实际端口>}"
  mooncake_log_to_terminal
  warn_host_ip
  echo "========== INITIATOR 卡 逻辑=${INIT_LOGIC} 物理=${INIT_PHY} HOST_IP=${HOST_IP} segment_id=${SEGMENT_ID} =========="
  echo "执行: ${MOONCAKE_BIN} ... (GLOG_logtostderr=1)" >&2
  exec "${MOONCAKE_BIN}" $(common_flags) \
    --local_server_name="${HOST_IP}:${PORT_INIT}:npu_${INIT_PHY}" \
    --segment_id="${SEGMENT_ID}" \
    --device_id="${DEVICE_SENTINEL}" \
    --device_logicid="${INIT_LOGIC}" \
    --device_phyid="${INIT_PHY}" \
    --mode=initiator \
    --block_size="${BLOCK_SIZE}" \
    --batch_size="${BATCH_SIZE}"
}

print_config() {
  local iphy="${INIT_PHY:-<export INIT_PHY>}"
  local tphy="${TGT_PHY:-<export TGT_PHY>}"
  local il="${INIT_LOGIC:-$iphy}"
  local tl="${TGT_LOGIC:-$tphy}"
  cat <<EOF
=== 节点内 Mooncake D2D 参数 ===
INIT_PHY=${iphy}  TGT_PHY=${tphy}
INIT_LOGIC=${il}  TGT_LOGIC=${tl}
HOST_IP=${HOST_IP}  PORT_INIT=${PORT_INIT}  PORT_TGT=${PORT_TGT}
MOONCAKE_BIN=${MOONCAKE_BIN}
BLOCK_SIZE=${BLOCK_SIZE}  BATCH_SIZE=${BATCH_SIZE}

（请先 source CANN：/usr/local/Ascend/ascend-toolkit/set_env.sh 与 latest/bin/setenv.bash）

--- 终端 A：target ---
export INIT_PHY=... TGT_PHY=...
${MOONCAKE_BIN} --metadata_server=${METADATA} --protocol=${PROTOCOL} --operation=${OPERATION} \\
    --local_server_name=${HOST_IP}:${PORT_TGT} \\
    --device_id=${DEVICE_SENTINEL} --device_logicid=${tl} --device_phyid=${tphy} \\
    --mode=target --block_size=${BLOCK_SIZE} --batch_size=${BATCH_SIZE}

--- 终端 B：initiator（把 SEGMENT_ID 换成 target 日志里的实际 IP:端口）---
export SEGMENT_ID=<例如 11.87.191.100:23456>
${MOONCAKE_BIN} --metadata_server=${METADATA} --protocol=${PROTOCOL} --operation=${OPERATION} \\
    --local_server_name=${HOST_IP}:${PORT_INIT} \\
    --segment_id=\${SEGMENT_ID} \\
    --device_id=${DEVICE_SENTINEL} --device_logicid=${il} --device_phyid=${iphy} \\
    --mode=initiator --block_size=${BLOCK_SIZE} --batch_size=${BATCH_SIZE}

EOF
}

case "${1:-print}" in
  target) run_target ;;
  initiator) run_initiator ;;
  print) print_config ;;
  *)
    echo "Usage: $0 {print|target|initiator}" >&2
    echo "需先 export INIT_PHY TGT_PHY；initiator 前还需 export SEGMENT_ID" >&2
    exit 1
    ;;
esac
