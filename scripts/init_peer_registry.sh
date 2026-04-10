#!/usr/bin/env bash
# init_peer_registry.sh – 初始化 PeerRegistry 的 JSON 文件和共享内存。
#
# 说明：
#   在所有 xllm 实例完成 link_d2d 后调用。
#   remote_addrs 是 per (sender, receiver) 对的，因此每对都需要单独建链并抓取地址。
#   脚本完成后各实例在下次 try_d2d_load 时自动加载（懒初始化）。
#
# 使用示例：
#   ./init_peer_registry.sh \
#     --instances inst_0,inst_1,inst_2 \
#     --logs /path/node_0.log,/path/node_1.log,/path/node_2.log \
#     --models Qwen3-8B:28,Qwen3-1.7B:24 \
#     --inst-models "inst_0:Qwen3-8B,Qwen3-1.7B;inst_1:Qwen3-8B;inst_2:Qwen3-1.7B" \
#     --tp 2 \
#     --out /tmp/xllm_peer_table.json \
#     --shm /xllm_peer_state
#
# --inst-models 格式：分号分隔每个实例的模型列表
#   "inst_0:ModelA,ModelB;inst_1:ModelA"
#
# 关于 remote_addrs 采集：
#   脚本假设每对 (sender, receiver) 建链后，receiver 的日志中会出现：
#     weight_transfer_addr=<ip:port>
#   每个 receiver 日志里的地址数量 = tp * num_senders。
#   脚本按出现顺序逐 sender 分配（每个 sender 占 tp 条）。
#
# 仅初始化共享内存（不写 JSON、不读日志）：
#   ./init_peer_registry.sh --shm-only --in-json /path/xllm_peer_table.json
#   要求 JSON 中含 model_slots、instances、model_num_layers（完整模式生成的 JSON 已含）。
#   若 JSON 无 model_num_layers，可再传 --models Qwen3-8B:28,... 仅用于层数。
#
#   或不用 JSON，全部从命令行：
#   ./init_peer_registry.sh --shm-only --instances i0,i1 --models Qwen3-8B:28 \
#     --inst-models 'i0:Qwen3-8B;i1:Qwen3-8B' [--shm /xllm_peer_state]

set -euo pipefail

INSTANCES=""
LOGS=""
MODELS=""
INST_MODELS=""
TP=1
OUT="/tmp/xllm_peer_table.json"
SHM_NAME="/xllm_peer_state"
SHM_ONLY=0
IN_JSON=""

usage() {
  echo "Full (JSON + SHM from logs):"
  echo "  $0 --instances i0,i1 --logs l0,l1 --models m:n,... --inst-models 'i0:m0,m1;i1:m0' [--tp N] [--out path] [--shm name]"
  echo "SHM only from existing JSON:"
  echo "  $0 --shm-only --in-json /path/table.json [--models m:n,... if JSON lacks model_num_layers] [--shm name]"
  echo "SHM only from CLI (no JSON file):"
  echo "  $0 --shm-only --instances i0,i1 --models m:n,... --inst-models '...' [--shm name]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --instances)   INSTANCES="$2";   shift 2 ;;
    --logs)        LOGS="$2";        shift 2 ;;
    --models)      MODELS="$2";      shift 2 ;;
    --inst-models) INST_MODELS="$2"; shift 2 ;;
    --tp)          TP="$2";          shift 2 ;;
    --out)         OUT="$2";         shift 2 ;;
    --shm)         SHM_NAME="$2";    shift 2 ;;
    --shm-only)    SHM_ONLY=1;       shift ;;
    --in-json)     IN_JSON="$2";     shift 2 ;;
    *) usage ;;
  esac
done

if [[ "$SHM_ONLY" -eq 1 ]]; then
  if [[ -n "$IN_JSON" ]]; then
    [[ -f "$IN_JSON" ]] || { echo "ERROR: --in-json not a file: $IN_JSON" >&2; exit 1; }
  else
    [[ -n "$INSTANCES" && -n "$MODELS" && -n "$INST_MODELS" ]] || usage
  fi
else
  [[ -z "$INSTANCES" || -z "$LOGS" || -z "$MODELS" || -z "$INST_MODELS" ]] && usage
fi

export SHM_ONLY IN_JSON

python3 - <<PYEOF
import json, sys, re, struct, os, mmap as _mmap

def parse_models_arg(models_raw):
    model_list = []
    for m in models_raw:
        if not m.strip():
            continue
        parts = m.strip().split(":")
        if len(parts) != 2:
            print(f"ERROR: model spec '{m}' must be name:num_layers", file=sys.stderr)
            sys.exit(1)
        model_list.append((parts[0], int(parts[1])))
    return model_list

def parse_inst_models(inst_models_raw):
    inst_model_map = {}
    for part in inst_models_raw.split(";"):
        part = part.strip()
        if not part:
            continue
        idx = part.index(":")
        iid = part[:idx].strip()
        mids = [x.strip() for x in part[idx+1:].split(",")]
        inst_model_map[iid] = mids
    return inst_model_map

def init_posix_shm(shm_name, instances_out, model_slots, model_num_layers,
                   MAX_INSTANCES, MAX_MODELS):
    shm_size = MAX_INSTANCES * MAX_MODELS * 4
    shm_path = f"/dev/shm{shm_name}"
    fd = os.open(shm_path, os.O_CREAT | os.O_RDWR, 0o666)
    os.ftruncate(fd, shm_size)
    buf = _mmap.mmap(fd, shm_size)
    buf.seek(0)
    buf.write(struct.pack(f"{MAX_INSTANCES * MAX_MODELS}i",
                          *[-1] * (MAX_INSTANCES * MAX_MODELS)))
    for inst in instances_out:
        s = inst["inst_slot"]
        for mid in inst["model_ids"]:
            if mid not in model_slots:
                continue
            if mid not in model_num_layers:
                print(f"WARNING: no layer count for model {mid}, skip SHM cell",
                      file=sys.stderr)
                continue
            ms = model_slots[mid]
            offset = (s * MAX_MODELS + ms) * 4
            buf.seek(offset)
            buf.write(struct.pack("i", model_num_layers[mid] - 1))
    buf.flush()
    buf.close()
    os.close(fd)
    print(f"[init_peer_registry] SHM '{shm_name}' ready ({shm_size} bytes, "
          f"{MAX_INSTANCES}×{MAX_MODELS} cells)")

instances_raw   = "$INSTANCES".split(",")
logs_raw        = "$LOGS".split(",")
models_raw      = [x for x in "$MODELS".split(",") if x.strip()]
inst_models_raw = "$INST_MODELS"
tp              = int("$TP")
out_path        = "$OUT"
shm_name_cli    = "$SHM_NAME"
shm_only        = os.environ.get("SHM_ONLY", "0") == "1"
in_json_path    = os.environ.get("IN_JSON", "").strip()

MAX_INSTANCES = 16
MAX_MODELS    = 16

if shm_only and in_json_path:
    # ── SHM only: load peer table JSON (written by you or by full run) ───────
    with open(in_json_path) as f:
        root = json.load(f)
    shm_name = root.get("shm_name") or shm_name_cli
    if shm_name_cli and shm_name_cli != "/xllm_peer_state":
        shm_name = shm_name_cli
    model_slots = {k: int(v) for k, v in root["model_slots"].items()}
    instances_out = root["instances"]
    if "model_num_layers" in root:
        model_num_layers = {k: int(v) for k, v in root["model_num_layers"].items()}
    elif models_raw:
        ml = parse_models_arg(models_raw)
        model_num_layers = {m[0]: m[1] for m in ml}
    else:
        print("ERROR: JSON has no model_num_layers; pass --models m:n,... for layer counts",
              file=sys.stderr)
        sys.exit(1)
    if len(model_slots) > MAX_MODELS:
        print(f"ERROR: too many models (> {MAX_MODELS})", file=sys.stderr)
        sys.exit(1)
    init_posix_shm(shm_name, instances_out, model_slots, model_num_layers,
                   MAX_INSTANCES, MAX_MODELS)
    sys.exit(0)

# ── Parse model_id:num_layers ────────────────────────────────────────────────
if not models_raw:
    print("ERROR: --models required", file=sys.stderr)
    sys.exit(1)
model_list = parse_models_arg(models_raw)
if len(model_list) > MAX_MODELS:
    print(f"ERROR: {len(model_list)} model types exceed MAX_MODELS={MAX_MODELS}", file=sys.stderr)
    sys.exit(1)

model_slots = {m[0]: i for i, m in enumerate(model_list)}
model_num_layers = {m[0]: m[1] for m in model_list}

inst_model_map = parse_inst_models(inst_models_raw)

if shm_only:
    # ── SHM only: all placement from CLI (no logs, no JSON file) ────────────
    if len(instances_raw) > MAX_INSTANCES:
        print(f"ERROR: too many instances (> {MAX_INSTANCES})", file=sys.stderr)
        sys.exit(1)
    instances_out = []
    for slot, inst_id in enumerate(instances_raw):
        inst_id = inst_id.strip()
        instances_out.append({
            "instance_id": inst_id,
            "inst_slot":   slot,
            "model_ids":   inst_model_map.get(inst_id, []),
        })
    shm_name = shm_name_cli
    init_posix_shm(shm_name, instances_out, model_slots, model_num_layers,
                   MAX_INSTANCES, MAX_MODELS)
    sys.exit(0)

# ── Full mode: logs + write JSON + SHM ───────────────────────────────────────
logs_raw = [x for x in "$LOGS".split(",")]

if len(instances_raw) != len(logs_raw):
    print("ERROR: --instances and --logs must have same count", file=sys.stderr)
    sys.exit(1)

if len(instances_raw) > MAX_INSTANCES:
    print(f"ERROR: {len(instances_raw)} instances exceed MAX_INSTANCES={MAX_INSTANCES}", file=sys.stderr)
    sys.exit(1)

instances_out = []
inst_slot_map = {}
for slot, inst_id in enumerate(instances_raw):
    inst_id = inst_id.strip()
    inst_slot_map[inst_id] = slot
    instances_out.append({
        "instance_id": inst_id,
        "inst_slot":   slot,
        "model_ids":   inst_model_map.get(inst_id, []),
    })

links_out = []

for recv_slot, (recv_id, log_path) in enumerate(zip(instances_raw, logs_raw)):
    recv_id  = recv_id.strip()
    log_path = log_path.strip()

    all_addrs = []
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(r"weight_transfer_addr=(\S+)", line)
                if m:
                    all_addrs.append(m.group(1))
    except FileNotFoundError:
        print(f"WARNING: log not found: {log_path}", file=sys.stderr)

    senders = [inst_id.strip() for inst_id in instances_raw if inst_id.strip() != recv_id]

    expected = len(senders) * tp
    if len(all_addrs) < expected:
        print(f"WARNING: receiver={recv_id}: found {len(all_addrs)} addrs, "
              f"expected {expected} ({len(senders)} senders × tp={tp})",
              file=sys.stderr)

    for i, sender_id in enumerate(senders):
        start = i * tp
        addrs = all_addrs[start:start + tp]
        if not addrs:
            print(f"WARNING: no addrs for sender={sender_id} → receiver={recv_id}",
                  file=sys.stderr)
            continue
        links_out.append({
            "sender_slot":   inst_slot_map[sender_id],
            "receiver_slot": recv_slot,
            "remote_addrs":  addrs,
        })

shm_name = shm_name_cli
root = {
    "shm_name":         shm_name,
    "model_slots":      model_slots,
    "model_num_layers": model_num_layers,
    "instances":        instances_out,
    "links":            links_out,
}
with open(out_path, "w") as f:
    json.dump(root, f, indent=2)
print(f"[init_peer_registry] JSON written to {out_path} "
      f"({len(instances_out)} instances, {len(links_out)} links)")

init_posix_shm(shm_name, instances_out, model_slots, model_num_layers,
               MAX_INSTANCES, MAX_MODELS)
PYEOF

echo "[init_peer_registry] Done."
