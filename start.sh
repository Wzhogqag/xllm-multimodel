# # 1. 环境变量设置
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/  # NPU 版 PyTorch 路径
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"  # PyTorch 安装路径
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"  # LibTorch 路径
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH  # 添加 NPU 库路径
export HCCL_CONNECT_TIMEOUT=7200
# # 2. 加载环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=4,5
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=INFO
export ASDOPS_LOG_TO_FILE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=0
export ATB_ACLNN_CACHE_GLOABL_COUNT=0
export INF_NAN_MODE_FORCE_DISABLE=1
# 3. 清理旧日志
# \rm -rf core.*

# 4. 启动分布式服务
#MODEL_PATH="/export/home/models/DeepSeek-V3-sub"  # 模型路径
#MODEL_PATH="/export/home/models/Qwen2.5-14B"
MODEL_PATH="/export/home/models/Qwen3-8B"
DRAFT_MODEL_PATH="/export/home/models/Eagle3-Qwen3-32B-zh"
#MODEL_PATH="/export/home/models/Qwen3-Embedding-0.6B"
#MODEL_PATH="/export/home/wangyunlong.115/workspace/Qwen2.5-VL-7B"
#MODEL_PATH="/export/home/models/Qwen3-32B"
#
#MODEL_PATH="/export/home/wangyi.480/1000006"
MASTER_NODE_ADDR="127.0.0.1:9929"                  # Master 节点地址（需全局一致）
#export MC_MS_AUTO_DISC=0
START_PORT=13212                                  # 服务起始端口
START_DEVICE=0                                     # 起始 NPU 逻辑设备号
LOG_DIR="log"                                      # 日志目录
NNODES=1                                         # 节点数（当前脚本启动 2 个进程）
export HCCL_IF_BASE_PORT=47532  # HCCL 通信基础端口
TRANSFER_START_PORT=25550
export XLLM_INSTANCE_ID=inst_0 
export XLLM_PEER_TABLE_PATH=/tmp/xllm_peer_table.json
phy_ids=(4 5)
for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))  
  curr_phy_id=${phy_ids[i]}
  TRANSFER_PORT=$((TRANSFER_START_PORT + i))
  echo $TRANSFER_PORT
  echo $curr_phy_id
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="node_$i.log"
  export NPU_PHY_ID=$curr_phy_id
  ./build/xllm/core/server/xllm \
    --model $MODEL_PATH \
    --devices="npu:$DEVICE" \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --max_memory_utilization=0.86 \
    --max_tokens_per_batch=16184 \
    --max_seqs_per_batch=256 \
    --enable_mla=false \
    --enable_shm=false \
    --block_size=128 \
    --dp_size=1 \
    --enable_xtensor=true \
    --npu_phy_id=$curr_phy_id \
    --transfer_listen_port=$TRANSFER_PORT \
    --communication_backend="hccl" \
    --enable_prefix_cache=true \
    --enable_chunked_prefill=true \
    --enable_schedule_overlap=false \
    --xtensor_master_node_addr=127.0.0.1:13534 \
    --priority_level=3 \
    --enable_activation_pooling=true \
    --host_blocks_factor=1.1 \
    --enable_watermark_degrade_restore_mvp=true \
    --node_rank=$i >> $LOG_FILE 2>&1 & 
done

    # --log_mooncake_weight_pull_timing=false \
