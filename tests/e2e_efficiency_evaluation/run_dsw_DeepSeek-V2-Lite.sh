#!/bin/bash
# End-to-end efficiency evaluation: DeepSeek-V2-Lite (16B)
# Requires: 1 node (8 GPUs)
#
# Usage:
#   bash run_dsw_DeepSeek-V2-Lite.sh <CASE>
#
# Cases:
#   bf16         - BF16 baseline
#   blockwise    - Blockwise FP8
#   tensorwise   - Tensorwise FP8
#   fp8flowmoe   - FP8-FLOW-MoE (blockwise + FP8 dataflow)

export NVSHMEM_IBRC_PROXY_EP_NUM=2
export NVSHMEM_IBRC_DISABLE_FENCE=true
export NVSHMEM_IBRC_PROXY_EP_AFFINITY=4
export NVSHMEM_IBRC_ROCE_LAG_PORT_SELECTION=2
export NVSHMEM_IBGDA_ROCE_LAG_PORT_SELECTION=6
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IB_TRAFFIC_CLASS=16

export NCCL_IB_RETRY_CNT=15
export NCCL_IB_TIMEOUT=21
export NCCL_MIN_NCHANNELS=32
export NCCL_IB_TC=166
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_SL=0
export NCCL_DEBUG=warn

export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NCCL_NVLS_ENABLE=0
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1

# --- Working directory ---
export BASE_DIR=${BASE_DIR:-/mnt/data/FP8FLOW-MOE-AE}
cd "${BASE_DIR}/Megatron-LM-FP8FlowMoe"
export OUTPUT_DIR=${BASE_DIR}/output/dsw
rm -rf ${OUTPUT_DIR}

# ---------------------------------------------------------------------------
# Precision case selection
# ---------------------------------------------------------------------------
CASE="${1:?Usage: bash $0 <bf16|blockwise|tensorwise|fp8flowmoe>}"

case "${CASE}" in
    bf16)
        echo ">>> Running: BF16 baseline"
        export PR=bf16
        ;;
    blockwise)
        echo ">>> Running: Blockwise FP8"
        export PR=blockwise
        ;;
    tensorwise)
        echo ">>> Running: Tensorwise FP8"
        export PR=tensorwise
        ;;
    fp8flowmoe)
        echo ">>> Running: FP8-FLOW-MoE (blockwise + FP8 dataflow)"
        export PR=blockwise
        export FP8_DATAFLOW=true
        ;;
    *)
        echo "Error: unknown case '${CASE}'"
        echo "Usage: bash $0 <bf16|blockwise|tensorwise|fp8flowmoe>"
        exit 1
        ;;
esac

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=warn
export MANUAL_GC=5
export NUM_WORKERS=6
export CKPT_FORMAT=torch_dist
export TRAIN_ITERS=6
export SAVE_INTERVAL=1000
export CHECK_NAN=false

export AC=full

export MODEL_SIZE=16B
export PP=2
export EP=4
export GLOBAL_BATCH_SIZE=256

export PAO=grads
export OFFLOAD_OPTIMIZER=true
export DISPATCHER_TYPE=flex_deepep

export ENV=dsw
export TOKENIZER_PATH=/mnt/data/chenqun/zjllm-llama3-tokenizer
export TOKENIZER_TYPE=HuggingFaceTokenizer
export DATASET_FILE=/mnt/data/chenqun/datasets/v2_test.datalist

export MOE_AUX_LOSS_COEFF=0.0001
export ROUTER_BIAS_UPDATE_RATE=0
export ROUTER_SCORE_FUNC=sigmod
export LOAD_BALANCE_TYPE=seq_aux_loss
export ENABLE_RAMPUP_BS=false

bash examples/pretrain_ds.sh