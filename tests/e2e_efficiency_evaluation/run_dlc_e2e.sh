#!/bin/bash
# End-to-end efficiency evaluation
# Requires: 32 nodes (256 GPUs)
#
# Usage:
#   bash run_dlc_e2e.sh <MODEL> <PRECISION> <EP> <AC>
#
# MODEL:     deepseekv3  - DeepSeek-V3 (671B)
#            deepseekv2  - DeepSeek-V2 (236B)
#            qwen3       - Qwen3-235B-A22B
# PRECISION: bf16        - BF16 baseline
#            blockwise   - Blockwise FP8
#            tensorwise  - Tensorwise FP8
#            fp8flowmoe  - FP8-FLOW-MoE (blockwise + FP8 dataflow)
# EP:        ep8  | ep16 | ep32
# AC:        full - full activation checkpointing
#            sel  - selective recomputation (attn layernorm moe_expert)
#
# Examples:
#   bash run_dlc_e2e.sh deepseekv3 bf16 ep8 full
#   bash run_dlc_e2e.sh deepseekv2 fp8flowmoe ep16 sel
#   bash run_dlc_e2e.sh qwen3 blockwise ep32 full

# ===========================================================================
# Common environment settings
# ===========================================================================

# --- NVSHMEM (for DeepEP communication) ---
export NVSHMEM_IBRC_PROXY_EP_NUM=2
export NVSHMEM_IBRC_DISABLE_FENCE=true
export NVSHMEM_IBRC_PROXY_EP_AFFINITY=4
export NVSHMEM_IBRC_ROCE_LAG_PORT_SELECTION=2
export NVSHMEM_IBGDA_ROCE_LAG_PORT_SELECTION=6
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IB_TRAFFIC_CLASS=16

# --- NCCL (for multi-node communication) ---
export NCCL_IB_RETRY_CNT=15
export NCCL_IB_TIMEOUT=21
export NCCL_MIN_NCHANNELS=32
export NCCL_IB_TC=166
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_SL=0
export NCCL_DEBUG=warn

# --- PyTorch / TransformerEngine ---
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NCCL_NVLS_ENABLE=0
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1

# --- Training settings ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MANUAL_GC=5
export NUM_WORKERS=6
export NO_MMAP_BIN_FILES=true
export CKPT_FORMAT=torch_dist
export DISPATCHER_TYPE=flex_deepep
export TRAIN_ITERS=6
export SAVE_INTERVAL=1000

# --- Memory optimization ---
export PAO=grads
export OFFLOAD_OPTIMIZER=true

# --- Working directory ---
export BASE_DIR=${BASE_DIR:-/mnt/data/FP8FLOW-MOE-AE}
cd "${BASE_DIR}/Megatron-LM-FP8FlowMoe"
export OUTPUT_DIR=${BASE_DIR}/output/
export ENV=dlc

# ===========================================================================
# Parameter parsing
# ===========================================================================
USAGE="Usage: bash $0 <deepseekv3|deepseekv2|qwen3> <bf16|blockwise|tensorwise|fp8flowmoe> <ep8|ep16|ep32> <full|sel>"

MODEL_CASE="${1:?${USAGE}}"
PR_CASE="${2:?${USAGE}}"
EP_CASE="${3:?${USAGE}}"
AC_CASE="${4:?${USAGE}}"

# ---------------------------------------------------------------------------
# Precision selection
# ---------------------------------------------------------------------------
case "${PR_CASE}" in
    bf16)
        echo ">>> Precision: BF16 baseline"
        export PR=bf16
        ;;
    blockwise)
        echo ">>> Precision: Blockwise FP8"
        export PR=blockwise
        ;;
    tensorwise)
        echo ">>> Precision: Tensorwise FP8"
        export PR=tensorwise
        ;;
    fp8flowmoe)
        echo ">>> Precision: FP8-FLOW-MoE (blockwise + FP8 dataflow)"
        export PR=blockwise
        export FP8_DATAFLOW=true
        ;;
    *)
        echo "Error: unknown precision '${PR_CASE}'"
        echo "${USAGE}"
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Activation checkpointing selection
# ---------------------------------------------------------------------------
case "${AC_CASE}" in
    full)
        echo ">>> Activation checkpointing: full"
        export AC=full
        ;;
    sel)
        echo ">>> Activation checkpointing: selective (attn layernorm moe_expert)"
        export AC=sel
        export RECOMPUTE_MODULES="attn layernorm moe_expert"
        ;;
    *)
        echo "Error: unknown AC '${AC_CASE}'"
        echo "${USAGE}"
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
case "${MODEL_CASE}" in
    deepseekv3)
        echo ">>> Model: DeepSeek-V3 (671B)"

        case "${EP_CASE}" in
            ep8)
                echo ">>> Parallelism: EP=8, PP=32"
                export PP_LAYOUT="E\|\(tt\|\)*30,tL"
                export PP=32
                export EP=8
                ;;
            ep16)
                echo ">>> Parallelism: EP=16, PP=16"
                export PP=16
                export EP=16
                export MP_PPN_LAYERS=1
                ;;
            ep32)
                echo ">>> Parallelism: EP=32, PP=8"
                export PP=8
                export EP=32
                export MP_PPN_LAYERS=5
                ;;
            *)
                echo "Error: unknown EP '${EP_CASE}'"
                echo "${USAGE}"
                exit 1
                ;;
        esac

        export MODEL_SIZE=600B
        export CPT_PRETRAIN_CHECKPOINT_PATH=/mnt/data/chenqun/models/DeepSeek-V3-0324-bf16_torch_dist
        export TOKENIZER_TYPE=HuggingFaceTokenizer
        export TOKENIZER_PATH=/mnt/data/chenqun/models/DeepSeek-V3-0324-bf16_torch_dist/release/
        export DATASET_FILE=/mnt/cpfs/tokens/deepseek-v3/train-sm.datalist
        export ROUTER_SCORE_FUNC=sigmod
        export LOAD_BALANCE_TYPE=seq_aux_loss
        export ROUTER_BIAS=true
        export GLOBAL_BATCH_SIZE=15360
        export LR=7.3E-6
        export MIN_LR=0
        export MOE_AUX_LOSS_COEFF=0.0001
        export ROUTER_BIAS_UPDATE_RATE=0
        export TRAIN_TOKENS=278284541614
        export ENABLE_RAMPUP_BS=fix_lr
        ENTRYPOINT="examples/pretrain_ds.sh"
        ;;
    deepseekv2)
        echo ">>> Model: DeepSeek-V2 (236B)"

        case "${EP_CASE}" in
            ep8)
                echo ">>> Parallelism: EP=8, PP=16"
                export MP_PP0_LAYERS=2
                export MP_PPN_LAYERS=2
                export PP=16
                export EP=8
                ;;
            ep16)
                echo ">>> Parallelism: EP=16, PP=8"
                export MP_PPN_LAYERS=4
                export PP=8
                export EP=16
                ;;
            ep32)
                echo ">>> Parallelism: EP=32, PP=8"
                export MP_PPN_LAYERS=4
                export PP_LAYOUT="Ett\|\(tt\|\)*6,t\|\(tt\|\)*7,t\|\(tt\|\)*7,t\|\(tt\|\)*7,tL"
                export PP=8
                export EP=32
                ;;
            *)
                echo "Error: unknown EP '${EP_CASE}'"
                echo "${USAGE}"
                exit 1
                ;;
        esac

        export MODEL_SIZE=200B
        export CPT_PRETRAIN_CHECKPOINT_PATH=/mnt/data/mousun/output/236B-1117/checkpoints/pretrain-dsv3-200B-bs-1-gbs-8192-pp-8-ep-8-ac-full_dlc1byf7k971ff3v
        export TOKENIZER_PATH=/mnt/data/chenqun/zjllm-llama3-tokenizer
        export DATASET_FILE=/mnt/data/mousun/version-20250711.datalist
        export ROUTER_SCORE_FUNC=pre_softmax
        export LOAD_BALANCE_TYPE=aux_loss
        export ROUTER_BIAS=false
        export GLOBAL_BATCH_SIZE=8192
        export INIT_METHOD_STD=0.006
        export LR=3E-6
        export MIN_LR=3E-7
        export TRAIN_TOKENS=1365016803413
        export ENABLE_RAMPUP_BS=false
        export LR_WARMUP_ITERS=100
        ENTRYPOINT="examples/pretrain_ds.sh"
        ;;
    qwen3)
        echo ">>> Model: Qwen3-235B-A22B"

        case "${EP_CASE}" in
            ep8)
                echo ">>> Parallelism: EP=8, PP=8"
                export PP=8
                export EP=8
                export MP_PPN_LAYERS=10
                ;;
            ep16)
                echo ">>> Parallelism: EP=16, PP=8"
                export PP=8
                export EP=16
                export MP_PPN_LAYERS=10
                ;;
            ep32)
                echo ">>> Parallelism: EP=32, PP=8"
                export PP=8
                export EP=32
                export MP_PPN_LAYERS=10
                ;;
            *)
                echo "Error: unknown EP '${EP_CASE}'"
                echo "${USAGE}"
                exit 1
                ;;
        esac

        export MODEL_SIZE=qwen_A22B
        export CPT_PRETRAIN_CHECKPOINT_PATH=/mnt/cpfs/users/lfu/llm/models/Qwen3-235B-A22B-Instruct-2507-mcore/
        export TOKENIZER_TYPE=HuggingFaceTokenizer
        export TOKENIZER_PATH=/mnt/cpfs/users/lfu/llm/models/Qwen3-235B-A22B-Instruct-2507-mcore/
        export DATASET_FILE=/mnt/cpfs/tokens/deepseek-v3/train-sm.datalist
        export ROUTER_SCORE_FUNC=sigmod
        export LOAD_BALANCE_TYPE=seq_aux_loss
        export ROUTER_BIAS=true
        export GLOBAL_BATCH_SIZE=8192
        export LR=7.3E-6
        export MIN_LR=0
        export MOE_AUX_LOSS_COEFF=0.0001
        export ROUTER_BIAS_UPDATE_RATE=0
        export TRAIN_TOKENS=278284541614
        export ENABLE_RAMPUP_BS=fix_lr
        ENTRYPOINT="examples/pretrain_qwen3.sh"
        ;;
    *)
        echo "Error: unknown model '${MODEL_CASE}'"
        echo "${USAGE}"
        exit 1
        ;;
esac

# ===========================================================================
# Run
# ===========================================================================
echo ""
echo ">>> Configuration summary: MODEL=${MODEL_CASE} PRECISION=${PR_CASE} EP=${EP_CASE} AC=${AC_CASE}"
echo ">>> Entrypoint: ${ENTRYPOINT}"
echo ""

bash "${ENTRYPOINT}"
