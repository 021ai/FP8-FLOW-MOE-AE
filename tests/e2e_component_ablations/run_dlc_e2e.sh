#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# FP8-FLOW-MoE (AE) — End-to-end Component Ablations (Paper Section 4.3)
#
# Experiment target:
#   We conduct component ablations on DeepSeek-V3 (671B) under the same setup as
#   the main efficiency evaluation (Section 4.2) with FULL activation
#   checkpointing (AC=full).
#
# Metrics & evaluation (same as Section 4.2):
#   - TGS (tokens/GPU/s): read from Megatron training logs. Ignore iteration 1
#     (warm-up); use iterations 2–6 (TRAIN_ITERS=6).
#   - Peak memory (GB/GPU): measure externally via nvidia-smi/DCGM/cluster monitor.
#
# Usage:
#   bash run_dsv3_component_ablations.sh <perm_pad|double_quant> <ep8|ep16|ep32>
#
# Arguments:
#   1) METHOD_CASE:
#      - perm_pad     : ablation for fused permute-and-padding (disable it)
#      - double_quant : enable DOUBLE_QUANT path (for ablation/comparison)
#   2) EP_CASE:
#      - ep8  : EP=8
#      - ep16 : EP=16
#      - ep32 : EP=32
#
# Notes:
#   - This script contains cluster-specific absolute paths (checkpoint/tokenizer/data).
#     AE reviewers must update them to match their environment.
#   - This script is a thin wrapper that sets env vars and launches:
#       Megatron-LM-FP8FlowMoe/examples/pretrain_ds.sh
# ==============================================================================


# ==============================================================================
# Common environment settings
# ==============================================================================

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


# ==============================================================================
# Parameter parsing
# ==============================================================================

USAGE="Usage: bash $0 <perm_pad|double_quant> <ep8|ep16|ep32>"

METHOD_CASE="${1:-}"
EP_CASE="${2:-}"
if [[ -z "${METHOD_CASE}" || -z "${EP_CASE}" ]]; then
  echo "${USAGE}" >&2
  exit 2
fi

# --- Component ablation selection ---
case "${METHOD_CASE}" in
    perm_pad)
        echo ">>> Ablation: not FUSED_PERM_PAD"
        export FUSED_PERM_PAD=false
        ;;
    double_quant)
        echo ">>> Ablation: DOUBLE_QUANT"
        export DOUBLE_QUANT=true
        ;;
    *)
        echo "Error: unknown method '${METHOD_CASE}'" >&2
        echo "${USAGE}" >&2
        exit 1
        ;;
esac

# --- Parallelism selection ---
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
        echo "Error: unknown EP '${EP_CASE}'" >&2
        echo "${USAGE}" >&2
        exit 1
        ;;
esac


# ==============================================================================
# Fixed experiment settings for Section 4.3 (kept as in the original script)
# ==============================================================================

export PR=blockwise
export FP8_DATAFLOW=true

# Full activation checkpointing for ablations
export AC=full

# DeepSeek-V3 (671B) settings (cluster-specific paths; update for AE environment)
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


# ==============================================================================
# Launch
# ==============================================================================

echo "==================================================================="
echo "FP8-FLOW-MoE AE — DeepSeek-V3 (671B) component ablation run"
echo "  METHOD_CASE  : ${METHOD_CASE}"
echo "  EP_CASE      : ${EP_CASE} (EP=${EP}, PP=${PP})"
echo "  PR           : ${PR}"
echo "  AC           : ${AC}"
echo "  OUTPUT_DIR   : ${OUTPUT_DIR}"
echo "  CHECKPOINT   : ${CPT_PRETRAIN_CHECKPOINT_PATH}"
echo "  TOKENIZER    : ${TOKENIZER_PATH}"
echo "  DATASET_FILE : ${DATASET_FILE}"
echo "==================================================================="

bash examples/pretrain_ds.sh