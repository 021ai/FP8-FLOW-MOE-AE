#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# FP8-FLOW-MoE (AE) — Convergence / Loss Validation Runner (Section 4.1)
#
# Purpose:
#   Run a short convergence/loss validation job and compare BF16 vs FP8-FLOW-MoE.
#
# Metrics & where to check loss:
#   - Training loss is logged by Megatron-LM.
#   - In this artifact, loss curves should be inspected via TensorBoard event files
#     under the output log directory, e.g.:
#       ${OUTPUT_DIR}/.../tensorboard/events.out.tfevents.*
#
# Usage:
#   bash run_convergence.sh <bf16|fp8flowmoe>
#
# ==============================================================================


# ==============================================================================
# Parameter parsing
# ==============================================================================
USAGE="Usage: bash $0 <bf16|fp8flowmoe>"
PR_CASE="${1:-}"
if [[ -z "${PR_CASE}" ]]; then
  echo "${USAGE}" >&2
  exit 2
fi

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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=warn
export MANUAL_GC=5
export NUM_WORKERS=6
export NO_MMAP_BIN_FILES=true
export CKPT_FORMAT=torch_dist
export DISPATCHER_TYPE=flex_deepep
export SAVE_INTERVAL=1000
export CHECK_NAN=false


# --- Working directory ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export BASE_DIR=${BASE_DIR:-/mnt/data/FP8FLOW-MOE-AE}
if [[ ! -d "${BASE_DIR}" ]]; then
    echo "Error: BASE_DIR does not exist: ${BASE_DIR}" >&2
    echo "Please set BASE_DIR or BASE_DIR to point to your installation." >&2
    exit 1
fi
export MEGATRON_PATH=${MEGATRON_PATH:-${BASE_DIR}/Megatron-LM-FP8FlowMoe}
if [[ ! -d "${MEGATRON_PATH}" ]]; then
    echo "Error: MEGATRON_PATH does not exist: ${MEGATRON_PATH}" >&2
    echo "Please set BASE_DIR or MEGATRON_PATH to point to your installation." >&2
    exit 1
fi
export OUTPUT_DIR=${BASE_DIR}/output/loss
rm -rf ${OUTPUT_DIR}

export ENV=dlc
export PAO=none
export GLOBAL_BATCH_SIZE=4800
export BATCH_SIZE=5

# For release
export TOKENIZER_PATH=${TOKENIZER_PATH:-/mnt/data/models/DeepSeek-V2-Lite}
export TOKENIZER_TYPE=HuggingFaceTokenizer
export DATASET_FILE=${DATASET_FILE:-/dev/null}  # 021-v1-8t+fweduv120+fwdomain1wp+nemosynth-optm+stackv2_train_full+cot_math.datalist
export MOCK_DATASET=${MOCK_DATASET:-true}  # use mock data
if [[ ! -d "${TOKENIZER_PATH}" ]]; then
    echo "Error: TOKENIZER_PATH does not exist: ${TOKENIZER_PATH}" >&2
    echo "Please set TOKENIZER_PATH or TOKENIZER_PATH to point to your installation." >&2
    exit 1
fi

# ==============================================================================
# Precision selection
# ==============================================================================
case "${PR_CASE}" in
  bf16)
    echo ">>> Precision: BF16 baseline"
    export PR=bf16
    ;;
  fp8flowmoe)
    echo ">>> Precision: FP8-FLOW-MoE (blockwise + FP8 dataflow)"
    export PR=blockwise
    export FP8_DATAFLOW=true
    ;;
  *)
    echo "Error: unknown precision '${PR_CASE}'" >&2
    echo "${USAGE}" >&2
    exit 1
    ;;
esac


# ==============================================================================
# Run
# ==============================================================================
echo ""
echo ">>> Run loss configuration summary: PRECISION=${PR_CASE}"
echo ">>> OUTPUT_DIR=${OUTPUT_DIR}"
echo ""

bash "${SCRIPT_DIR}/run_16B_loss.sh"