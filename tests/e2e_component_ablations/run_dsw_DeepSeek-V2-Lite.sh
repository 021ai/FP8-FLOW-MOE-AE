#!/bin/bash
# Component ablation smoke-test: DeepSeek-V2-Lite (16B)
# Requires: 1 node (8 GPUs)
#
# Usage:
#   bash run_dsw_DeepSeek-V2-Lite.sh <CASE>
#
# Cases:
#   fp8flowmoe   - FP8-FLOW-MoE baseline (all optimizations enabled)
#   perm_pad     - Ablation: disable fused permute-and-padding
#   double_quant - Ablation: disable scaling-aware transpose (use double quantization)

# --- Working directory ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# method case selection
# ---------------------------------------------------------------------------
USAGE="Usage: bash $0 <fp8flowmoe|perm_pad|double_quant>"

METHOD_CASE="${1:-}"
if [[ -z "${METHOD_CASE}" ]]; then
  echo "${USAGE}" >&2
  exit 2
fi
# --- Component ablation selection ---
case "${METHOD_CASE}" in
    fp8flowmoe)
        echo ">>> Ablation: fp8flowmoe"
        # export FUSED_PERM_PAD=true  # default
        # export SCALING_AWARE_TRANSPOSE=true  # default
        ;;
    perm_pad)
        echo ">>> Ablation: not FUSED_PERM_PAD"
        export FUSED_PERM_PAD=false
        ;;
    double_quant)
        echo ">>> Ablation: DOUBLE_QUANT"
        export SCALING_AWARE_TRANSPOSE=false
        ;;
    *)
        echo ">>> Ablation: not support"
        echo "${USAGE}" >&2
        exit 1
        ;;
esac

bash "${SCRIPT_DIR}/../e2e_efficiency_evaluation/run_dsw_DeepSeek-V2-Lite.sh" fp8flowmoe