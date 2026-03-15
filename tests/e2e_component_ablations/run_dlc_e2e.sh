#!/usr/bin/env bash
set -euo pipefail

# --- Working directory ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# method case selection
# ---------------------------------------------------------------------------
USAGE="Usage: bash $0 <fp8flowmoe|perm_pad|double_quant> <ep8|ep16|ep32>"

METHOD_CASE="${1:-}"
EP_CASE="${2:-}"
if [[ -z "${METHOD_CASE}" || -z "${EP_CASE}" ]]; then
  echo "${USAGE}" >&2
  exit 2
fi
# --- Component ablation selection ---
case "${METHOD_CASE}" in
    fp8flowmoe)
        echo ">>> dlc ablation: fp8flowmoe"
        # export FUSED_PERM_PAD=true  # default
        # export SCALING_AWARE_TRANSPOSE=true  # default
        ;;
    perm_pad)
        echo ">>> dlc ablation: not FUSED_PERM_PAD"
        export FUSED_PERM_PAD=false
        ;;
    double_quant)
        echo ">>> dlc ablation: DOUBLE_QUANT"
        export SCALING_AWARE_TRANSPOSE=false
        ;;
    *)
        echo ">>> dlc ablation: not support"
        echo "${USAGE}" >&2
        exit 1
        ;;
esac

bash "${SCRIPT_DIR}/../e2e_efficiency_evaluation/run_dlc_e2e.sh" deepseekv3 fp8flowmoe "${EP_CASE}" full