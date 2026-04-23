#!/bin/bash
# Var+Cov weight sweep for dual_branch_scl detector.
# Round 1: fix cov_weight, sweep var_weight.
# Round 2: fix best var_weight, sweep cov_weight.
# Each run patches a temp config and trains sequentially.

set -e

BASE_CONFIG="./training/config/detector/dual_branch_scl.yaml"
SWEEP_CONFIG_DIR="/tmp/dual_branch_scl_sweep"
GPU=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$SWEEP_CONFIG_DIR"

# ── Sweep grid ────────────────────────────────────────────────────────────────
VAR_WEIGHTS=(0.5 1.0 2.0 5.0)   # Round 1: sweep var_weight, fix cov_weight below
FIXED_COV=0.04

BEST_VAR=1.0                  # Round 2: set after Round 1; sweep cov_weight
COV_WEIGHTS=(0.02 0.04 0.1)

# Set ROUND=1 to sweep var_weight, ROUND=2 to sweep cov_weight, ROUND=all for both
ROUND=${ROUND:-all}
# ─────────────────────────────────────────────────────────────────────────────

patch_and_run() {
    local var_w=$1
    local cov_w=$2

    local tag="var${var_w}_cov${cov_w}"
    local tmp_cfg="${SWEEP_CONFIG_DIR}/${tag}.yaml"

    echo "=========================================="
    echo "Run: $tag  (var_weight=${var_w}, cov_weight=${cov_w})"
    echo "=========================================="

    python3 - <<PYEOF
import yaml

with open("${BASE_CONFIG}") as f:
    cfg = yaml.safe_load(f)

cfg["var_weight"] = ${var_w}
cfg["cov_weight"] = ${cov_w}

with open("${tmp_cfg}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
PYEOF

    CUDA_VISIBLE_DEVICES=${GPU} python3 training/train.py \
        --detector_path "${tmp_cfg}" \
        --task_target "${tag}" \
        --no-save_feat

    echo "Done: $tag"
    echo ""
}

if [ "$ROUND" = "1" ] || [ "$ROUND" = "all" ]; then
    echo "=== Round 1: sweep var_weight (cov_weight fixed at ${FIXED_COV}) ==="
    for vw in "${VAR_WEIGHTS[@]}"; do
        patch_and_run "$vw" "$FIXED_COV"
    done
fi

if [ "$ROUND" = "2" ] || [ "$ROUND" = "all" ]; then
    echo "=== Round 2: sweep cov_weight (var_weight fixed at ${BEST_VAR}) ==="
    for cw in "${COV_WEIGHTS[@]}"; do
        patch_and_run "$BEST_VAR" "$cw"
    done
fi

echo "All runs complete. Configs in: ${SWEEP_CONFIG_DIR}"
echo "Logs in: ./logs/dual_branch/"
