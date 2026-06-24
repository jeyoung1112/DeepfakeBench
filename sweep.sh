#!/bin/bash
# Var+Cov weight sweep for dual_branch_scl detector.
#
# ROUND=grid    — full 2D grid (default); use for sensitivity analysis
# ROUND=resume  — restart from (5.0, 0.25), then var=7 and var=10
# ROUND=1       — fix cov at FIXED_COV, sweep var_weight only
# ROUND=2       — fix var at BEST_VAR,  sweep cov_weight only
#
# Best known so far: var=5.0, cov=0.1

set -e

BASE_CONFIG="./training/config/detector/dual_branch_scl.yaml"
SWEEP_CONFIG_DIR="/tmp/dual_branch_scl_sweep"
GPU=${CUDA_VISIBLE_DEVICES:-1}

mkdir -p "$SWEEP_CONFIG_DIR"

# ── Sweep grid ────────────────────────────────────────────────────────────────
# 2D grid centered on the current best (var=5, cov=0.1).
# Covariance is log-spaced; variance covers a ×5 range around the best.
VAR_WEIGHTS=(2.0 5.0 7.0 10.0)
COV_WEIGHTS=(0.04 0.08 0.1 0.15 0.25)

# Coordinate-descent anchors (used when ROUND=1 or ROUND=2)
FIXED_COV=0.1        # anchor for Round 1 — use the current best, not 0.04
BEST_VAR=5.0         # anchor for Round 2
VAR_WEIGHTS_R1=(2.0 5.0 7.0 10.0)
COV_WEIGHTS_R2=(0.04 0.08 0.1 0.15 0.25)

ROUND=${ROUND:-grid}
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

if [ "$ROUND" = "grid" ]; then
    total=$(( ${#VAR_WEIGHTS[@]} * ${#COV_WEIGHTS[@]} ))
    echo "=== 2D grid sweep: ${#VAR_WEIGHTS[@]} var × ${#COV_WEIGHTS[@]} cov = ${total} runs ==="
    for vw in "${VAR_WEIGHTS[@]}"; do
        for cw in "${COV_WEIGHTS[@]}"; do
            patch_and_run "$vw" "$cw"
        done
    done
fi

if [ "$ROUND" = "resume" ]; then
    echo "=== Resume: var=10.0, cov in (0.08 0.1 0.15 0.25) ==="
    for cw in 0.08 0.1 0.15 0.25; do
        patch_and_run 10.0 "$cw"
    done
fi

if [ "$ROUND" = "1" ]; then
    echo "=== Round 1: sweep var_weight (cov_weight fixed at ${FIXED_COV}) ==="
    for vw in "${VAR_WEIGHTS_R1[@]}"; do
        patch_and_run "$vw" "$FIXED_COV"
    done
fi

if [ "$ROUND" = "2" ]; then
    echo "=== Round 2: sweep cov_weight (var_weight fixed at ${BEST_VAR}) ==="
    for cw in "${COV_WEIGHTS_R2[@]}"; do
        patch_and_run "$BEST_VAR" "$cw"
    done
fi

echo "All runs complete. Configs in: ${SWEEP_CONFIG_DIR}"
echo "Logs in: ./logs/dual_branch/"
