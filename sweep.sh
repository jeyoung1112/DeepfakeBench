#!/bin/bash
# VICReg hyperparameter sweep for dual_branch detector
# Each run generates a temp config with overridden VICReg params.
# Logs are written to ./logs/dual_branch/ with the task_target label.

set -e

BASE_CONFIG="./training/config/detector/dual_branch.yaml"
SWEEP_CONFIG_DIR="/tmp/dual_branch_sweep"
GPU=${CUDA_VISIBLE_DEVICES:-2}

mkdir -p "$SWEEP_CONFIG_DIR"

# ── Sweep grid ────────────────────────────────────────────────────────────────
# Adjust these arrays to change the search space.
VICREG_WEIGHTS=(0.01 0.001)
LAMBDA_INVS=(25.0)
MU_VARS=(25.0)
NU_COVS=(5.0 10.0)

# Set to 1 to sweep all combinations; 0 to sweep each param independently
# (keeping the others at their base-config defaults).
FULL_GRID=0
# ─────────────────────────────────────────────────────────────────────────────

run_training() {
    local vicreg_weight=$1
    local lambda_inv=$2
    local mu_var=$3
    local nu_cov=$4

    local tag="vw${vicreg_weight}_li${lambda_inv}_mv${mu_var}_nc${nu_cov}"
    local tmp_cfg="${SWEEP_CONFIG_DIR}/${tag}.yaml"

    echo "=========================================="
    echo "Run: $tag"
    echo "  vicreg_weight=${vicreg_weight}  lambda_inv=${lambda_inv}"
    echo "  mu_var=${mu_var}  nu_cov=${nu_cov}"
    echo "=========================================="

    # Patch the base config with new VICReg values
    python3 - <<PYEOF
import yaml, copy

with open("${BASE_CONFIG}") as f:
    cfg = yaml.safe_load(f)

cfg["vicreg_weight"] = ${vicreg_weight}
cfg["lambda_inv"]    = ${lambda_inv}
cfg["mu_var"]        = ${mu_var}
cfg["nu_cov"]        = ${nu_cov}

with open("${tmp_cfg}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
PYEOF

    CUDA_VISIBLE_DEVICES=${GPU} python3 training/train.py \
        --detector_path "${tmp_cfg}" \
        --task_target "${tag}" \
        --no-save_feat

    echo "Completed: $tag"
    echo ""
}

# ── Base defaults (read from config for independent sweeps) ───────────────────
BASE_VW=$(python3 -c "import yaml; c=yaml.safe_load(open('${BASE_CONFIG}')); print(c['vicreg_weight'])")
BASE_LI=$(python3 -c "import yaml; c=yaml.safe_load(open('${BASE_CONFIG}')); print(c['lambda_inv'])")
BASE_MV=$(python3 -c "import yaml; c=yaml.safe_load(open('${BASE_CONFIG}')); print(c['mu_var'])")
BASE_NC=$(python3 -c "import yaml; c=yaml.safe_load(open('${BASE_CONFIG}')); print(c['nu_cov'])")

echo "Base defaults: vicreg_weight=${BASE_VW}  lambda_inv=${BASE_LI}  mu_var=${BASE_MV}  nu_cov=${BASE_NC}"
echo ""

if [ "$FULL_GRID" -eq 1 ]; then
    echo "Mode: full grid search (${#VICREG_WEIGHTS[@]} x ${#LAMBDA_INVS[@]} x ${#MU_VARS[@]} x ${#NU_COVS[@]} runs)"
    echo ""
    for vw in "${VICREG_WEIGHTS[@]}"; do
        for li in "${LAMBDA_INVS[@]}"; do
            for mv in "${MU_VARS[@]}"; do
                for nc in "${NU_COVS[@]}"; do
                    run_training "$vw" "$li" "$mv" "$nc"
                done
            done
        done
    done
else
    total=$(( ${#VICREG_WEIGHTS[@]} + ${#NU_COVS[@]} ))
    echo "Mode: independent sweep (${total} runs — vicreg_weight and nu_cov)"
    echo ""

    echo "--- Sweeping vicreg_weight ---"
    for vw in "${VICREG_WEIGHTS[@]}"; do
        run_training "$vw" "$BASE_LI" "$BASE_MV" "$BASE_NC"
    done

    echo "--- Sweeping nu_cov ---"
    for nc in "${NU_COVS[@]}"; do
        run_training "$BASE_VW" "$BASE_LI" "$BASE_MV" "$nc"
    done
fi

echo "All sweep runs completed."
echo "Configs saved in: ${SWEEP_CONFIG_DIR}"
echo "Logs in: ./logs/dual_branch/"
