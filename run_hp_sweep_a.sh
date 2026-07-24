#!/bin/bash
# HP sensitivity sweep -- HALF A (rho-floor + VICReg var_weight), 6 configs.
# Runs sequentially on ONE GPU. Pair with run_hp_sweep_b.sh on the other GPU.
# All seed 1024, each a single-knob change off v5px_rel_w001.yaml (main run).
#
# Usage (detach so it survives logout):
#   GPU=0 nohup bash run_hp_sweep_a.sh > logs/hp_sweep_a_driver.out 2>&1 &
# Monitor:
#   tail -f logs/hp_sweep_a_driver.out ; tail -f logs/v5px_rho*.out
set -u
cd "$(dirname "$0")"

PY=python
GPU=${GPU:-0}
CONFIGS=(
  v5px_rho0 v5px_rho1e4 v5px_rho1e2 v5px_rho1e1   # rho floor: 0, 1e-4, 1e-2, 1e-1
  v5px_var1 v5px_var10                            # var_weight: 1.0, 10.0
)

echo "HALF A: ${#CONFIGS[@]} configs on GPU ${GPU}, sequential"
for cfg in "${CONFIGS[@]}"; do
  CFG="training/config/detector/_sweep_gen/${cfg}.yaml"
  [ -f "$CFG" ] || { echo "MISSING $CFG"; exit 1; }
  echo "[$(date '+%F %T')] GPU ${GPU}  START  ${cfg}"
  CUDA_VISIBLE_DEVICES=${GPU} "$PY" training/train.py \
      --detector_path "$CFG" --task_target "$cfg" \
      > "logs/${cfg}.out" 2>&1
  echo "[$(date '+%F %T')] GPU ${GPU}  DONE   ${cfg}  (exit $?, log: logs/${cfg}.out)"
done
echo "HALF A complete"
