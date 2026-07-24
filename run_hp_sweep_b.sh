#!/bin/bash
# HP sensitivity sweep -- HALF B (patch_center_weight + fishr_warmup), 5 configs.
# Runs sequentially on ONE GPU. Pair with run_hp_sweep_a.sh on the other GPU.
# All seed 1024, each a single-knob change off v5px_rel_w001.yaml (main run).
#
# Usage (detach so it survives logout):
#   GPU=1 nohup bash run_hp_sweep_b.sh > logs/hp_sweep_b_driver.out 2>&1 &
# Monitor:
#   tail -f logs/hp_sweep_b_driver.out ; tail -f logs/v5px_pc*.out logs/v5px_wu*.out
set -u
cd "$(dirname "$0")"

PY=python
GPU=${GPU:-1}
CONFIGS=(
  v5px_pc0p05 v5px_pc0p2 v5px_pc0p5   # patch_center_weight: 0.05, 0.2, 0.5
  v5px_wu500 v5px_wu3000              # fishr_warmup: 500, 3000
)

echo "HALF B: ${#CONFIGS[@]} configs on GPU ${GPU}, sequential"
for cfg in "${CONFIGS[@]}"; do
  CFG="training/config/detector/_sweep_gen/${cfg}.yaml"
  [ -f "$CFG" ] || { echo "MISSING $CFG"; exit 1; }
  echo "[$(date '+%F %T')] GPU ${GPU}  START  ${cfg}"
  CUDA_VISIBLE_DEVICES=${GPU} "$PY" training/train.py \
      --detector_path "$CFG" --task_target "$cfg" \
      > "logs/${cfg}.out" 2>&1
  echo "[$(date '+%F %T')] GPU ${GPU}  DONE   ${cfg}  (exit $?, log: logs/${cfg}.out)"
done
echo "HALF B complete"
