#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CKPT_DIR="/media/NAS/USERS/jeyoung/checkpoints/FF_trained"
OUTPUT_DIR="./tsne_results"
DETECTOR_DIR="training/config/detector"
GPU_NO="2"

run_tsne() {
    local model_name="$1"
    local detector_yaml="$2"
    local weights="$3"
    echo "========================================"
    echo "  Running t-SNE: ${model_name}"
    echo "========================================"
    CUDA_VISIBLE_DEVICES="${GPU_NO}" python  training/tsne_plot.py \
        --model_name       "${model_name}" \
        --detector_path    "${DETECTOR_DIR}/${detector_yaml}" \
        --weights_path     "${weights}" \
        --output_dir       "${OUTPUT_DIR}" \
        --color_by         both
}

run_tsne "dual_branch_scl" "dual_branch_scl.yaml" "${CKPT_DIR}/covariance_sweep/dual_scl_cov_0.1.pth"
run_tsne "effort"          "effort.yaml"           "${CKPT_DIR}/effort_ckpt_best.pth"
run_tsne "xception"        "xception.yaml"         "${CKPT_DIR}/xception_ckpt_best.pth"

echo ""
echo "All t-SNE runs complete. Results in: ${OUTPUT_DIR}"
