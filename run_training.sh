#!/bin/bash

set -e

for detector in \
    "./training/config/detector/clip_simple_lora.yaml" \
    "./training/config/detector/dino_lora.yaml"
do
    echo "=========================================="
    echo "Running detector: $detector"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=6 python training/train.py --detector_path "$detector"
done

echo "All training runs completed."
