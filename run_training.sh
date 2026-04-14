#!/bin/bash

set -e

for detector in \
     "./training/config/detector/sbi.yaml" 
do
    echo "=========================================="
    echo "Running detector: $detector"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=2 python training/train.py --detector_path "$detector"
done

echo "All training runs completed."
