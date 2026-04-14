"""
Iterate over confusion_config.yaml and run test_fpr_degradation.py for each model.
Mirrors the structure of run_confusion_matrix_all.py.

Usage:
    python training/run_fpr_degradation_all.py
    python training/run_fpr_degradation_all.py --threshold 0.5 --output_dir ./fpr_results
"""
import os
import subprocess
import sys
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='./training/confusion_config_nonvit.yaml')
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--output_dir', type=str,
                    default='./fpr_degradation_results')
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

model_names    = cfg.get('model_name', [])
detector_paths = cfg.get('detector_path', [])
test_datasets  = cfg.get('test_dataset', [])
weights_paths  = cfg.get('weights_path', [])
gpus           = cfg.get('gpu', [])

if len(model_names) != len(detector_paths):
    print(f'WARNING: model_name ({len(model_names)}) and detector_path '
          f'({len(detector_paths)}) counts differ.')

if len(weights_paths) < len(model_names):
    print(f'WARNING: weights_path has {len(weights_paths)} entries but '
          f'model_name has {len(model_names)}. Skipping: '
          f'{model_names[len(weights_paths):]}')

entries = list(zip(model_names, detector_paths, weights_paths))
print(f'Running FPR-degradation evaluation for '
      f'{len(entries)} model(s): {[e[0] for e in entries]}\n')

script = os.path.join(os.path.dirname(__file__), 'test_fpr_degradation.py')

for i, (model_name, detector_path, weights_path) in enumerate(entries):
    gpu_id = str(gpus[i % len(gpus)]) if gpus else '0'

    print(f"\n{'='*60}")
    print(f'  Model:    {model_name}  (GPU {gpu_id})')
    print(f'  Datasets: {test_datasets}')
    print(f'  Detector: {detector_path}')
    print(f'  Weights:  {weights_path}')
    print(f"{'='*60}")

    cmd = [
        sys.executable, script,
        '--model_name',    model_name,
        '--detector_path', detector_path,
        '--weights_path',  weights_path,
        '--test_dataset',  *test_datasets,
        '--threshold',     str(args.threshold),
        '--output_dir',    args.output_dir,
        '--batch_size',    str(args.batch_size),
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f'[WARN] {model_name} exited with code {result.returncode}.')

print('\n===> All models processed.')