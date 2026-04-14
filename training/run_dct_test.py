"""
Iterate over confusion_config.yaml and run test_confusion_matrix.py for each model.
Runs models sequentially, pinning each to a specific GPU.
"""
import argparse
import os
import subprocess
import sys
import yaml


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', default='./training/confusion_config_nonvit.yaml',
                    help='Path to confusion config YAML (default: %(default)s)')
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

model_names    = cfg.get('model_name', [])
detector_paths = cfg.get('detector_path', [])
test_datasets  = cfg.get('test_dataset', [])
weights_paths  = cfg.get('weights_path', [])

if len(model_names) != len(detector_paths):
    print(f"WARNING: model_name ({len(model_names)}) and detector_path ({len(detector_paths)}) counts differ.")

if len(weights_paths) < len(model_names):
    print(
        f"WARNING: weights_path has {len(weights_paths)} entries but model_name has {len(model_names)}. "
        f"Models without weights will be skipped: {model_names[len(weights_paths):]}"
    )

gpus = cfg.get('gpu', [])
entries = list(zip(model_names, detector_paths, weights_paths))

if len(gpus) < len(entries):
    print(f"WARNING: only {len(gpus)} GPU(s) for {len(entries)} model(s) — some will share a GPU.")

print(f"Running confusion matrix evaluation for {len(entries)} model(s): {[e[0] for e in entries]}\n")

for i, (model_name, detector_path, weights_path) in enumerate(entries):
    gpu_id = str(gpus[i % len(gpus)])

    print(f"\n{'='*60}")
    print(f"  Model:    {model_name}  (GPU {gpu_id})")
    print(f"  Datasets: {test_datasets}")
    print(f"  Detector: {detector_path}")
    print(f"  Weights:  {weights_path}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), 'test_clip_lora.py'),
        '--model_name',    model_name,
        '--detector_path', detector_path,
        '--weights_path',  weights_path,
        '--test_dataset',  *test_datasets,
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"[WARN] {model_name} exited with code {result.returncode}.")

print('\n===> All models processed.')
