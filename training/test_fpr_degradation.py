"""
Compute FPR on real images under three families of degradation:
  - JPEG compression at varying quality levels
  - Gaussian blur at varying sigma values
  - Unsharp-mask sharpening at varying strength values

Only real images (label == 0) from each test dataset are used, so every
positive prediction is a false positive.  FPR = (# predicted fake) / (# real).

Usage (standalone):
    python training/test_fpr_degradation.py \
        --model_name clip_probing \
        --detector_path ./training/config/detector/clip_probing.yaml \
        --weights_path /path/to/ckpt.pth \
        --test_dataset FaceForensics++ DFDCP Celeb-DF-v2 DFDC DeepFakeDetection \
        --output_dir ./fpr_degradation_results

Or run via run_fpr_degradation_all.py which reads confusion_config.yaml.
"""

import os
import sys
import random
import argparse
import json
import warnings
from copy import deepcopy

import numpy as np
import cv2
import yaml
import torch
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append('.')

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description='Compute FPR under JPEG / blur / sharpening degradations (real images only).'
)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--detector_path', type=str, required=True,
                    help='Path to detector YAML config.')
parser.add_argument('--test_dataset', nargs='+', required=True)
parser.add_argument('--weights_path', type=str, default=None)
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Probability threshold for fake class.')
parser.add_argument('--output_dir', type=str, default='./fpr_degradation_results')
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Degradation levels
# ---------------------------------------------------------------------------

JPEG_QUALITIES = [10, 20, 30, 50, 70, 90]   # lower = more compression
BLUR_SIGMAS    = [0.5, 1.0, 2.0, 3.0, 5.0]
SHARPEN_AMOUNTS = [0.5, 1.0, 2.0, 3.0, 5.0]


def apply_jpeg(img_np: np.ndarray, quality: int) -> np.ndarray:
    """JPEG encode then decode (HWC uint8 RGB)."""
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img_np
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def apply_blur(img_np: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with the given sigma (HWC uint8 RGB)."""
    k = max(3, int(6 * sigma + 1))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img_np, (k, k), sigma)


def apply_sharpen(img_np: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp masking: sharpened = img + amount * (img - blurred(img, sigma=3))."""
    blurred = cv2.GaussianBlur(img_np, (0, 0), 3)
    sharpened = cv2.addWeighted(img_np, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class RealOnlyDegradedDataset(DeepfakeAbstractBaseDataset):
    """
    Wraps DeepfakeAbstractBaseDataset keeping only real (label==0) images.
    Applies an optional degradation function between image load and normalise.
    """

    def __init__(self, config: dict, degradation_fn=None):
        super().__init__(config=config, mode='test')

        # Keep only real images
        real_idx = [i for i, lbl in enumerate(self.label_list) if lbl == 0]
        self.image_list = [self.image_list[i] for i in real_idx]
        self.label_list = [0] * len(self.image_list)
        self.data_dict  = {'image': self.image_list, 'label': self.label_list}

        self.degradation_fn = degradation_fn

    def __getitem__(self, index):
        image_path = self.data_dict['image'][index]
        label      = self.data_dict['label'][index]   # always 0

        # Handle video-level (clip) entries – just use the first frame
        if isinstance(image_path, list):
            image_path = image_path[0]

        # Same path normalisation as the parent class
        if 'dataset' in image_path:
            image_path = image_path.split('dataset')[-1]
        image_path = image_path.replace('\\', '/').lstrip('/')

        try:
            image = self.load_rgb(image_path)   # PIL image
        except Exception as e:
            warnings.warn(f'Skipping {image_path}: {e}')
            return self.__getitem__(0)

        image = np.array(image, dtype=np.uint8)   # HWC RGB

        if self.degradation_fn is not None:
            image = self.degradation_fn(image)

        image_tensor = self.normalize(self.to_tensor(image))
        return image_tensor, label, None, None

    def __len__(self):
        return len(self.image_list)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, dataset, batch_size: int, num_workers: int = 4) -> np.ndarray:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )
    all_probs = []
    for data_dict in tqdm(loader, leave=False, desc='  infer'):
        imgs   = data_dict['image'].to(device)
        labels = torch.zeros(imgs.size(0), dtype=torch.long, device=device)
        d = {'image': imgs, 'label': labels, 'mask': None, 'landmark': None}
        out = model(d, inference=True)
        all_probs.extend(out['prob'].cpu().numpy().tolist())
    return np.array(all_probs, dtype=np.float32)


def compute_fpr(probs: np.ndarray, threshold: float) -> float:
    """All items are real; FPR = fraction predicted as fake."""
    return float((probs.squeeze() > threshold).mean())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(all_results: dict, output_dir: str, model_name: str):
    deg_meta = {
        'jpeg':    (JPEG_QUALITIES,    'JPEG Quality',       'lower = more compressed'),
        'blur':    (BLUR_SIGMAS,       'Gaussian Blur σ',    ''),
        'sharpen': (SHARPEN_AMOUNTS,   'Sharpen Amount',     ''),
    }

    for deg_type, (levels, xlabel, note) in deg_meta.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for dataset_name, results in all_results.items():
            if deg_type not in results:
                continue
            fprs = [results[deg_type].get(k, float('nan'))
                    for k in results[deg_type]]
            x    = levels[:len(fprs)]
            ax.plot(x, fprs, marker='o', label=dataset_name)

        ax.set_xlabel(f'{xlabel}' + (f'  ({note})' if note else ''))
        ax.set_ylabel('FPR')
        ax.set_ylim(0, 1)
        ax.set_title(f'{model_name} — FPR vs {xlabel}')
        ax.legend(fontsize=8)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{model_name}_fpr_{deg_type}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f'  plot  → {save_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']

    config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path

    # Seed
    seed = config.get('manualSeed') or 42
    random.seed(seed)
    torch.manual_seed(seed)
    if config.get('cuda'):
        torch.cuda.manual_seed_all(seed)
    if config.get('cudnn'):
        cudnn.benchmark = True

    # Load model
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    weights_path = config.get('weights_path')
    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)
        if all(k.startswith('module.') for k in ckpt.keys()):
            ckpt = {k[len('module.'):]: v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=True)
        print('===> Checkpoint loaded.')
    else:
        print('No weights — using random initialisation.')
    model.eval()

    num_workers = int(config.get('workers', 4))

    # Degradation schedule
    degradations = [
        ('clean',   [(None, None)]),
        ('jpeg',    [(f'Q{q}',      lambda img, q=q: apply_jpeg(img, q))   for q in JPEG_QUALITIES]),
        ('blur',    [(f's{s}',      lambda img, s=s: apply_blur(img, s))   for s in BLUR_SIGMAS]),
        ('sharpen', [(f'a{a:.1f}',  lambda img, a=a: apply_sharpen(img, a)) for a in SHARPEN_AMOUNTS]),
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for dataset_name in args.test_dataset:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        all_results[dataset_name] = {}

        cfg = config.copy()
        cfg['test_dataset'] = dataset_name

        for deg_type, levels in degradations:
            all_results[dataset_name][deg_type] = {}

            for level_key, fn in levels:
                ds = RealOnlyDegradedDataset(cfg, degradation_fn=fn)
                n_real = len(ds)
                if n_real == 0:
                    print(f'  [{deg_type:8s}] {str(level_key):8s}  — no real images, skipped')
                    continue

                probs = run_inference(model, ds,
                                      batch_size=args.batch_size,
                                      num_workers=num_workers)
                fpr = compute_fpr(probs, args.threshold)
                key = level_key if level_key is not None else 'clean'
                all_results[dataset_name][deg_type][key] = fpr
                print(f'  [{deg_type:8s}] {str(key):8s}  '
                      f'FPR={fpr:.4f}  n_real={n_real}')

    # Save JSON
    json_path = os.path.join(args.output_dir,
                             f'{args.model_name}_fpr_degradation.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nJSON → {json_path}')

    # Plots
    plot_results(all_results, args.output_dir, args.model_name)
    print('===> Done.')


if __name__ == '__main__':
    main()