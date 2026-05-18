"""
Evaluate pretrained model under image perturbations:
  - clean (baseline)
  - JPEG compression Q=70
  - JPEG compression Q=30
  - Gaussian blur sigma=2
  - Gaussian noise sigma=0.1
"""
import io
import os
import numpy as np
import random
import yaml
import torch
import torch.backends.cudnn as cudnn
import argparse
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score, roc_curve

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR

parser = argparse.ArgumentParser(description='Test model robustness under image perturbations.')
parser.add_argument('--model_name', type=str)
parser.add_argument('--detector_path', type=str,
                    default='training/config/detector/effort.yaml')
parser.add_argument('--test_dataset', nargs='+')
parser.add_argument('--weights_path', type=str,
                    default='training/pretrained/effort_clip_L14_trainOn_FaceForensic.pth')
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--output_dir', type=str, default='./perturbation_results')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalisation assumed by most vision/CLIP backbones
_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Perturbation helpers — operate on a (B, C, H, W) tensor
# ---------------------------------------------------------------------------

def _denorm(t: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalised (C,H,W) → [0,1] float32."""
    mean = _MEAN[:, None, None].to(t)
    std  = _STD[:, None, None].to(t)
    return (t * std + mean).clamp(0.0, 1.0)


def _renorm(t: torch.Tensor) -> torch.Tensor:
    """[0,1] float32 (C,H,W) → ImageNet-normalised tensor."""
    mean = _MEAN[:, None, None].to(t)
    std  = _STD[:, None, None].to(t)
    return (t - mean) / std


def apply_jpeg_batch(batch: torch.Tensor, quality: int) -> torch.Tensor:
    out = []
    for img in batch:
        img_01 = _denorm(img.cpu())
        pil = TF.to_pil_image(img_01)
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        img_01_comp = TF.to_tensor(Image.open(buf).convert('RGB'))
        out.append(_renorm(img_01_comp).to(batch.device))
    return torch.stack(out)


def apply_blur_batch(batch: torch.Tensor, sigma: float) -> torch.Tensor:
    ks = int(2 * np.ceil(3 * sigma) + 1)
    if ks % 2 == 0:
        ks += 1
    ks = max(ks, 3)
    return TF.gaussian_blur(batch, kernel_size=[ks, ks], sigma=[sigma, sigma])


def apply_noise_batch(batch: torch.Tensor, sigma: float) -> torch.Tensor:
    return batch + torch.randn_like(batch) * sigma


PERTURBATIONS = {
    'jpeg_q70':   lambda x: apply_jpeg_batch(x, quality=70),
    'jpeg_q30':   lambda x: apply_jpeg_batch(x, quality=30),
    'blur_s2':    lambda x: apply_blur_batch(x, sigma=2.0),
    'noise_s01':  lambda x: apply_noise_batch(x, sigma=0.1),
}


# ---------------------------------------------------------------------------
# Boilerplate (mirrors test_confusion_matrix.py)
# ---------------------------------------------------------------------------

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_loader(config, test_name):
        cfg = config.copy()
        cfg['test_dataset'] = test_name
        test_set = DeepfakeAbstractBaseDataset(config=cfg, mode='test')
        return torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config['test_batchSize'],
            shuffle=False,
            num_workers=int(config['workers']),
            collate_fn=test_set.collate_fn,
            drop_last=False,
        )
    return {name: get_loader(config, name) for name in config['test_dataset']}


def load_checkpoint(model, weights_path):
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        ckpt = ckpt['state_dict']
    if all(k.startswith('module.') for k in ckpt.keys()):
        ckpt = {k[len('module.'):]: v for k, v in ckpt.items()}
    model_keys = set(model.state_dict().keys())
    model_expects_backbone = any(k.startswith('backbone.') for k in model_keys)
    if model_expects_backbone and any(k.startswith('pixel_branch.vision.') for k in ckpt.keys()):
        ckpt = {
            k.replace('pixel_branch.vision.', 'backbone.base_model.model.'): v
            for k, v in ckpt.items()
        }
    if (not model_expects_backbone) and any(k.startswith('backbone.') for k in ckpt.keys()):
        ckpt = {k: v for k, v in ckpt.items() if not k.startswith('backbone.')}
    current_sd = model.state_dict()
    missing = {k for k in current_sd if k not in ckpt}
    spectral_keys = {k for k in missing if k.startswith('spectral_projector.')}
    if spectral_keys and not (missing - spectral_keys):
        for k in spectral_keys:
            ckpt[k] = current_sd[k]
        print(f'Backfilled {len(spectral_keys)} spectral_projector buffer(s).')
    model.load_state_dict(ckpt, strict=True)
    print('===> Checkpoint loaded.')


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, data_dict):
    return model(data_dict, inference=True)


def test_perturbed(model, data_loader, perturb_fn):
    preds, labels = [], []
    for data_dict in tqdm(data_loader, total=len(data_loader), leave=False):
        data     = perturb_fn(data_dict['image'])
        label    = torch.where(data_dict['label'] != 0, 1, 0)
        mask     = data_dict['mask']
        landmark = data_dict['landmark']

        data_dict['image'] = data.to(device)
        data_dict['label'] = label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        out = run_inference(model, data_dict)
        labels += list(label.cpu().numpy())
        preds  += list(out['prob'].cpu().numpy())

    return np.array(preds), np.array(labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_pred, y_true, threshold):
    y_bin = (y_pred.squeeze() > threshold).astype(int)
    tp = int(((y_bin == 1) & (y_true == 1)).sum())
    tn = int(((y_bin == 0) & (y_true == 0)).sum())
    fp = int(((y_bin == 1) & (y_true == 0)).sum())
    fn = int(((y_bin == 0) & (y_true == 1)).sum())
    total = tp + tn + fp + fn
    acc  = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp)    if (tp + fp) else 0.0
    rec  = tp / (tp + fn)    if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp)    if (tn + fp) else 0.0
    auc  = float(roc_auc_score(y_true, y_pred.squeeze()))
    fpr_c, tpr_c, _ = roc_curve(y_true, y_pred.squeeze())
    fnr_c   = 1.0 - tpr_c
    eer_idx = np.nanargmin(np.abs(fpr_c - fnr_c))
    eer     = float((fpr_c[eer_idx] + fnr_c[eer_idx]) / 2.0)
    tpr_at_fpr5 = float(tpr_c[np.searchsorted(fpr_c, 0.05, side='right') - 1])
    return dict(
        tp=tp, tn=tn, fp=fp, fn=fn,
        accuracy=acc, precision=prec, recall=rec, f1=f1, specificity=spec,
        auc=auc, eer=eer, tpr_at_fpr5=tpr_at_fpr5,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_perturbation_bar(dataset_results, dataset_name, safe_name):
    pert_names = list(dataset_results.keys())
    aucs = [dataset_results[p]['auc']      for p in pert_names]
    accs = [dataset_results[p]['accuracy'] for p in pert_names]

    x = np.arange(len(pert_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, aucs, w, label='AUC',      color='steelblue')
    ax.bar(x + w/2, accs, w, label='Accuracy', color='darkorange')
    ax.set_xticks(x)
    ax.set_xticklabels(pert_names, rotation=20, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title(f'Perturbation robustness — {dataset_name}')
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8)
    ax.legend()
    plt.tight_layout()

    tag = f"{args.model_name}_{args.test_dataset[0]}" if args.test_dataset else args.model_name
    path = os.path.join(args.output_dir, f"{tag}_pert_bar_{safe_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Bar chart saved → {path}")


def plot_auc_heatmap(all_results):
    datasets = list(all_results.keys())
    perts    = list(next(iter(all_results.values())).keys())
    matrix   = np.array([[all_results[d][p]['auc'] for p in perts] for d in datasets])

    fig, ax = plt.subplots(figsize=(max(6, len(perts) * 1.4), max(4, len(datasets) * 0.8)))
    im = ax.imshow(matrix, vmin=0.4, vmax=1.0, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(perts)));    ax.set_xticklabels(perts, rotation=30, ha='right')
    ax.set_yticks(range(len(datasets))); ax.set_yticklabels(datasets)
    for i in range(len(datasets)):
        for j in range(len(perts)):
            ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, label='AUC')
    ax.set_title(f'AUC — {args.model_name}')
    plt.tight_layout()

    tag = f"{args.model_name}_{args.test_dataset[0]}" if args.test_dataset else args.model_name
    path = os.path.join(args.output_dir, f"{tag}_pert_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  AUC heatmap saved → {path}")


def print_summary_table(all_results):
    perts = list(next(iter(all_results.values())).keys())
    col_w = 12
    header = f"{'Dataset':<32}" + "".join(f"{p:>{col_w}}" for p in perts)
    sep    = '=' * len(header)
    print(f"\n{sep}\nAUC by perturbation\n{header}\n{'-'*len(header)}")
    for ds, results in all_results.items():
        row = f"{ds:<32}" + "".join(f"{results[p]['auc']:>{col_w}.4f}" for p in perts)
        print(row)
    print(sep)


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

    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path

    init_seed(config)
    if config.get('cudnn'):
        cudnn.benchmark = True

    test_data_loaders = prepare_testing_data(config)

    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)

    if config.get('weights_path'):
        load_checkpoint(model, config['weights_path'])
    else:
        print('No weights loaded — using random initialisation.')

    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    tag = f"{args.model_name}_{args.test_dataset[0]}" if args.test_dataset else args.model_name

    for dataset_name, loader in test_data_loaders.items():
        print(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")
        all_results[dataset_name] = {}

        for pert_name, perturb_fn in PERTURBATIONS.items():
            print(f"\n  [{pert_name}]")
            preds, labels = test_perturbed(model, loader, perturb_fn)
            m = compute_metrics(preds, labels, args.threshold)
            m.update(dataset=dataset_name, perturbation=pert_name, threshold=args.threshold)
            all_results[dataset_name][pert_name] = m
            print(f"    acc={m['accuracy']:.4f}  auc={m['auc']:.4f}"
                  f"  eer={m['eer']:.4f}  f1={m['f1']:.4f}  tpr@fpr5={m['tpr_at_fpr5']:.4f}")

        safe_name = dataset_name.replace('/', '_').replace('\\', '_')
        json_path = os.path.join(args.output_dir, f"{tag}_pert_{safe_name}.json")
        with open(json_path, 'w') as f:
            json.dump(all_results[dataset_name], f, indent=2)
        print(f"\n  Metrics saved → {json_path}")

        plot_perturbation_bar(all_results[dataset_name], dataset_name, safe_name)

    plot_auc_heatmap(all_results)
    print_summary_table(all_results)
    print('\n===> All done.')


if __name__ == '__main__':
    main()
