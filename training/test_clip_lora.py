"""
Evaluate pretrained model and compute confusion matrix per dataset.
"""
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
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
from metrics.utils import get_test_metrics
from scipy.fft import dctn

parser = argparse.ArgumentParser(description='Test with confusion matrix output.')
parser.add_argument('--model_name', type=str)
parser.add_argument('--detector_path', type=str,
                    default='training/config/detector/effort.yaml',
                    help='path to detector YAML file')
parser.add_argument('--test_dataset', nargs='+')
parser.add_argument('--weights_path', type=str,
                    default='training/pretrained/effort_clip_L14_trainOn_FaceForensic.pth')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='probability threshold for positive (fake) class')
parser.add_argument('--output_dir', type=str, default='./confusion_matrix_results',
                    help='directory to save confusion matrix plots')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        config = config.copy()
        config['test_dataset'] = test_name
        test_set = DeepfakeAbstractBaseDataset(config=config, mode='test')
        return torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config['test_batchSize'],
            shuffle=False,
            num_workers=int(config['workers']),
            collate_fn=test_set.collate_fn,
            drop_last=False,
        )

    return {name: get_test_data_loader(config, name) for name in config['test_dataset']}


@torch.no_grad()
def inference(model, data_dict):
    return model(data_dict, inference=True)


def test_one_dataset(model, data_loader):
    prediction_lists, label_lists, feature_lists = [], [], []
    for data_dict in tqdm(data_loader, total=len(data_loader)):
        data, label, mask, landmark = (
            data_dict['image'], data_dict['label'],
            data_dict['mask'], data_dict['landmark'],
        )
        label = torch.where(data_dict['label'] != 0, 1, 0)
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        predictions = inference(model, data_dict)
        label_lists     += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists   += list(predictions['feat'].cpu().detach().numpy())

    return np.array(prediction_lists), np.array(label_lists), np.array(feature_lists)


def save_image_paths_by_category(y_pred, y_true, img_paths, threshold, dataset_name, output_dir):
    """Categorise image paths into TP/TN/FP/FN and save as JSON."""
    y_binary = (y_pred.squeeze() > threshold).astype(int)
    img_paths = np.array(img_paths)

    categorised = {
        "tp": img_paths[(y_binary == 1) & (y_true == 1)].tolist(),
        "tn": img_paths[(y_binary == 0) & (y_true == 0)].tolist(),
        "fp": img_paths[(y_binary == 1) & (y_true == 0)].tolist(),
        "fn": img_paths[(y_binary == 0) & (y_true == 1)].tolist(),
    }

    os.makedirs(output_dir, exist_ok=True)
    safe_name = dataset_name.replace('/', '_').replace('\\', '_')
    json_path = os.path.join(output_dir, f"{args.model_name}_{safe_name}_image_paths.json")
    with open(json_path, 'w') as f:
        json.dump(categorised, f, indent=2)
    print(f"  Image paths saved → {json_path}  "
          f"(TP={len(categorised['tp'])} TN={len(categorised['tn'])} "
          f"FP={len(categorised['fp'])} FN={len(categorised['fn'])})")

def _dct_heatmap(images_chw):
    """Average log-DCT heatmap over a batch of CHW images."""
    if len(images_chw) == 0:
        return None
    maps = []
    for img in images_chw:
        # CHW → grayscale HW by averaging channels
        gray = img.mean(axis=0)
        # Normalise to [0,1] so different normalisations don't distort the DCT
        gray = (gray - gray.min()) / (gray.ptp() + 1e-8)
        dct = dctn(gray, norm='ortho')
        maps.append(np.log(np.abs(dct) + 1e-8))
    return np.mean(maps, axis=0)

def plot_dct_heatmaps(images, labels, y_pred, threshold, dataset_name, output_dir, model_name):
    """
    Plots 4-panel DCT heatmap: Real | Fake | FP | FN
    """
    y_binary = (y_pred.squeeze() > threshold).astype(int)
    is_real = labels == 0
    is_fake = labels == 1
    fp_mask = (y_binary == 1) & is_real   # predicted fake, actually real
    fn_mask = (y_binary == 0) & is_fake   # predicted real, actually fake

    groups = {
        f'Real  (n={is_real.sum()})': images[is_real],
        f'Fake  (n={is_fake.sum()})': images[is_fake],
        f'FP    (n={fp_mask.sum()})': images[fp_mask],
        f'FN    (n={fn_mask.sum()})': images[fn_mask],
    }

    heatmaps = {title: _dct_heatmap(imgs) for title, imgs in groups.items()}

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f'DCT heatmaps — {model_name} | {dataset_name}', fontsize=12)

    vmin = min(h.min() for h in heatmaps.values() if h is not None)
    vmax = max(h.max() for h in heatmaps.values() if h is not None)

    for ax, (title, hmap) in zip(axes, heatmaps.items()):
        if hmap is None:
            ax.set_title(f'{title}\n(no samples)')
            ax.axis('off')
            continue
        im = ax.imshow(hmap, cmap='inferno', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    safe_name = dataset_name.replace('/', '_').replace('\\', '_')
    save_path = os.path.join(output_dir, f'{model_name}_{safe_name}_dct_heatmap.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'  DCT heatmap saved → {save_path}')

def compute_and_display_confusion_matrix(y_pred, y_true, dataset_name, threshold, output_dir):
    y_binary = (y_pred.squeeze() > threshold).astype(int)
    cm = confusion_matrix(y_true, y_binary)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*50}")
    print(f"Confusion Matrix — {dataset_name}  (threshold={threshold})")
    print(f"{'='*50}")
    print(f"                Predicted")
    print(f"                Real   Fake")
    print(f"Actual  Real  [{tn:>6} {fp:>6}]")
    print(f"        Fake  [{fn:>6} {tp:>6}]")
    print(f"\n  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    accuracy = (tp + fp) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"  Accuracy={accuracy:.3f}  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}  Specificity={specificity:.3f}")

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    # fig, ax = plt.subplots(figsize=(5, 4))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
    # disp.plot(ax=ax, colorbar=True, cmap='Blues')
    # ax.set_title(f"{dataset_name}\n(threshold={threshold})")
    safe_name = dataset_name.replace('/', '_').replace('\\', '_')
    # save_path = os.path.join(output_dir, f"{args.model_name}_{args.test_dataset[0]}_cm_{safe_name}.png")
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=150)
    # plt.close()
    # print(f"  Plot saved → {save_path}")

    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_pred.squeeze())
    sklearn_auc = roc_auc_score(y_true, y_pred.squeeze())
    fnr_curve = 1.0 - tpr_curve
    eer_idx = np.nanargmin(np.abs(fpr_curve - fnr_curve))
    eer = (fpr_curve[eer_idx] + fnr_curve[eer_idx]) / 2.0
    tpr_at_fpr1 = tpr_curve[np.searchsorted(fpr_curve, 0.01, side='right') - 1]
    tpr_at_fpr5 = tpr_curve[np.searchsorted(fpr_curve, 0.05, side='right') - 1]
    fpr_at_tpr1 = fpr_curve[np.searchsorted(tpr_curve, 0.01, side='left')]
    fpr_at_tpr5 = fpr_curve[np.searchsorted(tpr_curve, 0.05, side='left')]

    metrics = {
    "dataset": dataset_name,
    "threshold": threshold,
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    "accuracy": accuracy, "precision": precision,
    "recall": recall, "f1": f1, "specificity": specificity,
    "sklearn_auc": float(sklearn_auc),
    "eer": float(eer),
    "tpr_at_fpr1pct": float(tpr_at_fpr1),
    "tpr_at_fpr5pct": float(tpr_at_fpr5)
    }
    json_path = os.path.join(output_dir, f"{args.model_name}_{args.test_dataset[0]}_cm_{safe_name}.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {json_path}")

    return cm


def plot_all_score_distributions(results, threshold, output_dir):
    """
    All datasets overlaid on a single plot.
    Real = solid line, Fake = dashed line, one color per dataset.
    results: list of (dataset_name, y_pred, y_true)
    """
    palette = sns.color_palette('tab10', n_colors=len(results))
    fig, ax = plt.subplots(figsize=(8, 5))

    for color, (dataset_name, y_pred, y_true) in zip(palette, results):
        probs = y_pred.squeeze()
        real_probs = probs[y_true == 0]
        fake_probs = probs[y_true == 1]
        sns.kdeplot(real_probs, ax=ax, color=color, linestyle='-',  linewidth=1.8,
                    label=f'{dataset_name} — real', clip=(0, 1))
        sns.kdeplot(fake_probs, ax=ax, color=color, linestyle='--', linewidth=1.8,
                    label=f'{dataset_name} — fake', clip=(0, 1))

    ax.axvline(threshold, color='black', linestyle=':', linewidth=1.2, label=f'threshold={threshold}')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Predicted probability (fake)')
    ax.set_ylabel('Density')
    ax.set_title(f'Score distributions — {args.model_name}')
    ax.legend(fontsize=8)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{args.model_name}_kde_all.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Score distributions saved → {save_path}")


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
    if config['cudnn']:
        cudnn.benchmark = True

    test_data_loaders = prepare_testing_data(config)

    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)

    weights_path = config.get('weights_path')
    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)
        # Strip 'module.' prefix added by nn.DataParallel during training
        if all(k.startswith('module.') for k in ckpt.keys()):
            ckpt = {k[len('module.'):]: v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=True)
        print('===> Checkpoint loaded.')
    else:
        print('No weights loaded — using random initialisation.')

    model.eval()

    all_cms = {}
    score_results = []
    for dataset_name, loader in test_data_loaders.items():
        print(f"\nTesting on: {dataset_name}")
        predictions_nps, label_nps, _ = test_one_dataset(model, loader)

        # Standard metrics (auc, eer, acc, ap)
        data_dict = loader.dataset.data_dict
        metric = get_test_metrics(
            y_pred=predictions_nps, y_true=label_nps, img_names=data_dict['image']
        )
        tqdm.write(f"  acc={metric['acc']:.4f}  auc={metric['auc']:.4f}"
                   f"  eer={metric['eer']:.4f}  ap={metric['ap']:.4f}")

        # Confusion matrix
        cm = compute_and_display_confusion_matrix(
            predictions_nps, label_nps, dataset_name,
            threshold=args.threshold, output_dir=args.output_dir,
        )
        all_cms[dataset_name] = cm
        score_results.append((dataset_name, predictions_nps, label_nps))

        save_image_paths_by_category(
            predictions_nps, label_nps, data_dict['image'],
            threshold=args.threshold,
            dataset_name=dataset_name,
            output_dir=args.output_dir,
        )

    # Score distributions — all datasets in one figure
    # plot_all_score_distributions(score_results, threshold=args.threshold, output_dir=args.output_dir)
    print('\n===> All done.')


if __name__ == '__main__':
    main()
