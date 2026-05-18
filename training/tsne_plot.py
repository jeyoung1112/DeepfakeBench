"""
Plot t-SNE of feature embeddings from a deepfake detector,
colored by real/fake label and/or dataset.
"""
import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR


parser = argparse.ArgumentParser(
    description="Plot t-SNE of detector feature embeddings."
)
parser.add_argument(
    "--model_name",
    type=str,
    default="dual_scl",
    help="Name of the model for testing",
)
parser.add_argument(
    "--detector_path",
    type=str,
    default="training/config/detector/dual_branch_scl.yaml",
    help="Path to detector YAML file.",
)
parser.add_argument(
    "--weights_path",
    type=str,
    default="/media/NAS/USERS/jeyoung/checkpoints/FF_trained/covariance_sweep/dual_scl_cov_0.1.pth",
    help="Checkpoint to evaluate.",
)
parser.add_argument(
    "--test_dataset",
    nargs="+",
    default=["FaceForensics++", "Celeb-DF-v2", "DFDC", "DFDCP", "DeepFakeDetection"],
    help="Dataset(s) to test. If omitted, uses detector config test_dataset.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./tsne_results",
    help="Directory to save figures and stats.",
)
parser.add_argument(
    "--max_samples_per_class",
    type=int,
    default=500,
    help="Max real and fake samples each to collect per dataset. 0 = no limit.",
)
parser.add_argument(
    "--perplexity",
    type=float,
    default=30.0,
    help="t-SNE perplexity.",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1000,
    help="t-SNE iterations.",
)
parser.add_argument(
    "--color_by",
    choices=["label", "dataset", "both"],
    default="both",
    help="Color points by real/fake label, dataset, or produce both plots.",
)
parser.add_argument(
    "--pca_components",
    type=int,
    default=50,
    help="Apply PCA before t-SNE to this many components. 0 = skip PCA.",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_seed(config):
    if config["manualSeed"] is None:
        config["manualSeed"] = random.randint(1, 10000)
    random.seed(config["manualSeed"])
    np.random.seed(config["manualSeed"])
    torch.manual_seed(config["manualSeed"])
    if config["cuda"]:
        torch.cuda.manual_seed_all(config["manualSeed"])


def prepare_testing_data(config):
    def get_test_data_loader(base_config, test_name):
        cfg = base_config.copy()
        cfg["test_dataset"] = test_name
        test_set = DeepfakeAbstractBaseDataset(config=cfg, mode="test")
        return torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=cfg["test_batchSize"],
            shuffle=False,
            num_workers=int(cfg["workers"]),
            collate_fn=test_set.collate_fn,
            drop_last=False,
        )

    return {
        dataset_name: get_test_data_loader(config, dataset_name)
        for dataset_name in config["test_dataset"]
    }


def unwrap_checkpoint_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint format is not a valid state_dict dict.")
    return ckpt


def normalize_state_dict_keys(state_dict):
    keys = list(state_dict.keys())
    if not keys:
        return state_dict

    if all(k.startswith("module.") for k in keys):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        keys = list(state_dict.keys())

    if all(k.startswith("model.") for k in keys):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}

    return state_dict


def load_model_weights(model, weights_path):
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = unwrap_checkpoint_state_dict(ckpt)
    state_dict = normalize_state_dict_keys(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
        print("===> Checkpoint loaded with strict=True.")
    except RuntimeError as err:
        print(f"Strict load failed: {err}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("===> Checkpoint loaded with strict=False.")
        if missing:
            print(f"Missing keys ({len(missing)}): {missing[:10]}")
        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}")


@torch.no_grad()
def inference(model, data_dict):
    return model(data_dict, inference=True)


def flatten_features(feat: torch.Tensor) -> np.ndarray:
    """Flatten spatial feature maps to 1-D vectors via global average pooling."""
    feat = feat.detach()
    if feat.ndim == 4:
        # [B, C, H, W] -> [B, C]
        feat = feat.mean(dim=[2, 3])
    elif feat.ndim == 3:
        # [B, T, C] or [B, C, L] -> [B, C]
        feat = feat.mean(dim=1)
    return feat.cpu().numpy().astype(np.float32)


class ReservoirSampler:
    """Reservoir sampling (Algorithm R) for a fixed-size random sample."""

    def __init__(self, k):
        self.k = k
        self.reservoir = []  # list of 1-D numpy arrays
        self.seen = 0

    def add_batch(self, batch: np.ndarray):
        for row in batch:
            self.seen += 1
            if len(self.reservoir) < self.k:
                self.reservoir.append(row.copy())
            else:
                j = random.randint(0, self.seen - 1)
                if j < self.k:
                    self.reservoir[j] = row.copy()

    def result(self):
        return np.stack(self.reservoir, axis=0) if self.reservoir else np.empty((0,), dtype=np.float32)


def collect_features(model, data_loader, max_samples_per_class):
    """Reservoir-sample up to max_samples_per_class real and fake embeddings each."""
    unlimited = max_samples_per_class <= 0
    real_sampler = ReservoirSampler(max_samples_per_class) if not unlimited else None
    fake_sampler = ReservoirSampler(max_samples_per_class) if not unlimited else None
    real_all, fake_all = ([], []) if unlimited else (None, None)

    for data_dict in tqdm(data_loader, total=len(data_loader)):
        data = data_dict["image"]
        label = data_dict["label"]
        mask = data_dict["mask"]
        landmark = data_dict["landmark"]

        binary_label = torch.where(label != 0, 1, 0)

        data_dict["image"] = data.to(device)
        data_dict["label"] = binary_label.to(device)
        if mask is not None:
            data_dict["mask"] = mask.to(device)
        if landmark is not None:
            data_dict["landmark"] = landmark.to(device)

        pred = inference(model, data_dict)

        feat = pred.get("cls_feat")
        if feat is None:
            feat = pred.get("feat")
        if feat is None:
            raise ValueError("Model prediction dict has no 'cls_feat' or 'feat' key.")

        batch_feats = flatten_features(feat)       # [B, D]
        batch_labels = binary_label.cpu().numpy()  # [B]

        real_batch = batch_feats[batch_labels == 0]
        fake_batch = batch_feats[batch_labels == 1]

        if unlimited:
            if len(real_batch): real_all.append(real_batch)
            if len(fake_batch): fake_all.append(fake_batch)
        else:
            if len(real_batch): real_sampler.add_batch(real_batch)
            if len(fake_batch): fake_sampler.add_batch(fake_batch)

    if unlimited:
        real_feats = np.concatenate(real_all, axis=0) if real_all else np.empty((0,), dtype=np.float32)
        fake_feats = np.concatenate(fake_all, axis=0) if fake_all else np.empty((0,), dtype=np.float32)
    else:
        real_feats = real_sampler.result()
        fake_feats = fake_sampler.result()

    feats = np.concatenate([real_feats, fake_feats], axis=0)
    labels = np.array([0] * len(real_feats) + [1] * len(fake_feats), dtype=np.int32)
    print(f"  real={len(real_feats)}, fake={len(fake_feats)}")
    return feats, labels


def run_tsne(feats_all, perplexity, n_iter, pca_components):
    if pca_components > 0 and feats_all.shape[1] > pca_components:
        print(f"Applying PCA: {feats_all.shape[1]}d -> {pca_components}d ...")
        feats_all = PCA(n_components=pca_components, random_state=42).fit_transform(feats_all)
    print(f"Running t-SNE on {len(feats_all)} samples with {feats_all.shape[1]}-dim features...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init="pca",
        random_state=42,
        verbose=1,
    )
    return tsne.fit_transform(feats_all)


def plot_by_label(embedding, labels_all, output_dir):
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    label_names = {0: "Real", 1: "Fake"}
    colors = {0: "#4C72B0", 1: "#C44E52"}

    for lv in [0, 1]:
        mask = labels_all == lv
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=6,
            alpha=0.5,
            color=colors[lv],
            label=f"{label_names[lv]} (n={mask.sum()})",
            linewidths=0,
        )

    ax.set_title("t-SNE colored by Real / Fake")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=2.5, fontsize=9)
    plt.tight_layout()

    path = os.path.join(output_dir, f"{args.model_name}_tsne_by_label.png")
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    with open(args.detector_path, "r") as f:
        config = yaml.safe_load(f)
    with open("./training/config/test_config.yaml", "r") as f:
        config_test = yaml.safe_load(f)
    config.update(config_test)

    if args.test_dataset:
        config["test_dataset"] = args.test_dataset
    if args.weights_path:
        config["weights_path"] = args.weights_path

    init_seed(config)
    if config["cudnn"]:
        cudnn.benchmark = True

    test_data_loaders = prepare_testing_data(config)

    model_class = DETECTOR[config["model_name"]]
    model = model_class(config).to(device)

    if not config.get("weights_path"):
        raise ValueError("weights_path is required.")
    load_model_weights(model, config["weights_path"])
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    all_feats = []
    all_labels = []
    all_dataset_ids = []
    dataset_names = list(test_data_loaders.keys())

    for idx, (dataset_name, loader) in enumerate(test_data_loaders.items()):
        print(f"\nCollecting features for: {dataset_name}")
        feats, labels = collect_features(model, loader, args.max_samples_per_class)
        print(f"  -> {len(feats)} samples, feature dim={feats.shape[1]}")
        all_feats.append(feats)
        all_labels.append(labels)
        all_dataset_ids.append(np.full(len(feats), idx, dtype=np.int32))

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_dataset_ids = np.concatenate(all_dataset_ids, axis=0)

    embedding = run_tsne(all_feats, args.perplexity, args.n_iter, args.pca_components)

    plot_by_label(embedding, all_labels, args.output_dir)

    print("===> Done.")


if __name__ == "__main__":
    main()
