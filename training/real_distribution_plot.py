"""
Plot distribution of predicted real probabilities for real samples
on one or more datasets, with Gaussian fit curves.
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
from tqdm import tqdm

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR


parser = argparse.ArgumentParser(
    description="Plot real-probability distributions with Gaussian fit."
)
parser.add_argument(
    "--detector_path",
    type=str,
    default="training/config/detector/xception.yaml",
    help="Path to detector YAML file.",
)
parser.add_argument(
    "--weights_path",
    type=str,
    default="/media/NAS/USERS/jeyoung/checkpoints/DeepfakeBench/xception_best.pth",
    help="Checkpoint to evaluate.",
)
parser.add_argument(
    "--test_dataset",
    nargs="+",
    default=None,
    help="Dataset(s) to test. If omitted, uses detector config test_dataset.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./real_distribution_results",
    help="Directory to save figures and stats.",
)
parser.add_argument(
    "--bins",
    type=int,
    default=40,
    help="Histogram bins for probability distribution.",
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


def real_probability_from_prediction(pred):
    """Convert model output to P(real), prioritizing class logits if available."""
    cls = pred.get("cls")
    if isinstance(cls, torch.Tensor):
        cls = cls.detach()
        if cls.ndim == 2 and cls.shape[1] >= 2:
            return torch.softmax(cls[:, :2], dim=1)[:, 0].cpu().numpy().astype(np.float32)
        if cls.ndim == 1 or (cls.ndim == 2 and cls.shape[1] == 1):
            fake_prob = torch.sigmoid(cls.reshape(-1)).cpu().numpy()
            return (1.0 - fake_prob).astype(np.float32)

    prob = pred.get("prob")
    if isinstance(prob, torch.Tensor):
        fake_prob = prob.detach().reshape(-1).cpu().numpy()
        return (1.0 - fake_prob).astype(np.float32)

    raise ValueError("Prediction dict must contain tensor key 'cls' or 'prob'.")


def collect_real_probabilities(model, data_loader):
    """Collect predicted real probabilities from GT-real images only."""
    real_probs = []

    for data_dict in tqdm(data_loader, total=len(data_loader)):
        data, label, mask, landmark = (
            data_dict["image"],
            data_dict["label"],
            data_dict["mask"],
            data_dict["landmark"],
        )
        binary_label = torch.where(label != 0, 1, 0)

        real_mask = binary_label == 0
        if not torch.any(real_mask):
            continue

        real_data_dict = dict(data_dict)
        real_data_dict["image"] = data[real_mask].to(device)
        real_data_dict["label"] = binary_label[real_mask].to(device)
        if mask is not None:
            real_data_dict["mask"] = mask[real_mask].to(device)
        if landmark is not None:
            real_data_dict["landmark"] = landmark[real_mask].to(device)

        pred = inference(model, real_data_dict)
        pred_real_prob = real_probability_from_prediction(pred)
        real_probs.extend(pred_real_prob.tolist())

    return np.array(real_probs, dtype=np.float32)


def gaussian_pdf(x, mean, std):
    std = max(float(std), 1e-6)
    coeff = 1.0 / (std * np.sqrt(2.0 * np.pi))
    expo = np.exp(-0.5 * ((x - mean) / std) ** 2)
    return coeff * expo


def save_dataset_plot(dataset_name, real_probs, output_dir, bins):
    os.makedirs(output_dir, exist_ok=True)
    safe_name = dataset_name.replace("/", "_").replace("\\", "_")

    mean = float(np.mean(real_probs))
    std = float(np.std(real_probs))
    count = int(real_probs.shape[0])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(
        real_probs,
        bins=bins,
        range=(0.0, 1.0),
        density=True,
        alpha=0.45,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.4,
        label="Real samples histogram",
    )

    x = np.linspace(0.0, 1.0, 400)
    y = gaussian_pdf(x, mean, std)
    ax.plot(x, y, color="#C44E52", linewidth=2.0, label="Gaussian fit")

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Predicted probability of real class")
    ax.set_ylabel("Density")
    ax.set_title(f"{dataset_name} | mean={mean:.4f}, std={std:.4f}, n={count}")
    ax.legend()
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"{safe_name}_real_prob_gaussian.png")
    plt.savefig(fig_path, dpi=170)
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    return {
        "dataset": dataset_name,
        "num_real_samples": count,
        "mean_real_probability": mean,
        "std_real_probability": std,
        "figure_path": fig_path,
    }


def save_combined_gaussian_plot(stats, output_dir):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = np.linspace(0.0, 1.0, 600)

    for one in stats:
        y = gaussian_pdf(x, one["mean_real_probability"], one["std_real_probability"])
        ax.plot(
            x,
            y,
            linewidth=1.8,
            label=f"{one['dataset']} (mu={one['mean_real_probability']:.3f}, sigma={one['std_real_probability']:.3f})",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Predicted probability of real class")
    ax.set_ylabel("Density")
    ax.set_title("Gaussian fits of real-sample probabilities by dataset")
    ax.legend(fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "all_datasets_real_prob_gaussian_fit.png")
    plt.savefig(save_path, dpi=170)
    plt.close(fig)
    print(f"Saved combined Gaussian plot: {save_path}")


def save_overlapped_histogram_plot(real_probs_by_dataset, output_dir, bins):
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    cmap = plt.get_cmap("tab20")

    for idx, (dataset_name, real_probs) in enumerate(real_probs_by_dataset.items()):
        color = cmap(idx % cmap.N)
        ax.hist(
            real_probs,
            bins=bins,
            range=(0.0, 1.0),
            density=True,
            alpha=0.28,
            color=color,
            edgecolor=color,
            linewidth=0.6,
            label=f"{dataset_name} (n={len(real_probs)})",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Predicted probability of real class")
    ax.set_ylabel("Density")
    ax.set_title("Overlapped real-sample probability distributions by dataset")
    ax.legend(fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "all_datasets_real_prob_hist_overlap.png")
    plt.savefig(save_path, dpi=170)
    plt.close(fig)
    print(f"Saved overlapped histogram plot: {save_path}")


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
        raise ValueError("weights_path is required for evaluation.")
    load_model_weights(model, config["weights_path"])
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    stats = []
    real_probs_by_dataset = {}
    for dataset_name, loader in test_data_loaders.items():
        print(f"\nEvaluating dataset: {dataset_name}")
        real_probs = collect_real_probabilities(model, loader)
        if real_probs.size == 0:
            print(f"Skipping {dataset_name}: no real samples found.")
            continue
        real_probs_by_dataset[dataset_name] = real_probs
        stats.append(save_dataset_plot(dataset_name, real_probs, args.output_dir, args.bins))

    if not stats:
        raise RuntimeError("No real-sample probabilities were collected for plotting.")

    save_combined_gaussian_plot(stats, args.output_dir)
    save_overlapped_histogram_plot(real_probs_by_dataset, args.output_dir, args.bins)

    stats_path = os.path.join(args.output_dir, "real_probability_distribution_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats: {stats_path}")
    print("===> Done.")


if __name__ == "__main__":
    main()
