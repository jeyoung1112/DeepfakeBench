#!/usr/bin/env python3
"""
Precompute train-set spectral statistics for LPSND.

This utility supports three input modes:
  1. --config detector.yaml  (reads train_dataset, compression, resolution,
                              and lpsnd_stats_path directly from the YAML)
  2. --csv file with columns: path,label
  3. --real_dir and --fake_dir class folders

Labels must be:
  0 = real
  1 = fake

The output .npz contains:
  spectral_mean, spectral_std, mu0, mu1, sigma_w, sigma_gamma,
  whitening, v_hat, mu_mid, labels, count_real, count_fake

Use the produced path as lpsnd_stats_path in the detector config.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from scipy.fft import dctn as scipy_dctn
from tqdm import tqdm

SRM_K1 = np.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]], dtype=np.float32)
SRM_K2 = np.array([[0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 1, -4, 1, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0]], dtype=np.float32) / 2.0
SRM_K3 = np.array([[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_spectral_one(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    gray_f = gray_u8.astype(np.float32)

    blurred = cv2.medianBlur(gray_u8, 9).astype(np.float32)
    hp = gray_f - blurred

    fft_img = np.fft.fftshift(np.fft.fft2(gray_f))
    power = np.abs(fft_img) ** 2
    total_e = power.sum() + 1e-12
    fft_log_energy = np.log1p(power.sum())

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    nyq_20 = 0.20 * min(h, w) / 2.0
    fft_hf_ratio = power[radius > nyq_20].sum() / total_e
    prob = power / total_e
    fft_entropy = -np.sum(np.clip(prob, 1e-20, None) * np.log(np.clip(prob, 1e-20, None)))
    fft_centroid = np.sum(radius * power) / total_e

    dct_coeff = scipy_dctn(gray_f, norm="ortho")
    dct_power = np.abs(dct_coeff) ** 2
    dct_total = dct_power.sum() + 1e-12
    dct_log_energy = np.log1p(dct_power.sum())
    dct_hf_ratio = 1.0 - dct_power[:32, :32].sum() / dct_total
    dct_prob = dct_power / dct_total
    dct_entropy = -np.sum(np.clip(dct_prob, 1e-20, None) * np.log(np.clip(dct_prob, 1e-20, None)))

    hp_std = np.std(hp)
    srm_feats = []
    for kernel in (SRM_K1, SRM_K2, SRM_K3):
        residual = cv2.filter2D(gray_f, -1, kernel)
        srm_feats.append(np.mean(np.abs(residual)))

    return np.array([
        fft_log_energy, fft_hf_ratio, fft_entropy, fft_centroid,
        dct_log_energy, dct_hf_ratio, dct_entropy, hp_std,
        *srm_feats,
    ], dtype=np.float64)


def read_image_gray(path: str, resolution: int = None) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resolution is not None and resolution > 0:
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32)


def collect_from_csv(csv_path: str, root: str = None) -> List[Tuple[str, int]]:
    root_path = Path(root) if root else None
    items = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must contain columns: path,label")
        for row in reader:
            p = Path(row["path"])
            if root_path is not None and not p.is_absolute():
                p = root_path / p
            items.append((str(p), int(row["label"])))
    return items


def collect_from_config(
    detector_yaml: str,
    train_config_yaml: str,
    all_frames: bool = False,
) -> Tuple[List[Tuple[str, int]], dict]:
    """Build items list from detector + train_config YAMLs, mirroring abstract_dataset logic."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for --config mode: pip install pyyaml")

    with open(train_config_yaml) as f:
        cfg = yaml.safe_load(f)
    with open(detector_yaml) as f:
        det = yaml.safe_load(f)
    cfg.update(det)  # detector keys override train_config

    rgb_dir = cfg["rgb_dir"]
    json_folder = cfg["dataset_json_folder"]
    label_dict = cfg["label_dict"]
    compression = cfg.get("compression", "c23")
    frame_num = cfg.get("frame_num", {}).get("train", 32)
    train_datasets = cfg["train_dataset"]
    if isinstance(train_datasets, str):
        train_datasets = [train_datasets]

    FF_POOL = {"FaceForensics++", "FaceShifter", "DeepFakeDetection",
               "FF-DF", "FF-F2F", "FF-FS", "FF-NT"}

    items: List[Tuple[str, int]] = []
    for dataset_name in train_datasets:
        json_path = os.path.join(json_folder, dataset_name + ".json")
        with open(json_path) as f:
            dataset_info = json.load(f)

        for split_label, videos in dataset_info[dataset_name].items():
            sub = videos["train"]
            if dataset_name in FF_POOL:
                sub = sub[compression]
            for _vid, vid_info in sub.items():
                raw_label = vid_info["label"]
                if raw_label not in label_dict:
                    continue
                numeric_label = label_dict[raw_label]
                frames = vid_info["frames"]
                frames = [f.replace("\\", "/").lstrip("/") for f in frames]
                frames = sorted(frames, key=lambda x: int(x.split("/")[-1].split(".")[0]))
                if not all_frames:
                    frames = frames[:frame_num]
                for frame in frames:
                    full_path = os.path.join(rgb_dir, frame)
                    items.append((full_path, numeric_label))

    return items, cfg


def collect_from_dirs(real_dir: str, fake_dir: str) -> List[Tuple[str, int]]:
    items = []
    for label, folder in [(0, real_dir), (1, fake_dir)]:
        for p in Path(folder).rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                items.append((str(p), label))
    return items


def inverse_sqrt_psd(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, eps)
    return (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T).astype(np.float32)


def compute_lpsnd_arrays(stats: np.ndarray, labels: np.ndarray, gamma: float = -1.0, eps: float = 1e-6):
    labels = labels.astype(np.int64)
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("Need both labels 0=real and 1=fake to compute LPSND stats.")

    spectral_mean = stats.mean(axis=0)
    spectral_std = stats.std(axis=0)
    spectral_std = np.maximum(spectral_std, eps)
    s_norm = (stats - spectral_mean) / spectral_std

    real = labels == 0
    fake = labels == 1
    mu0 = s_norm[real].mean(axis=0)
    mu1 = s_norm[fake].mean(axis=0)

    centered0 = s_norm[real] - mu0
    centered1 = s_norm[fake] - mu1
    denom = max(int(real.sum() + fake.sum() - 2), 1)
    sigma_w = (centered0.T @ centered0 + centered1.T @ centered1) / denom

    if gamma <= 0:
        gamma = 1e-3 * float(np.trace(sigma_w) / sigma_w.shape[0])
    sigma_gamma = sigma_w + gamma * np.eye(sigma_w.shape[0], dtype=np.float64)
    whitening = inverse_sqrt_psd(sigma_gamma, eps=eps)

    delta = mu1 - mu0
    v = whitening @ delta
    v_hat = v / (np.linalg.norm(v) + eps)
    mu_mid = 0.5 * (mu0 + mu1)

    return {
        "spectral_mean": spectral_mean.astype(np.float32),
        "spectral_std": spectral_std.astype(np.float32),
        "mu0": mu0.astype(np.float32),
        "mu1": mu1.astype(np.float32),
        "sigma_w": sigma_w.astype(np.float32),
        "sigma_gamma": sigma_gamma.astype(np.float32),
        "whitening": whitening.astype(np.float32),
        "v_hat": v_hat.astype(np.float32),
        "mu_mid": mu_mid.astype(np.float32),
        "gamma": np.array(gamma, dtype=np.float32),
        "count_real": np.array(real.sum(), dtype=np.int64),
        "count_fake": np.array(fake.sum(), dtype=np.int64),
    }


def main():
    parser = argparse.ArgumentParser()
    # Config-driven mode
    parser.add_argument("--config", type=str, default=None,
                        help="Path to detector YAML (e.g. config/detector/xception_lpsnd.yaml). "
                             "Reads train_dataset, compression, resolution, and lpsnd_stats_path.")
    parser.add_argument("--train_config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "config", "train_config.yaml"),
                        help="Path to train_config.yaml (default: config/train_config.yaml next to this script)")
    parser.add_argument("--all_frames", action="store_true",
                        help="Use all frames per video instead of capping at frame_num['train']")
    # Manual modes
    parser.add_argument("--csv", type=str, default=None, help="CSV with columns path,label")
    parser.add_argument("--root", type=str, default=None, help="Optional root prefix for relative CSV paths")
    parser.add_argument("--real_dir", type=str, default=None)
    parser.add_argument("--fake_dir", type=str, default=None)
    # Output (optional when --config is used and lpsnd_stats_path is set)
    parser.add_argument("--out", type=str, default=None,
                        help="Output .npz path. Defaults to lpsnd_stats_path from detector YAML.")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Resize images to this square resolution. Defaults to 'resolution' in YAML or 256.")
    parser.add_argument("--gamma", type=float, default=-1.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--max_samples", type=int, default=-1, help="Debug/subsample option")
    args = parser.parse_args()

    cfg_resolution = 256
    if args.config:
        items, merged_cfg = collect_from_config(args.config, args.train_config, args.all_frames)
        cfg_resolution = merged_cfg.get("resolution", 256)
        if args.out is None:
            args.out = merged_cfg.get("lpsnd_stats_path")
        if args.out is None:
            raise ValueError("--out is required when lpsnd_stats_path is not set in the detector YAML")
    elif args.csv:
        items = collect_from_csv(args.csv, args.root)
    elif args.real_dir and args.fake_dir:
        items = collect_from_dirs(args.real_dir, args.fake_dir)
    else:
        raise ValueError("Use --config, --csv, or both --real_dir and --fake_dir")

    resolution = args.resolution if args.resolution is not None else cfg_resolution

    if args.max_samples > 0:
        rng = np.random.default_rng(1024)
        idx = rng.choice(len(items), size=min(args.max_samples, len(items)), replace=False)
        items = [items[i] for i in idx]

    if not items:
        raise ValueError("No images found.")

    stats = []
    labels = []
    skipped = 0
    for path, label in tqdm(items, desc="Extracting spectral stats"):
        try:
            gray = read_image_gray(path, resolution=resolution)
            stats.append(extract_spectral_one(gray))
            labels.append(label)
        except Exception as exc:
            skipped += 1
            print(f"[skip] {path}: {exc}")

    stats = np.stack(stats, axis=0)
    labels = np.asarray(labels, dtype=np.int64)
    arrays = compute_lpsnd_arrays(stats, labels, gamma=args.gamma, eps=args.eps)
    arrays["labels"] = labels

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)

    summary = {
        "out": str(out_path),
        "num_used": int(len(labels)),
        "num_skipped": int(skipped),
        "count_real": int(arrays["count_real"]),
        "count_fake": int(arrays["count_fake"]),
        "gamma": float(arrays["gamma"]),
        "resolution": resolution,
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
