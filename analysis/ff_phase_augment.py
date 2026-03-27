"""
Phase spectra of real images under photometric/compression augmentations.

Loads real frames from FaceForensics++, applies a set of augmentation
variants, and compares their phase coherence and mean phase maps.

Augmentation groups:
  baseline     — no change
  bright+      — gamma 0.6  (brighter)
  bright-      — gamma 1.8  (darker)
  blur_s       — Gaussian blur σ=1
  blur_m       — Gaussian blur σ=2
  blur_l       — Gaussian blur σ=4
  sharp_s      — unsharp-mask strength 0.5
  sharp_m      — unsharp-mask strength 1.0
  sharp_l      — unsharp-mask strength 2.0
  jpeg_90      — JPEG quality 90
  jpeg_75      — JPEG quality 75
  jpeg_50      — JPEG quality 50
  jpeg_30      — JPEG quality 30

Usage:
    python ff_phase_augment.py
    python ff_phase_augment.py --max 200 --out ff_phase_augment.png
    python ff_phase_augment.py --groups baseline,blur_s,blur_l,jpeg_50 --radial
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

JSON_PATH = "/media/NAS/DATASET/DeepfakeBench/original/dataset_json/FaceForensics++.json"
BASE_DIR  = "/media/NAS/DATASET/DeepfakeBench/original/dataset"
REAL_KEY  = "FF-real"

# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def _gamma(img: np.ndarray, g: float) -> np.ndarray:
    lut = (255.0 * (np.arange(256) / 255.0) ** g).astype(np.uint8)
    return lut[img]


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    # kernel size: smallest odd number >= 6*sigma
    k = int(np.ceil(sigma * 6)) | 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)


def _unsharp_mask(img: np.ndarray, sigma: float = 1.0, strength: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
    sharp = img.astype(np.float32) + strength * (img.astype(np.float32) - blur)
    return np.clip(sharp, 0, 255).astype(np.uint8)


# Each entry: (label, callable img→img)
ALL_AUGMENTS: dict[str, tuple[str, callable]] = {
    "baseline":  ("Baseline",       lambda img: img),
    "bright+":   ("Bright+ (γ=0.6)",lambda img: _gamma(img, 0.6)),
    "bright-":   ("Bright− (γ=1.8)",lambda img: _gamma(img, 1.8)),
    "blur_s":    ("Blur σ=1",        lambda img: _gaussian_blur(img, 1.0)),
    "blur_m":    ("Blur σ=2",        lambda img: _gaussian_blur(img, 2.0)),
    "blur_l":    ("Blur σ=4",        lambda img: _gaussian_blur(img, 4.0)),
    "sharp_s":   ("Sharp ×0.5",      lambda img: _unsharp_mask(img, 1.0, 0.5)),
    "sharp_m":   ("Sharp ×1.0",      lambda img: _unsharp_mask(img, 1.0, 1.0)),
    "sharp_l":   ("Sharp ×2.0",      lambda img: _unsharp_mask(img, 1.0, 2.0)),
    "jpeg_90":   ("JPEG q=90",       lambda img: _jpeg(img, 90)),
    "jpeg_75":   ("JPEG q=75",       lambda img: _jpeg(img, 75)),
    "jpeg_50":   ("JPEG q=50",       lambda img: _jpeg(img, 50)),
    "jpeg_30":   ("JPEG q=30",       lambda img: _jpeg(img, 30)),
}

DEFAULT_GROUPS = list(ALL_AUGMENTS.keys())

# ---------------------------------------------------------------------------
# Path collection
# ---------------------------------------------------------------------------

def win_to_posix(p: str) -> str:
    return p.replace("\\", "/")


def collect_paths(data: dict, key: str, split: str, compression: str = "c23") -> list[str]:
    group = data[key].get(split, {}).get(compression, {})
    paths = []
    for video in group.values():
        for rel in video.get("frames", []):
            paths.append(str(Path(BASE_DIR) / win_to_posix(rel)))
    return paths


# ---------------------------------------------------------------------------
# Phase accumulator
# ---------------------------------------------------------------------------

class PhaseAccumulator:
    def __init__(self, shape: tuple[int, int]):
        h, w = shape
        self.sum_cos = np.zeros((h, w), dtype=np.float64)
        self.sum_sin = np.zeros((h, w), dtype=np.float64)
        self.count = 0

    def update(self, img: np.ndarray):
        F = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
        phase = np.angle(F)
        self.sum_cos += np.cos(phase)
        self.sum_sin += np.sin(phase)
        self.count += 1

    def coherence(self) -> np.ndarray:
        if self.count == 0:
            raise RuntimeError("No images accumulated.")
        return np.sqrt(
            (self.sum_cos / self.count) ** 2 +
            (self.sum_sin / self.count) ** 2
        ).astype(np.float32)

    def mean_phase(self) -> np.ndarray:
        if self.count == 0:
            raise RuntimeError("No images accumulated.")
        return np.arctan2(self.sum_sin, self.sum_cos).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-augment computation
# ---------------------------------------------------------------------------

def compute_all_augments(
    paths: list[str],
    size: tuple[int, int],
    max_images: int,
    groups: list[str],
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Returns {aug_key: (coherence_map, mean_phase_map)} for each requested group.
    All augments share the same sampled image set.
    """
    random.seed(seed)
    sampled = random.sample(paths, min(max_images, len(paths)))

    accumulators = {k: PhaseAccumulator(shape=(size[1], size[0])) for k in groups}

    for p in sampled:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        for k in groups:
            _, fn = ALL_AUGMENTS[k]
            aug = fn(img)
            # restore size if augment changed it
            if aug.shape[:2] != (size[1], size[0]):
                aug = cv2.resize(aug, size, interpolation=cv2.INTER_AREA)
            accumulators[k].update(aug)

    n = next(iter(accumulators.values())).count
    print(f"  Accumulated {n} frames across {len(groups)} augment groups.")

    return {k: (acc.coherence(), acc.mean_phase()) for k, acc in accumulators.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_coherence(coh: np.ndarray) -> np.ndarray:
    return np.log1p(coh * 99) / np.log1p(99)


def radial_profile(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    r_max = min(cx, cy)
    radii = np.arange(r_max)
    profile = np.array([img[r == ri].mean() if (r == ri).any() else 0.0
                        for ri in radii])
    return radii, profile


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase spectra of real FF++ images under augmentation variants."
    )
    parser.add_argument("--split",  default="test")
    parser.add_argument("--max",    type=int, default=300)
    parser.add_argument("--size",   nargs=2, type=int, default=[256, 256], metavar=("W", "H"))
    parser.add_argument("--out",    help="Save main figure to this path")
    parser.add_argument("--radial", action="store_true",
                        help="Also save/show radial coherence profile plot")
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS),
                        help=f"Comma-separated subset of augments (default: all). "
                             f"Available: {', '.join(ALL_AUGMENTS)}")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    size = tuple(args.size)
    groups = [g.strip() for g in args.groups.split(",")]
    unknown = [g for g in groups if g not in ALL_AUGMENTS]
    if unknown:
        parser.error(f"Unknown groups: {unknown}. Available: {list(ALL_AUGMENTS)}")

    print(f"Loading JSON: {JSON_PATH}")
    with open(JSON_PATH) as f:
        raw = json.load(f)
    data = raw["FaceForensics++"]

    print(f"\n[Real] collecting paths — split={args.split}")
    real_paths = collect_paths(data, REAL_KEY, args.split, "c23")
    print(f"  Found {len(real_paths)} frames, sampling up to {args.max}")

    stats = compute_all_augments(real_paths, size, args.max, groups, args.seed)

    # ---- main figure: 3 rows × N cols ----------------------------------------
    n_cols = len(groups)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 9.6))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f"Real Image Phase Spectra — Augmentation Comparison\n"
        f"(FF++ {args.split}, n≤{args.max}, {size[0]}×{size[1]})",
        fontsize=12,
    )

    baseline_coh = stats["baseline"][0] if "baseline" in stats else stats[groups[0]][0]

    row_labels = [
        "Phase Coherence\n(MRL, log-scaled)",
        "Circular Mean Phase\n(−π to π)",
        "Coherence Diff\nvs Baseline (bwr)",
    ]
    for row, lbl in enumerate(row_labels):
        axes[row, 0].set_ylabel(lbl, fontsize=8)

    for col, key in enumerate(groups):
        label, _ = ALL_AUGMENTS[key]
        coh, mph = stats[key]

        # Row 0: coherence
        axes[0, col].imshow(_log_coherence(coh), cmap="jet", vmin=0, vmax=1)
        axes[0, col].set_title(label, fontsize=9)
        axes[0, col].axis("off")

        # Row 1: mean phase
        axes[1, col].imshow((mph + np.pi) / (2 * np.pi), cmap="hsv", vmin=0, vmax=1)
        axes[1, col].axis("off")

        # Row 2: diff from baseline
        ax2 = axes[2, col]
        if key == "baseline" or (key == groups[0] and "baseline" not in groups):
            ax2.axis("off")
            ax2.set_title("(reference)", fontsize=8, color="grey")
        else:
            diff = coh - baseline_coh
            vmax = np.abs(diff).max() or 1e-6
            im = ax2.imshow(diff, cmap="bwr", vmin=-vmax, vmax=vmax)
            ax2.set_title(f"− Baseline", fontsize=8)
            ax2.axis("off")
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.02)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {args.out}")
    else:
        plt.show()

    # ---- radial profile -------------------------------------------------------
    if args.radial:
        # Group augments by category for cleaner lines
        categories = {
            "brightness": [g for g in groups if g.startswith("bright")],
            "blur":       [g for g in groups if g.startswith("blur")],
            "sharpening": [g for g in groups if g.startswith("sharp")],
            "jpeg":       [g for g in groups if g.startswith("jpeg")],
            "baseline":   [g for g in groups if g == "baseline"],
        }
        active_cats = {k: v for k, v in categories.items() if v}
        n_cat = len(active_cats)

        fig2, cat_axes = plt.subplots(1, n_cat, figsize=(5 * n_cat, 4), sharey=True)
        if n_cat == 1:
            cat_axes = [cat_axes]
        fig2.suptitle(
            f"Radial Phase Coherence — Real Images by Augmentation Category\n"
            f"(FF++ {args.split}, n≤{args.max})",
            fontsize=11,
        )

        for ax, (cat_name, cat_keys) in zip(cat_axes, active_cats.items()):
            ax.set_title(cat_name.capitalize(), fontsize=10)
            ax.set_xlabel("Radial frequency (px from DC)")
            ax.grid(True, alpha=0.3)

            # always draw baseline as reference
            if "baseline" in stats:
                b_radii, b_prof = radial_profile(stats["baseline"][0])
                ax.plot(b_radii, b_prof, color="black", lw=2.0, label="Baseline", zorder=5)

            palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(cat_keys)))
            for ck, color in zip(cat_keys, palette):
                if ck == "baseline":
                    continue
                label, _ = ALL_AUGMENTS[ck]
                radii, profile = radial_profile(stats[ck][0])
                ax.plot(radii, profile, label=label, color=color, lw=1.5, linestyle="--")

            ax.legend(fontsize=8)

        cat_axes[0].set_ylabel("Mean Phase Coherence (MRL)")
        plt.tight_layout()

        if args.out:
            radial_out = str(Path(args.out).with_stem(Path(args.out).stem + "_radial"))
            fig2.savefig(radial_out, dpi=150, bbox_inches="tight")
            print(f"Saved radial plot: {radial_out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
