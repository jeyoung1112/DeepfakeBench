"""
Average DCT heatmap for FaceForensics++ real vs. fake images.

Reads frame paths from FaceForensics++.json, samples up to --max frames per
group, computes the per-pixel average DCT log-magnitude heatmap, and plots
real vs. each fake method (and a combined fake average).

Usage:
    python ff_avg_dct_heatmap.py
    python ff_avg_dct_heatmap.py --split test --max 200
    python ff_avg_dct_heatmap.py --compression c40 --out ff_avg_dct_c40.png
    python ff_avg_dct_heatmap.py --out ff_avg_dct.png

Pipelines (--pipeline):
  c23        raw frames (default)
  c40        JPEG quality 40  (FF++ c40 definition)
  whatsapp   resize to 1600px longest side + JPEG q=75
  facebook   JPEG q=85 + mild sharpening
  twitter    resize to 1280px longest side + JPEG q=85
  instagram  JPEG q=85 + unsharp-mask sharpening + slight saturation boost
  phone      sensor noise (Gaussian σ=8) + slight blur + JPEG q=92
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

JSON_PATH = "/media/NAS/DATASET/DeepfakeBench/original/dataset_json/DFDCP.json"
BASE_DIR  = "/media/NAS/DATASET/DeepfakeBench/original/dataset"

REAL_KEY  = "FF-real"
FAKE_KEYS = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]
FAKE_LABELS = {
    "FF-DF":  "DeepFakes",
    "FF-F2F": "Face2Face",
    "FF-FS":  "FaceSwap",
    "FF-NT":  "NeuralTextures",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def win_to_posix(p: str) -> str:
    return p.replace("\\", "/")


def collect_paths(data: dict, key: str, split: str, compression: str) -> list[str]:
    """Extract absolute frame paths for one group from the JSON."""
    group = data[key].get(split, {}).get(compression, {})
    paths = []
    for video in group.values():
        for rel in video.get("frames", []):
            abs_path = str(Path(BASE_DIR) / win_to_posix(rel))
            paths.append(abs_path)
    return paths


# ---------------------------------------------------------------------------
# Pipeline transforms  (all operate on uint8 grayscale unless noted)
# ---------------------------------------------------------------------------

PIPELINES = ["c23", "c40", "whatsapp", "facebook", "twitter", "instagram", "phone"]


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def _resize_long_side(img: np.ndarray, max_px: int) -> np.ndarray:
    h, w = img.shape[:2]
    long = max(h, w)
    if long <= max_px:
        return img
    scale = max_px / long
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _unsharp_mask(img: np.ndarray, sigma: float = 1.0, strength: float = 0.5) -> np.ndarray:
    blur = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
    sharp = img.astype(np.float32) + strength * (img.astype(np.float32) - blur)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _add_gaussian_noise(img: np.ndarray, sigma: float = 8.0, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_pipeline(img: np.ndarray, pipeline: str, rng: np.random.Generator) -> np.ndarray:
    """Apply a named pipeline to a uint8 grayscale image."""
    if pipeline == "c23":
        return img

    if pipeline == "c40":
        return _jpeg(img, 40)

    if pipeline == "whatsapp":
        img = _resize_long_side(img, 1600)
        return _jpeg(img, 75)

    if pipeline == "facebook":
        img = _jpeg(img, 85)
        return _unsharp_mask(img, sigma=0.8, strength=0.3)

    if pipeline == "twitter":
        img = _resize_long_side(img, 1280)
        return _jpeg(img, 85)

    if pipeline == "instagram":
        img = _jpeg(img, 85)
        return _unsharp_mask(img, sigma=1.0, strength=0.5)

    if pipeline == "phone":
        img = _add_gaussian_noise(img, sigma=8.0, rng=rng)
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.8)
        return _jpeg(img, 92)

    raise ValueError(f"Unknown pipeline: {pipeline}")


def compute_avg_dct_heatmap(
    paths: list[str],
    size: tuple[int, int],
    max_images: int,
    seed: int = 42,
    pipeline: str = "c23",
) -> np.ndarray:
    """
    Load up to max_images frames, apply the named pipeline, compute
    DCT log-magnitude for each, and return the pixel-wise mean in [0, 1].
    """
    random.seed(seed)
    sampled = random.sample(paths, min(max_images, len(paths)))
    rng = np.random.default_rng(seed)

    accumulator = None
    count = 0

    for p in sampled:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = apply_pipeline(img, pipeline, rng)
        # after pipeline, img may have been resized again — restore target size
        if img.shape[:2] != (size[1], size[0]):
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        dct = dctn(img, norm="ortho")
        log_mag = np.log1p(np.abs(dct))

        if accumulator is None:
            accumulator = log_mag
        else:
            accumulator += log_mag
        count += 1

    if count == 0:
        raise RuntimeError("No valid images loaded.")

    mean_map = accumulator / count
    # Normalise to [0, 1] for display
    mean_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-8)
    print(f"  Averaged {count} frames.")
    return mean_map


def heatmap_rgb(norm_map: np.ndarray) -> np.ndarray:
    """Apply jet colormap to a [0,1] float map → uint8 RGB."""
    return (plt.cm.jet(norm_map)[:, :, :3] * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",       default="test",  help="train / val / test")
    parser.add_argument("--pipeline", default="c23", choices=PIPELINES,
                        help="Transform pipeline applied before DCT (default: c23)")
    parser.add_argument("--max",         type=int, default=300,
                        help="Max frames sampled per group (default: 300)")
    parser.add_argument("--size",        nargs=2, type=int, default=[256, 256],
                        metavar=("W", "H"))
    parser.add_argument("--out",         help="Save figure to this path")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    size = tuple(args.size)

    print(f"Loading JSON: {JSON_PATH}")
    print(f"Pipeline: {args.pipeline}")
    with open(JSON_PATH) as f:
        raw = json.load(f)
    data = raw["FaceForensics++"]

    # --- collect and average each group ---
    groups = {}

    print(f"\n[Real] {REAL_KEY}")
    real_paths = collect_paths(data, REAL_KEY, args.split, "c23")
    print(f"  Found {len(real_paths)} frames, sampling up to {args.max}")
    groups[REAL_KEY] = compute_avg_dct_heatmap(real_paths, size, args.max, args.seed, args.pipeline)

    fake_maps = []
    for fk in FAKE_KEYS:
        print(f"\n[Fake] {fk}")
        paths = collect_paths(data, fk, args.split, "c23")
        print(f"  Found {len(paths)} frames, sampling up to {args.max}")
        m = compute_avg_dct_heatmap(paths, size, args.max, args.seed, args.pipeline)
        groups[fk] = m
        fake_maps.append(m)

    # Combined fake average
    combined_fake = np.mean(fake_maps, axis=0)
    combined_fake = (combined_fake - combined_fake.min()) / (combined_fake.max() - combined_fake.min() + 1e-8)

    # --- plot ---
    # Row 0: heatmaps  (real | DF | F2F | FS | NT | combined fake)
    # Row 1: difference from real
    col_keys   = [REAL_KEY] + FAKE_KEYS + ["combined_fake"]
    col_titles = ["Real"] + [FAKE_LABELS[k] for k in FAKE_KEYS] + ["Avg Fake"]
    n_cols = len(col_keys)

    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))
    fig.suptitle(
        f"Average DCT Heatmap — FaceForensics++ ({args.split}, pipeline={args.pipeline})",
        fontsize=13,
    )

    real_map = groups[REAL_KEY]

    for col, (key, title) in enumerate(zip(col_keys, col_titles)):
        m = groups[key] if key != "combined_fake" else combined_fake

        # Row 0: heatmap
        axes[0, col].imshow(heatmap_rgb(m))
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].axis("off")

        # Row 1: difference from real (skip real itself)
        ax_diff = axes[1, col]
        if key == REAL_KEY:
            ax_diff.axis("off")
            ax_diff.set_title("(reference)", fontsize=8, color="grey")
        else:
            diff = m - real_map          # positive → more energy than real
            vmax = np.abs(diff).max()
            im = ax_diff.imshow(diff, cmap="bwr", vmin=-vmax, vmax=vmax)
            ax_diff.set_title(f"{title} − Real", fontsize=9)
            ax_diff.axis("off")
            plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.02)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
