"""
Compare how different image conditions affect the DCT profile of real images.

Applies a configurable set of transforms to the same real frames and plots:
  - Average DCT heatmap per condition
  - Radial energy curves overlaid on one plot
  - Difference heatmaps relative to the baseline condition

Usage:
    python dct_condition_comparison.py
    python dct_condition_comparison.py --conditions jpeg_40 jpeg_85 blur_3 noise_15
    python dct_condition_comparison.py --max 200 --out dct_conditions.png

Built-in conditions:
    baseline          no transform

    -- JPEG recompression (overlaps with block-boundary forgery artifacts) --
    jpeg_<q>          JPEG re-encode at quality q        e.g. jpeg_40, jpeg_75, jpeg_95

    -- Resizing / interpolation (mid-freq patterns similar to GAN upsampling) --
    bicubic_<f>       upscale by factor f then back      e.g. bicubic_2, bicubic_4
    lanczos_<f>       same with Lanczos-4 kernel         e.g. lanczos_2, lanczos_4
    downup_<f>        downsample then upsample (bicubic) e.g. downup_2, downup_4

    -- Social media pipelines (resize → compress → sharpen chains) --
    whatsapp          resize 1600px + JPEG q=75
    facebook          JPEG q=85 + mild sharpen
    twitter           resize 1280px + JPEG q=85
    instagram         JPEG q=85 + strong sharpen
    wechat            resize 1080px + JPEG q=65  (more aggressive than WhatsApp)
    line_app          resize 1024px + JPEG q=80  (LINE messenger)

    -- Sharpening (boosts mid-high freqs, overlaps with fake spectral signature) --
    sharpen_<s>       Unsharp mask, strength=s (0–1)     e.g. sharpen_03, sharpen_07

    -- Smoothing / noise --
    blur_<r>          Gaussian blur, sigma=r             e.g. blur_1, blur_3, blur_5
    median_<k>        Median filter, kernel size k       e.g. median_3, median_5
    noise_<s>         Additive Gaussian noise, sigma=s   e.g. noise_5, noise_15, noise_30

    -- Brightness shift (affects DC component and low-freq energy) --
    brightness_<d>    Add delta d to pixel values        e.g. brightness_30, brightness_n30

    -- Device pipelines --
    phone             noise σ=8 + blur + JPEG q=92
    flagship          bilateral skin-smooth + unsharp mask
"""

import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn

# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

JSON_PATH = "/media/NAS/DATASET/DeepfakeBench/original/dataset_json/FaceForensics++.json"
BASE_DIR  = "/media/NAS/DATASET/DeepfakeBench/original/dataset"

DEFAULT_CONDITIONS = [
    "baseline", "jpeg_40", "jpeg_75", "jpeg_95",
    "blur_1", "blur_3", "noise_10", "noise_25",
    "sharpen_05", "whatsapp", "instagram", "phone",
]

# Conditions grouped by the FPR-risk mechanism they target.
# Use these as --conditions presets for focused analysis.
FPR_CONDITIONS = {
    "jpeg":    ["baseline", "jpeg_40", "jpeg_60", "jpeg_75", "jpeg_85", "jpeg_95"],
    "resize":  ["baseline", "bicubic_2", "bicubic_4", "lanczos_2", "lanczos_4", "downup_2", "downup_4"],
    "social":  ["baseline", "whatsapp", "facebook", "twitter", "instagram", "wechat", "line_app"],
    "sharpen": ["baseline", "sharpen_05", "sharpen_10", "sharpen_20", "sharpen_30"],
    "smooth":      ["baseline", "blur_1", "blur_3", "median_3", "median_5"],
    "brightness":  ["baseline", "brightness_n50", "brightness_n30", "brightness_30", "brightness_50", "brightness_80"],
}

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def flagship_selfie_filter(img):
    # 1. Skin smoothing — bilateral keeps edges sharp (grayscale-safe)
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. Smart sharpening — unsharp mask (iPhone 15 / Galaxy S25 style)
    gaussian_3 = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0)

    # Note: colour/saturation boost is skipped — images are grayscale
    return img


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


def _noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _brightness(img: np.ndarray, delta: int) -> np.ndarray:
    return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def _sharpen(img: np.ndarray, strength: float) -> np.ndarray:
    blur = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=1.0)
    sharp = img.astype(np.float32) + strength * (img.astype(np.float32) - blur)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _resize_long_side(img: np.ndarray, max_px: int, interp=cv2.INTER_AREA) -> np.ndarray:
    h, w = img.shape[:2]
    long = max(h, w)
    if long <= max_px:
        return img
    scale = max_px / long
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=interp)


def _upscale_downscale(img: np.ndarray, factor: int, up_interp, down_interp) -> np.ndarray:
    """Upscale by factor then back to original size — introduces interpolation artifacts."""
    h, w = img.shape[:2]
    big = cv2.resize(img, (w * factor, h * factor), interpolation=up_interp)
    return cv2.resize(big, (w, h), interpolation=down_interp)


def _downscale_upscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Downsample then upsample — simulates lossy pipeline resize."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // factor), max(1, h // factor)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def apply_condition(img: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    """Apply a named condition to a uint8 grayscale image."""

    if condition == "baseline":
        return img

    # parametric: jpeg_<q>
    m = re.fullmatch(r"jpeg_(\d+)", condition)
    if m:
        return _jpeg(img, int(m.group(1)))

    # parametric: blur_<sigma>  (e.g. blur_3 → sigma=3, blur_15 → sigma=1.5)
    m = re.fullmatch(r"blur_(\d+)", condition)
    if m:
        sigma = int(m.group(1)) / 10 if int(m.group(1)) > 9 else int(m.group(1))
        return _blur(img, float(sigma))

    # parametric: noise_<sigma>
    m = re.fullmatch(r"noise_(\d+)", condition)
    if m:
        return _noise(img, float(m.group(1)), rng)

    # parametric: sharpen_<strength*10>  e.g. sharpen_05 → 0.5, sharpen_10 → 1.0
    m = re.fullmatch(r"sharpen_(\d+)", condition)
    if m:
        strength = int(m.group(1)) / 10
        return _sharpen(img, strength)

    # parametric: bicubic_<factor>  — upscale then back (bicubic)
    m = re.fullmatch(r"bicubic_(\d+)", condition)
    if m:
        return _upscale_downscale(img, int(m.group(1)), cv2.INTER_CUBIC, cv2.INTER_AREA)

    # parametric: lanczos_<factor>  — upscale then back (Lanczos-4)
    m = re.fullmatch(r"lanczos_(\d+)", condition)
    if m:
        return _upscale_downscale(img, int(m.group(1)), cv2.INTER_LANCZOS4, cv2.INTER_AREA)

    # parametric: downup_<factor>  — downsample then upsample (bicubic)
    m = re.fullmatch(r"downup_(\d+)", condition)
    if m:
        return _downscale_upscale(img, int(m.group(1)))

    # parametric: median_<kernel>
    m = re.fullmatch(r"median_(\d+)", condition)
    if m:
        k = int(m.group(1))
        if k % 2 == 0:
            k += 1  # kernel must be odd
        return cv2.medianBlur(img, k)

    # parametric: brightness_<delta>  — positive or negative offset
    # Use 'n' prefix for negative: brightness_n30 → delta=-30
    m = re.fullmatch(r"brightness_n(\d+)", condition)
    if m:
        return _brightness(img, -int(m.group(1)))
    m = re.fullmatch(r"brightness_(\d+)", condition)
    if m:
        return _brightness(img, int(m.group(1)))

    # named social-media / device pipelines
    if condition == "whatsapp":
        return _jpeg(_resize_long_side(img, 1600), 75)

    if condition == "facebook":
        return _sharpen(_jpeg(img, 85), 0.3)

    if condition == "twitter":
        return _jpeg(_resize_long_side(img, 1280), 85)

    if condition == "instagram":
        return _sharpen(_jpeg(img, 85), 0.5)

    if condition == "wechat":
        # More aggressive than WhatsApp: smaller cap + lower quality
        return _jpeg(_resize_long_side(img, 1080), 65)

    if condition == "line_app":
        return _jpeg(_resize_long_side(img, 1024), 80)

    if condition == "phone":
        img = _noise(img, 8.0, rng)
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.8)
        return _jpeg(img, 92)

    if condition == "flagship":
        return flagship_selfie_filter(img)

    raise ValueError(f"Unknown condition: '{condition}'. See --help for valid options.")


def condition_label(condition: str) -> str:
    """Short human-readable label for plot titles."""
    replacements = {
        "baseline": "Baseline",
        "whatsapp": "WhatsApp",
        "facebook": "Facebook",
        "twitter": "Twitter",
        "instagram": "Instagram",
        "wechat": "WeChat",
        "line_app": "LINE",
        "phone": "Phone",
        "flagship": "Flagship",
    }
    if condition in replacements:
        return replacements[condition]
    m = re.fullmatch(r"jpeg_(\d+)", condition)
    if m:
        return f"JPEG q={m.group(1)}"
    m = re.fullmatch(r"bicubic_(\d+)", condition)
    if m:
        return f"Bicubic ×{m.group(1)}"
    m = re.fullmatch(r"lanczos_(\d+)", condition)
    if m:
        return f"Lanczos ×{m.group(1)}"
    m = re.fullmatch(r"downup_(\d+)", condition)
    if m:
        return f"Down↓Up ×{m.group(1)}"
    m = re.fullmatch(r"median_(\d+)", condition)
    if m:
        return f"Median k={m.group(1)}"
    m = re.fullmatch(r"blur_(\d+)", condition)
    if m:
        sigma = int(m.group(1)) / 10 if int(m.group(1)) > 9 else int(m.group(1))
        return f"Blur σ={sigma}"
    m = re.fullmatch(r"noise_(\d+)", condition)
    if m:
        return f"Noise σ={m.group(1)}"
    m = re.fullmatch(r"sharpen_(\d+)", condition)
    if m:
        return f"Sharpen ×{int(m.group(1))/10:.1f}"
    m = re.fullmatch(r"brightness_n(\d+)", condition)
    if m:
        return f"Bright −{m.group(1)}"
    m = re.fullmatch(r"brightness_(\d+)", condition)
    if m:
        return f"Bright +{m.group(1)}"
    return condition


# ---------------------------------------------------------------------------
# DCT computation
# ---------------------------------------------------------------------------

def compute_dct_profile(
    paths: list[str],
    condition: str,
    size: tuple[int, int],
    max_images: int,
    seed: int,
) -> dict:
    """
    Returns:
        mean_heatmap  : [0,1] float32 (H, W) — averaged log-DCT magnitude
        mean_radial   : 1-D energy vs. radial frequency
        std_radial    : std of radial energy across frames
        freqs         : radial frequency bins
    """
    random.seed(seed)
    sampled = random.sample(paths, min(max_images, len(paths)))
    rng = np.random.default_rng(seed)

    heatmap_acc = None
    radial_curves = []
    count = 0

    for p in sampled:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = apply_condition(img, condition, rng)
        if img.shape[:2] != (size[1], size[0]):
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        dct = dctn(img.astype(np.float32), norm="ortho")
        log_mag = np.log1p(np.abs(dct))

        if heatmap_acc is None:
            heatmap_acc = log_mag.copy()
        else:
            heatmap_acc += log_mag

        # radial energy
        h, w = dct.shape
        cy, cx = h // 2, w // 2
        shifted = np.fft.fftshift(dct)
        power = shifted ** 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
        max_r = min(cx, cy)
        energies = np.array([power[r == rad].mean() for rad in range(max_r)])
        radial_curves.append(energies)
        count += 1

    if count == 0:
        raise RuntimeError(f"No valid images for condition '{condition}'.")

    mean_map = heatmap_acc / count
    mean_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-8)

    radial_arr = np.array(radial_curves)
    print(f"  [{condition_label(condition):20s}] averaged {count} frames")
    return {
        "condition": condition,
        "label": condition_label(condition),
        "mean_heatmap": mean_map,
        "mean_radial": radial_arr.mean(axis=0),
        "std_radial": radial_arr.std(axis=0),
        "freqs": np.arange(max_r),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def heatmap_rgb(norm_map: np.ndarray) -> np.ndarray:
    return (plt.cm.jet(norm_map)[:, :, :3] * 255).astype(np.uint8)


def plot_results(profiles: list[dict], baseline_label: str, out_path: str | None):
    n = len(profiles)
    baseline = next((p for p in profiles if p["condition"] == "baseline"), profiles[0])

    # Layout:
    #   Row 0: DCT heatmaps
    #   Row 1: difference vs baseline (skip baseline itself)
    #   Row 2 (full width): radial energy curves
    fig = plt.figure(figsize=(max(14, 3.2 * n), 12))
    fig.suptitle("DCT Profile — Real Images Under Different Conditions", fontsize=13)

    gs = fig.add_gridspec(3, n, height_ratios=[1, 1, 0.9], hspace=0.45, wspace=0.15)

    colors = plt.cm.tab10.colors

    for col, prof in enumerate(profiles):
        # --- row 0: heatmap ---
        ax_h = fig.add_subplot(gs[0, col])
        ax_h.imshow(heatmap_rgb(prof["mean_heatmap"]))
        ax_h.set_title(prof["label"], fontsize=9, pad=3)
        ax_h.axis("off")

        # --- row 1: diff vs baseline ---
        ax_d = fig.add_subplot(gs[1, col])
        if prof["condition"] == baseline["condition"]:
            ax_d.axis("off")
            ax_d.text(0.5, 0.5, "baseline", ha="center", va="center",
                      fontsize=9, color="grey", transform=ax_d.transAxes)
        else:
            diff = prof["mean_heatmap"] - baseline["mean_heatmap"]
            vmax = max(np.abs(diff).max(), 1e-6)
            im = ax_d.imshow(diff, cmap="bwr", vmin=-vmax, vmax=vmax)
            ax_d.set_title(f"vs baseline", fontsize=7, pad=2)
            ax_d.axis("off")
            plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.02, format="%.2f")

    # --- row 2: radial energy curves (full width) ---
    ax_r = fig.add_subplot(gs[2, :])
    for idx, prof in enumerate(profiles):
        freqs = prof["freqs"]
        mean_e = prof["mean_radial"]
        std_e = prof["std_radial"]
        c = colors[idx % len(colors)]
        ax_r.semilogy(freqs, mean_e, label=prof["label"], color=c, linewidth=1.5)
        ax_r.fill_between(freqs,
                          np.maximum(mean_e - std_e, 1e-10),
                          mean_e + std_e,
                          alpha=0.12, color=c)

    ax_r.set_xlabel("Radial frequency (pixels from DC)", fontsize=10)
    ax_r.set_ylabel("Mean energy (log scale)", fontsize=10)
    ax_r.set_title("Radial frequency energy — all conditions", fontsize=10)
    ax_r.legend(ncol=min(n, 6), fontsize=8, loc="upper right")
    ax_r.grid(True, which="both", linestyle="--", alpha=0.4)

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_real_paths(split: str) -> list[str]:
    with open(JSON_PATH) as f:
        raw = json.load(f)
    data = raw["FaceForensics++"]["FF-real"].get(split, {}).get("c23", {})
    paths = []
    for video in data.values():
        for rel in video.get("frames", []):
            paths.append(str(Path(BASE_DIR) / rel.replace("\\", "/")))
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Compare DCT profiles of real images under different conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="List of conditions to compare")
    parser.add_argument("--preset", choices=list(FPR_CONDITIONS.keys()),
                        help=f"Named FPR-risk preset: {list(FPR_CONDITIONS.keys())}")
    parser.add_argument("--split",  default="test", help="train / val / test")
    parser.add_argument("--max",    type=int, default=300,
                        help="Max frames sampled (default: 300)")
    parser.add_argument("--size",   nargs=2, type=int, default=[256, 256],
                        metavar=("W", "H"))
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--out",    help="Save figure to this path")
    args = parser.parse_args()

    size = tuple(args.size)

    if args.preset:
        conditions = FPR_CONDITIONS[args.preset]
    elif args.conditions:
        conditions = args.conditions
    else:
        conditions = DEFAULT_CONDITIONS

    print(f"Collecting real frame paths ({args.split} split)...")
    paths = collect_real_paths(args.split)
    print(f"Found {len(paths)} frames, sampling up to {args.max} per condition.\n")

    profiles = []
    for cond in conditions:
        try:
            prof = compute_dct_profile(paths, cond, size, args.max, args.seed)
            profiles.append(prof)
        except ValueError as e:
            print(f"  [skip] {e}")

    if not profiles:
        raise SystemExit("No valid conditions to plot.")

    plot_results(profiles, baseline_label="baseline", out_path=args.out)


if __name__ == "__main__":
    main()
