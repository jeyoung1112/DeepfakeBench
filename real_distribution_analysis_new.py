"""
Distribution analysis using SDNormNew-style normalization.

Key difference from real_distribution_analysis.py:
  OLD (SDNorm):    per-ring instance norm → collapses real/fake ring-level differences
  NEW (SDNormNew): global instance norm on log-magnitude → removes dataset energy bias
                   while preserving the relative frequency profile across rings

After global instance norm, ring means reflect how each band compares to the
sample-level average. If fake images boost high frequencies relative to low,
that cross-ring pattern survives normalization — making it visible in the plots.
"""

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde
import os
import random
from glob import glob

# ── Dataset paths ─────────────────────────────────────────────────────────────
BASE = "/media/NAS/DATASET/DeepfakeBench/original/dataset"

DATASETS_REAL = {
    "Celeb-real":       os.path.join(BASE, "Celeb-DF-v2",  "Celeb-real",      "frames"),
    "YouTube-real":     os.path.join(BASE, "Celeb-DF-v2",  "YouTube-real",    "frames"),
    "DFDCP":            os.path.join(BASE, "DFDCP",         "original_videos", "frames"),
    "FaceForensics++":  None,
}

FF_ROOTS_REAL = [
    os.path.join(BASE, "FaceForensics++", "original_sequences", "actors",  "c23", "frames"),
    os.path.join(BASE, "FaceForensics++", "original_sequences", "youtube", "c23", "frames"),
]

DATASETS_FAKE = {
    "Celeb-synthesis":      os.path.join(BASE, "Celeb-DF-v2", "Celeb-synthesis", "frames"),
    "DFDCP-fake-A":         os.path.join(BASE, "DFDCP", "method_A", "frames"),
    "DFDCP-fake-B":         os.path.join(BASE, "DFDCP", "method_B", "frames"),
    "FaceForensics++-fake": None,
}

FF_ROOTS_FAKE = [
    os.path.join(BASE, "FaceForensics++", "manipulated_sequences", method, "c23", "frames")
    for method in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
]

N_SAMPLES   = 500
TARGET_SIZE = (256, 256)
RANDOM_SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "sdnorm_new_distribution_analysis.png")

COLORS = {
    "Celeb-real":           "#4fc3f7",
    "YouTube-real":         "#81c784",
    "DFDCP":                "#ffb74d",
    "FaceForensics++":      "#f06292",
    "Celeb-synthesis":      "#e040fb",
    "DFDCP-fake":           "#ff5252",
    "FaceForensics++-fake": "#ff6d00",
}

LINESTYLES = {name: "-"  for name in ("Celeb-real", "YouTube-real", "DFDCP", "FaceForensics++")}
LINESTYLES.update({name: "--" for name in ("Celeb-synthesis", "DFDCP-fake", "FaceForensics++-fake")})

# ── Ring mask config (mirrors sdnorm_new.py) ──────────────────────────────────
NUM_RINGS  = 4
SHARPNESS  = 5.0
_log_bounds = np.logspace(np.log10(0.01), np.log10(1.0), NUM_RINGS + 1)
RING_INNER = _log_bounds[:-1]
RING_OUTER = _log_bounds[1:]


# ── Helpers ───────────────────────────────────────────────────────────────────
def collect_frame_paths(root_dirs):
    paths = []
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]
    for root in root_dirs:
        paths.extend(glob(os.path.join(root, "**", "*.png"), recursive=True))
        paths.extend(glob(os.path.join(root, "**", "*.jpg"), recursive=True))
    return paths


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, TARGET_SIZE).astype(np.float32)


def high_pass(gray):
    blurred = cv2.medianBlur(gray.astype(np.uint8), 9).astype(np.float32)
    return gray - blurred


def _build_ring_masks(dist_norm):
    """Soft ring masks with softmax partition-of-unity. [K, H, W]"""
    masks = []
    for k in range(NUM_RINGS):
        raw = (1 / (1 + np.exp(-SHARPNESS * (dist_norm - RING_INNER[k])))) * \
              (1 / (1 + np.exp(-SHARPNESS * (RING_OUTER[k] - dist_norm))))
        masks.append(raw)
    raw_stack = np.stack(masks, axis=0)
    exp_stack = np.exp(raw_stack - raw_stack.max(axis=0, keepdims=True))
    return exp_stack / exp_stack.sum(axis=0, keepdims=True)


def _ring_mag_stats(mag_norm, dist_norm):
    """Mask-weighted mean and std of globally-normalised magnitude per ring."""
    masks = _build_ring_masks(dist_norm)
    stats = {}
    eps = 1e-5
    for k in range(NUM_RINGS):
        mask_k   = masks[k]
        mask_sum = mask_k.sum().clip(min=1.0)
        mean_k   = (mag_norm * mask_k).sum() / mask_sum
        var_k    = ((mag_norm - mean_k) ** 2 * mask_k).sum() / mask_sum
        stats[f"ring{k}_mean"] = float(mean_k)
        stats[f"ring{k}_std"]  = float(np.sqrt(var_k + eps))
    return stats


# ── SDNormNew-style normalization applied per-frame ───────────────────────────
def per_frame_stats(gray):
    """
    Compute statistics under SDNormNew normalization.

    Pipeline:
      1. FFT2D → magnitude
      2. log1p compression (matches FFTTransform in frequency_branch.py)
      3. Global instance norm: subtract per-sample mean, divide by per-sample std
         → removes dataset energy bias, keeps relative frequency profile
      4. Ring stats on normalised magnitude
         → ring means now reflect deviation from sample average per frequency band
    """
    spec     = np.fft.fftshift(np.fft.fft2(gray))
    mag      = np.abs(spec)

    # Step 1: log1p compression — reduces heavy skew in raw FFT magnitudes
    mag_log  = np.log1p(mag)

    # Step 2: global instance normalization — key change from original SDNorm
    global_mean = mag_log.mean()
    global_std  = mag_log.std() + 1e-5
    mag_norm    = (mag_log - global_mean) / global_std   # zero-mean, unit-std

    # Spatial frequency distance (DC at centre)
    H, W  = mag.shape
    cy, cx = H // 2, W // 2
    dist   = np.sqrt(
        (np.arange(H)[:, None] - cy) ** 2 +
        (np.arange(W)[None, :] - cx) ** 2
    )
    dist_norm = dist / (dist.max() + 1e-12)

    # Step 3: per-ring stats on globally normalised magnitude
    ring_stats = _ring_mag_stats(mag_norm, dist_norm)

    # High-pass energy ratio (unchanged — uses raw hp signal for diagnostic)
    hp          = high_pass(gray)
    spec_hp     = np.fft.fftshift(np.fft.fft2(hp))
    mag2_hp     = np.abs(spec_hp) ** 2
    total_energy = mag2_hp.sum()
    r_max        = min(cy, cx)
    high_mask    = dist > 0.20 * r_max
    hf_ratio     = mag2_hp[high_mask].sum() / (total_energy + 1e-12)

    return {
        "fft_log_energy": np.log1p(total_energy),
        "fft_hf_ratio":   hf_ratio,
        **ring_stats,
    }


def sample_stats(paths, n, seed=RANDOM_SEED):
    rng    = random.Random(seed)
    chosen = rng.sample(paths, min(n, len(paths)))
    records = []
    for p in chosen:
        gray = load_gray(p)
        if gray is not None:
            records.append(per_frame_stats(gray))
    return records


# ── Collect paths ─────────────────────────────────────────────────────────────
print("Collecting frame paths …")
all_paths = {
    "Celeb-real":           collect_frame_paths(DATASETS_REAL["Celeb-real"]),
    "YouTube-real":         collect_frame_paths(DATASETS_REAL["YouTube-real"]),
    "DFDCP":                collect_frame_paths(DATASETS_REAL["DFDCP"]),
    "FaceForensics++":      collect_frame_paths(FF_ROOTS_REAL),
    "Celeb-synthesis":      collect_frame_paths(DATASETS_FAKE["Celeb-synthesis"]),
    "DFDCP-fake":           collect_frame_paths([DATASETS_FAKE["DFDCP-fake-A"], DATASETS_FAKE["DFDCP-fake-B"]]),
    "FaceForensics++-fake": collect_frame_paths(FF_ROOTS_FAKE),
}
for name, paths in all_paths.items():
    tag = "real" if LINESTYLES.get(name) == "-" else "fake"
    print(f"  [{tag}] {name}: {len(paths):,} frames found")

# ── Compute per-frame statistics ──────────────────────────────────────────────
print(f"\nSampling {N_SAMPLES} frames per dataset …")
all_stats = {}
for name, paths in all_paths.items():
    if not paths:
        raise RuntimeError(f"No frames found for {name}")
    records = sample_stats(paths, N_SAMPLES)
    print(f"  {name}: {len(records)} frames processed")
    all_stats[name] = records

stat_keys = list(all_stats[next(iter(all_stats))][0].keys())
data = {
    name: {k: np.array([r[k] for r in records]) for k in stat_keys}
    for name, records in all_stats.items()
}

# ── Plot ──────────────────────────────────────────────────────────────────────
_RING_BOUNDS_STR = [
    f"r∈[{RING_INNER[k]:.2f},{RING_OUTER[k]:.2f}]" for k in range(NUM_RINGS)
]

STAT_META = {
    "fft_log_energy": (
        "FFT Log Energy (high-pass)\nlog(1 + Σ|F|²)",
        "Log Energy (au)"
    ),
    "fft_hf_ratio": (
        "FFT High-Freq Ratio\n(r > 20% Nyquist)",
        "Fraction of Total Energy"
    ),
    **{f"ring{k}_mean": (
        f"SDNormNew Ring {k} Mean\n{_RING_BOUNDS_STR[k]}  [global-norm, →0=avg band]",
        "Normalised FFT Magnitude Mean"
    ) for k in range(NUM_RINGS)},
    **{f"ring{k}_std": (
        f"SDNormNew Ring {k} Std\n{_RING_BOUNDS_STR[k]}  [global-norm]",
        "Normalised FFT Magnitude Std"
    ) for k in range(NUM_RINGS)},
}

stat_keys = [k for k in stat_keys if not k.startswith("_")]
n_stats = len(stat_keys)
n_cols  = 2
n_rows  = (n_stats + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5 * n_rows))
fig.patch.set_facecolor("#0d0d0d")
axes = np.array(axes).flatten()

KDE_POINTS = 512

for ax_idx, key in enumerate(stat_keys):
    ax = axes[ax_idx]
    ax.set_facecolor("#1a1a1a")

    title, xlabel = STAT_META[key]
    x_min = min(data[name][key].min() for name in data)
    x_max = max(data[name][key].max() for name in data)
    x_pad = (x_max - x_min) * 0.05
    xs    = np.linspace(x_min - x_pad, x_max + x_pad, KDE_POINTS)

    for name, color in COLORS.items():
        vals = data[name][key]
        ls   = LINESTYLES[name]
        kde  = gaussian_kde(vals, bw_method="scott")
        ys   = kde(xs)
        ax.plot(xs, ys, color=color, lw=2.0, ls=ls, label=name)
        ax.fill_between(xs, ys, alpha=0.10, color=color)
        ax.axvline(vals.mean(), color=color, lw=1.0, ls=ls, alpha=0.7)

    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.set_xlabel(xlabel, color="gray", fontsize=8)
    ax.set_ylabel("Density", color="gray", fontsize=8)
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

handles, labels = axes[0].get_legend_handles_labels()
real_patch = mlines.Line2D([], [], color="white", lw=1.5, ls="-",  label="── Real")
fake_patch = mlines.Line2D([], [], color="white", lw=1.5, ls="--", label="-- Fake")
handles = [real_patch, fake_patch] + handles
labels  = ["── Real", "-- Fake"] + labels

fig.legend(handles, labels,
           loc="upper center", ncol=5,
           frameon=True, framealpha=0.15,
           facecolor="#222222", edgecolor="#555555",
           fontsize=9, labelcolor="white",
           bbox_to_anchor=(0.5, 1.02))

for ax in axes[n_stats:]:
    ax.set_visible(False)

fig.suptitle(
    "SDNormNew — Global Instance Norm (log-magnitude)\n"
    "Dataset bias removed · Real/fake frequency profile preserved\n"
    "Solid = Real  ·  Dashed = Fake",
    color="white", fontsize=13, fontweight="bold", y=1.07,
)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"\nSaved → {OUTPUT_PATH}")
