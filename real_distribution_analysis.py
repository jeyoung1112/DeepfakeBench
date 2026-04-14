import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import dctn
from scipy.stats import gaussian_kde
import os
import random
from glob import glob

# ── Dataset paths ────────────────────────────────────────────────────────────
BASE = "/media/NAS/DATASET/DeepfakeBench/original/dataset"

DATASETS_REAL = {
    "Celeb-real":    os.path.join(BASE, "Celeb-DF-v2",  "Celeb-real",      "frames"),
    "YouTube-real":  os.path.join(BASE, "Celeb-DF-v2",  "YouTube-real",    "frames"),
    "DFDCP":         os.path.join(BASE, "DFDCP",         "original_videos", "frames"),
    "FaceForensics++": None,  # handled via FF_ROOTS_REAL
}

FF_ROOTS_REAL = [
    os.path.join(BASE, "FaceForensics++", "original_sequences", "actors",  "c23", "frames"),
    os.path.join(BASE, "FaceForensics++", "original_sequences", "youtube", "c23", "frames"),
]

DATASETS_FAKE = {
    "Celeb-synthesis": os.path.join(BASE, "Celeb-DF-v2",  "Celeb-synthesis", "frames"),
    "DFDCP-fake-A":      os.path.join(BASE, "DFDCP",         "method_A", "frames"),
    "DFDCP-fake-B":      os.path.join(BASE, "DFDCP",         "method_B", "frames"),
    "FaceForensics++-fake": None,  # handled via FF_ROOTS_FAKE
}

# Aggregate all FF++ manipulation methods as one fake entry
FF_ROOTS_FAKE = [
    os.path.join(BASE, "FaceForensics++", "manipulated_sequences", method, "c23", "frames")
    for method in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
]

N_SAMPLES   = 500
TARGET_SIZE = (256, 256)
RANDOM_SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "overall_distribution_analysis.png")

# Real datasets: cool/neutral colours; Fake datasets: warm/vivid colours
COLORS = {
    # Real
    "Celeb-real":           "#4fc3f7",   # light blue
    "YouTube-real":         "#81c784",   # green
    "DFDCP":                "#ffb74d",   # amber
    "FaceForensics++":      "#f06292",   # pink
    # Fake
    "Celeb-synthesis":      "#e040fb",   # purple
    "DFDCP-fake":           "#ff5252",   # red
    "FaceForensics++-fake": "#ff6d00",   # deep orange
}

LINESTYLES = {name: "-"  for name in ("Celeb-real", "YouTube-real", "DFDCP", "FaceForensics++")}
LINESTYLES.update({name: "--" for name in ("Celeb-synthesis", "DFDCP-fake", "FaceForensics++-fake")})

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


def per_frame_stats(gray):
    """Compute a dict of scalar statistics for one frame."""
    hp = high_pass(gray)

    # ── FFT stats ──────────────────────────────────────────────────────────
    spec  = np.fft.fftshift(np.fft.fft2(hp))
    mag2  = np.abs(spec) ** 2
    total_energy = mag2.sum()

    H, W  = mag2.shape
    cy, cx = H // 2, W // 2
    r_max = min(cy, cx)

    # Low vs high split at 20 % of Nyquist radius
    Y, X = np.ogrid[:H, :W]
    R    = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    low_mask  = R <= 0.20 * r_max
    high_mask = R >  0.20 * r_max
    low_energy  = mag2[low_mask].sum()
    high_energy = mag2[high_mask].sum()
    hf_ratio    = high_energy / (total_energy + 1e-12)

    # Spectral entropy (Shannon, over normalised magnitude)
    p    = mag2.ravel() / (total_energy + 1e-12)
    p    = p[p > 0]
    fft_entropy = -np.sum(p * np.log(p))

    # Radial energy profile – summarise as centre-of-mass radius
    radii  = R.ravel()
    weights = mag2.ravel()
    fft_spectral_centroid = (radii * weights).sum() / (weights.sum() + 1e-12)

    # ── DCT stats ──────────────────────────────────────────────────────────
    coef  = dctn(hp, norm="ortho")
    cabs  = np.abs(coef)
    dct_total    = (cabs ** 2).sum()
    # fraction of DCT energy in top-left 32×32 block (low freq) vs rest
    dct_low  = (cabs[:32, :32] ** 2).sum()
    dct_high = dct_total - dct_low
    dct_hf_ratio = dct_high / (dct_total + 1e-12)

    # DCT entropy
    p2   = (cabs.ravel() ** 2) / (dct_total + 1e-12)
    p2   = p2[p2 > 0]
    dct_entropy = -np.sum(p2 * np.log(p2))

    # ── Spatial stats ──────────────────────────────────────────────────────
    hp_std = hp.std()   # overall contrast after high-pass

    return {
        "fft_log_energy":       np.log1p(total_energy),
        "fft_hf_ratio":         hf_ratio,
        "fft_entropy":          fft_entropy,
        "fft_spectral_centroid": fft_spectral_centroid,
        "dct_log_energy":       np.log1p(dct_total),
        "dct_hf_ratio":         dct_hf_ratio,
        "dct_entropy":          dct_entropy,
        "hp_std":               hp_std,
    }


def sample_stats(paths, n, seed=RANDOM_SEED):
    rng     = random.Random(seed)
    chosen  = rng.sample(paths, min(n, len(paths)))
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
    tag = "real" if name in LINESTYLES and LINESTYLES[name] == "-" else "fake"
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

# Convert list-of-dicts → dict-of-arrays
stat_keys = list(all_stats[next(iter(all_stats))][0].keys())
data = {
    name: {k: np.array([r[k] for r in records]) for k in stat_keys}
    for name, records in all_stats.items()
}

# ── Plotting ───────────────────────────────────────────────────────────────────
STAT_META = {
    "fft_log_energy":          ("FFT Log Energy\nlog(1 + Σ|F|²)",             "Log Energy (au)"),
    "fft_hf_ratio":            ("FFT High-Freq Ratio\n(r > 20% Nyquist)",     "Fraction of Total Energy"),
    "fft_entropy":             ("FFT Spectral Entropy\n(−Σ p·log p)",         "Nats"),
    "fft_spectral_centroid":   ("FFT Spectral Centroid\n(energy-weighted radius)", "Pixels"),
    "dct_log_energy":          ("DCT Log Energy\nlog(1 + Σ|C|²)",             "Log Energy (au)"),
    "dct_hf_ratio":            ("DCT High-Freq Ratio\n(outside 32×32 block)", "Fraction of Total Energy"),
    "dct_entropy":             ("DCT Spectral Entropy\n(−Σ p·log p)",         "Nats"),
    "hp_std":                  ("High-Pass Std Dev\n(spatial texture contrast)", "Pixel Intensity"),
}

n_stats = len(stat_keys)
n_cols  = 4
n_rows  = (n_stats + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5 * n_rows))
fig.patch.set_facecolor("#0d0d0d")
axes = axes.flatten()

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
        # KDE curve
        kde  = gaussian_kde(vals, bw_method="scott")
        ys   = kde(xs)
        ax.plot(xs, ys, color=color, lw=2.0, ls=ls, label=name)
        ax.fill_between(xs, ys, alpha=0.10, color=color)
        # Vertical mean line
        ax.axvline(vals.mean(), color=color, lw=1.0, ls=ls, alpha=0.7)

    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.set_xlabel(xlabel, color="gray", fontsize=8)
    ax.set_ylabel("Density", color="gray", fontsize=8)
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.yaxis.label.set_color("gray")

# Shared legend – add linestyle hint entries
handles, labels = axes[0].get_legend_handles_labels()

import matplotlib.lines as mlines
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

# Hide unused axes
for ax in axes[n_stats:]:
    ax.set_visible(False)

fig.suptitle(
    "Real vs Fake Frequency Statistics — Overlapping Distributions\n"
    "Solid = Real  ·  Dashed = Fake",
    color="white", fontsize=14, fontweight="bold", y=1.06,
)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"\nSaved → {OUTPUT_PATH}")
