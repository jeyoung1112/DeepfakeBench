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
SR_N_BINS   = 64
_SR_CENTERS = (np.arange(SR_N_BINS) + 0.5) / SR_N_BINS   # normalised freq [0,1]
TARGET_SIZE = (256, 256)
RANDOM_SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "spectral_residual_analysis.png")

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



def _spectral_residual_curve(mag, dist_norm, n=3):
    """
    Hou & Zhang (CVPR 2007) spectral residual, radially averaged.
      R(f) = log|F| − h_n * log|F|   (h_n = n×n mean filter)
    Returns _sr_curve: [SR_N_BINS] mean R per radial band.
    Stored with _ prefix so the KDE loop skips it.
    """
    L = np.log(mag + 1e-8)
    kernel = np.ones((n, n), dtype=np.float32) / (n * n)
    R = L - cv2.filter2D(L, -1, kernel)
    edges = np.linspace(0.0, 1.0, SR_N_BINS + 1)
    curve = np.zeros(SR_N_BINS)
    for i in range(SR_N_BINS):
        mask = (dist_norm >= edges[i]) & (dist_norm < edges[i + 1])
        if mask.any():
            curve[i] = R[mask].mean()
    return {"_sr_curve": curve}


def _radial_log_envelope(mag, dist_norm, n_bins=64):
    """Radially average log(1+mag) into n_bins bands; returns (centers, envelope)."""
    log_mag = np.log1p(mag)
    edges   = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    envelope = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (dist_norm >= edges[i]) & (dist_norm < edges[i + 1])
        if mask.any():
            envelope[i] = log_mag[mask].mean()
    return centers, envelope


def _log_envelope_stats(mag, dist_norm):
    """
    Fit a line to the radial log-spectrum in log-log space and return:
      envelope_slope : spectral decay rate (steeper → more low-freq dominated)
      envelope_resid : RMS deviation from the linear fit (higher → more irregular)
    """
    centers, envelope = _radial_log_envelope(mag, dist_norm)
    valid   = centers > 0.02                     # skip near-DC bin
    log_r   = np.log(centers[valid])
    env_v   = envelope[valid]
    slope, intercept = np.polyfit(log_r, env_v, 1)
    resid   = float(np.sqrt(np.mean((env_v - (slope * log_r + intercept)) ** 2)))
    return {"envelope_slope": float(slope), "envelope_resid": resid}


def per_frame_stats(gray):
    """Compute a dict of scalar statistics for one frame."""
    spec     = np.fft.fftshift(np.fft.fft2(gray))
    mag      = np.abs(spec)

    H, W  = mag.shape
    cy, cx = H // 2, W // 2
    dist   = np.sqrt(
        (np.arange(H)[:, None] - cy) ** 2 +
        (np.arange(W)[None, :] - cx) ** 2
    )
    dist_norm = dist / (dist.max() + 1e-12)

    # ── Overall high-pass FFT energy ───────────────────────────────────────
    hp       = high_pass(gray)
    spec_hp  = np.fft.fftshift(np.fft.fft2(hp))
    mag2_hp  = np.abs(spec_hp) ** 2
    total_energy = mag2_hp.sum()

    r_max = min(cy, cx)
    R     = dist                                 # already computed
    high_mask   = R > 0.20 * r_max
    hf_ratio    = mag2_hp[high_mask].sum() / (total_energy + 1e-12)

    # ── Log-spectrum envelope ───────────────────────────────────────────────
    env_stats = _log_envelope_stats(mag, dist_norm)

    # ── Spectral residual curve (Hou & Zhang 2007) ─────────────────────────
    sr_stats = _spectral_residual_curve(mag, dist_norm)

    return {
        "fft_log_energy": np.log1p(total_energy),
        "fft_hf_ratio":   hf_ratio,
        **env_stats,
        **sr_stats,   # _sr_curve — skipped by KDE loop, plotted separately
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
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

STAT_META = {
    "fft_log_energy": ("FFT Log Energy (high-pass)\nlog(1 + Σ|F|²)",  "Log Energy (au)"),
    "fft_hf_ratio":   ("FFT High-Freq Ratio\n(r > 20% Nyquist)",       "Fraction of Total Energy"),
    "envelope_slope": ("Log-Spectrum Envelope Slope\n(log|F| vs log r, linear fit)", "Slope"),
    "envelope_resid": ("Log-Spectrum Envelope Residual\n(RMS deviation from linear envelope)", "RMS Residual"),
}

stat_keys = [k for k in stat_keys if not k.startswith("_")]   # drop array fields

dataset_names = list(COLORS.keys())   # 7 datasets
n_sr          = len(dataset_names)    # 7

# Layout: 4 KDE plots (2×2) on top, 7 SR-per-dataset (2×4) on bottom
SR_COLS = 4
SR_ROWS = (n_sr + SR_COLS - 1) // SR_COLS   # 2

fig = plt.figure(figsize=(26, 5 * (2 + SR_ROWS)))
fig.patch.set_facecolor("#0d0d0d")

gs = GridSpec(2 + SR_ROWS, 4, figure=fig, hspace=0.55, wspace=0.35)

# KDE axes — top 2 rows, left 2 columns each
kde_positions = [(0, 0), (0, 2), (1, 0), (1, 2)]
kde_axes = [fig.add_subplot(gs[r, c:c+2]) for r, c in kde_positions]

# SR axes — bottom SR_ROWS rows, 4 columns
sr_axes = []
for i in range(n_sr):
    r = 2 + i // SR_COLS
    c = i % SR_COLS
    sr_axes.append(fig.add_subplot(gs[r, c]))

KDE_POINTS = 512

for ax, key in zip(kde_axes, stat_keys):
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

# ── Spectral residual curve — one subplot per dataset ────────────────────────
# Shared y-axis range across all SR subplots for easy comparison
all_means = [data[n]["_sr_curve"].mean(axis=0) for n in dataset_names]
all_stds  = [data[n]["_sr_curve"].std(axis=0)  for n in dataset_names]
y_min = min((m - s).min() for m, s in zip(all_means, all_stds))
y_max = max((m + s).max() for m, s in zip(all_means, all_stds))
y_pad = (y_max - y_min) * 0.10

for ax, name, mean, std in zip(sr_axes, dataset_names, all_means, all_stds):
    color = COLORS[name]
    ls    = LINESTYLES[name]
    ax.set_facecolor("#1a1a1a")
    ax.axhline(0, color="#555555", lw=0.8)
    ax.plot(_SR_CENTERS, mean, color=color, lw=2.0, ls=ls)
    ax.fill_between(_SR_CENTERS, mean - std, mean + std, color=color, alpha=0.20)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_title(name, color=color, fontsize=9, pad=6, fontweight="bold")
    ax.set_xlabel("Norm. Freq.", color="gray", fontsize=7)
    ax.set_ylabel("Mean R(f)", color="gray", fontsize=7)
    ax.tick_params(colors="gray", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

# Section label above SR rows
fig.text(0.5, (SR_ROWS * 5) / (5 * (2 + SR_ROWS)) + 0.005,
         "Spectral Residual Curve per Dataset  [Hou & Zhang 2007]  "
         "R(f) = log|F| − h₃∗log|F|,  mean ± std",
         ha="center", va="bottom", color="white", fontsize=10,
         transform=fig.transFigure)

# Shared legend
handles, labels = kde_axes[0].get_legend_handles_labels()
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

fig.suptitle(
    "Real vs Fake Frequency Statistics — Overlapping Distributions\n"
    "Solid = Real  ·  Dashed = Fake",
    color="white", fontsize=14, fontweight="bold", y=1.05,
)

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"\nSaved → {OUTPUT_PATH}")
