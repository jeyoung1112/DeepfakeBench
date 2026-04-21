import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, kurtosis
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
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "phase_distribution_analysis.png")

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
    """Compute phase-specific statistics for one frame."""
    hp = high_pass(gray)

    # 2D FFT → shift DC to centre
    spec  = np.fft.fftshift(np.fft.fft2(hp))
    phase = np.angle(spec)          # ∈ [−π, π]

    H, W   = phase.shape
    cy, cx = H // 2, W // 2
    r_max  = min(cy, cx)

    Y, X    = np.ogrid[:H, :W]
    R       = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    lf_mask = R <= 0.20 * r_max    # inner 20 % of Nyquist radius
    hf_mask = R >  0.20 * r_max    # outer 80 %

    # 1. Phase entropy  (histogram over [−π, π], 64 bins)
    #    Uniform phase → max entropy; structured phase → lower entropy.
    #    GANs tend to push phase toward uniformity in HF regions.
    hist, _ = np.histogram(phase.ravel(), bins=64, range=(-np.pi, np.pi))
    hist    = hist / (hist.sum() + 1e-12)
    hist    = hist[hist > 0]
    phase_entropy = -np.sum(hist * np.log(hist))

    # 2 & 3. Phase std in LF / HF bands
    phase_lf_std = phase[lf_mask].std()
    phase_hf_std = phase[hf_mask].std()

    # 4. HF/LF phase std ratio — captures relative phase disorder between bands.
    #    Deepfakes often show disproportionately high HF phase noise.
    phase_hf_lf_ratio = phase_hf_std / (phase_lf_std + 1e-12)

    # 5. Global phase gradient mean  (measures smoothness of the phase landscape)
    #    Abrupt phase transitions are a hallmark of GAN blending artefacts.
    gy, gx   = np.gradient(phase)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    phase_grad_mean = grad_mag.mean()

    # 6. Phase gradient restricted to the high-frequency region
    phase_grad_hf_mean = grad_mag[hf_mask].mean()

    # 7. Phase coherence — circular mean resultant length |mean(e^{iφ})|
    #    → 1: all phases perfectly aligned; 0: uniformly random (incoherent).
    #    Real textures tend to be more incoherent; GAN outputs can be more
    #    structured or more chaotic depending on the method.
    phase_coherence = float(np.abs(np.exp(1j * phase).mean()))

    # 8. Excess kurtosis of the phase distribution
    #    Measures peakedness / heavy-tailedness relative to a Gaussian.
    phase_kurtosis = float(kurtosis(phase.ravel(), fisher=True))

    return {
        "phase_entropy":      phase_entropy,
        "phase_lf_std":       phase_lf_std,
        "phase_hf_std":       phase_hf_std,
        "phase_hf_lf_ratio":  phase_hf_lf_ratio,
        "phase_grad_mean":    phase_grad_mean,
        "phase_grad_hf_mean": phase_grad_hf_mean,
        "phase_coherence":    phase_coherence,
        "phase_kurtosis":     phase_kurtosis,
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
    "phase_entropy":      ("Phase Entropy\n(−Σ p·log p over [−π, π])",             "Nats"),
    "phase_lf_std":       ("Phase Std — Low Freq\n(r ≤ 20% Nyquist)",              "Radians"),
    "phase_hf_std":       ("Phase Std — High Freq\n(r > 20% Nyquist)",             "Radians"),
    "phase_hf_lf_ratio":  ("Phase HF/LF Std Ratio\n(relative band disorder)",      "Ratio"),
    "phase_grad_mean":    ("Phase Gradient Mean\n(global phase smoothness)",        "rad / pixel"),
    "phase_grad_hf_mean": ("Phase Gradient HF Mean\n(HF-only phase smoothness)",   "rad / pixel"),
    "phase_coherence":    ("Phase Coherence\n|mean(e^{iφ})| — circular mean",      "0 – 1"),
    "phase_kurtosis":     ("Phase Kurtosis\n(peakedness vs Gaussian baseline)",     "Excess kurtosis"),
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
    "Real vs Fake — Phase-Specific Distribution Analysis\n"
    "Solid = Real  ·  Dashed = Fake",
    color="white", fontsize=14, fontweight="bold", y=1.06,
)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"\nSaved → {OUTPUT_PATH}")
