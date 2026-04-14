import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import dctn
import os
import random
from glob import glob

# ── Dataset paths ───────────────────────────────────────────────────────────
BASE = "/media/NAS/DATASET/DeepfakeBench/original/dataset"

DATASETS = {
    "Celeb-real":    os.path.join(BASE, "Celeb-DF-v2",  "Celeb-real",      "frames"),
    "YouTube-real":  os.path.join(BASE, "Celeb-DF-v2",  "YouTube-real",    "frames"),
    "DFDCP":         os.path.join(BASE, "DFDCP",         "original_videos", "frames"),
    "FaceForensics++": None,  # two sub-trees – handled below
}

FF_ROOTS = [
    os.path.join(BASE, "FaceForensics++", "original_sequences", "actors",  "c23", "frames"),
    os.path.join(BASE, "FaceForensics++", "original_sequences", "youtube", "c23", "frames"),
]

N_SAMPLES   = 500   # frames sampled per dataset
TARGET_SIZE = (256, 256)
RANDOM_SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "real_frequency_heatmaps.png")


# ── Helpers ──────────────────────────────────────────────────────────────────
def collect_frame_paths(root_dirs):
    """Recursively collect all .png/.jpg paths under root_dirs."""
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
    """Subtract median-blurred version to isolate high-frequency content."""
    blurred = cv2.medianBlur(gray.astype(np.uint8), 9).astype(np.float32)
    return gray - blurred


def average_fft_spectrum(frames):
    """Return log-scale average magnitude spectrum (DC centred)."""
    acc = np.zeros(TARGET_SIZE, dtype=np.float64)
    for f in frames:
        hp   = high_pass(f)
        spec = np.fft.fftshift(np.fft.fft2(hp))
        acc += np.abs(spec)
    acc /= len(frames)
    return np.log1p(acc)


def average_dct_spectrum(frames):
    """Return log-scale average 2-D DCT-II coefficient magnitude."""
    acc = np.zeros(TARGET_SIZE, dtype=np.float64)
    for f in frames:
        hp   = high_pass(f)
        coef = dctn(hp, norm="ortho")       # scipy 2-D DCT-II
        acc += np.abs(coef)
    acc /= len(frames)
    return np.log1p(acc)


def sample_frames(paths, n, seed=RANDOM_SEED):
    rng = random.Random(seed)
    chosen = rng.sample(paths, min(n, len(paths)))
    frames = [f for p in chosen if (f := load_gray(p)) is not None]
    return frames


# ── Gather data ───────────────────────────────────────────────────────────────
print("Collecting frame paths …")
all_paths = {
    "Celeb-real":    collect_frame_paths(DATASETS["Celeb-real"]),
    "YouTube-real":  collect_frame_paths(DATASETS["YouTube-real"]),
    "DFDCP":         collect_frame_paths(DATASETS["DFDCP"]),
    "FaceForensics++": collect_frame_paths(FF_ROOTS),
}

for name, paths in all_paths.items():
    print(f"  {name}: {len(paths):,} frames found")

print(f"\nSampling {N_SAMPLES} frames per dataset and computing spectra …")
fft_maps = {}
dct_maps = {}
for name, paths in all_paths.items():
    if not paths:
        raise RuntimeError(f"No frames found for {name}")
    frames = sample_frames(paths, N_SAMPLES)
    print(f"  {name}: {len(frames)} frames loaded")
    fft_maps[name] = average_fft_spectrum(frames)
    dct_maps[name] = average_dct_spectrum(frames)

names = list(all_paths.keys())   # ["Celeb-real", "YouTube-real", "DFDCP", "FaceForensics++"]

# ── Pairwise differences  (C(4,2) = 6 pairs) ─────────────────────────────────
from itertools import combinations
pairs = list(combinations(names, 2))

fft_diffs = {(a, b): np.abs(fft_maps[a] - fft_maps[b]) for a, b in pairs}
dct_diffs = {(a, b): np.abs(dct_maps[a] - dct_maps[b]) for a, b in pairs}


# ── Plotting ──────────────────────────────────────────────────────────────────
CMAP_HEAT = "viridis"
CMAP_DIFF = "inferno"

SHORT = {
    "Celeb-real":    "Celeb-real",
    "YouTube-real":  "YouTube-real",
    "DFDCP":         "DFDCP",
    "FaceForensics++": "FF++",
}
PAIR_LABELS = {(a, b): f"{SHORT[a]} vs {SHORT[b]}" for a, b in pairs}

# Layout:
#   section 0 – FFT heatmaps  : 1 image-row × 4 cols
#   section 1 – DCT heatmaps  : 1 image-row × 4 cols
#   section 2 – FFT diffs     : 2 image-rows × 3 cols  (6 pairs)
#   section 3 – DCT diffs     : 2 image-rows × 3 cols  (6 pairs)
# height_ratios give diff sections twice the space of heatmap sections
fig = plt.figure(figsize=(24, 26))
fig.patch.set_facecolor("#0d0d0d")

outer = gridspec.GridSpec(4, 1, figure=fig, hspace=0.55,
                          height_ratios=[1, 1, 2, 2])

sections = [
    ("FFT Frequency Heatmaps\n(log-scale average amplitude spectrum, high-pass filtered)",
     names, fft_maps, CMAP_HEAT, SHORT, 1, 4),
    ("DCT Heatmaps\n(log-scale average 2-D DCT-II coefficient magnitude, high-pass filtered)",
     names, dct_maps, CMAP_HEAT, SHORT, 1, 4),
    ("FFT Pairwise Differences  |dataset A − dataset B|",
     pairs, fft_diffs, CMAP_DIFF, PAIR_LABELS, 2, 3),
    ("DCT Pairwise Differences  |dataset A − dataset B|",
     pairs, dct_diffs, CMAP_DIFF, PAIR_LABELS, 2, 3),
]

for row_idx, (section_title, keys, data_dict, cmap, label_map,
              n_rows_inner, n_cols_inner) in enumerate(sections):

    inner = gridspec.GridSpecFromSubplotSpec(
        n_rows_inner, n_cols_inner,
        subplot_spec=outer[row_idx],
        wspace=0.08, hspace=0.35,
    )

    # Invisible overlay axis just for the section title
    ax_title = fig.add_subplot(outer[row_idx])
    ax_title.set_axis_off()
    ax_title.set_title(section_title, color="white", fontsize=13,
                       fontweight="bold", pad=32)

    for idx, key in enumerate(keys):
        r, c = divmod(idx, n_cols_inner)
        ax = fig.add_subplot(inner[r, c])
        heatmap = data_dict[key]

        im = ax.imshow(heatmap, cmap=cmap, aspect="auto",
                       interpolation="bilinear")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
        cbar.outline.set_edgecolor("white")

        title_str = label_map[key] if isinstance(key, str) else PAIR_LABELS[key]
        ax.set_title(title_str, color="white", fontsize=10, pad=6)
        ax.set_xlabel("Frequency (u)" if row_idx < 2 else "FFT / DCT bin (u)",
                      color="gray", fontsize=7)
        ax.set_ylabel("Frequency (v)", color="gray", fontsize=7)
        ax.tick_params(colors="gray", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        # Red DC crosshair on FFT heatmap panels
        if row_idx == 0:
            h, w = heatmap.shape
            ax.axhline(h // 2, color="red", lw=0.5, alpha=0.4)
            ax.axvline(w // 2, color="red", lw=0.5, alpha=0.4)
            ax.text(w // 2 + 4, h // 2 - 8, "DC", color="red",
                    fontsize=7, alpha=0.7)

fig.suptitle(
    "Real-Dataset Frequency Analysis\n"
    "Celeb-real · YouTube-real · DFDCP · FaceForensics++ (original sequences)",
    color="white", fontsize=15, fontweight="bold", y=0.998,
)

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"\nSaved → {OUTPUT_PATH}")
