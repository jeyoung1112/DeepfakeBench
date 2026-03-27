"""
Phase spectra analysis for FaceForensics++ real vs. fake images.

Reads frame paths from a dataset JSON, samples up to --max frames per group,
and analyses two complementary phase statistics per frequency bin:

  1. Phase Coherence (Mean Resultant Length, MRL)
       MRL = |mean(exp(i·φ))| ∈ [0, 1]
       High → phase is consistent across images at that frequency.
       Low  → phase is random / incoherent.
       Natural images have low MRL (near-random phase); GANs sometimes leak
       consistent phase offsets at certain frequencies.

  2. Circular Mean Phase
       The average phase angle (via atan2 of accumulated sin/cos), useful
       for spotting systematic phase biases (e.g. ringing from upsampling).

Both are computed on the FFT2 magnitude-normalised spectrum (shift DC to
centre) so that spatial-frequency position is interpretable.

Output figure (3 rows × N columns):
  Row 0 — Phase coherence heatmap (jet, 0–1)
  Row 1 — Circular mean phase (hsv colormap, –π to π)
  Row 2 — Coherence difference from real (bwr)

Usage:
    python ff_phase_spectra.py
    python ff_phase_spectra.py --split test --max 200
    python ff_phase_spectra.py --pipeline c40 --out ff_phase_c40.png

Pipelines (--pipeline): c23 | c40 | whatsapp | facebook | twitter | instagram | phone
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config — mirrors ff_avg_dct_heatmap.py
# ---------------------------------------------------------------------------

JSON_PATH = "/media/NAS/DATASET/DeepfakeBench/original/dataset_json/FaceForensics++.json"
BASE_DIR  = "/media/NAS/DATASET/DeepfakeBench/original/dataset"

REAL_KEY    = "FF-real"
FAKE_KEYS   = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]
FAKE_LABELS = {
    "FF-DF":  "DeepFakes",
    "FF-F2F": "Face2Face",
    "FF-FS":  "FaceSwap",
    "FF-NT":  "NeuralTextures",
}

# ---------------------------------------------------------------------------
# Pipeline helpers (identical to ff_avg_dct_heatmap.py)
# ---------------------------------------------------------------------------

PIPELINES = ["c23", "c40", "whatsapp", "facebook", "twitter", "instagram", "phone"]


def win_to_posix(p: str) -> str:
    return p.replace("\\", "/")


def collect_paths(data: dict, key: str, split: str, compression: str) -> list[str]:
    group = data[key].get(split, {}).get(compression, {})
    paths = []
    for video in group.values():
        for rel in video.get("frames", []):
            paths.append(str(Path(BASE_DIR) / win_to_posix(rel)))
    return paths


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


def _add_gaussian_noise(img: np.ndarray, sigma: float = 8.0,
                        rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_pipeline(img: np.ndarray, pipeline: str, rng: np.random.Generator) -> np.ndarray:
    if pipeline == "c23":
        return img
    if pipeline == "c40":
        return _jpeg(img, 40)
    if pipeline == "whatsapp":
        return _jpeg(_resize_long_side(img, 1600), 75)
    if pipeline == "facebook":
        return _unsharp_mask(_jpeg(img, 85), sigma=0.8, strength=0.3)
    if pipeline == "twitter":
        return _jpeg(_resize_long_side(img, 1280), 85)
    if pipeline == "instagram":
        return _unsharp_mask(_jpeg(img, 85), sigma=1.0, strength=0.5)
    if pipeline == "phone":
        img = _add_gaussian_noise(img, sigma=8.0, rng=rng)
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.8)
        return _jpeg(img, 92)
    raise ValueError(f"Unknown pipeline: {pipeline}")


# ---------------------------------------------------------------------------
# Phase statistics accumulation
# ---------------------------------------------------------------------------

class PhaseAccumulator:
    """
    Accumulates circular statistics of FFT phase across multiple images.

    Per frequency bin we track:
      sum_cos, sum_sin  — for circular mean and MRL
      count             — number of valid images contributed
    """

    def __init__(self, shape: tuple[int, int]):
        h, w = shape
        self.sum_cos = np.zeros((h, w), dtype=np.float64)
        self.sum_sin = np.zeros((h, w), dtype=np.float64)
        self.count = 0

    def update(self, img: np.ndarray):
        """Accept a float32 grayscale image (already at target size)."""
        F = np.fft.fft2(img.astype(np.float64))
        F = np.fft.fftshift(F)          # DC at centre
        phase = np.angle(F)             # ∈ [−π, π]
        self.sum_cos += np.cos(phase)
        self.sum_sin += np.sin(phase)
        self.count += 1

    def coherence(self) -> np.ndarray:
        """Mean Resultant Length ∈ [0, 1]. High = phase-consistent."""
        if self.count == 0:
            raise RuntimeError("No images accumulated.")
        mrl = np.sqrt((self.sum_cos / self.count) ** 2 +
                      (self.sum_sin / self.count) ** 2)
        return mrl.astype(np.float32)

    def mean_phase(self) -> np.ndarray:
        """Circular mean phase ∈ [−π, π]."""
        if self.count == 0:
            raise RuntimeError("No images accumulated.")
        return np.arctan2(self.sum_sin, self.sum_cos).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-group computation
# ---------------------------------------------------------------------------

def compute_phase_stats(
    paths: list[str],
    size: tuple[int, int],
    max_images: int,
    seed: int = 42,
    pipeline: str = "c23",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (coherence_map, mean_phase_map) for the sampled group.
    coherence_map ∈ [0, 1], mean_phase_map ∈ [−π, π].
    """
    random.seed(seed)
    sampled = random.sample(paths, min(max_images, len(paths)))
    rng = np.random.default_rng(seed)

    acc = PhaseAccumulator(shape=(size[1], size[0]))  # (H, W)

    for p in sampled:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = apply_pipeline(img, pipeline, rng)
        if img.shape[:2] != (size[1], size[0]):
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        acc.update(img)

    print(f"  Accumulated {acc.count} frames.")
    return acc.coherence(), acc.mean_phase()


# ---------------------------------------------------------------------------
# Radial profile helper (optional supplementary plot)
# ---------------------------------------------------------------------------

def radial_profile(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average img values in concentric rings from DC centre."""
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    r_max = min(cx, cy)
    radii = np.arange(0, r_max)
    profile = np.array([img[r == ri].mean() if (r == ri).any() else 0.0
                        for ri in radii])
    return radii, profile


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _norm01(m: np.ndarray) -> np.ndarray:
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo + 1e-8)


def _log_coherence(coh: np.ndarray) -> np.ndarray:
    """Log-compress coherence for better contrast at low values."""
    return np.log1p(coh * 99) / np.log1p(99)  # maps [0,1]→[0,1] with log stretch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase spectra analysis for FaceForensics++ real vs. fake."
    )
    parser.add_argument("--split",    default="test",  help="train / val / test")
    parser.add_argument("--pipeline", default="c23",   choices=PIPELINES)
    parser.add_argument("--max",      type=int, default=300,
                        help="Max frames sampled per group (default: 300)")
    parser.add_argument("--size",     nargs=2, type=int, default=[256, 256],
                        metavar=("W", "H"))
    parser.add_argument("--out",      help="Save figure to this path")
    parser.add_argument("--radial",   action="store_true",
                        help="Also plot radial coherence profiles")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    size = tuple(args.size)  # (W, H)

    print(f"Loading JSON: {JSON_PATH}")
    print(f"Pipeline   : {args.pipeline}  |  Split: {args.split}  |  Max: {args.max}")
    with open(JSON_PATH) as f:
        raw = json.load(f)
    data = raw["FaceForensics++"]

    # ---- compute stats for each group ----------------------------------------
    groups_coh  = {}   # coherence maps
    groups_mph  = {}   # mean phase maps

    print(f"\n[Real] {REAL_KEY}")
    real_paths = collect_paths(data, REAL_KEY, args.split, "c23")
    print(f"  Found {len(real_paths)} frames")
    groups_coh[REAL_KEY], groups_mph[REAL_KEY] = compute_phase_stats(
        real_paths, size, args.max, args.seed, args.pipeline)

    fake_cohs = []
    for fk in FAKE_KEYS:
        print(f"\n[Fake] {fk}")
        paths = collect_paths(data, fk, args.split, "c23")
        print(f"  Found {len(paths)} frames")
        coh, mph = compute_phase_stats(paths, size, args.max, args.seed, args.pipeline)
        groups_coh[fk] = coh
        groups_mph[fk] = mph
        fake_cohs.append(coh)

    # Combined fake average coherence
    combined_coh = np.mean(fake_cohs, axis=0)
    groups_coh["combined_fake"] = combined_coh

    # ---- layout ---------------------------------------------------------------
    col_keys   = [REAL_KEY] + FAKE_KEYS + ["combined_fake"]
    col_titles = ["Real"] + [FAKE_LABELS[k] for k in FAKE_KEYS] + ["Avg Fake"]
    n_cols = len(col_keys)

    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    fig.suptitle(
        f"Phase Spectra — FaceForensics++ ({args.split}, pipeline={args.pipeline})",
        fontsize=13,
    )

    row_labels = [
        "Phase Coherence\n(MRL, log-scaled)",
        "Circular Mean Phase\n(−π to π)",
        "Coherence Diff\nvs Real (bwr)",
    ]
    for row, lbl in enumerate(row_labels):
        axes[row, 0].set_ylabel(lbl, fontsize=8)

    real_coh = groups_coh[REAL_KEY]

    for col, (key, title) in enumerate(zip(col_keys, col_titles)):
        coh = groups_coh[key]
        log_coh = _log_coherence(coh)

        # Row 0: phase coherence (log-scaled, jet)
        axes[0, col].imshow(log_coh, cmap="jet", vmin=0, vmax=1)
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].axis("off")

        # Row 1: circular mean phase (hsv) — only defined for non-combined groups
        ax1 = axes[1, col]
        if key in groups_mph:
            mph = groups_mph[key]
            # map [−π, π] → [0, 1] for hsv colormap
            ax1.imshow((mph + np.pi) / (2 * np.pi), cmap="hsv", vmin=0, vmax=1)
        else:
            # combined fake: show mean of fake phase maps instead
            fake_mphs = np.stack([groups_mph[fk] for fk in FAKE_KEYS], axis=0)
            combined_sin = np.mean(np.sin(fake_mphs), axis=0)
            combined_cos = np.mean(np.cos(fake_mphs), axis=0)
            avg_mph = np.arctan2(combined_sin, combined_cos)
            ax1.imshow((avg_mph + np.pi) / (2 * np.pi), cmap="hsv", vmin=0, vmax=1)
        ax1.axis("off")

        # Row 2: coherence difference from real
        ax2 = axes[2, col]
        if key == REAL_KEY:
            ax2.axis("off")
            ax2.set_title("(reference)", fontsize=8, color="grey")
        else:
            diff = coh - real_coh           # + → more coherent than real
            vmax = np.abs(diff).max()
            im = ax2.imshow(diff, cmap="bwr", vmin=-vmax, vmax=vmax)
            ax2.set_title(f"{title} − Real", fontsize=9)
            ax2.axis("off")
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.02)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {args.out}")
    else:
        plt.show()

    # ---- optional radial profile plot ----------------------------------------
    if args.radial:
        fig2, ax = plt.subplots(figsize=(9, 4))
        ax.set_title(
            f"Radial Phase Coherence Profile — {args.split}, pipeline={args.pipeline}",
            fontsize=11,
        )
        colors = plt.cm.tab10(np.linspace(0, 1, n_cols))

        for (key, title), color in zip(zip(col_keys, col_titles), colors):
            coh = groups_coh[key]
            radii, profile = radial_profile(coh)
            lw = 2.0 if key == REAL_KEY else 1.2
            ls = "-" if key == REAL_KEY else "--" if key != "combined_fake" else "-."
            ax.plot(radii, profile, label=title, color=color, linewidth=lw, linestyle=ls)

        ax.set_xlabel("Radial frequency (pixels from DC)")
        ax.set_ylabel("Mean Phase Coherence (MRL)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if args.out:
            radial_out = str(Path(args.out).with_stem(Path(args.out).stem + "_radial"))
            fig2.savefig(radial_out, dpi=150, bbox_inches="tight")
            print(f"Saved radial plot: {radial_out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
