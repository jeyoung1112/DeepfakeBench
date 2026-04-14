"""
DFM (Discrete Fourier Magnitude) Frequency Analysis for images.

Computes the 2D DFT magnitude of each image (grayscale), visualises the
log-magnitude spectrum, and plots the mean frequency energy distribution
across all supplied images.  Useful for comparing real vs. fake images in
deepfake research.

Usage:
    python dct_frequency_analysis.py --images img1.jpg img2.png ...
    python dct_frequency_analysis.py --dir /path/to/images --ext jpg png
    python dct_frequency_analysis.py --dir /path/to/images --compare real/ fake/
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR  = "/media/NAS/DATASET/DeepfakeBench/original/dataset"
# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_gray(path: str, size: tuple[int, int] | None = (256, 256)) -> np.ndarray:
    """Load image as float32 grayscale, optionally resized."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def compute_dfm(img: np.ndarray) -> np.ndarray:
    """2-D Discrete Fourier Magnitude of a single-channel float image (DC centred)."""
    return np.abs(np.fft.fftshift(np.fft.fft2(img)))


def log_spectrum(dfm: np.ndarray) -> np.ndarray:
    """Log-magnitude of a DFM array (DC already centred)."""
    return np.log1p(dfm)


def phase_spectrum(img: np.ndarray) -> np.ndarray:
    """
    Phase angle (radians) of the 2D FFT, shifted so DC is at centre.
    Values are in [-π, π].  Use a cyclic colormap (e.g. 'hsv') to display.
    """
    f = np.fft.fft2(img)
    return np.angle(np.fft.fftshift(f))


def radial_energy(dfm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean energy as a function of radial frequency (distance from DC).
    Returns (frequencies, energies) arrays.
    """
    h, w = dfm.shape
    cy, cx = h // 2, w // 2
    power = dfm ** 2

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r = min(cx, cy)
    energies = np.array([power[r == radius].mean() for radius in range(max_r)])
    freqs = np.arange(max_r)
    return freqs, energies


# ---------------------------------------------------------------------------
# Analysis and plotting
# ---------------------------------------------------------------------------

def analyse_images(
    paths: list[str],
    label: str = "images",
    size: tuple[int, int] = (256, 256),
) -> dict:
    """Return per-image DFMs and aggregate radial energy for a list of paths."""
    dfms, spectra, radial_curves = [], [], []

    phases = []
    for p in paths:
        try:
            img = load_gray(p, size)
            dfm = compute_dfm(img)
            freqs, energies = radial_energy(dfm)
            dfms.append(dfm)
            spectra.append(log_spectrum(dfm))
            phases.append(phase_spectrum(img))
            radial_curves.append(energies)
        except Exception as e:
            print(f"  [skip] {p}: {e}")

    if not dfms:
        raise RuntimeError(f"No valid images loaded for label '{label}'")

    return {
        "label": label,
        "paths": paths,
        "dfms": dfms,
        "spectra": spectra,
        "mean_spectrum": np.mean(spectra, axis=0),
        "mean_phase": np.mean(phases, axis=0),
        "freqs": freqs,
        "mean_radial": np.mean(radial_curves, axis=0),
        "std_radial": np.std(radial_curves, axis=0),
    }


def plot_single_group(result: dict, out_path: str | None = None):
    """Plot individual log-spectra + mean spectrum + radial energy."""
    n = min(len(result["spectra"]), 6)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(14, 4 * (nrows + 2)))
    fig.suptitle(f"DFM Frequency Analysis — {result['label']}", fontsize=14)

    # Individual spectra
    for i in range(n):
        ax = fig.add_subplot(nrows + 2, ncols, i + 1)
        ax.imshow(result["spectra"][i], cmap="inferno", origin="upper")
        ax.set_title(Path(result["paths"][i]).name, fontsize=7)
        ax.axis("off")

    # Mean spectrum + phase side by side
    ax_mean = fig.add_subplot(nrows + 3, 2, (nrows + 1) * 2 - 1)
    im = ax_mean.imshow(result["mean_spectrum"], cmap="inferno", origin="upper")
    ax_mean.set_title("Mean log-DFM spectrum")
    ax_mean.axis("off")
    plt.colorbar(im, ax=ax_mean, fraction=0.04)

    ax_phase = fig.add_subplot(nrows + 3, 2, (nrows + 1) * 2)
    im_p = ax_phase.imshow(result["mean_phase"], cmap="hsv", origin="upper",
                           vmin=-np.pi, vmax=np.pi)
    ax_phase.set_title("Mean FFT phase spectrum")
    ax_phase.axis("off")
    plt.colorbar(im_p, ax=ax_phase, fraction=0.04, label="phase (rad)")

    # Radial energy curve
    ax_r = fig.add_subplot(nrows + 3, 1, nrows + 3)
    freqs, mean_e, std_e = result["freqs"], result["mean_radial"], result["std_radial"]
    ax_r.semilogy(freqs, mean_e, label=result["label"])
    ax_r.fill_between(freqs, mean_e - std_e, mean_e + std_e, alpha=0.2)
    ax_r.set_xlabel("Radial frequency (pixels from DC)")
    ax_r.set_ylabel("Mean energy (log scale)")
    ax_r.set_title("Radial frequency energy")
    ax_r.legend()
    ax_r.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()


def plot_four_group_analysis(results_by_key: dict, out_path: str | None = None):
    """
    Full dashboard for TP / TN / FP / FN categories, mirroring the reference figure style.

    Layout (using GridSpec):
      Row 0  — mean DCT log-spectra for each of the 4 groups
      Row 1  — difference maps vs TP (baseline); TP column shows a text label
      Rows 2-3 — 2×2 pairwise radial-energy comparisons:
                   TP vs TN  |  FP vs FN
                   TP vs FN  |  TN vs FP
    """
    import matplotlib.gridspec as gridspec

    KEYS = ["TP", "TN", "FP", "FN"]
    results = {k: results_by_key[k] for k in KEYS if k in results_by_key}
    present = [k for k in KEYS if k in results]
    n = len(present)
    if n < 2:
        raise RuntimeError("Need at least 2 non-empty categories.")

    colors = {"TP": "#2196F3", "TN": "#4CAF50", "FP": "#FF9800", "FN": "#F44336"}
    baseline = present[0]  # TP when available

    fig = plt.figure(figsize=(6 * n, 34))
    fig.suptitle("DFM Frequency Analysis — TP / TN / FP / FN", fontsize=15, y=0.99)
    gs = gridspec.GridSpec(6, n, figure=fig, hspace=0.45, wspace=0.35,
                           height_ratios=[1, 1, 1, 1, 1.2, 1.2])

    # ── Row 0: mean log-DFM spectra ────────────────────────────────────────
    for col, key in enumerate(present):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(results[key]["mean_spectrum"], cmap="inferno", origin="upper")
        ax.set_title(key, fontsize=12, fontweight="bold", color=colors.get(key, "black"))
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # ── Row 2: difference maps vs baseline (TP) ────────────────────────────
    comparison_target = [("TP", "TN"), ("TP", "FP"), ("TN", "FN"),]
    for col, target_pairs in enumerate(comparison_target):
        ax = fig.add_subplot(gs[1, col])
        diff = results[target_pairs[0]]["mean_spectrum"] - results[target_pairs[1]]["mean_spectrum"]
        vmax = np.abs(diff).max()
        im2 = ax.imshow(diff, cmap="bwr", origin="upper", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{target_pairs[0]} vs {target_pairs[1]}", fontsize=9)
        ax.axis("off")
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # ── Row 1: mean FFT phase spectra ──────────────────────────────────────
    for col, key in enumerate(present):
        ax = fig.add_subplot(gs[2, col])
        im_p = ax.imshow(results[key]["mean_phase"], cmap="hsv", origin="upper",
                         vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"{key} phase", fontsize=10, color=colors.get(key, "black"))
        ax.axis("off")
        plt.colorbar(im_p, ax=ax, fraction=0.046, pad=0.04, label="rad")


    for col, target_pairs in enumerate(comparison_target):
        ax = fig.add_subplot(gs[3, col])
        diff = results[target_pairs[0]]["mean_phase"] - results[target_pairs[1]]["mean_phase"]
        vmax = np.abs(diff).max()
        im2 = ax.imshow(diff, cmap="bwr", origin="upper", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{target_pairs[0]} vs {target_pairs[1]}", fontsize=9)
        ax.axis("off")
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # ── Rows 3-4: 2×2 pairwise radial-energy comparisons ──────────────────
    half = n // 2
    pair_layout = [
        (4, slice(0, half),    "TP", "TN"),
        (4, slice(half, n),    "FP", "FN"),
        (5, slice(0, half),    "TP", "FN"),
        (5, slice(half, n),    "TN", "FP"),
    ]
    for row, cols, key_a, key_b in pair_layout:
        if key_a not in results or key_b not in results:
            continue
        ax = fig.add_subplot(gs[row, cols])
        for key, ls in [(key_a, "-"), (key_b, "--")]:
            res = results[key]
            freqs, mean_e, std_e = res["freqs"], res["mean_radial"], res["std_radial"]
            c = colors.get(key)
            ax.semilogy(freqs, mean_e, label=key, color=c, linestyle=ls, linewidth=1.5)
            ax.fill_between(freqs, np.maximum(mean_e - std_e, 1e-10),
                            mean_e + std_e, alpha=0.15, color=c)
        ax.set_xlabel("Radial frequency (pixels from DC)", fontsize=8)
        ax.set_ylabel("Mean energy (log scale)", fontsize=8)
        ax.set_title(f"{key_a} vs {key_b}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()


def plot_comparison(results: list[dict], out_path: str | None = None):
    """Compare mean spectra and radial energy curves across groups."""
    n_groups = len(results)
    fig, axes = plt.subplots(3, n_groups + 1, figsize=(5 * (n_groups + 1), 14))
    fig.suptitle("DFM Frequency Comparison", fontsize=14)

    colors = plt.cm.tab10.colors

    for col, res in enumerate(results):
        # Mean log-DFM spectrum per group
        im = axes[0, col].imshow(res["mean_spectrum"], cmap="inferno", origin="upper")
        axes[0, col].set_title(f"Mean spectrum\n{res['label']}", fontsize=9)
        axes[0, col].axis("off")
        plt.colorbar(im, ax=axes[0, col], fraction=0.04)

        # Mean FFT phase spectrum per group
        im_p = axes[1, col].imshow(res["mean_phase"], cmap="hsv", origin="upper",
                                   vmin=-np.pi, vmax=np.pi)
        axes[1, col].set_title(f"Mean phase\n{res['label']}", fontsize=9)
        axes[1, col].axis("off")
        plt.colorbar(im_p, ax=axes[1, col], fraction=0.04, label="rad")

    # Difference map (first two groups) — magnitude row
    if n_groups >= 2:
        diff = results[0]["mean_spectrum"] - results[1]["mean_spectrum"]
        im2 = axes[0, -1].imshow(diff, cmap="bwr", origin="upper",
                                  vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
        axes[0, -1].set_title(f"Diff (magnitude)\n{results[0]['label']} − {results[1]['label']}", fontsize=9)
        axes[0, -1].axis("off")
        plt.colorbar(im2, ax=axes[0, -1], fraction=0.04)

        # Phase difference (circular: wrap to [-π, π])
        phase_diff = results[0]["mean_phase"] - results[1]["mean_phase"]
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
        im_pd = axes[1, -1].imshow(phase_diff, cmap="bwr", origin="upper",
                                    vmin=-np.pi, vmax=np.pi)
        axes[1, -1].set_title(f"Diff (phase)\n{results[0]['label']} − {results[1]['label']}", fontsize=9)
        axes[1, -1].axis("off")
        plt.colorbar(im_pd, ax=axes[1, -1], fraction=0.04, label="rad")
    else:
        axes[0, -1].axis("off")
        axes[1, -1].axis("off")

    # Radial energy curves — all groups on one plot (row 2)
    ax_r = axes[2, :]
    for a in ax_r[1:]:
        a.axis("off")
    ax_r = axes[2, 0]
    ax_r.set_position([0.08, 0.05, 0.88, 0.25])

    for idx, res in enumerate(results):
        freqs, mean_e, std_e = res["freqs"], res["mean_radial"], res["std_radial"]
        c = colors[idx % len(colors)]
        ax_r.semilogy(freqs, mean_e, label=res["label"], color=c)
        ax_r.fill_between(freqs, np.maximum(mean_e - std_e, 1e-10),
                          mean_e + std_e, alpha=0.15, color=c)

    ax_r.set_xlabel("Radial frequency (pixels from DC)")
    ax_r.set_ylabel("Mean energy (log scale)")
    ax_r.set_title("Radial frequency energy — comparison")
    ax_r.legend()
    ax_r.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Heatmap-style visualisation (image + DFM heatmap pairs, like ref figure)
# ---------------------------------------------------------------------------

def dfm_heatmap(dfm: np.ndarray) -> np.ndarray:
    """
    Convert DFM to a uint8 heatmap image (jet colormap).
    Uses log-magnitude of the DFM — matches the spatial layout typical in
    deepfake-detection figures.
    """
    magnitude = np.log1p(dfm)
    norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    heatmap = (plt.cm.jet(norm)[:, :, :3] * 255).astype(np.uint8)
    return heatmap


def plot_heatmap_pairs(
    pairs: list[tuple[str, str]],
    labels: tuple[str, str] = ("Real", "Fake"),
    size: tuple[int, int] = (256, 256),
    out_path: str | None = None,
):
    """
    Show (image, DFM heatmap) pairs side by side, mirroring the reference figure.

    pairs: list of (real_path, fake_path) tuples.  Pass [(path, None)] for
           single-image mode.
    """
    n_pairs = len(pairs)
    # Layout: each pair occupies 2 columns (real | fake), each column has 2 rows
    # (original image on top, heatmap below).
    fig, axes = plt.subplots(2, 2 * n_pairs, figsize=(4 * 2 * n_pairs, 8))
    if n_pairs == 1:
        axes = axes.reshape(2, -1)  # ensure 2-D indexing works

    col_labels = []
    for pair_idx, (path_a, path_b) in enumerate(pairs):
        for side, path, lbl in [(0, path_a, labels[0]), (1, path_b, labels[1])]:
            col = pair_idx * 2 + side
            ax_img = axes[0, col]
            ax_heat = axes[1, col]

            if path is None:
                ax_img.axis("off")
                ax_heat.axis("off")
                continue

            img = load_gray(path, size)
            dfm = compute_dfm(img)
            heat = dfm_heatmap(dfm)

            # Original image (shown as grayscale but with colour for context)
            orig_bgr = cv2.imread(path)
            if orig_bgr is not None:
                orig_rgb = cv2.cvtColor(
                    cv2.resize(orig_bgr, size, interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB,
                )
                ax_img.imshow(orig_rgb)
            else:
                ax_img.imshow(img, cmap="gray")

            ax_img.set_title(lbl, fontsize=11, style="italic", fontweight="bold")
            ax_img.axis("off")

            ax_heat.imshow(heat)
            ax_heat.axis("off")

            # Panel label (a), (b), …
            panel_letter = chr(ord("a") + col)
            ax_heat.set_xlabel(f"({panel_letter})", fontsize=10, labelpad=4)
            ax_heat.xaxis.set_label_position("bottom")
            ax_heat.tick_params(bottom=False, labelbottom=False)
            ax_heat.set_frame_on(False)
            # Re-enable xlabel rendering without ticks
            ax_heat.set_xlabel(f"({panel_letter})", fontsize=10)

    # Dashed divider between pairs (drawn as a vertical line in figure coords)
    if n_pairs > 1:
        for i in range(1, n_pairs):
            x = i / n_pairs
            fig.add_artist(
                plt.Line2D([x, x], [0.02, 0.98], transform=fig.transFigure,
                           color="black", linewidth=1.2, linestyle="--")
            )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_images(directory: str, exts: list[str]) -> list[str]:
    exts = {e.lower().lstrip(".") for e in exts}
    paths = []
    for f in sorted(Path(directory).rglob("*")):
        if f.suffix.lstrip(".").lower() in exts:
            paths.append(str(f))
    return paths


def main():
    parser = argparse.ArgumentParser(description="DFM frequency analysis of images.")
    parser.add_argument("--images", nargs="+", help="Explicit image paths")
    parser.add_argument("--dir", help="Directory to scan for images")
    parser.add_argument("--ext", nargs="+", default=["jpg", "jpeg", "png"],
                        help="Extensions to include when scanning a directory (default: jpg jpeg png)")
    parser.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"),
                        help="Compare two directories (e.g. real/ fake/)")
    parser.add_argument("--labels", nargs=2, default=["group_A", "group_B"],
                        help="Labels for --compare groups")
    parser.add_argument("--size", nargs=2, type=int, default=[256, 256],
                        metavar=("W", "H"), help="Resize images to W×H before DFM")
    parser.add_argument("--max", type=int, default=200,
                        help="Max images per group (default: 50)")
    parser.add_argument("--out", help="Save plot to this path instead of displaying", default="analysis/classification_comparison.png")
    parser.add_argument("--heatmap", nargs="+",
                        help="Show image+heatmap pairs. Provide paths as: real1 fake1 real2 fake2 ...")
    parser.add_argument("--heatmap-single", nargs="+", dest="heatmap_single",
                        help="Show heatmap for individual images (no pairing)")
    parser.add_argument("--from-json", dest="from_json",
                        help="Path to image-paths JSON (TP/TN/FP/FN) produced by test_clip_lora.py",
                        default="confusion_matrix_results/clip_lora_image_paths_detected.json")
    args = parser.parse_args()

    size = tuple(args.size)

    if args.from_json:
        import json
        with open(args.from_json) as f:
            categorised = json.load(f)

        results_by_key = {}
        for key in ("tp", "tn", "fp", "fn"):
            paths = categorised.get(key, [])[:args.max]
            if not paths:
                print(f"  [skip] '{key}': no images in JSON")
                continue
            print(f"  {key.upper()}: {len(paths)} images")
            paths = [os.path.join(BASE_DIR, p) for p in paths]
            results_by_key[key.upper()] = analyse_images(paths, label=key.upper(), size=size)

        if len(results_by_key) < 2:
            raise RuntimeError("Need at least 2 non-empty categories to compare.")
        plot_four_group_analysis(results_by_key, out_path=args.out)

    elif args.heatmap:
        paths = args.heatmap
        if len(paths) % 2 != 0:
            parser.error("--heatmap requires an even number of paths (real fake real fake ...)")
        pairs = [(paths[i], paths[i + 1]) for i in range(0, len(paths), 2)]
        lbl = tuple(args.labels) if args.labels != ["group_A", "group_B"] else ("Real", "Fake")
        plot_heatmap_pairs(pairs, labels=lbl, size=size, out_path=args.out)

    elif args.heatmap_single:
        pairs = [(p, None) for p in args.heatmap_single]
        plot_heatmap_pairs(pairs, labels=("Image", ""), size=size, out_path=args.out)

    elif args.compare:
        dir_a, dir_b = args.compare
        paths_a = collect_images(dir_a, args.ext)[: args.max]
        paths_b = collect_images(dir_b, args.ext)[: args.max]
        print(f"Group '{args.labels[0]}': {len(paths_a)} images from {dir_a}")
        print(f"Group '{args.labels[1]}': {len(paths_b)} images from {dir_b}")
        res_a = analyse_images(paths_a, label=args.labels[0], size=size)
        res_b = analyse_images(paths_b, label=args.labels[1], size=size)
        plot_comparison([res_a, res_b], out_path=args.out)

    elif args.dir:
        paths = collect_images(args.dir, args.ext)[: args.max]
        print(f"Found {len(paths)} images in {args.dir}")
        result = analyse_images(paths, label=os.path.basename(args.dir.rstrip("/")), size=size)
        plot_single_group(result, out_path=args.out)

    elif args.images:
        paths = args.images[: args.max]
        result = analyse_images(paths, label="input images", size=size)
        plot_single_group(result, out_path=args.out)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
