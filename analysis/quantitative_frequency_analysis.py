"""
TP/TN/FP/FN Frequency Domain Quantitative Analysis
===================================================
모델의 예측 결과를 4가지 카테고리로 분류하고,
각 카테고리 간 magnitude/phase 차이를 정량적으로 측정합니다.

출력: 논문에 바로 넣을 수 있는 테이블 수치
"""

import argparse
import json
import os

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from collections import defaultdict
import pandas as pd

BASE_DIR = "/media/NAS/DATASET/DeepfakeBench/original/dataset"


# ============================================================
# 1. 데이터를 TP/TN/FP/FN으로 분류
# ============================================================

def split_predictions(images, labels, scores, threshold=0.5):
    """
    images:  (N, C, H, W) tensor
    labels:  (N,) — 0=Real, 1=Fake
    scores:  (N,) — 모델의 Fake 확률 (0~1)
    threshold: 판정 기준
    
    Returns: dict of {"TP": tensor, "TN": tensor, "FP": tensor, "FN": tensor}
    """
    preds = (scores > threshold).long()
    
    tp_mask = (preds == 1) & (labels == 1)  # Fake을 Fake로 맞힘
    tn_mask = (preds == 0) & (labels == 0)  # Real을 Real로 맞힘
    fp_mask = (preds == 1) & (labels == 0)  # Real을 Fake로 오분류
    fn_mask = (preds == 0) & (labels == 1)  # Fake를 Real로 미탐
    
    result = {
        "TP": images[tp_mask],
        "TN": images[tn_mask],
        "FP": images[fp_mask],
        "FN": images[fn_mask],
    }
    
    print(f"TP: {tp_mask.sum()}, TN: {tn_mask.sum()}, "
          f"FP: {fp_mask.sum()}, FN: {fn_mask.sum()}")
    
    return result


# ============================================================
# 2. Frequency 특징 추출 함수들
# ============================================================

def compute_magnitude_spectrum(imgs):
    """배치의 평균 magnitude spectrum 계산"""
    # imgs: (N, C, H, W)
    gray = imgs.mean(dim=1, keepdim=True)  # (N, 1, H, W)
    X = torch.fft.fft2(gray, dim=(-2, -1))
    X_shifted = torch.fft.fftshift(X, dim=(-2, -1))
    magnitude = torch.log1p(torch.abs(X_shifted))  # log scale
    return magnitude  # (N, 1, H, W)


def compute_phase_spectrum(imgs):
    """배치의 phase spectrum 계산"""
    gray = imgs.mean(dim=1, keepdim=True)
    X = torch.fft.fft2(gray, dim=(-2, -1))
    X_shifted = torch.fft.fftshift(X, dim=(-2, -1))
    phase = torch.angle(X_shifted)  # [-pi, pi]
    return phase  # (N, 1, H, W)


def compute_phase_per_channel(imgs):
    """채널별 phase spectrum 계산 (inter-channel coherence 분석용)"""
    X = torch.fft.fft2(imgs, dim=(-2, -1))
    X_shifted = torch.fft.fftshift(X, dim=(-2, -1))
    phase = torch.angle(X_shifted)
    return phase  # (N, C, H, W)


def compute_radial_profile(spectrum_2d, num_bins=128):
    """2D spectrum → 1D radial profile"""
    # spectrum_2d: (H, W)
    H, W = spectrum_2d.shape
    cy, cx = H // 2, W // 2
    
    y = torch.arange(H, device=spectrum_2d.device).float() - cy
    x = torch.arange(W, device=spectrum_2d.device).float() - cx
    radius = torch.sqrt(y[:, None]**2 + x[None, :]**2)
    
    max_r = min(cy, cx)
    bin_idx = (radius / max_r * num_bins).long().clamp(0, num_bins - 1)
    
    profile = torch.zeros(num_bins, device=spectrum_2d.device)
    counts = torch.zeros(num_bins, device=spectrum_2d.device)
    
    flat_spec = spectrum_2d.flatten()
    flat_bins = bin_idx.flatten()
    
    profile.scatter_add_(0, flat_bins, flat_spec)
    counts.scatter_add_(0, flat_bins, torch.ones_like(flat_spec))
    
    profile = profile / counts.clamp(min=1)
    return profile.cpu().numpy()


# ============================================================
# 3. 정량적 비교 Metric 함수들
# ============================================================

def magnitude_difference_stats(group_a, group_b):
    """
    두 그룹 간 magnitude spectrum 차이의 통계량
    
    Returns:
      mean_abs_diff:  평균 절대 차이 (그림의 colorbar 스케일에 대응)
      max_abs_diff:   최대 절대 차이
      std_diff:       차이의 표준편차
      energy_ratio:   에너지 비율 (group_a / group_b)
    """
    mag_a = compute_magnitude_spectrum(group_a).mean(dim=0).squeeze()  # (H, W)
    mag_b = compute_magnitude_spectrum(group_b).mean(dim=0).squeeze()  # (H, W)
    
    diff = mag_a - mag_b
    
    return {
        "mean_abs_diff": diff.abs().mean().item(),
        "max_abs_diff": diff.abs().max().item(),
        "std_diff": diff.std().item(),
        "median_abs_diff": diff.abs().median().item(),
        "energy_ratio": (mag_a.pow(2).sum() / mag_b.pow(2).sum()).item(),
    }


def phase_difference_stats(group_a, group_b):
    """
    두 그룹 간 phase spectrum 차이의 통계량
    Phase는 circular이므로 circular statistics 사용
    """
    phase_a = compute_phase_spectrum(group_a).mean(dim=0).squeeze()  # (H, W)
    phase_b = compute_phase_spectrum(group_b).mean(dim=0).squeeze()  # (H, W)
    
    # Circular difference
    diff = phase_a - phase_b
    # Wrap to [-pi, pi]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    
    return {
        "mean_abs_diff": diff.abs().mean().item(),
        "max_abs_diff": diff.abs().max().item(),
        "std_diff": diff.std().item(),
        "median_abs_diff": diff.abs().median().item(),
        # Circular variance (0=identical, 1=completely random)
        "circular_variance": (1 - torch.sqrt(
            torch.cos(diff).mean()**2 + torch.sin(diff).mean()**2
        )).item(),
    }


def phase_coherence_comparison(group_a, group_b, num_bins=64):
    """
    두 그룹의 radial phase coherence (MRL) 비교
    
    Returns:
      mrl_diff_mean:    MRL 차이의 평균
      mrl_diff_by_freq: 주파수별 MRL 차이
    """
    def compute_mrl(imgs, num_bins):
        """Mean Resultant Length per radial frequency"""
        phase = compute_phase_per_channel(imgs)  # (N, C, H, W)
        # 채널 평균
        phase = phase.mean(dim=1)  # (N, H, W)
        N, H, W = phase.shape
        cy, cx = H // 2, W // 2
        
        y = torch.arange(H, device=phase.device).float() - cy
        x = torch.arange(W, device=phase.device).float() - cx
        radius = torch.sqrt(y[:, None]**2 + x[None, :]**2)
        
        max_r = min(cy, cx)
        bin_idx = (radius / max_r * num_bins).long().clamp(0, num_bins - 1)
        
        mrl_per_bin = []
        for b in range(num_bins):
            mask = bin_idx == b
            if mask.sum() == 0:
                mrl_per_bin.append(0.0)
                continue
            phases_in_bin = phase[:, mask]  # (N, num_pixels_in_bin)
            # MRL = |mean of exp(j*phase)|
            complex_mean = torch.exp(1j * phases_in_bin.float()).mean()
            mrl = torch.abs(complex_mean).item()
            mrl_per_bin.append(mrl)
        
        return np.array(mrl_per_bin)
    
    mrl_a = compute_mrl(group_a, num_bins)
    mrl_b = compute_mrl(group_b, num_bins)
    
    diff = mrl_a - mrl_b
    
    return {
        "mrl_diff_mean": np.abs(diff).mean(),
        "mrl_diff_max": np.abs(diff).max(),
        "mrl_diff_low_freq": np.abs(diff[:num_bins//4]).mean(),   # 저주파 차이
        "mrl_diff_high_freq": np.abs(diff[num_bins//4:]).mean(),  # 고주파 차이
        "mrl_profile_a": mrl_a,
        "mrl_profile_b": mrl_b,
    }


def radial_energy_divergence(group_a, group_b, num_bins=128):
    """
    두 그룹의 radial energy profile 간 KL divergence와 correlation
    """
    mag_a = compute_magnitude_spectrum(group_a).mean(dim=0).squeeze()
    mag_b = compute_magnitude_spectrum(group_b).mean(dim=0).squeeze()
    
    profile_a = compute_radial_profile(mag_a, num_bins)
    profile_b = compute_radial_profile(mag_b, num_bins)
    
    # Normalize to probability distributions
    p = np.abs(profile_a) + 1e-10
    q = np.abs(profile_b) + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    
    # KL divergence
    kl_div = stats.entropy(p, q)
    
    # Pearson correlation
    correlation, _ = stats.pearsonr(profile_a, profile_b)
    
    # L2 distance (log scale에서)
    log_diff = np.log1p(np.abs(profile_a)) - np.log1p(np.abs(profile_b))
    l2_log = np.sqrt((log_diff**2).mean())
    
    return {
        "kl_divergence": kl_div,
        "pearson_correlation": correlation,
        "l2_log_distance": l2_log,
        "profile_a": profile_a,
        "profile_b": profile_b,
    }


def angular_anisotropy_score(group, freq_range=(10, 60), num_angle_bins=36):
    """
    Phase difference map의 방향성(anisotropy) 정량화
    높을수록 anisotropic (방향 의존적)
    """
    phase = compute_phase_spectrum(group).mean(dim=0).squeeze()  # (H, W)
    H, W = phase.shape
    cy, cx = H // 2, W // 2
    
    y = torch.arange(H).float() - cy
    x = torch.arange(W).float() - cx
    radius = torch.sqrt(y[:, None]**2 + x[None, :]**2)
    angle = torch.atan2(y[:, None], x[None, :])
    
    freq_mask = (radius >= freq_range[0]) & (radius < freq_range[1])
    
    angle_bins = torch.linspace(-torch.pi, torch.pi, num_angle_bins + 1)
    energy_per_angle = []
    
    for i in range(num_angle_bins):
        mask = freq_mask & (angle >= angle_bins[i]) & (angle < angle_bins[i + 1])
        if mask.sum() > 0:
            energy_per_angle.append(phase[mask].pow(2).mean().item())
        else:
            energy_per_angle.append(0)
    
    energy = np.array(energy_per_angle)
    # Anisotropy = CoV (Coefficient of Variation)
    anisotropy = energy.std() / (energy.mean() + 1e-8)
    
    return {
        "anisotropy_score": anisotropy,
        "angular_energy": energy,
    }


# ============================================================
# 4. 전체 분석 실행 및 테이블 생성
# ============================================================

def full_frequency_analysis(groups, device='cuda'):
    """
    groups: {"TP": tensor, "TN": tensor, "FP": tensor, "FN": tensor}
    
    모든 pairwise 비교에 대해 frequency metric을 계산하고
    논문용 테이블을 생성합니다.
    """
    
    # 비교할 pair 정의
    pairs = [
        ("TP", "TN"),   # 정탐 Fake vs 정탐 Real (기본 차이)
        ("TP", "FN"),   # 정탐 Fake vs 미탐 Fake (왜 놓쳤나?)
        ("TN", "FP"),   # 정탐 Real vs 오탐 Real (왜 오분류했나?)
        ("FP", "FN"),   # 오탐 Real vs 미탐 Fake (오류끼리 비슷한가?)
    ]
    
    results = []
    
    for name_a, name_b in pairs:
        g_a = groups[name_a].to(device)
        g_b = groups[name_b].to(device)
        
        if len(g_a) < 10 or len(g_b) < 10:
            print(f"Skipping {name_a} vs {name_b}: insufficient samples")
            continue
        
        # 샘플 수 제한 (메모리 절약)
        max_n = min(500, len(g_a), len(g_b))
        g_a = g_a[:max_n]
        g_b = g_b[:max_n]
        
        print(f"\nAnalyzing {name_a} vs {name_b} "
              f"({len(g_a)} vs {len(g_b)} samples)...")
        
        # Magnitude 비교
        mag_stats = magnitude_difference_stats(g_a, g_b)
        
        # Phase 비교
        phase_stats = phase_difference_stats(g_a, g_b)
        
        # Radial energy divergence
        energy_stats = radial_energy_divergence(g_a, g_b)
        
        # Phase coherence 비교
        coherence_stats = phase_coherence_comparison(g_a, g_b)
        
        # Anisotropy (각 그룹)
        aniso_a = angular_anisotropy_score(g_a)
        aniso_b = angular_anisotropy_score(g_b)
        
        results.append({
            "Comparison": f"{name_a} vs {name_b}",
            # Magnitude metrics
            "Mag_MeanAbsDiff": mag_stats["mean_abs_diff"],
            "Mag_MaxAbsDiff": mag_stats["max_abs_diff"],
            "Mag_StdDiff": mag_stats["std_diff"],
            # Phase metrics
            "Phase_MeanAbsDiff": phase_stats["mean_abs_diff"],
            "Phase_MaxAbsDiff": phase_stats["max_abs_diff"],
            "Phase_CircularVar": phase_stats["circular_variance"],
            # Radial energy metrics
            "RadialEnergy_KL": energy_stats["kl_divergence"],
            "RadialEnergy_Corr": energy_stats["pearson_correlation"],
            "RadialEnergy_L2Log": energy_stats["l2_log_distance"],
            # Phase coherence
            "MRL_DiffMean": coherence_stats["mrl_diff_mean"],
            "MRL_DiffLowFreq": coherence_stats["mrl_diff_low_freq"],
            "MRL_DiffHighFreq": coherence_stats["mrl_diff_high_freq"],
            # Anisotropy
            f"Anisotropy_{name_a}": aniso_a["anisotropy_score"],
            f"Anisotropy_{name_b}": aniso_b["anisotropy_score"],
        })
    
    df = pd.DataFrame(results)
    return df


def generate_paper_tables(df):
    """분석 결과를 논문에 넣을 수 있는 형태로 정리"""
    
    # ---- Table 1: Magnitude vs Phase 차이 스케일 비교 ----
    print("=" * 70)
    print("TABLE 1: Magnitude vs Phase Difference Scale")
    print("=" * 70)
    
    table1 = df[["Comparison", "Mag_MeanAbsDiff", "Phase_MeanAbsDiff",
                  "Mag_MaxAbsDiff", "Phase_MaxAbsDiff"]].copy()
    table1.columns = ["Comparison", "Mag Mean|Δ|", "Phase Mean|Δ|",
                       "Mag Max|Δ|", "Phase Max|Δ|"]
    print(table1.to_string(index=False, float_format="%.4f"))
    
    print()
    
    # ---- Table 2: Feature 분리도 관련 지표 ----
    print("=" * 70)
    print("TABLE 2: Feature Separability Metrics")
    print("=" * 70)
    
    table2 = df[["Comparison", "RadialEnergy_KL", "RadialEnergy_Corr",
                  "Phase_CircularVar", "MRL_DiffMean"]].copy()
    table2.columns = ["Comparison", "KL Divergence", "Radial Corr",
                       "Phase Circ.Var", "MRL Δ Mean"]
    print(table2.to_string(index=False, float_format="%.4f"))
    
    print()
    
    # ---- Key Finding 자동 추출 ----
    print("=" * 70)
    print("KEY FINDINGS (자동 추출)")
    print("=" * 70)
    
    tn_fp = df[df["Comparison"] == "TN vs FP"]
    tp_tn = df[df["Comparison"] == "TP vs TN"]
    
    if len(tn_fp) > 0 and len(tp_tn) > 0:
        mag_ratio = (tn_fp["Mag_MeanAbsDiff"].values[0] / 
                     tp_tn["Mag_MeanAbsDiff"].values[0])
        phase_ratio = (tn_fp["Phase_MeanAbsDiff"].values[0] / 
                       tp_tn["Phase_MeanAbsDiff"].values[0])
        
        print(f"1. TN-FP magnitude diff / TP-TN magnitude diff = {mag_ratio:.1f}x")
        print(f"   → FP가 되는 Real의 magnitude 이탈이 Real-Fake 차이의 "
              f"{mag_ratio:.1f}배")
        print()
        print(f"2. TN-FP phase diff / TP-TN phase diff = {phase_ratio:.1f}x")
        print(f"   → FP가 되는 Real의 phase 이탈이 Real-Fake 차이의 "
              f"{phase_ratio:.1f}배")
        print()
        
        kl_tn_fp = tn_fp["RadialEnergy_KL"].values[0]
        kl_tp_tn = tp_tn["RadialEnergy_KL"].values[0]
        print(f"3. KL divergence: TP-TN={kl_tp_tn:.4f}, TN-FP={kl_tn_fp:.4f}")
        print(f"   → TN vs FP의 radial energy 분포 차이가 "
              f"{'더 큼' if kl_tn_fp > kl_tp_tn else '더 작음'}")


# ============================================================
# 5. JSON에서 이미지 로드
# ============================================================

def load_image_as_tensor(path: str, size: tuple = (256, 256)) -> torch.Tensor | None:
    """Load a single image as a (C, H, W) float32 tensor in [0, 1]."""
    img = cv2.imread(path)
    if img is None:
        print(f"  [skip] Cannot read: {path}")
        return None
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


def load_groups_from_json(
    json_path: str,
    max_n: int = 500,
    size: tuple = (256, 256),
) -> dict:
    """
    Load TP/TN/FP/FN image groups from a JSON file produced by test_clip_lora.py.
    Keys in the JSON are lowercase (tp/tn/fp/fn); paths are relative to BASE_DIR.

    Returns: {"TP": Tensor(N,C,H,W), ...} for each non-empty key.
    """
    with open(json_path) as f:
        categorised = json.load(f)

    groups = {}
    for key in ("tp", "tn", "fp", "fn"):
        paths = categorised.get(key, [])[:max_n]
        if not paths:
            print(f"  [skip] '{key}': no images in JSON")
            continue
        paths = [os.path.join(BASE_DIR, p) for p in paths]
        print(f"  {key.upper()}: loading {len(paths)} images …")
        tensors = [t for p in paths if (t := load_image_as_tensor(p, size)) is not None]
        if tensors:
            groups[key.upper()] = torch.stack(tensors)
            print(f"  {key.upper()}: {len(tensors)} images loaded")
        else:
            print(f"  [skip] '{key}': all images failed to load")

    return groups


# ============================================================
# 6. 사용 예시 (직접 호출용)
# ============================================================

def example_usage():
    """
    실제 사용 예시 — 당신의 학습된 모델과 테스트 데이터에 맞게 수정하세요
    """
    
    # ----- Step 1: 모델로 예측하고 TP/TN/FP/FN 분류 -----
    
    # model = load_your_model("clip_lora_checkpoint.pth")
    # test_loader = load_ff_test_set()
    
    all_images, all_labels, all_scores = [], [], []
    
    # with torch.no_grad():
    #     for imgs, labels in test_loader:
    #         scores = model(imgs.cuda()).softmax(dim=1)[:, 1]  # fake probability
    #         all_images.append(imgs)
    #         all_labels.append(labels)
    #         all_scores.append(scores.cpu())
    
    # images = torch.cat(all_images)
    # labels = torch.cat(all_labels)
    # scores = torch.cat(all_scores)
    
    # ----- Step 2: JSON에서 이미지 로드하여 그룹화 -----
    # JSON 경로를 실제 경로로 변경하세요
    json_path = "image_paths.json"
    groups = load_groups_from_json(json_path, max_n=500)

    if len(groups) < 2:
        raise RuntimeError("Need at least 2 non-empty categories to compare.")

    # ----- Step 3: 분석 실행 -----
    df = full_frequency_analysis(groups, device='cuda')
    
    # ----- Step 4: 논문용 테이블 출력 -----
    generate_paper_tables(df)
    
    # ----- Step 5: CSV 저장 -----
    df.to_csv("frequency_analysis_tp_tn_fp_fn.csv", index=False)
    print("\nSaved to frequency_analysis_tp_tn_fp_fn.csv")
    
    return df


# ============================================================
# 7. 추가: Perturbation 환경에서의 동일 분석
# ============================================================

def perturbation_frequency_analysis(model, test_data, perturbations, device='cuda'):
    """
    각 perturbation 조건에서 TP/TN/FP/FN의 frequency 특성이 어떻게 변하는지 추적
    
    핵심 질문: blur가 적용되면 TN→FP로 이동하는 이미지들의 
              frequency 특성이 어떻게 바뀌는가?
    """
    
    results_per_condition = {}
    
    for pert_name, pert_fn in perturbations.items():
        print(f"\n{'='*50}")
        print(f"Condition: {pert_name}")
        print(f"{'='*50}")
        
        # Perturbation 적용
        perturbed_data = pert_fn(test_data)
        
        # 모델 예측
        with torch.no_grad():
            images, labels = perturbed_data
            scores = model(images.to(device)).softmax(dim=1)[:, 1].cpu()
        
        # TP/TN/FP/FN 분류
        groups = split_predictions(images, labels, scores)
        
        # Frequency 분석
        df = full_frequency_analysis(groups, device)
        results_per_condition[pert_name] = df
    
    # 조건 간 비교 테이블
    print("\n" + "=" * 70)
    print("TN vs FP Phase Difference Across Conditions")
    print("=" * 70)
    
    comparison_rows = []
    for cond_name, df in results_per_condition.items():
        tn_fp = df[df["Comparison"] == "TN vs FP"]
        if len(tn_fp) > 0:
            comparison_rows.append({
                "Condition": cond_name,
                "Phase_MeanAbsDiff": tn_fp["Phase_MeanAbsDiff"].values[0],
                "Mag_MeanAbsDiff": tn_fp["Mag_MeanAbsDiff"].values[0],
                "FP_count": "see split output",
            })
    
    comp_df = pd.DataFrame(comparison_rows)
    print(comp_df.to_string(index=False, float_format="%.4f"))
    
    return results_per_condition


# ============================================================
# 8. 논문 Figure 생성용 시각화
# ============================================================

def plot_paper_figures(groups, save_dir="./figures"):
    """논문에 넣을 figure들을 생성"""
    import matplotlib.pyplot as plt
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    pairs = [("TP", "TN"), ("TP", "FN"), ("TN", "FP")]
    
    # ---- Figure A: Radial Energy 비교 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (na, nb) in enumerate(pairs):
        g_a, g_b = groups[na][:500], groups[nb][:500]
        stats = radial_energy_divergence(g_a, g_b)
        
        axes[idx].semilogy(stats["profile_a"], label=na, linewidth=2)
        axes[idx].semilogy(stats["profile_b"], label=nb, linewidth=2, linestyle='--')
        axes[idx].set_title(f"{na} vs {nb}\n"
                           f"KL={stats['kl_divergence']:.4f}, "
                           f"r={stats['pearson_correlation']:.4f}")
        axes[idx].set_xlabel("Radial frequency")
        axes[idx].set_ylabel("Mean energy (log)")
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/radial_energy_comparison.pdf", dpi=300)
    plt.close()
    
    # ---- Figure B: Phase Coherence 비교 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (na, nb) in enumerate(pairs):
        g_a, g_b = groups[na][:500], groups[nb][:500]
        stats = phase_coherence_comparison(g_a, g_b)
        
        axes[idx].plot(stats["mrl_profile_a"], label=na, linewidth=2)
        axes[idx].plot(stats["mrl_profile_b"], label=nb, linewidth=2, linestyle='--')
        axes[idx].set_title(f"{na} vs {nb}\nMRL Δ={stats['mrl_diff_mean']:.4f}")
        axes[idx].set_xlabel("Radial frequency")
        axes[idx].set_ylabel("Phase Coherence (MRL)")
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/phase_coherence_comparison.pdf", dpi=300)
    plt.close()
    
    # ---- Figure C: Bar chart — 핵심 metric 비교 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    pair_names = [f"{a} vs {b}" for a, b in pairs + [("FP", "FN")]]
    
    # Magnitude differences
    mag_diffs = []
    phase_diffs = []
    for na, nb in pairs + [("FP", "FN")]:
        g_a, g_b = groups[na][:500], groups[nb][:500]
        ms = magnitude_difference_stats(g_a, g_b)
        ps = phase_difference_stats(g_a, g_b)
        mag_diffs.append(ms["mean_abs_diff"])
        phase_diffs.append(ps["mean_abs_diff"])
    
    x = np.arange(len(pair_names))
    width = 0.35
    
    axes[0].bar(x - width/2, mag_diffs, width, label='Magnitude', color='steelblue')
    axes[0].bar(x + width/2, phase_diffs, width, label='Phase', color='coral')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pair_names, rotation=15)
    axes[0].set_ylabel('Mean |Difference|')
    axes[0].set_title('Magnitude vs Phase Difference by Pair')
    axes[0].legend()
    
    # TN vs FP 강조
    axes[0].annotate('Largest gap\n(FPR cause)',
                     xy=(2, max(mag_diffs[2], phase_diffs[2])),
                     fontsize=10, ha='center', color='red',
                     fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/metric_comparison_bar.pdf", dpi=300)
    plt.close()
    
    print(f"Figures saved to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="TP/TN/FP/FN frequency domain quantitative analysis."
    )
    parser.add_argument(
        "--from-json", dest="from_json", required=True,
        help="Path to image-paths JSON (tp/tn/fp/fn) produced by test_clip_lora.py",
    )
    parser.add_argument("--max", type=int, default=500,
                        help="Max images per group (default: 500)")
    parser.add_argument("--size", nargs=2, type=int, default=[256, 256],
                        metavar=("W", "H"), help="Resize images to W×H (default: 256 256)")
    parser.add_argument("--out", default="frequency_analysis_tp_tn_fp_fn.csv",
                        help="Output CSV path (default: frequency_analysis_tp_tn_fp_fn.csv)")
    parser.add_argument("--figures", default=None,
                        help="Directory to save paper figures (optional)")
    args = parser.parse_args()

    size = tuple(args.size)
    groups = load_groups_from_json(args.from_json, max_n=args.max, size=size)

    if len(groups) < 2:
        raise RuntimeError("Need at least 2 non-empty categories to compare.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = full_frequency_analysis(groups, device=device)
    generate_paper_tables(df)

    df.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")

    if args.figures:
        plot_paper_figures(groups, save_dir=args.figures)


if __name__ == "__main__":
    main()