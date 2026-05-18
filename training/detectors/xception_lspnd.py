"""
Xception + Label-Preserving Spectral Nuisance Decorrelation (LPSND)

This detector is based on the conditional spectral decorrelation detector,
but replaces full spectral decorrelation with a label-preserving nuisance
projection:

    1. Extract 11-D spectral statistics S from the inverse-normalized image.
    2. Normalize S with fixed train-set mean/std.
    3. Whiten S using the train-set pooled within-class covariance.
    4. Remove the Fisher/LDA real-vs-fake spectral direction.
    5. Penalize class-conditional dependence between Xception features F
       and the residual spectral nuisance U_perp.

Objective:
    L = CE + lambda_decorr / 2 * [D(F_real, U_perp_real)
                                + D(F_fake, U_perp_fake)]

where D is either:
    - cross_cov: linear squared cross-covariance
    - hsic:      kernel dependence, default linear kernel on F and RBF on U_perp

Expected config keys:
    model_name: xception_lpsnd
    backbone_name: xception
    lambda_decorr: 0.01
    loss_mode: hsic | cross_cov
    lpsnd_stats_path: /path/to/lpsnd_stats.npz

The .npz file should contain fixed train-set spectral statistics. Use the
companion script precompute_lpsnd_stats.py, or create the arrays manually:
    spectral_mean, spectral_std, mu0, mu1, sigma_w
Optional arrays:
    sigma_gamma, whitening, v_hat, mu_mid
"""

import logging
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from scipy.fft import dctn as scipy_dctn

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


# =============================================================================
# Spectral statistics
# =============================================================================

SRM_K1 = np.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]], dtype=np.float32)
SRM_K2 = np.array([[0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 1, -4, 1, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0]], dtype=np.float32) / 2.0
SRM_K3 = np.array([[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0


def _extract_spectral_one(gray: np.ndarray) -> np.ndarray:
    """Extract 11 spectral statistics from one grayscale image in [0, 255]."""
    h, w = gray.shape
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    gray_f = gray_u8.astype(np.float32)

    blurred = cv2.medianBlur(gray_u8, 9).astype(np.float32)
    hp = gray_f - blurred

    fft_img = np.fft.fftshift(np.fft.fft2(gray_f))
    power = np.abs(fft_img) ** 2
    total_e = power.sum() + 1e-12

    fft_log_energy = np.log1p(power.sum())

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    nyq_20 = 0.20 * min(h, w) / 2.0
    fft_hf_ratio = power[radius > nyq_20].sum() / total_e

    prob = power / total_e
    prob_safe = np.clip(prob, 1e-20, None)
    fft_entropy = -np.sum(prob_safe * np.log(prob_safe))
    fft_centroid = np.sum(radius * power) / total_e

    dct_coeff = scipy_dctn(gray_f, norm="ortho")
    dct_power = np.abs(dct_coeff) ** 2
    dct_total = dct_power.sum() + 1e-12
    dct_log_energy = np.log1p(dct_power.sum())
    dct_hf_ratio = 1.0 - dct_power[:32, :32].sum() / dct_total
    dct_prob = dct_power / dct_total
    dct_prob_safe = np.clip(dct_prob, 1e-20, None)
    dct_entropy = -np.sum(dct_prob_safe * np.log(dct_prob_safe))

    hp_std = np.std(hp)

    srm_feats = []
    for kernel in (SRM_K1, SRM_K2, SRM_K3):
        residual = cv2.filter2D(gray_f, -1, kernel)
        srm_feats.append(np.mean(np.abs(residual)))

    return np.array([
        fft_log_energy, fft_hf_ratio, fft_entropy, fft_centroid,
        dct_log_energy, dct_hf_ratio, dct_entropy, hp_std,
        *srm_feats,
    ], dtype=np.float64)


def inverse_normalize_images(
    images_tensor: torch.Tensor,
    mean,
    std,
    clamp: bool = True,
) -> torch.Tensor:
    """Undo model normalization. For mean/std=[0.5], maps [-1,1] -> [0,1]."""
    device = images_tensor.device
    dtype = images_tensor.dtype
    c = images_tensor.shape[1]

    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, c, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(1, c, 1, 1)
    img = images_tensor * std_t + mean_t
    if clamp:
        img = img.clamp(0.0, 1.0)
    return img


def extract_spectral_batch(
    images_tensor: torch.Tensor,
    mean,
    std,
    input_is_normalized: bool = True,
) -> torch.Tensor:
    """
    Compute 11-D spectral statistics for a batch.

    Args:
        images_tensor: model input tensor, usually normalized to [-1,1].
        mean, std: normalization values from config.
        input_is_normalized: if True, inverse-normalize before extraction.

    Returns:
        Tensor [B, 11] on the original device. No gradient is used.
    """
    device = images_tensor.device
    with torch.no_grad():
        img = images_tensor.detach()
        if input_is_normalized:
            img = inverse_normalize_images(img, mean=mean, std=std, clamp=True)

        img_np = img.detach().cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = img_np * 255.0
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        stats = []
        for i in range(img_np.shape[0]):
            if img_np.shape[1] == 3:
                gray = cv2.cvtColor(img_np[i].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np[i, 0]
            stats.append(_extract_spectral_one(gray.astype(np.float32)))

    return torch.tensor(np.stack(stats), dtype=torch.float32, device=device)


# =============================================================================
# LPSND spectral projection
# =============================================================================


def _inverse_sqrt_psd_np(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Symmetric inverse square root for a positive semi-definite matrix."""
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, eps)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return inv_sqrt.astype(np.float32)


class LabelPreservingSpectralProjector(nn.Module):
    """
    Fixed spectral transform:
        S_raw -> S_norm -> U = A(S_norm - mu_mid) -> U_perp.

    Uses precomputed train-set statistics by default. A batch fallback is
    included only for debugging; it is not recommended for final experiments.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.num_stats = int(config.get("lpsnd_num_stats", 11))
        self.eps = float(config.get("lpsnd_eps", 1e-6))
        self.gamma = float(config.get("lpsnd_gamma", -1.0))
        self.batch_fallback = bool(config.get("lpsnd_batch_fallback", False))
        self.require_stats = bool(config.get("lpsnd_require_stats", True))
        self.stats_path = config.get("lpsnd_stats_path", None)

        self.loaded = False
        if self.stats_path:
            self._load_stats(self.stats_path)
        elif self.require_stats and not self.batch_fallback:
            raise ValueError(
                "LPSND requires lpsnd_stats_path. Run precompute_lpsnd_stats.py "
                "or set lpsnd_batch_fallback: true for debugging only."
            )
        else:
            logger.warning(
                "LPSND running without precomputed train-set stats. "
                "Using per-batch spectral projection; this is noisy and should "
                "not be used for final experiments."
            )
            self._init_identity_buffers()

    def _init_identity_buffers(self):
        m = self.num_stats
        self.register_buffer("spectral_mean", torch.zeros(m))
        self.register_buffer("spectral_std", torch.ones(m))
        self.register_buffer("mu_mid", torch.zeros(m))
        self.register_buffer("whitening", torch.eye(m))
        v = torch.zeros(m)
        v[0] = 1.0
        self.register_buffer("v_hat", v)

    def _load_stats(self, path: str):
        if not os.path.isfile(path):
            if self.require_stats and not self.batch_fallback:
                raise FileNotFoundError(f"lpsnd_stats_path not found: {path}")
            logger.warning("lpsnd_stats_path not found; using batch fallback: %s", path)
            self._init_identity_buffers()
            return

        arr = np.load(path)
        spectral_mean = arr["spectral_mean"].astype(np.float32)
        spectral_std = arr["spectral_std"].astype(np.float32)
        spectral_std = np.maximum(spectral_std, self.eps)
        mu0 = arr["mu0"].astype(np.float32)
        mu1 = arr["mu1"].astype(np.float32)

        if "whitening" in arr:
            whitening = arr["whitening"].astype(np.float32)
        else:
            if "sigma_gamma" in arr:
                sigma_gamma = arr["sigma_gamma"].astype(np.float32)
            else:
                sigma_w = arr["sigma_w"].astype(np.float32)
                if self.gamma <= 0:
                    self.gamma = 1e-3 * float(np.trace(sigma_w) / sigma_w.shape[0])
                sigma_gamma = sigma_w + self.gamma * np.eye(sigma_w.shape[0], dtype=np.float32)
            whitening = _inverse_sqrt_psd_np(sigma_gamma, eps=self.eps)

        if "mu_mid" in arr:
            mu_mid = arr["mu_mid"].astype(np.float32)
        else:
            mu_mid = 0.5 * (mu0 + mu1)

        if "v_hat" in arr:
            v_hat = arr["v_hat"].astype(np.float32)
        else:
            delta = mu1 - mu0
            v = whitening @ delta
            v_norm = np.linalg.norm(v) + self.eps
            v_hat = (v / v_norm).astype(np.float32)

        if spectral_mean.shape[0] != self.num_stats:
            raise ValueError(
                f"Expected {self.num_stats} spectral stats but got {spectral_mean.shape[0]}"
            )

        self.register_buffer("spectral_mean", torch.tensor(spectral_mean, dtype=torch.float32))
        self.register_buffer("spectral_std", torch.tensor(spectral_std, dtype=torch.float32))
        self.register_buffer("mu_mid", torch.tensor(mu_mid, dtype=torch.float32))
        self.register_buffer("whitening", torch.tensor(whitening, dtype=torch.float32))
        self.register_buffer("v_hat", torch.tensor(v_hat, dtype=torch.float32))
        self.loaded = True
        logger.info("Loaded LPSND spectral stats from %s", path)

    def _batch_project(self, spectral_raw: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Debug-only per-batch version of the LPSND projection."""
        labels = labels.view(-1).long()
        eps = self.eps
        s_mean = spectral_raw.mean(dim=0)
        s_std = spectral_raw.std(dim=0, unbiased=False).clamp_min(eps)
        s_norm = (spectral_raw - s_mean) / s_std

        real = labels == 0
        fake = labels == 1
        if real.sum() < 2 or fake.sum() < 2:
            return s_norm

        mu0 = s_norm[real].mean(dim=0)
        mu1 = s_norm[fake].mean(dim=0)
        centered0 = s_norm[real] - mu0
        centered1 = s_norm[fake] - mu1
        denom = max(int(real.sum() + fake.sum() - 2), 1)
        sigma_w = (centered0.T @ centered0 + centered1.T @ centered1) / denom
        if self.gamma <= 0:
            gamma = 1e-3 * torch.trace(sigma_w) / sigma_w.shape[0]
        else:
            gamma = torch.tensor(self.gamma, device=spectral_raw.device, dtype=spectral_raw.dtype)
        sigma_gamma = sigma_w + gamma * torch.eye(sigma_w.shape[0], device=spectral_raw.device)

        eigvals, eigvecs = torch.linalg.eigh(0.5 * (sigma_gamma + sigma_gamma.T))
        eigvals = eigvals.clamp_min(eps)
        whitening = eigvecs @ torch.diag(torch.rsqrt(eigvals)) @ eigvecs.T
        mu_mid = 0.5 * (mu0 + mu1)
        u = (s_norm - mu_mid) @ whitening.T
        v = whitening @ (mu1 - mu0)
        v_hat = v / (v.norm() + eps)
        return u - (u @ v_hat).unsqueeze(1) * v_hat.unsqueeze(0)

    def forward(self, spectral_raw: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.loaded and self.batch_fallback:
            if labels is None:
                raise ValueError("labels are required for lpsnd_batch_fallback")
            return self._batch_project(spectral_raw, labels).detach()

        s_norm = (spectral_raw - self.spectral_mean) / (self.spectral_std + self.eps)
        u = (s_norm - self.mu_mid) @ self.whitening.T
        v = self.v_hat
        u_perp = u - (u @ v).unsqueeze(1) * v.unsqueeze(0)
        return u_perp.detach()


# =============================================================================
# Conditional dependence losses
# =============================================================================


def _cross_cov_loss(features: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
    n, d = features.shape
    k = spectral.shape[1]
    if n < 2:
        return features.new_tensor(0.0)
    f_c = features - features.mean(dim=0, keepdim=True)
    s_c = spectral - spectral.mean(dim=0, keepdim=True)
    cov = (f_c.T @ s_c) / (n - 1)
    return (cov ** 2).sum() / max(d * k, 1)


def _csica_loss(f_real, s_real, f_fake, s_fake):
    """
    CSICA: penalise the difference between within-class cross-covariances.
    
    L_CSICA = ||C_R - C_F||²_F / (d * k)
    
    where C_R, C_F are within-class cross-covariance matrices.
    
    f_real: (n_R, d), s_real: (n_R, k)
    f_fake: (n_F, d), s_fake: (n_F, k)
    """
    n_R = f_real.shape[0]
    n_F = f_fake.shape[0]
    
    if n_R < 2 or n_F < 2:
        return f_real.new_tensor(0.0)
    
    # Center within each class
    f_R_c = f_real - f_real.mean(dim=0, keepdim=True)
    s_R_c = s_real - s_real.mean(dim=0, keepdim=True)
    f_F_c = f_fake - f_fake.mean(dim=0, keepdim=True)
    s_F_c = s_fake - s_fake.mean(dim=0, keepdim=True)
    
    # Within-class cross-covariances
    C_R = (f_R_c.T @ s_R_c) / (n_R - 1)  # (d, k)
    C_F = (f_F_c.T @ s_F_c) / (n_F - 1)  # (d, k)
    
    # Penalise the difference
    diff = C_R - C_F
    d, k = diff.shape
    return (diff ** 2).sum() / (d * k)


def _median_bandwidth(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    if n < 2:
        return x.new_tensor(1.0)
    with torch.no_grad():
        dist2 = torch.cdist(x, x).pow(2)
        mask = torch.triu(torch.ones_like(dist2, dtype=torch.bool), diagonal=1)
        vals = dist2[mask]
        if vals.numel() == 0:
            return x.new_tensor(1.0)
        med = vals.median().clamp_min(1e-6)
        # For exp(-d^2 / (2 sigma^2)), median heuristic sets 2 sigma^2 = median(d^2).
        return torch.sqrt(med / 2.0).clamp_min(1e-3)


def _kernel_matrix(x: torch.Tensor, kernel: str, sigma: float) -> torch.Tensor:
    if kernel == "linear":
        return x @ x.T
    if kernel == "rbf":
        if sigma <= 0:
            sigma_t = _median_bandwidth(x)
        else:
            sigma_t = x.new_tensor(float(sigma)).clamp_min(1e-3)
        dist2 = torch.cdist(x, x).pow(2)
        return torch.exp(-dist2 / (2.0 * sigma_t.pow(2)))
    raise ValueError(f"Unknown kernel: {kernel}")


def _hsic_loss(
    features: torch.Tensor,
    spectral: torch.Tensor,
    kernel_f: str = "linear",
    kernel_s: str = "rbf",
    sigma_f: float = -1.0,
    sigma_s: float = -1.0,
) -> torch.Tensor:
    n = features.shape[0]
    if n < 2:
        return features.new_tensor(0.0)

    k_f = _kernel_matrix(features, kernel_f, sigma_f)
    k_s = _kernel_matrix(spectral, kernel_s, sigma_s)
    h = torch.eye(n, device=features.device, dtype=features.dtype) - (1.0 / n)
    return torch.trace(k_f @ h @ k_s @ h) / ((n - 1) ** 2)


class ClassConditionalDependenceLoss(nn.Module):
    """
    Class-conditional dependence loss.

    class_weighting:
        equal        -> 0.5 * D(real) + 0.5 * D(fake), recommended for paper theorem.
        proportional -> n_c / B weighting, closer to original uploaded code.
    """

    def __init__(
        self,
        mode: str = "hsic",
        kernel_f: str = "linear",
        kernel_s: str = "rbf",
        sigma_f: float = -1.0,
        sigma_s: float = -1.0,
        min_class_samples: int = 4,
        class_weighting: str = "equal",
    ):
        super().__init__()
        if mode not in ("cross_cov", "hsic"):
            raise ValueError("loss_mode must be 'cross_cov' or 'hsic'")
        if kernel_f not in ("linear", "rbf") or kernel_s not in ("linear", "rbf"):
            raise ValueError("hsic kernels must be 'linear' or 'rbf'")
        if class_weighting not in ("equal", "proportional"):
            raise ValueError("class_weighting must be 'equal' or 'proportional'")

        self.mode = mode
        self.kernel_f = kernel_f
        self.kernel_s = kernel_s
        self.sigma_f = sigma_f
        self.sigma_s = sigma_s
        self.min_class_samples = int(min_class_samples)
        self.class_weighting = class_weighting

    def _compute(self, f: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.mode == "cross_cov":
            return _cross_cov_loss(f, s)
        return _hsic_loss(
            f, s,
            kernel_f=self.kernel_f,
            kernel_s=self.kernel_s,
            sigma_f=self.sigma_f,
            sigma_s=self.sigma_s,
        )

    def forward(self, features: torch.Tensor, spectral_nuisance: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1).long()
        batch_size = features.shape[0]
        total = features.new_tensor(0.0)
        used_weight = 0.0

        for cls in (0, 1):
            mask = labels == cls
            n_cls = int(mask.sum().item())
            if n_cls < self.min_class_samples:
                continue

            dep = self._compute(features[mask], spectral_nuisance[mask])
            if self.class_weighting == "equal":
                weight = 0.5
            else:
                weight = n_cls / max(batch_size, 1)
            total = total + weight * dep
            used_weight += weight

        # If one class is skipped, keep the scale stable by normalizing to total nominal weight.
        if used_weight > 0:
            return total / used_weight
        return total

class CSICALoss(nn.Module):
    """
    Cross-class Spectral Invariance via Cross-covariance Alignment.
    
    L_CSICA = ||C_R - C_F||²_F / (d * k)
    
    Penalises asymmetry between within-class cross-covariances,
    preserving within-class structure that is symmetric across classes.
    """
    def __init__(self, min_class_samples=4):
        super().__init__()
        self.min_class_samples = min_class_samples
    
    def forward(self, features, spectral_nuisance, labels):
        labels = labels.view(-1).long()
        real_mask = labels == 0
        fake_mask = labels == 1
        n_R = int(real_mask.sum().item())
        n_F = int(fake_mask.sum().item())
        
        if n_R < self.min_class_samples or n_F < self.min_class_samples:
            return features.new_tensor(0.0)
        
        # Within-class centered values
        f_R = features[real_mask]
        s_R = spectral_nuisance[real_mask]
        f_R_c = f_R - f_R.mean(dim=0, keepdim=True)
        s_R_c = s_R - s_R.mean(dim=0, keepdim=True)
        
        f_F = features[fake_mask]
        s_F = spectral_nuisance[fake_mask]
        f_F_c = f_F - f_F.mean(dim=0, keepdim=True)
        s_F_c = s_F - s_F.mean(dim=0, keepdim=True)
        
        # Within-class cross-covariances
        C_R = (f_R_c.T @ s_R_c) / (n_R - 1)
        C_F = (f_F_c.T @ s_F_c) / (n_F - 1)
        
        # CSICA penalty: asymmetry only
        diff = C_R - C_F
        d, k = diff.shape
        return (diff ** 2).sum() / max(d * k, 1)

# =============================================================================
# Detector
# =============================================================================


@DETECTOR.register_module(module_name="xception_lpsnd")
class XceptionLPSNDDetector(AbstractDetector):
    """Xception detector with Label-Preserving Spectral Nuisance Decorrelation."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

        self.lambda_decorr = float(config.get("lambda_decorr", 0.01))
        self.mean = config.get("mean", [0.5, 0.5, 0.5])
        self.std = config.get("std", [0.5, 0.5, 0.5])
        self.spectral_input_is_normalized = bool(config.get("spectral_input_is_normalized", True))

        # Projection head for the HSIC/decorrelation representation.
        self.use_projection_head = bool(config.get("lpsnd_use_projection_head", True))
        feature_dim = int(config.get("lpsnd_feature_dim", 2048))
        proj_dim = int(config.get("lpsnd_proj_dim", 128))
        if self.use_projection_head:
            self.decorr_projector = nn.Sequential(
                nn.Linear(feature_dim, proj_dim),
                nn.LayerNorm(proj_dim),
            )
        else:
            self.decorr_projector = nn.Identity()

        self.spectral_projector = LabelPreservingSpectralProjector(config)

        loss_mode = config.get("loss_mode", "csica")
        if loss_mode == "csica":
            self.decorr_loss = CSICALoss()
        else:
            self.decorr_loss = ClassConditionalDependenceLoss(
                mode=config.get("loss_mode", "hsic"),
                kernel_f=config.get("hsic_kernel_f", "linear"),
                kernel_s=config.get("hsic_kernel_s", "rbf"),
                sigma_f=float(config.get("hsic_sigma_f", -1.0)),
                sigma_s=float(config.get("hsic_sigma_s", -1.0)),
                min_class_samples=int(config.get("min_class_samples", 4)),
                class_weighting=config.get("class_weighting", "equal"),
            )

        logger.info(
            "XceptionLPSND: lambda=%s, mode=%s, kernels=(%s,%s), proj_dim=%s, stats=%s",
            self.lambda_decorr,
            config.get("loss_mode", "hsic"),
            config.get("hsic_kernel_f", "linear"),
            config.get("hsic_kernel_s", "rbf"),
            proj_dim if self.use_projection_head else "identity",
            config.get("lpsnd_stats_path", None),
        )

    def build_backbone(self, config):
        backbone_class = BACKBONE[config["backbone_name"]]
        model_config = config["backbone_config"]
        backbone = backbone_class(model_config)

        if "pretrained" in config and config["pretrained"]:
            state_dict = torch.load(config["pretrained"], map_location="cpu")
            for name, weights in list(state_dict.items()):
                if "pointwise" in name:
                    state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
            state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
            backbone.load_state_dict(state_dict, False)
            logger.info("Backbone loaded from %s", config["pretrained"])
        return backbone

    def build_loss(self, config):
        loss_class = LOSSFUNC[config["loss_func"]]
        return loss_class()

    def features(self, data_dict: dict) -> torch.Tensor:
        return self.backbone.features(data_dict["image"])

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.classifier(features)

    @staticmethod
    def _feature_vector(features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 4:
            return torch_F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        if features.dim() > 2:
            return features.flatten(1)
        return features

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        labels = data_dict["label"].view(-1).long()
        pred = pred_dict["cls"]
        loss_cls = self.loss_func(pred, labels)

        if self.lambda_decorr <= 0 or pred_dict.get("spectral_nuisance", None) is None:
            return {"overall": loss_cls, "cls": loss_cls}

        feat_vec = self._feature_vector(pred_dict["feat"])
        try:
            feat_decorr = self.decorr_projector(feat_vec)
        except RuntimeError as exc:
            raise RuntimeError(
                "LPSND projection head input dimension mismatch. "
                "Set lpsnd_feature_dim in the config to match the pooled Xception feature dimension."
            ) from exc

        loss_decorr = self.decorr_loss(feat_decorr, pred_dict["spectral_nuisance"], labels)
        loss_total = loss_cls + self.lambda_decorr * loss_decorr
        return {
            "overall": loss_total,
            "cls": loss_cls,
            "decorr": loss_decorr,
        }

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        labels = data_dict["label"]
        pred = pred_dict["cls"]
        auc, eer, acc, ap = calculate_metrics_for_train(labels.detach(), pred.detach())
        metrics = {"acc": acc, "auc": auc, "eer": eer, "ap": ap}
        if "losses" in pred_dict and "decorr" in pred_dict["losses"]:
            metrics["decorr"] = pred_dict["losses"]["decorr"].item()
        return metrics

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]

        pred_dict = {"cls": pred, "prob": prob, "feat": features}

        if not inference and self.lambda_decorr > 0:
            with torch.no_grad():
                spectral_raw = extract_spectral_batch(
                    data_dict["image"],
                    mean=self.mean,
                    std=self.std,
                    input_is_normalized=self.spectral_input_is_normalized,
                )
                spectral_nuisance = self.spectral_projector(
                    spectral_raw,
                    labels=data_dict.get("label", None),
                )
            pred_dict["spectral_raw"] = spectral_raw
            pred_dict["spectral_nuisance"] = spectral_nuisance
        else:
            pred_dict["spectral_raw"] = None
            pred_dict["spectral_nuisance"] = None

        return pred_dict
