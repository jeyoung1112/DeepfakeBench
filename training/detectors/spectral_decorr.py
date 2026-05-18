"""
Xception + Conditional Spectral Decorrelation

Replaces global decorrelation with class-conditional decorrelation:
  - Computes decorrelation within REAL samples (removes pipeline bias from real)
  - Computes decorrelation within FAKE samples (removes pipeline bias from fake)
  - Weighted average by class proportion

Mathematical justification:
  Total f-s dependence = within-class dependence + between-class dependence
                       = pipeline bias        + manipulation signal
                         (remove)               (keep)

  By computing HSIC/CrossCov separately within each class, we remove the
  bias component while preserving the manipulation signal that distinguishes
  real from fake on average.

Two loss modes via config:
  loss_mode: 'cross_cov'  — linear, fast (replicates current behaviour but conditional)
  loss_mode: 'hsic'       — nonlinear, captures kernel-level dependence

Config additions:
  loss_mode:        str   — 'cross_cov' | 'hsic'  (default: 'cross_cov')
  hsic_sigma_f:     float — RBF bandwidth for features (default: -1 = median heuristic)
  hsic_sigma_s:     float — RBF bandwidth for spectral (default: -1 = median heuristic)
"""

import os
import logging
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import dctn as scipy_dctn

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Spectral Statistics (UNCHANGED from previous version)
# ═══════════════════════════════════════════════════════════════════════════

SRM_K1 = np.array([[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  1, -2,  1,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]], dtype=np.float32)
SRM_K2 = np.array([[ 0,  0,  0,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  1, -4,  1,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0,  0,  0,  0]], dtype=np.float32) / 2.0
SRM_K3 = np.array([[-1,  2, -2,  2, -1],
                    [ 2, -6,  8, -6,  2],
                    [-2,  8,-12,  8, -2],
                    [ 2, -6,  8, -6,  2],
                    [-1,  2, -2,  2, -1]], dtype=np.float32) / 12.0


def _extract_spectral_one(gray):
    """Extract 11-dim spectral stats from one grayscale image."""
    h, w = gray.shape
    blurred = cv2.medianBlur(gray.astype(np.uint8), 9).astype(np.float32)
    hp = gray - blurred

    F_img = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(F_img) ** 2
    total_e = power.sum() + 1e-12

    fft_log_energy = np.log1p(power.sum())

    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    nyq_20 = 0.20 * min(h, w) / 2
    fft_hf_ratio = power[R > nyq_20].sum() / total_e

    p = power / total_e
    p_safe = np.clip(p, 1e-20, None)
    fft_entropy = -np.sum(p_safe * np.log(p_safe))
    fft_centroid = np.sum(R * power) / total_e

    C = scipy_dctn(gray, norm="ortho")
    Cp = np.abs(C) ** 2
    dct_total = Cp.sum() + 1e-12
    dct_log_energy = np.log1p(Cp.sum())
    dct_hf_ratio = 1.0 - Cp[:32, :32].sum() / dct_total
    pd = Cp / dct_total
    pd_safe = np.clip(pd, 1e-20, None)
    dct_entropy = -np.sum(pd_safe * np.log(pd_safe))

    hp_std = np.std(hp)

    srm_feats = []
    for kern in [SRM_K1, SRM_K2, SRM_K3]:
        res = cv2.filter2D(gray, -1, kern)
        srm_feats.append(np.mean(np.abs(res)))

    return np.array([
        fft_log_energy, fft_hf_ratio, fft_entropy, fft_centroid,
        dct_log_energy, dct_hf_ratio, dct_entropy, hp_std,
        *srm_feats
    ], dtype=np.float64)


def extract_spectral_batch(images_tensor):
    """Compute spectral stats for a batch. No gradient."""
    device = images_tensor.device
    imgs = images_tensor.detach().cpu().numpy()
    if imgs.max() <= 1.0:
        imgs = (imgs * 255)
    imgs = np.clip(imgs, 0, 255).astype(np.uint8)

    stats = []
    for i in range(imgs.shape[0]):
        if imgs.shape[1] == 3:
            gray = cv2.cvtColor(imgs[i].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
        else:
            gray = imgs[i, 0]
        gray = gray.astype(np.float32)
        stats.append(_extract_spectral_one(gray))

    return torch.tensor(np.stack(stats), dtype=torch.float32, device=device)


# ═══════════════════════════════════════════════════════════════════════════
#  CONDITIONAL Decorrelation Loss (NEW)
# ═══════════════════════════════════════════════════════════════════════════

def _cross_cov_loss(f, s):
    """
    Cross-covariance loss within a single subset.
    f: (n, d), s: (n, k), assumes n >= 2
    """
    n, d = f.shape
    k = s.shape[1]
    f_c = f - f.mean(dim=0, keepdim=True)
    s_c = s - s.mean(dim=0, keepdim=True)
    cov = (f_c.T @ s_c) / (n - 1)
    return (cov ** 2).sum() / (d * k)


def _median_bandwidth(x):
    """Median heuristic for RBF kernel bandwidth."""
    n = x.shape[0]
    if n < 2:
        return torch.tensor(1.0, device=x.device)
    with torch.no_grad():
        dist = torch.cdist(x, x) ** 2
        # Take upper triangle excluding diagonal
        mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
        med = dist[mask].median()
    return torch.sqrt(med / 2.0).clamp(min=1e-3)


def _hsic_loss(f, s, sigma_f, sigma_s):
    """
    HSIC with RBF kernels within a single subset.
    f: (n, d), s: (n, k), assumes n >= 2
    """
    n = f.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=f.device)

    # Auto bandwidth via median heuristic if not specified
    if sigma_f <= 0:
        sigma_f = _median_bandwidth(f)
    if sigma_s <= 0:
        sigma_s = _median_bandwidth(s)

    # RBF kernels
    f_dist = torch.cdist(f, f) ** 2
    s_dist = torch.cdist(s, s) ** 2
    K_f = torch.exp(-f_dist / (2 * sigma_f ** 2))
    K_s = torch.exp(-s_dist / (2 * sigma_s ** 2))

    # Centering matrix H = I - 1/n * 11ᵀ
    H = torch.eye(n, device=f.device) - 1.0 / n

    # HSIC = (1/(n-1)²) tr(K_f H K_s H)
    return torch.trace(K_f @ H @ K_s @ H) / ((n - 1) ** 2)


class ConditionalSpectralDecorrelationLoss(nn.Module):
    """
    Conditional decorrelation: compute decorrelation WITHIN each class
    separately, then take weighted average.

    L_cond = (n_real/N) * D(f_real, s_real) + (n_fake/N) * D(f_fake, s_fake)

    where D is either cross-covariance or HSIC.

    Why this preserves manipulation signal:
      Within each class, all spectral variation is pipeline-driven (no
      manipulation differences within real or within fake). Between
      classes, spectral differences include manipulation signal.
      Computing D within each class removes only the within-class
      (pipeline) component.
    """
    def __init__(self, mode='cross_cov', sigma_f=-1.0, sigma_s=-1.0):
        super().__init__()
        assert mode in ('cross_cov', 'hsic'), f"Unknown mode: {mode}"
        self.mode = mode
        self.sigma_f = sigma_f
        self.sigma_s = sigma_s

    def _compute_within_class(self, f, s):
        if self.mode == 'cross_cov':
            return _cross_cov_loss(f, s)
        else:  # hsic
            return _hsic_loss(f, s, self.sigma_f, self.sigma_s)

    def forward(self, features, spectral_stats, labels):
        """
        Args:
            features: (B, d) — backbone output, HAS gradient
            spectral_stats: (B, k) — from pixels, NO gradient
            labels: (B,) — 0 = real, 1 = fake
        Returns:
            scalar loss (weighted average of within-class decorrelations)
        """
        B = features.shape[0]
        device = features.device

        real_mask = (labels == 0)
        fake_mask = (labels == 1)

        n_real = real_mask.sum().item()
        n_fake = fake_mask.sum().item()

        loss = torch.tensor(0.0, device=device)

        if n_real >= 2:
            f_real = features[real_mask]
            s_real = spectral_stats[real_mask]
            loss = loss + (n_real / B) * self._compute_within_class(f_real, s_real)

        if n_fake >= 2:
            f_fake = features[fake_mask]
            s_fake = spectral_stats[fake_mask]
            loss = loss + (n_fake / B) * self._compute_within_class(f_fake, s_fake)

        return loss


# ═══════════════════════════════════════════════════════════════════════════
#  Detector
# ═══════════════════════════════════════════════════════════════════════════

@DETECTOR.register_module(module_name='xception_spec_decorr')
class XceptionSpecDecorrDetector(AbstractDetector):
    """
    Xception + Conditional Spectral Decorrelation.

    Config:
        lambda_decorr: float — decorrelation strength (default: 0.1)
        loss_mode:     str   — 'cross_cov' | 'hsic' (default: 'cross_cov')
        hsic_sigma_f:  float — RBF bandwidth for features, -1 = auto (default: -1)
        hsic_sigma_s:  float — RBF bandwidth for spectral, -1 = auto (default: -1)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

        # Conditional decorrelation loss
        loss_mode = config.get('loss_mode', 'cross_cov')
        sigma_f = config.get('hsic_sigma_f', -1.0)
        sigma_s = config.get('hsic_sigma_s', -1.0)
        self.decorr_loss = ConditionalSpectralDecorrelationLoss(
            mode=loss_mode, sigma_f=sigma_f, sigma_s=sigma_s)
        self.lambda_decorr = config.get('lambda_decorr', 0.1)

        logger.info(
            f"XceptionSpecDecorr (CONDITIONAL): "
            f"lambda={self.lambda_decorr}, mode={loss_mode}, "
            f"sigma_f={sigma_f}, sigma_s={sigma_s}"
        )

    def build_backbone(self, config):
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)

        if 'pretrained' in config and config['pretrained']:
            state_dict = torch.load(config['pretrained'])
            for name, weights in state_dict.items():
                if 'pointwise' in name:
                    state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
            state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
            backbone.load_state_dict(state_dict, False)
            logger.info('Backbone loaded from {}'.format(config['pretrained']))

        return backbone

    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        return loss_class()

    def features(self, data_dict: dict) -> torch.Tensor:
        return self.backbone.features(data_dict['image'])

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.classifier(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']

        loss_cls = self.loss_func(pred, label)

        if 'spectral_stats' in pred_dict and pred_dict['spectral_stats'] is not None:
            feat = pred_dict['feat']
            if feat.dim() > 2:
                feat = feat.flatten(1)

            # CONDITIONAL: pass labels to decorr loss
            loss_decorr = self.decorr_loss(
                feat, pred_dict['spectral_stats'], label)

            loss_total = loss_cls + self.lambda_decorr * loss_decorr
            return {
                'overall': loss_total,
                'cls': loss_cls,
                'decorr': loss_decorr,
            }
        else:
            return {'overall': loss_cls, 'cls': loss_cls}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach(), pred.detach())
        metrics = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        if 'losses' in pred_dict and 'decorr' in pred_dict['losses']:
            metrics['decorr'] = pred_dict['losses']['decorr'].item()
        return metrics

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]

        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

        if not inference:
            with torch.no_grad():
                spectral = extract_spectral_batch(data_dict['image'])
            pred_dict['spectral_stats'] = spectral
        else:
            pred_dict['spectral_stats'] = None

        return pred_dict