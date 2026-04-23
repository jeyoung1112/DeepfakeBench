import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics.base_metrics_class import calculate_metrics_for_train

from networks.pixel_branch import PixelBranch
from networks.frequency_branch import FrequencyBranch

from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='dual_branch_scl')
class DualBranchSCLDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.mode = config.get('mode', 'dual')

        self.pixel_branch, self.freq_branch = self.build_backbone(config)

        pixel_dim = self.pixel_branch.out_dim if self.pixel_branch else 0
        freq_dim = self.freq_branch.out_dim if self.freq_branch else 0

        self.loss_func, self.scl_loss, self.varcov_loss = self.build_loss(config, pixel_dim + freq_dim)

        head_input_dim = pixel_dim + freq_dim
        head_hidden = config.get('head_hidden_dim', 256)
        head_dropout = config.get('head_dropout', 0.3)
        num_classes = config.get('backbone_config', {}).get('num_classes', 2)

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_classes),
        )

        self.scl_weight = config.get('scl_weight', 0.1)
        self.var_weight = config.get('var_weight', 0.04)
        self.cov_weight = config.get('cov_weight', 0.01)

        self._diag_step = 0
        self._diag_log_every = config.get('diag_log_every', 500)

        self._log_params()

    def _log_params(self):
        total, trainable = 0, 0
        components = {
            'pixel_branch': self.pixel_branch,
            'freq_branch': self.freq_branch,
            'head': self.head,
        }

        for name, module in components.items():
            if module is not None:
                t = sum(p.numel() for p in module.parameters())
                tr = sum(p.numel() for p in module.parameters() if p.requires_grad)
                total += t
                trainable += tr
                logger.info(f"  {name}: {tr:,} trainable / {t:,} total")

        logger.info(f"DualBranch [{self.mode}]: {trainable:,} trainable / {total:,} total")

    def classifier(self, feat_dict):
        parts = []

        if feat_dict['y_pixel'] is not None:
            parts.append(feat_dict['y_pixel'])

        if feat_dict['y_freq'] is not None:
            parts.append(feat_dict['y_freq'])

        fused = torch.cat(parts, dim=-1)
        return self.head(fused)

    def build_backbone(self, config):
        pixel_branch = None
        freq_branch = None

        mode = config.get("mode", "dual")

        if mode in ('dual', 'pixel_only'):
            pixel_branch = PixelBranch(config)

        if mode in ('dual', 'freq_only'):
            freq_branch = FrequencyBranch(config)

        return pixel_branch, freq_branch

    def build_loss(self, config, fused_dim):
        cls_loss_class = LOSSFUNC[config.get('loss_func', 'cross_entropy')]
        cls_loss = cls_loss_class()

        scl_loss = None
        if config.get('mode', 'dual') == 'dual':
            scl_loss = LOSSFUNC['scl'](
                feat_dim=fused_dim,
                ema_tau=config.get('scl_ema_tau', 0.99),
            )

        varcov_loss = LOSSFUNC['varcov']()

        return cls_loss, scl_loss, varcov_loss

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def features(self, data_dict: dict) -> dict:
        img = data_dict['image']
        feat_dict = {'y_pixel': None, 'y_freq': None}
        if self.pixel_branch is not None:
            feat_dict['y_pixel'] = self.pixel_branch(img)
        if self.freq_branch is not None:
            feat_dict['y_freq'] = self.freq_branch(img)
        return feat_dict

    @torch.no_grad()
    def _log_feature_diagnostics(self, feat: torch.Tensor, label: torch.Tensor) -> None:
        real_mask = label == 0
        fake_mask = label == 1

        # Per-class L2 norms
        for cls_name, mask in (('real', real_mask), ('fake', fake_mask)):
            if mask.sum() > 0:
                mean_norm = feat[mask].norm(dim=1).mean().item()
                logger.info(f"  feat_norm/{cls_name}: {mean_norm:.4f}")

        # Real-feature variance (mean per-dimension variance)
        if real_mask.sum() > 1:
            real_feats = feat[real_mask].float()
            real_var = real_feats.var(dim=0).mean().item()
            logger.info(f"  real_feat_var: {real_var:.4f}")

            # Effective rank of real covariance: exp(H) where H = entropy of normalised eigenvalues
            # Use SVD on mean-centred features for numerical stability
            centered = real_feats - real_feats.mean(dim=0, keepdim=True)
            # Clamp to at most 512 dims to keep SVD tractable on large feature spaces
            if centered.size(1) > 512:
                centered = centered[:, :512]
            try:
                _, S, _ = torch.linalg.svd(centered, full_matrices=False)
                # Eigenvalues of cov are S^2 / (n-1)
                eigvals = S.pow(2) / max(centered.size(0) - 1, 1)
                eigvals = eigvals[eigvals > 0]
                p = eigvals / eigvals.sum()
                eff_rank = torch.exp(-(p * p.log()).sum()).item()
                logger.info(f"  real_cov_eff_rank: {eff_rank:.2f}")
            except Exception:
                pass

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        cls_loss = self.loss_func(pred_dict['cls'], label)
        overall = cls_loss
        losses = {'cls': cls_loss}

        if self.scl_loss is not None and self.mode == 'dual':
            scl_loss = self.scl_loss(pred_dict['feat'], label)
            losses['scl'] = scl_loss
            overall = overall + self.scl_weight * scl_loss

        feat = pred_dict['feat']
        # if real only:
        #   target_feat = feat[label == 0]
        if feat.size(0) > 1 and feat.size(1) > 0:
            var_loss, cov_loss = self.varcov_loss(feat)
            losses['var'] = var_loss
            losses['cov'] = cov_loss
            overall = overall + (self.var_weight * var_loss) + (self.cov_weight * cov_loss)

        losses['overall'] = overall

        self._diag_step += 1
        if self._diag_step % self._diag_log_every == 0:
            feat = pred_dict['feat']
            if feat.size(0) > 1 and feat.size(1) > 0:
                logger.info(f"[diag step={self._diag_step}]")
                self._log_feature_diagnostics(feat.detach(), label.detach())

        return losses

    def forward(self, data_dict, inference=False):

        feat_dict = self.features(data_dict)

        pred = self.classifier(feat_dict)

        prob = torch.softmax(pred, dim=1)[:, 1]

        parts = []
        if feat_dict['y_pixel'] is not None:
            parts.append(feat_dict['y_pixel'])
        if feat_dict['y_freq'] is not None:
            parts.append(feat_dict['y_freq'])
        feat_tensor = torch.cat(parts, dim=-1) if parts else pred.new_zeros(pred.size(0), 0)

        pred_dict = {
            'cls': pred,
            'prob': prob,
            'feat': feat_tensor,
            'feat_dict': feat_dict,
        }

        return pred_dict
