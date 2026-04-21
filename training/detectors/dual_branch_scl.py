import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics.base_metrics_class import calculate_metrics_for_train

from networks.pixel_branch import PixelBranch
from networks.frequency_branch_test import FrequencyBranch

from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC

logger = logging.getLogger(__name__)

# Optional: GPU-side augmentation for pixel branch
HAS_KORNIA = False
try:
    import kornia.augmentation as K
    HAS_KORNIA = True
except ImportError:
    logger.warning("kornia not installed, skipping pixel branch augmentation")


class PixelBranchAugmentation(nn.Module):
    """
    GPU-side differentiable augmentation for pixel branch only.
    Applied to already-normalized tensors from the dataloader.
    
    These augmentations are HARMFUL to frequency analysis
    but BENEFICIAL for CLIP-based pixel branch.
    """
    def __init__(self):
        super().__init__()
        if HAS_KORNIA:
            self.aug = nn.Sequential(
                K.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.2, hue=0.05, p=0.5
                ),
                K.RandomGaussianBlur(
                    kernel_size=(3, 7), sigma=(0.1, 2.0), p=0.2
                ),
            )
        else:
            self.aug = nn.Identity()
    
    def forward(self, x):
        return self.aug(x)


class Expander(nn.Module):
    def __init__(self, in_dim, embed_dim=4096):
        super().__init__()
        self.exp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.exp(x)


class SingleCenterLoss(nn.Module):
    """
    Li et al. (2021), "Frequency-aware Discriminative Feature 
    Learning Supervised by Single-Center Loss for Face Forgery Detection"
    """
    def __init__(self, feat_dim, margin=0.3):
        super().__init__()
        self.center = nn.Parameter(torch.randn(feat_dim) * 0.01)
        self.margin = margin
        self.feat_dim = feat_dim
    
    def forward(self, features, labels):
        # DeepfakeBench convention: 0 = real, 1 = fake
        real_mask = (labels == 0)
        fake_mask = (labels == 1)
        
        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device,
                                requires_grad=True)
        
        real_feats = features[real_mask]
        d_real = torch.norm(real_feats - self.center, dim=1)
        M_nat = d_real.mean()
        
        fake_feats = features[fake_mask]
        d_fake = torch.norm(fake_feats - self.center, dim=1)
        M_man = d_fake.mean()
        
        margin_scaled = self.margin * (self.feat_dim ** 0.5)
        hinge = torch.clamp(M_nat - M_man + margin_scaled, min=0.0)
        
        return M_nat + hinge


@DETECTOR.register_module(module_name='dual_branch_scl')
class DualBranchSCLDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.mode = config.get('mode', 'dual')

        self.pixel_branch, self.freq_branch = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        
        # Pixel branch augmentation (GPU-side, training only)
        self.pixel_aug = PixelBranchAugmentation()

        pixel_dim = self.pixel_branch.out_dim if self.pixel_branch else 0
        freq_dim = self.freq_branch.out_dim if self.freq_branch else 0
        fused_dim = pixel_dim + freq_dim

        # Single-Center Loss
        self.scl = SingleCenterLoss(
            feat_dim=fused_dim,
            margin=config.get('scl_margin', 0.3),
        )
        self.scl_weight = config.get('scl_weight', 0.5)
        self.scl_warmup_epochs = config.get('scl_warmup_epochs', 5)
        self.current_epoch = 0

        # Classification head
        head_hidden = config.get('head_hidden_dim', 256)
        head_dropout = config.get('head_dropout', 0.3)
        num_classes = config.get('backbone_config', {}).get(
            'num_classes', 2
        )
        self.head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_classes),
        )

        # Optional: asymmetric VICReg
        self.use_vicreg = config.get('use_vicreg', False)
        if self.use_vicreg:
            embed_dim = config.get('embed_dim', 4096)
            self.exp_pixel = Expander(pixel_dim, embed_dim)
            self.exp_freq = Expander(freq_dim, embed_dim)
            self.vicreg_loss = LOSSFUNC['vicreg'](
                lambda_inv=config.get('lambda_inv', 25.0),
                mu_var=config.get('mu_var', 25.0),
                nu_cov=config.get('nu_cov', 1.0),
            )
            self.vicreg_weight = config.get('vicreg_weight', 0.1)

        self._log_params(config)

    def build_backbone(self, config):
        pixel_branch = None
        freq_branch = None
        mode = config.get('mode', 'dual')

        if mode in ('dual', 'pixel_only'):
            pixel_branch = PixelBranch(config)
        if mode in ('dual', 'freq_only'):
            freq_branch = FrequencyBranch(config)

        return pixel_branch, freq_branch

    def build_loss(self, config):
        cls_loss_class = LOSSFUNC[config.get('loss_func', 'cross_entropy')]
        return cls_loss_class()

    def _log_params(self, config):
        total, trainable = 0, 0
        components = {
            'pixel_branch': self.pixel_branch,
            'freq_branch': self.freq_branch,
            'head': self.head,
            'scl_center': self.scl,
        }
        for name, module in components.items():
            if module is not None:
                t = sum(p.numel() for p in module.parameters())
                tr = sum(p.numel() for p in module.parameters() 
                         if p.requires_grad)
                total += t
                trainable += tr
                logger.info(f"  {name}: {tr:,} trainable / {t:,} total")

        if self.freq_branch is not None:
            sr = self.freq_branch.spec_residual
            sr_params = sum(p.numel() for p in sr.parameters())
            sr_train = sum(p.numel() for p in sr.parameters() 
                           if p.requires_grad)
            sr_type = config.get("sr_type")
            if sr_type == "mine":
                gate_val = torch.sigmoid(sr.gate_logit).item()
            else:
                gate_val = 0
            logger.info(
                f"  spec_residual: {sr_train:,} trainable / "
                f"{sr_params:,} total (gate={gate_val:.3f})"
            )

        logger.info(
            f"DualBranchSCL [{self.mode}]: "
            f"{trainable:,} trainable / {total:,} total"
        )

    def set_epoch(self, epoch):
        """Called by training loop to update epoch for SCL warmup."""
        self.current_epoch = epoch

    def _get_scl_weight(self):
        """Linear warmup: 0 → scl_weight over warmup epochs."""
        if self.scl_warmup_epochs <= 0:
            return self.scl_weight
        ramp = min(1.0, self.current_epoch / self.scl_warmup_epochs)
        return self.scl_weight * ramp

    def classifier(self, feat_dict):
        parts = []
        if feat_dict['y_pixel'] is not None:
            parts.append(feat_dict['y_pixel'])
        if feat_dict['y_freq'] is not None:
            parts.append(feat_dict['y_freq'])
        fused = torch.cat(parts, dim=-1)
        return self.head(fused)

    def features(self, data_dict):
        img = data_dict['image']
        feat_dict = {'y_pixel': None, 'y_freq': None}

        if self.pixel_branch is not None:
            feat_dict['y_pixel'] = self.pixel_branch(img)

        if self.freq_branch is not None:
            feat_dict['y_freq'] = self.freq_branch(img)

        return feat_dict

    def get_losses(self, data_dict, pred_dict):
        label = data_dict['label']

        # Classification loss
        cls_loss = self.loss_func(pred_dict['cls'], label)

        # SCL on fused features (with warmup)
        feat_dict = pred_dict['feat_dict']
        parts = []
        if feat_dict['y_pixel'] is not None:
            parts.append(feat_dict['y_pixel'])
        if feat_dict['y_freq'] is not None:
            parts.append(feat_dict['y_freq'])
        fused = torch.cat(parts, dim=-1)

        scl_loss = self.scl(fused, label)
        scl_w = self._get_scl_weight()

        losses = {
            'cls': cls_loss,
            'scl': scl_loss,
            'scl_weight_effective': scl_w,
            'overall': cls_loss + scl_w * scl_loss,
        }

        # Optional asymmetric VICReg
        if self.use_vicreg and self.mode == 'dual':
            z_pixel = self.exp_pixel(
                feat_dict['y_pixel'].detach()  # STOP GRADIENT
            )
            z_freq = self.exp_freq(feat_dict['y_freq'])
            vic_loss, vic_dict = self.vicreg_loss(z_pixel, z_freq)
            losses['vicreg'] = vic_loss
            losses.update(vic_dict)
            losses['overall'] += self.vicreg_weight * vic_loss

        return losses

    def get_train_metrics(self, data_dict, pred_dict):
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach(), pred.detach()
        )
        metrics = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

        # SCL monitoring
        metrics['scl_center_norm'] = torch.norm(self.scl.center).item()
        metrics['scl_weight_eff'] = self._get_scl_weight()

        # Spectral residual monitoring
        if (self.freq_branch is not None 
                and hasattr(self.freq_branch, 'spec_residual')):
            sr = self.freq_branch.spec_residual
            if hasattr(sr, 'gate_logit'):
                metrics['specres_gate'] = torch.sigmoid(sr.gate_logit).item()
            # Envelope filter norms per scale
            if hasattr(sr, 'envelope_filters'):
                for s, filt in enumerate(sr.envelope_filters):
                    metrics[f'specres_env_norm_s{s}'] = \
                        filt.weight.norm().item()

        return metrics

    def forward(self, data_dict, inference=False):
        feat_dict = self.features(data_dict)
        pred = self.classifier(feat_dict)
        prob = torch.softmax(pred, dim=1)[:, 1]

        parts = []
        if feat_dict['y_pixel'] is not None:
            parts.append(feat_dict['y_pixel'])
        if feat_dict['y_freq'] is not None:
            parts.append(feat_dict['y_freq'])
        feat_tensor = (
            torch.cat(parts, dim=-1) if parts
            else pred.new_zeros(pred.size(0), 0)
        )

        pred_dict = {
            'cls': pred,
            'prob': prob,
            'feat': feat_tensor,
            'feat_dict': feat_dict,
        }
        return pred_dict