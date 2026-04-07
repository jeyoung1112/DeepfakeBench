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
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.exp(x)

@DETECTOR.register_module(module_name='dual_branch')
class DualBranchDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.mode = config.get('mode', 'dual')

        self.pixel_branch, self.freq_branch = self.build_backbone(config)
        self.loss_func, self.vicreg_loss = self.build_loss(config)

        pixel_dim = self.pixel_branch.out_dim if self.pixel_branch else 0
        freq_dim = self.freq_branch.out_dim if self.freq_branch else 0
        embed_dim = config.get('embed_dim', 4096)

        self.exp_pixel = None
        self.exp_freq = None
        if self.mode == 'dual':
            self.exp_pixel = Expander(pixel_dim, embed_dim)
            self.exp_freq = Expander(freq_dim, embed_dim)
        
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
 
        # VICReg weight
        self.vicreg_weight = config.get('vicreg_weight', 0.1)
 
        self._log_params()
    
    def _log_params(self):
        """Log parameter counts for all components."""
        total, trainable = 0, 0
        components = {
            'pixel_branch': self.pixel_branch,
            'freq_branch': self.freq_branch,
            'exp_pixel': self.exp_pixel,
            'exp_freq': self.exp_freq,
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

    def build_loss(self, config):
        cls_loss_class = LOSSFUNC[config.get('loss_func', 'cross_entropy')]
        cls_loss = cls_loss_class()
 
        # VICReg loss (dual mode only)
        vicreg_loss = None
        if config.get('mode', 'dual') == 'dual':
            vicreg_loss = LOSSFUNC['vicreg'](
                lambda_inv=config.get('lambda_inv', 25.0),
                mu_var=config.get('mu_var', 25.0),
                nu_cov=config.get('nu_cov', 1.0),
            )
 
        return cls_loss, vicreg_loss

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
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

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        cls_loss = self.loss_func(pred_dict['cls'], label)
        losses = {'overall': cls_loss, 'cls': cls_loss}
        if self.vicreg_loss is not None and self.mode == 'dual':
            feat_dict = pred_dict['feat_dict']
            z_pixel = self.exp_pixel(feat_dict['y_pixel'])
            z_freq = self.exp_freq(feat_dict['y_freq'])
            vic_loss, vic_loss_dict = self.vicreg_loss(z_pixel, z_freq)
            losses['vicreg'] = vic_loss
            losses.update(vic_loss_dict)
            losses['overall'] = cls_loss + self.vicreg_weight * vic_loss
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