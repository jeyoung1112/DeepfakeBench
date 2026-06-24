# '''
# # author: Zhiyuan Yan
# # email: zhiyuanyan@link.cuhk.edu.cn
# # date: 2023-0706
# # description: Class for the F3netDetector

# Functions in the Class are summarized as:
# 1. __init__: Initialization
# 2. build_backbone: Backbone-building
# 3. build_loss: Loss-function-building
# 4. features: Feature-extraction
# 5. classifier: Classification
# 6. get_losses: Loss-computation
# 7. get_train_metrics: Training-metrics-computation
# 8. get_test_metrics: Testing-metrics-computation
# 9. forward: Forward-propagation

# Reference:
# @inproceedings{qian2020thinking,
#   title={Thinking in frequency: Face forgery detection by mining frequency-aware clues},
#   author={Qian, Yuyang and Yin, Guojun and Sheng, Lu and Chen, Zixuan and Shao, Jing},
#   booktitle={European conference on computer vision},
#   pages={86--103},
#   year={2020},
#   organization={Springer}
# }

# GitHub Reference:
# https://github.com/yyk-wew/F3Net

# Notes:
# We replicate the results by solely utilizing the FAD branch, following the reference GitHub implementation (https://github.com/yyk-wew/F3Net).
# '''

# import os
# import datetime
# import logging
# import numpy as np
# from sklearn import metrics
# from typing import Union
# from collections import defaultdict

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.nn import DataParallel
# from torch.utils.tensorboard import SummaryWriter

# from metrics.base_metrics_class import calculate_metrics_for_train

# from .base_detector import AbstractDetector
# from detectors import DETECTOR
# from networks import BACKBONE
# from loss import LOSSFUNC

# logger = logging.getLogger(__name__)

# @DETECTOR.register_module(module_name='f3net')
# class F3netDetector(AbstractDetector):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.backbone = self.build_backbone(config)
#         self.loss_func = self.build_loss(config)
#         # modules only use in FAD
#         img_size = config['resolution']
#         self.FAD_head = FAD_Head(img_size)

#     def build_backbone(self, config):
#         # prepare the backbone
#         backbone_class = BACKBONE[config['backbone_name']]
#         model_config = config['backbone_config']
#         backbone = backbone_class(model_config)
        
#         # To get a good performance, use the ImageNet-pretrained Xception model
#         state_dict = torch.load(config['pretrained'])
#         for name, weights in state_dict.items():
#             if 'pointwise' in name:
#                 state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
#         state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
#         conv1_data = state_dict['conv1.weight'].data
#         backbone.load_state_dict(state_dict, False)
#         logger.info('Load pretrained model from {}'.format(config['pretrained']))

#         # copy on conv1
#         # let new conv1 use old param to balance the network
#         backbone.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
#         for i in range(4):
#            backbone.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data / 4.0
#         logger.info('Copy conv1 from pretrained model')
#         return backbone
    
#     def build_loss(self, config):
#         # prepare the loss function
#         loss_class = LOSSFUNC[config['loss_func']]
#         loss_func = loss_class()
#         return loss_func
    
#     def features(self, data_dict: dict) -> torch.tensor:
#         fea_FAD = self.FAD_head(data_dict['image']) # [B, 12, 256, 256]
#         return self.backbone.features(fea_FAD)

#     def classifier(self, features: torch.tensor) -> torch.tensor:
#         return self.backbone.classifier(features)
    
#     def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
#         label = data_dict['label']
#         pred = pred_dict['cls']
#         loss = self.loss_func(pred, label)
#         loss_dict = {'overall': loss}
#         return loss_dict

#     def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
#         label = data_dict['label']
#         pred = pred_dict['cls']
#         # compute metrics for batch data
#         auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
#         metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
#         return metric_batch_dict

#     def forward(self, data_dict: dict, inference=False) -> dict:
#         # get the features by backbone
#         features = self.features(data_dict)
#         # get the prediction by classifier
#         pred = self.classifier(features)
#         # get the probability of the pred
#         prob = torch.softmax(pred, dim=1)[:, 1]
#         # build the prediction dict for each output
#         pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

#         return pred_dict


# # ===================================== other modules for F3Net # =====================================


# # Filter Module
# class Filter(nn.Module):
#     def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
#         super(Filter, self).__init__()
#         self.use_learnable = use_learnable

#         self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
#         if self.use_learnable:
#             self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
#             self.learnable.data.normal_(0., 0.1)

#         self.norm = norm
#         if norm:
#             self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


#     def forward(self, x):
#         if self.use_learnable:
#             filt = self.base + norm_sigma(self.learnable)
#         else:
#             filt = self.base

#         if self.norm:
#             y = x * filt / self.ft_num
#         else:
#             y = x * filt
#         return y


# # FAD Module
# class FAD_Head(nn.Module):
#     def __init__(self, size):
#         super(FAD_Head, self).__init__()

#         # init DCT matrix
#         self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
#         self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

#         # define base filters and learnable
#         # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
#         low_filter = Filter(size, 0, size // 2.82)
#         middle_filter = Filter(size, size // 2.82, size // 2)
#         high_filter = Filter(size, size // 2, size * 2)
#         all_filter = Filter(size, 0, size * 2)

#         self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

#     def forward(self, x):
#         # DCT
#         x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 299, 299]

#         # 4 kernel
#         y_list = []
#         for i in range(4):
#             x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
#             y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 299, 299]
#             y_list.append(y)
#         out = torch.cat(y_list, dim=1)    # [N, 12, 299, 299]
#         return out

# # utils
# def DCT_mat(size):
#     m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
#     return m

# def generate_filter(start, end, size):
#     return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

# def norm_sigma(x):
#     return 2. * torch.sigmoid(x) - 1.


'''
F3-Net with Real Frequency Normalization.

Three normalization methods applied between FAD_Head and backbone:
  N1 — FreqIN:  Instance Normalization on frequency feature maps
  N2 — LRS:     Log-Ratio Spectrum (pairwise band ratios)
  N3 — ASS:     Adaptive Spectral Standardization

Config usage:
  freq_norm: "none"    # original F3-Net (baseline)
  freq_norm: "freqin"  # N1
  freq_norm: "lrs"     # N2
  freq_norm: "ass"     # N3

Place as: detectors/f3net_norm.py
Register in: detectors/__init__.py
'''

import os
import logging
import numpy as np
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  N1: FreqIN — Instance Normalization on frequency feature maps
# ═══════════════════════════════════════════════════════════════════════════

class FreqInstanceNorm(nn.Module):
    """
    Instance Normalization applied to FAD output (B, 12, H, W).
    
    Each of the 12 channels (4 bands × 3 RGB) is normalized independently
    per sample. Removes absolute energy (dataset-specific) while preserving
    spatial activation pattern (forgery-relevant).
    
    Key property: no stored population statistics.
    μ and σ are computed per-sample, per-channel at both train and test time.
    """

    def __init__(self, num_channels=12, affine=True):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_channels, affine=affine)

    def forward(self, x):
        # x: (B, 12, H, W) — FAD output
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════════════════
#  N2: LRS — Log-Ratio Spectrum
# ═══════════════════════════════════════════════════════════════════════════

class LogRatioSpectrum(nn.Module):
    """
    Compute pairwise log-ratios between the 4 frequency band energies,
    then use these ratios to re-weight the original FAD feature maps.
    
    The 4 bands are: low, mid, high, all.
    This produces C(4,2) = 6 ratio features that are invariant to
    any multiplicative scaling of the entire spectrum.
    
    The ratios modulate the original 12-channel output via learned
    channel attention, so the backbone still receives 12-channel input.
    """

    def __init__(self, num_bands=4, channels_per_band=3):
        super().__init__()
        self.num_bands = num_bands
        self.channels_per_band = channels_per_band
        self.num_ratios = num_bands * (num_bands - 1) // 2  # 6

        # Attention: 6 ratios → 12 channel weights
        total_channels = num_bands * channels_per_band  # 12
        self.attention = nn.Sequential(
            nn.Linear(self.num_ratios, 32),
            nn.GELU(),
            nn.Linear(32, total_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, 12, H, W)
        B, C, H, W = x.shape

        # Compute per-band energy: average over spatial dims and RGB channels
        band_energies = []
        for i in range(self.num_bands):
            ch_start = i * self.channels_per_band
            ch_end = (i + 1) * self.channels_per_band
            band_map = x[:, ch_start:ch_end, :, :]       # (B, 3, H, W)
            energy = band_map.pow(2).mean(dim=(1, 2, 3))  # (B,)
            band_energies.append(energy)
        band_energies = torch.stack(band_energies, dim=1)  # (B, 4)

        # Compute pairwise log-ratios
        log_e = torch.log(band_energies + 1e-8)  # (B, 4)
        ratios = []
        for i in range(self.num_bands):
            for j in range(i + 1, self.num_bands):
                ratios.append(log_e[:, i] - log_e[:, j])
        ratios = torch.stack(ratios, dim=1)  # (B, 6)

        # Generate channel attention weights from ratios
        weights = self.attention(ratios)  # (B, 12)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, 12, 1, 1)

        # Re-weight original features
        return x * weights


# ═══════════════════════════════════════════════════════════════════════════
#  N3: ASS — Adaptive Spectral Standardization
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveSpectralStandardization(nn.Module):
    """
    Two-step instance-level normalization on FAD feature maps:
    
    Step 1 — Energy normalization:
        Divide each band's feature map by total energy across all bands.
        Removes absolute spectral magnitude (camera/compression dependent).
    
    Step 2 — Cross-band standardization:
        Standardize across the 4 bands (treating each band as one "dimension").
        Removes systematic spectral tilt.
    
    After both steps, what remains is the relative spectral shape —
    which bands deviate from the sample's own average.
    
    All statistics are per-sample. No stored parameters except optional affine.
    """

    def __init__(self, num_bands=4, channels_per_band=3, affine=True):
        super().__init__()
        self.num_bands = num_bands
        self.channels_per_band = channels_per_band
        total_channels = num_bands * channels_per_band

        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, total_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, total_channels, 1, 1))

    def forward(self, x):
        # x: (B, 12, H, W)
        B, C, H, W = x.shape
        eps = 1e-8

        # Step 1: Energy normalization
        # Compute total energy per sample across all channels and spatial locs
        total_energy = x.pow(2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + eps  # (B, 1, 1, 1)
        x = x / total_energy

        # Step 2: Cross-band standardization
        # Group channels by band, compute per-band statistics
        # Then standardize each band relative to the cross-band mean/std
        band_means = []
        band_stds = []
        for i in range(self.num_bands):
            ch_start = i * self.channels_per_band
            ch_end = (i + 1) * self.channels_per_band
            band = x[:, ch_start:ch_end, :, :]  # (B, 3, H, W)
            band_means.append(band.mean(dim=(1, 2, 3)))  # (B,)
            band_stds.append(band.std(dim=(1, 2, 3)))     # (B,)

        band_means = torch.stack(band_means, dim=1)  # (B, 4)
        band_stds = torch.stack(band_stds, dim=1)     # (B, 4)

        # Cross-band statistics
        global_mean = band_means.mean(dim=1, keepdim=True)  # (B, 1)
        global_std = band_means.std(dim=1, keepdim=True) + eps  # (B, 1)

        # Standardize each band
        x_normed = torch.zeros_like(x)
        for i in range(self.num_bands):
            ch_start = i * self.channels_per_band
            ch_end = (i + 1) * self.channels_per_band
            band = x[:, ch_start:ch_end, :, :]
            # Remove global mean and scale by global std
            band_m = band_means[:, i:i+1].unsqueeze(-1).unsqueeze(-1)  # (B,1,1,1)
            gm = global_mean.unsqueeze(-1).unsqueeze(-1)               # (B,1,1,1)
            gs = global_std.unsqueeze(-1).unsqueeze(-1)                 # (B,1,1,1)
            x_normed[:, ch_start:ch_end, :, :] = (band - gm) / gs

        if self.affine:
            x_normed = x_normed * self.gamma + self.beta

        return x_normed


# ═══════════════════════════════════════════════════════════════════════════
#  FAD_Head (unchanged from original F3-Net)
# ═══════════════════════════════════════════════════════════════════════════

class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super().__init__()
        self.use_learnable = use_learnable
        self.base = nn.Parameter(
            torch.tensor(generate_filter(band_start, band_end, size)),
            requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size))
            self.learnable.data.normal_(0., 0.1)
        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(
                torch.sum(torch.tensor(generate_filter(band_start, band_end, size))),
                requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base
        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


class FAD_Head(nn.Module):
    def __init__(self, size):
        super().__init__()
        self._DCT_all = nn.Parameter(
            torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(
            torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1),
            requires_grad=False)
        low_filter = Filter(size, 0, size // 2.82)
        middle_filter = Filter(size, size // 2.82, size // 2)
        high_filter = Filter(size, size // 2, size * 2)
        all_filter = Filter(size, 0, size * 2)
        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        x_freq = self._DCT_all @ x @ self._DCT_all_T
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)
            y = self._DCT_all_T @ x_pass @ self._DCT_all
            y_list.append(y)
        out = torch.cat(y_list, dim=1)
        return out


# ═══════════════════════════════════════════════════════════════════════════
#  Main Detector
# ═══════════════════════════════════════════════════════════════════════════

@DETECTOR.register_module(module_name='f3net')
class F3netDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

        img_size = config['resolution']
        self.FAD_head = FAD_Head(img_size)

        # ── Normalization module ──────────────────────────────────────────
        freq_norm = config.get('freq_norm', 'none')
        if freq_norm == 'freqin':
            self.freq_norm = FreqInstanceNorm(num_channels=12, affine=True)
            logger.info('Using FreqIN (Instance Normalization on FAD output)')
        elif freq_norm == 'lrs':
            self.freq_norm = LogRatioSpectrum(num_bands=4, channels_per_band=3)
            logger.info('Using LRS (Log-Ratio Spectrum)')
        elif freq_norm == 'ass':
            self.freq_norm = AdaptiveSpectralStandardization(
                num_bands=4, channels_per_band=3, affine=True)
            logger.info('Using ASS (Adaptive Spectral Standardization)')
        elif freq_norm == 'none':
            self.freq_norm = nn.Identity()
            logger.info('No frequency normalization (baseline)')
        else:
            raise ValueError(f'Unknown freq_norm: {freq_norm}')

    def build_backbone(self, config):
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)

        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        conv1_data = state_dict['conv1.weight'].data
        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model from {}'.format(config['pretrained']))

        backbone.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
        for i in range(4):
            backbone.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data / 4.0
        logger.info('Copy conv1 from pretrained model')
        return backbone

    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        return loss_class()

    def features(self, data_dict: dict) -> torch.Tensor:
        # Step 1: FAD frequency decomposition
        fea_FAD = self.FAD_head(data_dict['image'])  # (B, 12, H, W)
        
        # Step 2: Frequency normalization (the insertion point)
        fea_FAD = self.freq_norm(fea_FAD)            # (B, 12, H, W)
        
        # Step 3: Backbone feature extraction
        return self.backbone.features(fea_FAD)

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.classifier(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        return {'overall': loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {'cls': pred, 'prob': prob, 'feat': features}


# ═══════════════════════════════════════════════════════════════════════════
#  Utilities (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def DCT_mat(size):
    return [[(np.sqrt(1./size) if i == 0 else np.sqrt(2./size))
             * np.cos((j + 0.5) * np.pi * i / size)
             for j in range(size)]
            for i in range(size)]

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1.
             for j in range(size)]
            for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.