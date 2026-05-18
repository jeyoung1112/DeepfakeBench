'''
Frequency-aware Discriminative Feature Learning Supervised by Single-CenterLoss for Face Forgery Detection.
'''

import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


class SingleCenterLoss(nn.Module):
    """
    L_sc = M_nat + max(M_nat - M_man + m*sqrt(D), 0)
    """

    def __init__(self, margin = 0.3, feat_dim = 1000, use_gpu=True):
        super(SingleCenterLoss, self).__init__()
        self.m = margin
        self.D = feat_dim
        self.margin = self.m * (feat_dim ** 0.5)  # plain float — avoids CPU/GPU device mismatch
        self.l2loss = nn.MSELoss(reduction = 'none')
        self.C = nn.Parameter(torch.randn(self.D))
        # Paper supplementary: clip center gradient to 0.01 to prevent oscillation.
        self.C.register_hook(lambda g: g.clamp(-0.01, 0.01))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        eud_mat = torch.sqrt(self.l2loss(x, self.C.expand(batch_size, self.C.size(0))).sum(dim=1, keepdim=True))

        labels = labels.unsqueeze(1)

        fake_count = labels.sum()
        real_count = batch_size - fake_count

        dist_real = (eud_mat * (1 - labels.float())).clamp(min=1e-12, max=1e+12).sum()
        dist_fake = (eud_mat * labels.float()).clamp(min=1e-12, max=1e+12).sum()

        if real_count != 0:
            dist_real /= real_count

        if fake_count != 0:
            dist_fake /= fake_count

        loss = dist_real + torch.clamp(dist_real - dist_fake + self.margin, min=0)

        return loss


class DataPreprocessing(nn.Module):
    """
    RGB image → YCbCr → 2-D block DCT per 8×8 block → [B, 192, H/8, W/8].
    """

    def __init__(self, block_size: int = 8, num_channels: int = 192):
        super().__init__()
        self.block_size = block_size
        N = block_size
        n = np.arange(N, dtype=np.float64)
        k = n.reshape(N, 1)
        D = np.cos(np.pi / N * (n + 0.5) * k)
        D[0, :] *= np.sqrt(1.0 / N)
        D[1:, :] *= np.sqrt(2.0 / N)
        self.register_buffer('dct_mat', torch.FloatTensor(D))  # [8, 8]
        # Per-channel normalization statistics (paper §3.2); identity by default.
        # Call set_channel_stats() with training-dataset statistics before training.
        self.register_buffer('ch_mean', torch.zeros(1, num_channels, 1, 1))
        self.register_buffer('ch_std', torch.ones(1, num_channels, 1, 1))

    def set_channel_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.ch_mean.copy_(mean.reshape(1, -1, 1, 1))
        self.ch_std.copy_(std.reshape(1, -1, 1, 1))

    def _rgb_to_ycbcr(self, x: torch.Tensor) -> torch.Tensor:
        # https://m.blog.naver.com/ngic315/20013751140
        x = (x + 1.0) / 2.0  # → [0, 1]
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y  =  0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.16874 * r - 0.33126 * g + 0.5 * b
        cr =  0.5 * r - 0.41869 * g - 0.08131 * b
        return torch.cat([y, cb, cr], dim=1)  # [B, 3, H, W]

    def _block_dct(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        bs = self.block_size
        # Split into non-overlapping 8×8 blocks
        x = x.reshape(B, C, H // bs, bs, W // bs, bs)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()  # [B, C, H//8, W//8, 8, 8]
        # 2-D DCT: D @ X @ D^T
        D = self.dct_mat
        x = torch.matmul(D, x)      # apply along rows
        x = torch.matmul(x, D.t())  # apply along cols
        # Group by frequency band → [B, C*64, H//8, W//8]
        x = x.reshape(B, C, H // bs, W // bs, bs * bs)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.reshape(B, C * bs * bs, H // bs, W // bs)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = self.block_size
        H, W = x.shape[2], x.shape[3]
        # Pad to multiple of block_size
        pad_h = (-H) % bs
        pad_w = (-W) % bs
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        ycbcr = self._rgb_to_ycbcr(x)             # [B, 3, H, W]
        dct = self._block_dct(ycbcr)              # [B, 192, H//8, W//8]
        return (dct - self.ch_mean) / (self.ch_std + 1e-8)


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = F.adaptive_max_pool2d(x, 1).flatten(1)
        attn = F.relu(self.fc1(attn), inplace=True)
        attn = torch.sigmoid(self.fc2(attn))
        return x * attn.unsqueeze(-1).unsqueeze(-1)


class AFIMB(nn.Module):
    """
    Adaptive Frequency Information Mining Block (paper Section 3.2).
    """

    def __init__(self, in_channels: int = 192, out_channels: int = 192):
        super().__init__()
        self.grouped_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.ordinary_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(2, 2)
        self.channel_attn = _ChannelAttention(in_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.grouped_conv(x)
        x = self.ordinary_conv(x)
        x = self.maxpool(x)
        x = self.channel_attn(x)
        return self.proj(x)


class AFFGM(nn.Module):
    """Adaptive Frequency Feature Generation Module."""

    def __init__(self, out_channels: int = 192):
        super().__init__()
        self.preprocessing = DataPreprocessing(block_size=8)
        self.afimb = AFIMB(in_channels=192, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocessing(x)
        return self.afimb(x)


@DETECTOR.register_module(module_name='fdfl')
class FDFLDetector(AbstractDetector):
    """
    FDFL: Frequency-aware Discriminative Feature Learning.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lambda_scl = config.get('lambda_scl', 0.5)
        self.margin = config.get('margin', 0.3)

        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

        freq_out_ch = config.get('freq_out_channels', 192)
        self.affgm = AFFGM(out_channels=freq_out_ch)

        # Point-wise fusion: concat(728 + freq_out_ch) → 728
        self.fusion_module = nn.Sequential(
            nn.Conv2d(728 + freq_out_ch, 728, 1, 1, 0),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
        )

        # Paper §4.1: "including the final fully connected layer … D=1000"
        # embedding_fc replicates Xception's pretrained 1000-class FC (2048→1000).
        # binary_fc is the separate 2-node classifier described in §4.1.
        feat_dim = 1000
        self.embedding_fc = self._load_embedding_fc(config)  # 2048 → 1000
        self.binary_fc = nn.Linear(feat_dim, 2)
        self.scl_loss = SingleCenterLoss(feat_dim=feat_dim, margin=self.margin)

    def build_backbone(self, config):
        backbone_class = BACKBONE[config['backbone_name']]
        backbone = backbone_class(config['backbone_config'])
        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        # Stash full state dict for _load_embedding_fc before filtering.
        self._pretrained_state = state_dict
        # Exclude the pretrained FC ('fc.*'): loaded separately into embedding_fc.
        backbone_state = {k: v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(backbone_state, strict=False)
        logger.info('Loaded pretrained Xception from {}'.format(config['pretrained']))
        return backbone

    def _load_embedding_fc(self, config) -> nn.Linear:
        fc = nn.Linear(2048, 1000)
        state_dict = getattr(self, '_pretrained_state', None) or torch.load(config['pretrained'])
        # Pretrained Xception checkpoint uses key 'fc' (not 'last_linear').
        if 'fc.weight' in state_dict:
            fc.weight = nn.Parameter(state_dict['fc.weight'].clone())
            fc.bias = nn.Parameter(state_dict['fc.bias'].clone())
            logger.info('Loaded pretrained embedding FC (2048→1000) from fc.*')
        else:
            logger.warning('fc.weight not found in pretrained; embedding_fc randomly initialized')
        return fc

    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        return loss_class()

    def features(self, data_dict: dict) -> torch.Tensor:
        img = data_dict['image']

        # RGB entry flow (Xception conv1+conv2+block1-3)
        x_rgb = self.backbone.fea_part1(img)
        x_rgb = self.backbone.fea_part2(x_rgb)   # [B, 728, H_e, W_e]

        # Frequency branch
        x_freq = self.affgm(img)                  # [B, freq_ch, H_f, W_f]

        # Align spatial dimensions (handles rounding differences)
        if x_freq.shape[2:] != x_rgb.shape[2:]:
            x_freq = F.adaptive_avg_pool2d(x_freq, x_rgb.shape[2:])

        # Fusion: concatenate + point-wise conv
        x = self.fusion_module(torch.cat([x_rgb, x_freq], dim=1))  # [B, 728, H_e, W_e]

        # Middle + exit flow
        x = self.backbone.fea_part3(x)
        x = self.backbone.fea_part4(x)
        x = self.backbone.fea_part5(x)
        return x  # [B, 2048, H_out, W_out]

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        x = F.relu(features)
        if len(x.shape) == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)    # [B, 2048]
        self.last_emb = self.embedding_fc(x)  # [B, 1000]
        return self.binary_fc(self.last_emb)  # [B, 2]

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        cls_loss = self.loss_func(pred_dict['cls'], label)
        scl = self.scl_loss(pred_dict['feat'], label)
        overall = cls_loss + self.lambda_scl * scl
        return {'overall': overall, 'cls': cls_loss, 'scl': scl}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {'cls': pred, 'prob': prob, 'feat': self.last_emb}
