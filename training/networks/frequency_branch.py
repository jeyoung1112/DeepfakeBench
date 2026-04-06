import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights
from networks import BACKBONE

logger = logging.getLogger(__name__)

class DCTTransform(nn.Module):
 
    def __init__(self, size=224, log_scale=True):
        super().__init__()
        self.size = size
        self.log_scale = log_scale
        self.register_buffer("basis", self._build_basis(size))
 
    @staticmethod
    def _build_basis(N):
        n = torch.arange(N, dtype=torch.float32)
        k = torch.arange(N, dtype=torch.float32)
        basis = torch.cos(math.pi / N * (n.unsqueeze(0) + 0.5) * k.unsqueeze(1))
        basis[0, :] *= 1.0 / math.sqrt(N)
        basis[1:, :] *= math.sqrt(2.0 / N)
        return basis
 
    def forward(self, x):
        """[B, C, H, W] -> [B, C, H, W] DCT coefficients."""
        dct = torch.matmul(self.basis, x)
        dct = torch.matmul(dct, self.basis.t())
        if self.log_scale:
            dct = torch.sign(dct) * torch.log1p(torch.abs(dct))
        return dct

class FrequencyBranch(nn.Module):
    def __init__(self, config):
        super().__init__()

        backbone_name = config.get('freq_backbone', 'resnet18')
        out_dim = config.get('freq_out_dim', 512)
        img_size = config.get('resolution', 224)
        log_scale = config.get('dct_log_scale', True)

        self.encoder = self.build_backbone(config)

        self.out_dim = out_dim
        self.dct = DCTTransform(size=img_size, log_scale=log_scale)
        self.norm = nn.LayerNorm(out_dim)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"FrequencyBranch ({backbone_name}): {trainable:,} params")

    def build_backbone(self, config):
        backbone_class = BACKBONE[config['freq_backbone_name']]
        model_config = config['freq_backbone_config']
        backbone = backbone_class(model_config)
        pretrained = config.get('pretrained')
        if pretrained:
            state_dict = torch.load(pretrained)
            for name, weights in state_dict.items():
                if 'pointwise' in name:
                    state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
            state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
            backbone.load_state_dict(state_dict, strict=False)
            logger.info('Load pretrained model successfully!')
        return backbone

    def forward(self, x):
        x = self.dct(x)
        y_freq = self.encoder(x)
        y_freq = self.norm(y_freq)
        return y_freq


# class SeparableConv2d(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
#         super().__init__()
#         self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding,
#                                    groups=in_ch, bias=bias)
#         self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
 
#     def forward(self, x):
#         return self.pointwise(self.depthwise(x))
 
 
# class XceptionBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, stride=1):
#         super().__init__()
#         self.sep1 = SeparableConv2d(in_ch, out_ch, 3, 1, 1)
#         self.bn1 = nn.BatchNorm2d(out_ch)
#         self.sep2 = SeparableConv2d(out_ch, out_ch, 3, stride, 1)
#         self.bn2 = nn.BatchNorm2d(out_ch)
#         self.shortcut = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
#             nn.BatchNorm2d(out_ch),
#         ) if in_ch != out_ch or stride != 1 else nn.Identity()
 
#     def forward(self, x):
#         residual = self.shortcut(x)
#         out = F.relu(self.bn1(self.sep1(x)))
#         out = self.bn2(self.sep2(out))
#         return F.relu(out + residual)
 
 
# class XceptionBackbone(nn.Module):
#     """Lightweight Xception for frequency maps. ~1.8M params."""
 
#     def __init__(self, in_channels=3, out_dim=512):
#         super().__init__()
#         self.stem = nn.Sequential(
#             nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(32), nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64), nn.ReLU(inplace=True),
#         )
#         self.blocks = nn.Sequential(
#             XceptionBlock(64, 128, stride=2),
#             XceptionBlock(128, 256, stride=2),
#             XceptionBlock(256, 512, stride=2),
#             XceptionBlock(512, 512, stride=2),
#         )
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(512, out_dim)
#         self.last_channel = out_dim
 
#     def forward(self, x):
#         x = self.stem(x)
#         x = self.blocks(x)
#         x = self.pool(x).flatten(1)
#         return self.fc(x)
 
 
@BACKBONE.register_module(module_name="resnet18_freq")
class ResNetBackbone(nn.Module):
    """ResNet-18 adapted for frequency maps. ~11.4M params."""

    def __init__(self, config):
        super().__init__()
        in_channels = config.get('inc', 3)
        out_dim = config.get('num_classes', 512)
        backbone = resnet18(weights=None)  # no pretrained for DCT input
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        backbone.fc = nn.Linear(backbone.fc.in_features, out_dim)
        self.backbone = backbone
        self.last_channel = out_dim

    def forward(self, x):
        return self.backbone(x)