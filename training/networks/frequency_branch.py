import logging
import torch
import torch.nn as nn
from torchvision.models import resnet18
from networks import BACKBONE
from networks.sdnorm import SDNorm

logger = logging.getLogger(__name__)

class FFTTransform(nn.Module):
    def __init__(self, log_scale=True, center=True):
        super().__init__()
        self.log_scale = log_scale
        self.center = center

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft_transformed = torch.fft.fft2(x, norm='ortho')
        if self.center:
            fft_transformed = torch.fft.fftshift(fft_transformed, dim=(-2, -1))
        magnitude = fft_transformed.abs()
        phase = fft_transformed.angle()
        if self.log_scale:
            magnitude = torch.log1p(magnitude)
        phase = phase / torch.pi
        return torch.cat([magnitude, phase], dim=1)

class FrequencyBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone_name = config.get('freq_backbone', 'resnet18')
        out_dim = config.get('freq_out_dim', 512)
        img_size = config.get('resolution', 224)
        self.out_dim = out_dim
        self.fft = FFTTransform(log_scale=config.get('fft_log_scale', True), center=True)
        self.encoder = self.build_backbone(config)

        self.use_sdnorm = config.get('use_sdnorm', True)
        if self.use_sdnorm:
            sc = config.get('sdnorm_config', {})
            self.sdnorm = SDNorm(
                size=img_size,
                num_rings=sc.get('num_rings', 4),
                num_mag_channels=3,
                eps=sc.get('eps', 1e-5), 
                learnable_affine=sc.get('learnable_affine', True),
                sharpness=sc.get('sharpness', 5.0),
            )
        else:
            self.sdnorm = None
        self.norm = nn.LayerNorm(out_dim)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"FrequencyBranch-FFT ({backbone_name}): {trainable:,} params")

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
        return backbone

    def forward(self, x):
        x = self.fft(x)
        if self.sdnorm is not None:
            x = self.sdnorm(x)
        y_freq = self.encoder(x)
        y_freq = self.norm(y_freq)
        return y_freq


@BACKBONE.register_module(module_name="resnet18_freq")
class ResNetBackbone(nn.Module):
    """ResNet-18 for magnitude+phase. 6-channel input."""
    def __init__(self, config):
        super().__init__()
        in_channels = config.get('inc', 6)
        out_dim = config.get('num_classes', 512)
        backbone = resnet18(weights=None)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        backbone.fc = nn.Linear(backbone.fc.in_features, out_dim)
        self.backbone = backbone
        self.last_channel = out_dim

    def forward(self, x):
        return self.backbone(x)
