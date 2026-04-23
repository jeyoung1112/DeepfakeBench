import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class OriginalSpectralResidualNorm(nn.Module):
    def __init__(self, envelope_kernel=21):
        super().__init__()
        # Uniform box filter, n x n, sums to 1
        kernel = torch.ones(3, 1, envelope_kernel, envelope_kernel) / (envelope_kernel * envelope_kernel)
        self.register_buffer('kernel', kernel)
        self.padding = envelope_kernel // 2

    def forward(self, fft_coeffs):
        mag = fft_coeffs[:, :3]
        phase = fft_coeffs[:, 3:]
        log_mag = torch.log1p(mag)
        envelope = F.conv2d(log_mag, self.kernel, padding=self.padding, groups=3)
        residual = log_mag - envelope
        return torch.cat([residual, phase], dim=1)


class FrequencyBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone_name = config.get('freq_backbone', 'resnet18')
        out_dim = config.get('freq_out_dim', 512)
        img_size = config.get('resolution', 224)
        spec_residual_type = config.get('sr_type')
        self.out_dim = out_dim
        self.fft = FFTTransform(log_scale=config.get('fft_log_scale', True), center=True)
        self.encoder = self.build_backbone(config)
        self.spec_residual = None
        if spec_residual_type == "original":
            self.spec_residual = OriginalSpectralResidualNorm(
                envelope_kernel=config.get('envelope_kernel', 21)
                )

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
        if self.spec_residual is not None:
            x = self.spec_residual(x)
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
