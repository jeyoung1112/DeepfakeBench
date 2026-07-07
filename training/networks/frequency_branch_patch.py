import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from networks import BACKBONE

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


class FrequencyBranchPatch(nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone_name = config.get('freq_backbone', 'resnet18')
        out_dim = config.get('freq_out_dim', 512)
        spec_residual_type = config.get('sr_type')

        self.out_dim = out_dim
        self.fft = FFTTransform(log_scale=config.get('fft_log_scale', True), center=True)
        self.encoder = self.build_backbone(config)
        # channels of the pre-pool spatial map (used for patch tokens); decoupled from out_dim
        self.map_dim = getattr(self.encoder, 'map_dim', out_dim)

        self.spec_residual = None
        if spec_residual_type == "original":
            self.spec_residual = OriginalSpectralResidualNorm(
                envelope_kernel=config.get('envelope_kernel', 21))

        self.norm = nn.LayerNorm(out_dim)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"FrequencyBranchPatch-FFT ({backbone_name}): {trainable:,} params "
                    f"(pooled={out_dim}, map={self.map_dim})")

    def build_backbone(self, config):
        backbone_class = BACKBONE[config['freq_backbone_name']]
        backbone = backbone_class(config['freq_backbone_config'])
        pretrained = config.get('pretrained')
        if pretrained:
            state_dict = torch.load(pretrained, map_location='cpu')
            for name, weights in list(state_dict.items()):
                if 'pointwise' in name:
                    state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
            state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
            backbone.load_state_dict(state_dict, strict=False)
        return backbone

    def forward(self, x, return_map=False):
        x = self.fft(x)
        if self.spec_residual is not None:
            x = self.spec_residual(x)
        if return_map:
            pooled, fmap = self.encoder.forward_map(x)
            return self.norm(pooled), fmap            # pooled -> CE head, fmap -> patch tokens
        return self.norm(self.encoder(x))


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
        self.map_dim = backbone.fc.in_features        # layer4 channels (512 for resnet18)
        backbone.fc = nn.Linear(backbone.fc.in_features, out_dim)
        self.backbone = backbone
        self.last_channel = out_dim

    def forward(self, x):
        return self.backbone(x)

    def forward_map(self, x):
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x); x = b.layer2(x); x = b.layer3(x)
        fmap = b.layer4(x)                                    # [B, map_dim, h, w]
        pooled = b.fc(torch.flatten(b.avgpool(fmap), 1))      # [B, out_dim]
        return pooled, fmap