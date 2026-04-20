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

class SpectralResidualNorm(nn.Module):
    def __init__(self, envelope_kernel=21, num_scales=3, eps=1e-6):
        super().__init__()
        self.num_scales=num_scales
        self.eps = eps

        self.envelope_filters = nn.ModuleList()
        for s in range(num_scales):
            kernel_size = envelope_kernel + s * 10  # 21, 31, 41
            if kernel_size % 2 == 0:
                kernel_size += 1
            conv = nn.Conv2d(
                num_mag_channels, num_mag_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=num_mag_channels,  # depthwise
                bias=False
            )
            # Initialize as Gaussian
            sigma = kernel_size / 6.0
            ax = torch.arange(kernel_size).float() - kernel_size // 2
            xx, yy = torch.meshgrid(ax, ax, indexing='ij')
            gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            gaussian = gaussian / gaussian.sum()
            with torch.no_grad():
                for c in range(num_mag_channels):
                    conv.weight[c, 0] = gaussian
            self.envelope_filters.append(conv)
        
        # Learnable scale weights (which envelope scales to combine)
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # Per-scale learnable affine on residuals
        self.gamma = nn.Parameter(torch.ones(1, num_mag_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_mag_channels, 1, 1))
        
        # Residual gate: how much raw mag to preserve
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, fft_coeffs):
        mc = self.num_mag_channels
        mag = fft_coeffs[:, :mc]     # [B, 3, H, W]
        phase = fft_coeffs[:, mc:]   # [B, 3, H, W]
        
        # Step 1: Log-magnitude spectrum
        log_mag = torch.log1p(mag)   # log(1 + |F|), already done in FFTTransform
        # If FFTTransform already applies log1p, use mag directly as log_mag
        
        # Step 2: Multi-scale smooth envelope estimation
        scale_w = torch.softmax(self.scale_weights, dim=0)
        envelope = torch.zeros_like(log_mag)
        for s, filt in enumerate(self.envelope_filters):
            envelope = envelope + scale_w[s] * filt(log_mag)
        
        # Step 3: Spectral residual = deviation from smooth envelope
        residual = log_mag - envelope  # [B, 3, H, W]
        
        # Step 4: Learnable affine on residual
        residual = residual * self.gamma + self.beta
        
        # Step 5: Gated combination
        # Gate allows network to blend residual with original if needed
        gate = torch.sigmoid(self.gate_logit)
        mag_out = gate * residual + (1.0 - gate) * log_mag
        
        # Concatenate with phase (untouched)
        return torch.cat([mag_out, phase], dim=1)  # [B, 6, H, W]
    

class FrequencyBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone_name = config.get('freq_backbone', 'resnet18')
        out_dim = config.get('freq_out_dim', 512)
        img_size = config.get('resolution', 224)
        self.out_dim = out_dim
        self.fft = FFTTransform(log_scale=config.get('fft_log_scale', True), center=True)
        self.encoder = self.build_backbone(config)
        self.spec_residual = SpectralResidualNorm(
            envelope_kernel=config.get('envelope_kernel', 21),
            num_scales=config.get('num_envelope_scales', 3))
        self.norm = nn.LayerNorm(self.out_dim)
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
