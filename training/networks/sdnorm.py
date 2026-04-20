import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FrequencyRingMasks(nn.Module):
    """
    Center-origin ring masks for FFT spectrum (DC at center after fftshift).
    """

    def __init__(self, size=224, num_rings=4, sharpness=5.0):
        super().__init__()
        self.num_rings = num_rings

        u = torch.arange(size, dtype=torch.float32)
        v = torch.arange(size, dtype=torch.float32)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        dist = torch.sqrt((uu - size / 2.0) ** 2 + (vv - size / 2.0) ** 2)
        dist_norm = dist / dist.max()
        self.register_buffer('dist_norm', dist_norm)

        boundaries = torch.logspace(math.log10(0.01), math.log10(1.0), num_rings + 1)
        self.inner = nn.Parameter(boundaries[:-1].clone())
        self.outer = nn.Parameter(boundaries[1:].clone())
        self.sharpness_logit = nn.Parameter(torch.tensor(sharpness).log())

    def forward(self):
        """Returns [K, H, W] masks summing to 1 at every pixel."""
        sharpness = self.sharpness_logit.exp()
        d = self.dist_norm.unsqueeze(0)                # [1, H, W]
        inner = self.inner.unsqueeze(-1).unsqueeze(-1)  # [K, 1, 1]
        outer = self.outer.unsqueeze(-1).unsqueeze(-1)  # [K, 1, 1]
        raw = torch.sigmoid(sharpness * (d - inner)) * torch.sigmoid(sharpness * (outer - d))
        return F.softmax(raw, dim=0)                   # [K, H, W]


class SDNorm(nn.Module):
    def __init__(
        self,
        size=224,
        num_rings=4,
        num_mag_channels=3,
        eps=1e-5,
        learnable_affine=True,
        sharpness=5.0,
    ):
        super().__init__()
        self.num_rings = num_rings
        self.num_mag_channels = num_mag_channels
        self.eps = eps
        self.learnable_affine = learnable_affine

        self.ring_masks = FrequencyRingMasks(size, num_rings, sharpness)

        if learnable_affine:
            self.gamma = nn.ParameterList([
                nn.Parameter(torch.ones(1, num_mag_channels, 1, 1))
                for _ in range(num_rings)
            ])
            self.beta = nn.ParameterList([
                nn.Parameter(torch.zeros(1, num_mag_channels, 1, 1))
                for _ in range(num_rings)
            ])

        self.gate_logit = nn.Parameter(torch.tensor(0.0))

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"SDNorm: {n_params:,} params, "
            f"{num_rings} rings, {num_mag_channels} mag channels"
        )

    def forward(self, fft_coeffs):
        mc = self.num_mag_channels
        mag   = fft_coeffs[:, :mc]   # [B, C, H, W]
        phase = fft_coeffs[:, mc:]   # [B, C, H, W]

        masks = self.ring_masks()    # [K, H, W], partition-of-unity

        ring_feats = []
        for k in range(self.num_rings):
            mask_k   = masks[k].unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
            mask_sum = mask_k.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)

            mean_k = (mag * mask_k).sum(dim=(-2, -1), keepdim=True) / mask_sum
            var_k  = ((mag - mean_k) ** 2 * mask_k).sum(dim=(-2, -1), keepdim=True) / mask_sum
            std_k  = (var_k + self.eps).sqrt()

            normed_k = (mag - mean_k) / std_k * mask_k

            if self.learnable_affine:
                normed_k = normed_k * self.gamma[k] + self.beta[k] * mask_k

            ring_feats.append(normed_k)

        mag_normalised = sum(ring_feats)   # [B, C, H, W]

        gate    = torch.sigmoid(self.gate_logit)
        mag_out = gate * mag_normalised + (1.0 - gate) * mag

        return torch.cat([mag_out, phase], dim=1)
