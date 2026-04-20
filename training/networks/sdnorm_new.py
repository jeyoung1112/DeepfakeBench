import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FrequencyRingMasks(nn.Module):
    """Center-origin ring masks for FFT spectrum (DC at center after fftshift)."""

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
        d = self.dist_norm.unsqueeze(0)
        inner = self.inner.unsqueeze(-1).unsqueeze(-1)
        outer = self.outer.unsqueeze(-1).unsqueeze(-1)
        raw = torch.sigmoid(sharpness * (d - inner)) * torch.sigmoid(sharpness * (outer - d))
        return F.softmax(raw, dim=0)   # [K, H, W]


class SDNormNew(nn.Module):
    """
    Dataset-bias-free frequency normalization for cross-dataset generalization.

    Problem with per-ring normalization (SDNorm): normalizing each ring independently
    removes both dataset bias AND real/fake ring-level differences, collapsing the
    discriminative frequency profile.

    Solution: global instance normalization over all frequency components removes
    dataset-level energy bias (caused by compression, sensor, ISP differences) while
    preserving the relative energy distribution across rings — which is the primary
    real/fake cue. Learnable per-ring affine then re-weights each band.

    Works at inference without dataset labels — only per-sample statistics needed,
    making it suitable for training on FF++ and generalizing to unseen datasets.
    """

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
            # Per-ring scale/shift applied after global normalization.
            # Allows the model to emphasize discriminative frequency bands.
            self.gamma = nn.ParameterList([
                nn.Parameter(torch.ones(1, num_mag_channels, 1, 1))
                for _ in range(num_rings)
            ])
            self.beta = nn.ParameterList([
                nn.Parameter(torch.zeros(1, num_mag_channels, 1, 1))
                for _ in range(num_rings)
            ])

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"SDNormNew: {n_params:,} params, "
            f"{num_rings} rings, {num_mag_channels} mag channels"
        )

    def forward(self, fft_coeffs):
        """
        Args:
            fft_coeffs: [B, C, H, W] where first num_mag_channels are log-magnitude,
                        remaining are phase. Magnitude assumed to already be log1p-scaled
                        by FFTTransform.

        Returns:
            [B, C, H, W] with globally instance-normalised magnitude + original phase.
        """
        mc = self.num_mag_channels
        mag   = fft_coeffs[:, :mc]    # [B, C, H, W] — log1p magnitude
        phase = fft_coeffs[:, mc:]    # [B, C, H, W] — phase / pi

        # Global instance normalization: remove per-sample energy level.
        # mean/std computed over all spatial frequencies — NOT per-ring.
        # This collapses dataset-level bias (compression artifacts, sensor gain)
        # while keeping the relative frequency profile intact.
        mean = mag.mean(dim=(-2, -1), keepdim=True)   # [B, C, 1, 1]
        std  = mag.std(dim=(-2, -1), keepdim=True) + self.eps
        mag_norm = (mag - mean) / std                  # [B, C, H, W]

        if self.learnable_affine:
            # Partition-of-unity masks ensure each pixel belongs to exactly one ring
            # in a soft sense, so the weighted sum recovers the full spatial map.
            masks = self.ring_masks()   # [K, H, W]
            mag_out = torch.zeros_like(mag_norm)
            for k in range(self.num_rings):
                mask_k = masks[k].unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
                mag_out = mag_out + mask_k * (self.gamma[k] * mag_norm + self.beta[k])
        else:
            mag_out = mag_norm

        return torch.cat([mag_out, phase], dim=1)
