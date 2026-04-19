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

        # Distance from center
        u = torch.arange(size, dtype=torch.float32)
        v = torch.arange(size, dtype=torch.float32)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        center_u, center_v = size / 2.0, size / 2.0
        dist = torch.sqrt((uu - center_u) ** 2 + (vv - center_v) ** 2)
        dist_norm = dist / dist.max()  # [0, 1], 0 = DC center
        self.register_buffer('dist_norm', dist_norm)

        # Log-spaced ring boundaries (finer resolution at low frequencies)
        boundaries = torch.logspace(math.log10(0.01), math.log10(1.0), num_rings + 1)
        self.inner = nn.Parameter(boundaries[:-1].clone())
        self.outer = nn.Parameter(boundaries[1:].clone())

        # Learnable sharpness (stored as log for positivity)
        self.sharpness_logit = nn.Parameter(torch.tensor(sharpness).log())

    def forward(self):
        """Returns [K, H, W] soft masks centered on DC."""
        sharpness = self.sharpness_logit.exp()
        d = self.dist_norm.unsqueeze(0)              # [1, H, W]
        inner = self.inner.unsqueeze(-1).unsqueeze(-1)  # [K, 1, 1]
        outer = self.outer.unsqueeze(-1).unsqueeze(-1)  # [K, 1, 1]

        # Soft annular ring: ramp up at inner boundary, ramp down at outer
        mask_in = torch.sigmoid(sharpness * (d - inner))
        mask_out = torch.sigmoid(sharpness * (outer - d))
        return mask_in * mask_out  # [K, H, W]


class SpectralStyleAugmentation(nn.Module):
    def __init__(self, aug_prob=0.5):
        super().__init__()
        self.aug_prob = aug_prob

    def forward(self, ring_feats, ring_means, ring_stds):
        if not self.training:
            # Inference: restore original statistics
            return [f * s + m for f, m, s in zip(ring_feats, ring_means, ring_stds)]

        B = ring_feats[0].size(0)
        device = ring_feats[0].device
        out = []

        for f, m, s in zip(ring_feats, ring_means, ring_stds):
            # Per-sample coin flip: augment or keep original
            do_aug = torch.rand(B, device=device) < self.aug_prob  # [B]
            perm = torch.randperm(B, device=device)

            # Swap statistics from random donor where augmenting
            aug_mask = do_aug[:, None, None, None]  # [B, 1, 1, 1]
            m_aug = torch.where(aug_mask, m[perm], m)
            s_aug = torch.where(aug_mask, s[perm], s)

            # Denormalise with (possibly swapped) statistics
            out.append(f * s_aug + m_aug)

        return out


class SDNorm(nn.Module):
    def __init__(
        self,
        size=224,
        num_rings=4,
        num_mag_channels=3,
        eps=1e-5,
        aug_prob=0.5,
        learnable_affine=True,
        sharpness=5.0,
    ):
        super().__init__()
        self.num_rings = num_rings
        self.num_mag_channels = num_mag_channels
        self.eps = eps
        self.learnable_affine = learnable_affine

        self.ring_masks = FrequencyRingMasks(size, num_rings, sharpness)
        self.style_aug = SpectralStyleAugmentation(aug_prob)

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
            f"SDNorm-FFT (mag-only): {n_params:,} params, "
            f"{num_rings} rings, {num_mag_channels} mag channels, "
            f"phase pass-through, aug_prob={aug_prob:.2f}"
        )

    def forward(self, fft_coeffs):
        mc = self.num_mag_channels

        # Split magnitude and phase
        mag = fft_coeffs[:, :mc]     # [B, 3, H, W] 
        phase = fft_coeffs[:, mc:]   # [B, 3, H, W]

        # Ring decomposition and normalisation (magnitude only)
        masks = self.ring_masks()  # [K, H, W]

        ring_feats_norm = []
        ring_means = []
        ring_stds = []

        for k in range(self.num_rings):
            mask_k = masks[k].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

            # Extract ring from magnitude channels
            ring_k = mag * mask_k  # [B, 3, H, W]

            # Mask-weighted statistics (per-sample, per-channel)
            mask_sum = mask_k.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
            mean_k = ring_k.sum(dim=(-2, -1), keepdim=True) / mask_sum  # [B, 3, 1, 1]
            var_k = (
                (ring_k - mean_k * mask_k) ** 2 * mask_k
            ).sum(dim=(-2, -1), keepdim=True) / mask_sum
            std_k = torch.sqrt(var_k + self.eps)  # [B, 3, 1, 1]

            # Instance normalise within the ring
            normed_k = (ring_k - mean_k * mask_k) / std_k  # [B, 3, H, W]

            ring_feats_norm.append(normed_k)
            ring_means.append(mean_k)
            ring_stds.append(std_k)

        # Spectral style augmentation
        ring_feats_restyled = self.style_aug(
            ring_feats_norm, ring_means, ring_stds
        )

        # Learnable affine (magnitude only)
        if self.learnable_affine:
            ring_feats_restyled = [
                f * self.gamma[k] + self.beta[k]
                for k, f in enumerate(ring_feats_restyled)
            ]

        # Reassemble magnitude from rings
        mag_normalised = sum(ring_feats_restyled)  # [B, 3, H, W]

        # Residual gate (magnitude only)
        gate = torch.sigmoid(self.gate_logit)
        mag_out = gate * mag_normalised + (1.0 - gate) * mag  # [B, 3, H, W]

        return torch.cat([mag_out, phase], dim=1)  # [B, 6, H, W]
