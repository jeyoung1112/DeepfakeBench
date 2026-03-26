'''
# description: CLIPLoraFrequency detector
#   Extends CLIPSimpleLora by concatenating a DCT-based high-frequency
#   energy ratio scalar (inspired by DDA) with the CLIP CLS token before
#   the linear classifier.
#
# Feature vector layout:
#   [ CLIP pooler_output (1024) | HF energy ratio (1) ] -> Linear(1025, 2)
#
# The HF energy ratio is computed per image as:
#   ratio = sum(|FFT|^2 outside low-freq circle) / sum(|FFT|^2)
# where the low-freq circle has radius = min(H, W) / 4 in the shifted
# spectrum.  This is differentiable and captures the same frequency
# content as a DCT magnitude analysis for real-valued inputs.
#
# Reference (DDA):
#   Roy et al., "Dual-Data Alignment for Deepfake Detection", 2023.
#   https://github.com/roy-ch/Dual-Data-Alignment
#
# Reference (CLIP):
#   Radford et al., "Learning Transferable Visual Models from Natural
#   Language Supervision", ICML 2021.
'''

import logging

import torch
import torch.nn as nn

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC
from transformers import AutoProcessor, CLIPModel
from peft import get_peft_model, LoraConfig

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='clip_lora_frequency')
class CLIPLoraFrequency(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        # CLIP vit-l pooler_output (1024) + HF energy ratio scalar (1)
        self.head = nn.Linear(1024 + 1, 2)
        self.loss_func = self.build_loss(config)

        # Low-frequency radius as a fraction of the shorter spatial dimension.
        # Energy outside this circle is considered high-frequency.
        self.lf_radius_frac = config.get('lf_radius_frac', 0.25)

    def build_backbone(self, config):
        _, backbone = get_clip_visual(model_name="openai/clip-vit-large-patch14")
        lora_rank = config.get('lora_rank', 16)
        lora_alpha = config.get('lora_alpha', 16)
        backbone = apply_lora_to_clip(backbone, r=lora_rank, lora_alpha=lora_alpha)
        return backbone

    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        return loss_class()

    def features(self, data_dict: dict) -> torch.Tensor:
        img = data_dict['image']                              # (B, C, H, W)

        # CLIP CLS token
        clip_feat = self.backbone(img)['pooler_output']       # (B, 1024)

        # DCT-proxy high-frequency energy ratio
        hf_ratio = compute_hf_energy_ratio(img, self.lf_radius_frac)  # (B, 1)

        return torch.cat([clip_feat, hf_ratio], dim=1)        # (B, 1025)

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_clip_visual(model_name="openai/clip-vit-large-patch14"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return processor, model.vision_model


def apply_lora_to_clip(backbone, r=16, lora_alpha=16):
    """Wrap the CLIP vision backbone with PEFT LoRA on Q and V projections."""
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(backbone, lora_config)


def compute_hf_energy_ratio(img: torch.Tensor, lf_radius_frac: float = 0.25) -> torch.Tensor:
    """Compute the high-frequency energy ratio for a batch of images.

    Converts each image to grayscale (luma), computes the 2-D FFT magnitude
    spectrum (shifted so DC is at the centre), then returns the fraction of
    total spectral energy that lies *outside* a central low-frequency disc.

    Args:
        img: (B, C, H, W) float tensor, arbitrary value range.
        lf_radius_frac: radius of the low-frequency disc as a fraction of
            min(H, W).  Default 0.25 keeps the inner quarter as "low-freq".

    Returns:
        Tensor of shape (B, 1) with values in [0, 1].
    """
    # --- luma conversion (BT.601 coefficients) ---
    # Works for both [0,1] and [-1,1] normalised inputs because the ratio
    # is scale-invariant (energy cancels in numerator and denominator).
    gray = (0.299 * img[:, 0] +
            0.587 * img[:, 1] +
            0.114 * img[:, 2])          # (B, H, W)

    B, H, W = gray.shape

    # --- 2-D FFT magnitude (proxy for DCT on real inputs) ---
    fft = torch.fft.fft2(gray)          # complex (B, H, W)
    fft = torch.fft.fftshift(fft)       # shift DC to centre
    power = fft.real ** 2 + fft.imag ** 2  # (B, H, W), equivalent to |FFT|^2

    # --- circular low-frequency mask (shared across batch) ---
    cy, cx = H / 2.0, W / 2.0
    radius = lf_radius_frac * min(H, W)

    ys = torch.arange(H, device=img.device, dtype=torch.float32)
    xs = torch.arange(W, device=img.device, dtype=torch.float32)
    dist = torch.sqrt((ys.view(H, 1) - cy) ** 2 +
                      (xs.view(1, W) - cx) ** 2)   # (H, W)

    lf_mask = (dist <= radius).float()              # 1 inside disc, 0 outside
    hf_mask = 1.0 - lf_mask                         # (H, W)

    # --- per-image ratio ---
    total_energy = power.sum(dim=(1, 2))                          # (B,)
    hf_energy = (power * hf_mask.unsqueeze(0)).sum(dim=(1, 2))    # (B,)
    ratio = hf_energy / (total_energy + 1e-8)                     # (B,)

    return ratio.unsqueeze(1)   # (B, 1)