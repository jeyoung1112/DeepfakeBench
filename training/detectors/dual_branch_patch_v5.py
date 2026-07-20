"""dual_branch_patch_v5: DualBranchPatch with two new ablation axes.

1. adapter_type: 'lora' | 'svd' | 'frozen' (pixel branch adaptation mechanism).
   'svd' = Effort-style SVD-residual (freeze top-r singular subspace, train the
   bottom residual) -- targets the logged LoRA rank-collapse of pooled real
   features. Adds an optional orthonormality penalty on the residual factors
   (svd_ortho_weight).
2. mode: 'dual' | 'pixel_only' now fully works in forward() (the parent class
   assumed both branches). pixel_only drops the FFT/ResNet18 branch entirely:
   patch tokens = CLIP tokens [B,256,1024], fused = pooled CLS [B,1024].

Everything else (fixed Fishr with smoothed-CE residual + relative variant,
patch-level real-only center loss, patch/pooled var-cov guards, diagnostics)
is inherited unchanged from DualBranchPatch, so A/B runs against v4-generation
models differ only by the axes above.
"""

import logging
import torch
import torch.nn.functional as F

from networks.pixel_branch_patch_v5 import PixelBranchPatchV5
from networks.pixel_branch_dino_v5 import PixelBranchDINOv5
from networks.frequency_branch_patch import FrequencyBranchPatch

from detectors import DETECTOR
from .dual_branch_patch import DualBranchPatch

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='dual_branch_patch_v5')
class DualBranchPatchV5(DualBranchPatch):

    def __init__(self, config):
        super().__init__(config)
        self.svd_ortho_weight = config.get('svd_ortho_weight', 0.0)
        if self.mode == 'freq_only':
            raise NotImplementedError("v5 supports mode: 'dual' or 'pixel_only'")

    def build_backbone(self, config):
        pixel_branch, freq_branch = None, None
        mode = config.get("mode", "dual")
        if mode in ('dual', 'pixel_only'):
            pixel_type = config.get("pixel_backbone", "clip")
            if pixel_type == "clip":
                pixel_branch = PixelBranchPatchV5(config)
            elif pixel_type == "dino":
                pixel_branch = PixelBranchDINOv5(config)
            else:
                raise ValueError(f"unknown pixel_backbone: {pixel_type} "
                                "(must be 'clip' or 'dino')")
        if mode == 'dual':
            freq_branch = FrequencyBranchPatch(config)
        return pixel_branch, freq_branch

    def forward(self, data_dict, inference=False):
        img = data_dict['image']

        pooled_px, tok_px = self.pixel_branch(img, return_tokens=True)  # [B,Dp], [B,N,Dp]

        if self.freq_branch is not None:
            pooled_fq, fmap = self.freq_branch(img, return_map=True)    # [B,Df], [B,Cf,h,w]
            fmap = F.interpolate(fmap, size=(self.grid, self.grid),
                                 mode='bilinear', align_corners=False)
            tok_fq = self.freq_tok_norm(fmap.flatten(2).transpose(1, 2))
            patch_tokens = torch.cat([tok_px, tok_fq], dim=-1)          # [B,N,Dp+Cf]
            fused = torch.cat([pooled_px, pooled_fq], dim=-1)           # [B,Dp+Df]
        else:                                                           # pixel_only
            patch_tokens = tok_px                                       # [B,N,Dp]
            fused = pooled_px                                           # [B,Dp]

        cls_feat = self.head[:-1](fused)              # Linear->ReLU->Dropout (single mask)
        pred = self.head[-1](cls_feat)                # final Linear on the SAME activation
        prob = torch.softmax(pred, dim=1)[:, 1]

        return {
            'cls': pred,
            'prob': prob,
            'feat': fused,
            'cls_feat': cls_feat,
            'patch_tokens': patch_tokens,
        }

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        losses = super().get_losses(data_dict, pred_dict)
        # orthonormality of the SVD residual factors (Effort's constraint; keeps
        # the trainable residual a valid rotation-free subspace of the spectrum)
        if self.svd_ortho_weight > 0 and \
                getattr(self.pixel_branch, 'adapter_type', None) == 'svd':
            ortho = self.pixel_branch.svd_orthogonal_loss()
            losses['svd_ortho'] = ortho
            losses['overall'] = losses['overall'] + self.svd_ortho_weight * ortho
        return losses
