"""Pixel branch v5: CLIP ViT with a config-selectable adaptation mechanism.

adapter_type:
  'lora'   -- PEFT LoRA on q/v projections (the original PixelBranchPatch path)
  'svd'    -- Effort-style SVD-residual adaptation (Yan et al., arXiv 2411.15633):
              freeze the top-r singular components of every attention Linear,
              train only the bottom (n-r) residual factors (rank-1 at r = n-1).
              Targets the LoRA-induced spectral/rank collapse we logged
              (real_feat_var 0.37 -> 0.02, eff_rank -> 1.0).
  'frozen' -- no adaptation (linear-probe style).

SVD classes adapted from detectors/effort_detector.py:232-361 (copied to avoid
a networks -> detectors circular import). One fix relative to that file:
compute_orthogonal_loss there penalizes U U^T - I_n (n x n; unsatisfiable for a
rank-k residual) -- here we use the correct small-identity form U^T U = I_k.
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


class SVDResidualLinear(nn.Module):
    """weight = frozen top-r SVD reconstruction + trainable bottom-(n-r) residual."""

    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features),
                                        requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def compute_current_weight(self):
        if self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        return self.weight_main

    def forward(self, x):
        if getattr(self, 'S_residual', None) is not None:
            weight = self.weight_main + \
                self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            weight = self.weight_main
        return F.linear(x, weight, self.bias)

    def compute_orthogonal_loss(self):
        """||U^T U - I_k||_F + ||Vh Vh^T - I_k||_F (small-identity form)."""
        if getattr(self, 'S_residual', None) is None:
            return self.weight_main.new_zeros(())
        UtU = self.U_residual.t() @ self.U_residual            # [k, k]
        VVt = self.V_residual @ self.V_residual.t()            # [k, k]
        I = torch.eye(UtU.size(0), device=UtU.device, dtype=UtU.dtype)
        return 0.5 * torch.norm(UtU - I, p='fro') + 0.5 * torch.norm(VVt - I, p='fro')


def _replace_with_svd_residual(module, r):
    if not isinstance(module, nn.Linear):
        return module
    bias = module.bias is not None
    new_module = SVDResidualLinear(module.in_features, module.out_features, r,
                                   bias=bias, init_weight=module.weight.data.clone())
    if bias:
        new_module.bias.data.copy_(module.bias.data)

    # SVD on GPU when available: 96 x (1024x1024) decompositions are ~10ms each
    # on CUDA vs 10s+ on a CPU contended by training dataloader workers.
    svd_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    U, S, Vh = torch.linalg.svd(module.weight.data.to(svd_dev), full_matrices=False)
    U, S, Vh = U.cpu(), S.cpu(), Vh.cpu()
    r = min(r, len(S))
    new_module.weight_main.data.copy_(U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :])
    if len(S) - r > 0:
        new_module.S_residual = nn.Parameter(S[r:].clone())
        new_module.U_residual = nn.Parameter(U[:, r:].clone())
        new_module.V_residual = nn.Parameter(Vh[r:, :].clone())
    else:
        new_module.S_residual = None
        new_module.U_residual = None
        new_module.V_residual = None
    return new_module


def apply_svd_residual_to_self_attn(model, r):
    """Replace every nn.Linear inside self_attn modules; freeze all but residuals."""
    for name, module in model.named_children():
        if 'self_attn' in name:
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    parent = module
                    parts = sub_name.split('.')
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], _replace_with_svd_residual(sub_module, r))
        else:
            apply_svd_residual_to_self_attn(module, r)
    for pname, param in model.named_parameters():
        param.requires_grad = any(t in pname for t in
                                  ('S_residual', 'U_residual', 'V_residual'))
    return model


class PixelBranchPatchV5(nn.Module):

    VARIANTS = {
        "openai/clip-vit-base-patch16": 768,
        "openai/clip-vit-large-patch14": 1024,
    }

    def __init__(self, config):
        super().__init__()
        model_name = config.get("clip_model_name", "openai/clip-vit-large-patch14")
        self.adapter_type = config.get('adapter_type', 'lora')

        clip_model = CLIPVisionModel.from_pretrained(model_name)
        vision = clip_model.vision_model
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.out_dim = self.VARIANTS[model_name]

        if self.adapter_type == 'svd':
            r = config.get('svd_rank_keep', self.out_dim - 1)   # residual rank = dim - r
            self.encoder = apply_svd_residual_to_self_attn(vision, r=r)
        elif self.adapter_type == 'lora':
            lora_config = LoraConfig(
                r=config.get('lora_rank', 8),
                lora_alpha=config.get('lora_alpha', 16),
                lora_dropout=config.get('lora_dropout', 0.1),
                target_modules=config.get('lora_targets', ['q_proj', 'v_proj']),
                bias="none",
            )
            self.encoder = get_peft_model(vision, lora_config)
            for name, param in self.encoder.named_parameters():
                if "lora_" not in name:
                    param.requires_grad = False
        elif self.adapter_type == 'frozen':
            self.encoder = vision
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"unknown adapter_type: {self.adapter_type}")

        self.norm = nn.LayerNorm(self.out_dim)         # pooled CLS -> CE head
        self.token_norm = nn.LayerNorm(self.out_dim)   # patch tokens -> geometry losses

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"PixelBranchPatchV5[{self.adapter_type}]: "
                    f"{trainable:,} trainable / {total:,} total params")

    def svd_orthogonal_loss(self):
        """Sum of corrected orthonormality penalties over all SVD-residual layers."""
        losses = [m.compute_orthogonal_loss() for m in self.encoder.modules()
                  if isinstance(m, SVDResidualLinear)]
        if not losses:
            return next(self.parameters()).new_zeros(())
        return torch.stack(losses).mean()

    def train(self, mode=True):
        super().train(mode)
        if self.adapter_type == 'frozen':
            self.encoder.eval()
        return self

    def forward(self, x, return_tokens=False):
        outputs = self.encoder(pixel_values=x)
        pooled = self.norm(outputs.pooler_output)
        if return_tokens:
            tokens = self.token_norm(outputs.last_hidden_state[:, 1:, :])  # drop CLS
            return pooled, tokens
        return pooled
