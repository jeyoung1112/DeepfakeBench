"""
GenD / LNCLIP-DF detector for DeepfakeBench.  Registers as module_name='gend'.

Faithful to "Deepfake Detection that Generalizes Across Benchmarks"
(Yermakov, Cech, Matas, Fritz; arXiv:2508.06248).

Method:
  - Backbone: frozen CLIP ViT-L/14 image encoder; ONLY LayerNorm affine params are
    fine-tuned (~0.03% of params), plus the linear head.
  - The CLS token is L2-normalized onto the unit hypersphere before the linear head.
  - Loss (Eq.2):  L = CE + alpha * L_align + beta * L_uniform   (alpha=0.1, beta=0.5)
    on the L2-normalized embeddings (Wang & Isola alignment/uniformity).
  - Slerp latent augmentation: spherical interpolation between SAME-class normalized
    features within the batch, expanding it (paper: 128 -> 1024) to fill the hypersphere.
    Training-only; the interpolated unit vectors feed CE + align + uniform.

Implementation notes / decisions where the paper is silent (see the recon spec):
  - No logit temperature is added (the paper mentions none). Instead the config sets
    weight_decay=0, so the head weights are free to absorb the softmax scale -- this is
    why the paper uses Adam WITHOUT weight decay.
  - Slerp is applied to the L2-normalized embeddings (slerp of two unit vectors is unit
    norm); outputs are re-normalized for numerical safety, with a lerp fallback for
    near-parallel pairs (sin(theta) -> 0).
  - 'cls'/'prob'/'feat' in pred_dict always describe the ORIGINAL (un-augmented) batch so
    get_train_metrics stays aligned with data_dict['label']; the Slerp-expanded tensors
    live under 'cls_exp'/'feat_norm_exp'/'label_exp' and feed only the losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train


# ==========================================
# 1. LOSS FUNCTIONS (Wang & Isola, on the unit hypersphere)
# ==========================================
def alignment_loss(embeddings, labels, alpha=2, max_pairs=8192):
    """L_align = E_{(x,y)~P+}[ ||z_x - z_y||_2^alpha ] over same-class pairs.
    The positive-pair set is O(N^2); when it exceeds max_pairs we subsample a random subset
    (an unbiased estimate of the expectation) so memory stays bounded on the Slerp-expanded batch."""
    n_samples = embeddings.size(0)
    if n_samples < 2:
        return torch.zeros((), device=embeddings.device)

    # same-label pairs, upper triangle only (exclude self-pairs / double counting)
    labels_equal_mask = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    positive_indices = torch.nonzero(labels_equal_mask, as_tuple=False)
    if positive_indices.numel() == 0:
        return torch.zeros((), device=embeddings.device)

    if positive_indices.size(0) > max_pairs:
        sel = torch.randint(positive_indices.size(0), (max_pairs,), device=embeddings.device)
        positive_indices = positive_indices[sel]

    x = embeddings[positive_indices[:, 0]]
    y = embeddings[positive_indices[:, 1]]
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity_loss(x, t=2, clip_value=1e-6, max_points=2048):
    """L_uniform = log E_{(x,y)~P}[ e^{-t ||z_x - z_y||_2^2} ] over all pairs.
    pdist is O(N^2); subsample rows past max_points to keep memory bounded on the expanded batch."""
    if x.size(0) < 2:
        return torch.zeros((), device=x.device)
    if x.size(0) > max_points:
        sel = torch.randperm(x.size(0), device=x.device)[:max_points]
        x = x[sel]
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().clamp(min=clip_value).log()


# ==========================================
# 2. SLERP LATENT AUGMENTATION (training only)
# ==========================================
def _same_class_partners(labels):
    """For each index i return a random index j with labels[j]==labels[i]
    (j may equal i when a class is a singleton in the batch)."""
    n = labels.shape[0]
    partners = torch.arange(n, device=labels.device)
    for c in torch.unique(labels):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        pick = torch.randint(idx.numel(), (idx.numel(),), device=labels.device)
        partners[idx] = idx[pick]
    return partners


def slerp(z_i, z_j, t, eps=1e-6):
    """Spherical linear interpolation of unit vectors.
    z_i, z_j: [K, D] on the unit sphere; t: [K] in [0,1]. Returns [K, D] unit vectors.
      slerp(z_i,z_j;t) = sin((1-t)th)/sin(th) z_i + sin(t th)/sin(th) z_j,  th=arccos(z_i.z_j)
    Falls back to (renormalized) lerp when sin(theta)~0 (near-parallel / antipodal)."""
    dot_raw = (z_i * z_j).sum(-1)                            # [K], pre-clamp
    dot = dot_raw.clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.arccos(dot)
    sin_theta = torch.sin(theta)
    t = t.to(z_i.dtype)
    w_i = (torch.sin((1.0 - t) * theta) / sin_theta).unsqueeze(-1)
    w_j = (torch.sin(t * theta) / sin_theta).unsqueeze(-1)
    out = w_i * z_i + w_j * z_j
    # Degenerate geodesic: sin(theta) -> 0 at BOTH near-parallel (dot->1) and near-antipodal
    # (dot->-1) ends, where slerp is ill-defined/unstable (antipodal midpoint -> zero vector).
    # Fall back to an endpoint, which is unit-norm and safe. Keyed on the pre-clamp dot so it
    # actually fires (the clamp otherwise floors sin_theta ~1e-3 and the guard never triggers).
    degenerate = (dot_raw.abs() > 1.0 - eps).unsqueeze(-1)
    out = torch.where(degenerate, z_i, out)
    return F.normalize(out, p=2, dim=1)


def slerp_expand(z, labels, expand):
    """Expand a batch of unit features by (expand-1) rounds of same-class slerp.
    Returns z_exp [expand*N, D] (originals first) and labels_exp [expand*N]."""
    if expand is None or expand <= 1 or z.size(0) < 2:
        return z, labels
    zs, ls = [z], [labels]
    for _ in range(int(expand) - 1):
        j = _same_class_partners(labels)
        t = torch.rand(z.size(0), device=z.device)
        zs.append(slerp(z, z[j], t))
        ls.append(labels)
    return torch.cat(zs, 0), torch.cat(ls, 0)


# ==========================================
# 3. GenD DETECTOR
# ==========================================
@DETECTOR.register_module(module_name='gend')
class GenDDetector(AbstractDetector):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}

        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(1024, 2)          # CLIP ViT-L/14 hidden size = 1024
        self.loss_func = self.build_loss(config)

        self.lambda_align = self.config.get('lambda_align', 0.1)
        self.lambda_unif = self.config.get('lambda_unif', 0.5)
        # Slerp latent augmentation (paper expands 128 -> 1024, i.e. 8x)
        self.use_slerp = self.config.get('use_slerp', True)
        self.slerp_expand = self.config.get('slerp_expand', 8)

    def build_backbone(self, config):
        print("Loading CLIP ViT-L/14 for GenD...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        backbone = clip_model.vision_model

        # Freeze everything, then unfreeze the affine params of EVERY LayerNorm (LN-tuning).
        # isinstance() catches all of them robustly, including HF CLIP's misspelled input
        # `pre_layrnorm` that a name-substring match ('layer_norm'/'layernorm') would miss.
        for param in backbone.parameters():
            param.requires_grad = False
        trainable_params = 0
        for module in backbone.modules():
            if isinstance(module, nn.LayerNorm):
                for param in module.parameters(recurse=False):
                    param.requires_grad = True
                    trainable_params += param.numel()
        print(f"GenD Backbone: trainable params (LayerNorm only) = {trainable_params}")
        return backbone

    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        return loss_class()

    def features(self, data_dict: dict) -> torch.Tensor:
        # pooler_output is the post-LN CLS token for the HF CLIP vision model
        return self.backbone(data_dict['image']).pooler_output

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(self, data_dict: dict, inference=False) -> dict:
        raw_features = self.features(data_dict)                 # [N, 1024]
        norm_features = F.normalize(raw_features, p=2, dim=1)   # [N, 1024] on hypersphere
        pred = self.classifier(norm_features)                  # [N, 2]
        prob = torch.softmax(pred, dim=1)[:, 1]

        out = {'cls': pred, 'prob': prob, 'feat': raw_features, 'feat_norm': norm_features}

        # Slerp latent augmentation feeds the losses only, and only during training.
        if self.training and not inference and self.use_slerp and 'label' in data_dict:
            labels = data_dict['label']
            z_exp, label_exp = slerp_expand(norm_features, labels, self.slerp_expand)
            out['cls_exp'] = self.classifier(z_exp)
            out['feat_norm_exp'] = z_exp
            out['label_exp'] = label_exp
        return out

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        # Train on the Slerp-expanded batch when available; otherwise the raw batch.
        if self.training and 'cls_exp' in pred_dict:
            logits = pred_dict['cls_exp']
            feat = pred_dict['feat_norm_exp']
            label = pred_dict['label_exp']
        else:
            logits = pred_dict['cls']
            feat = pred_dict['feat_norm']
            label = data_dict['label']

        loss_cls = self.loss_func(logits, label)

        if self.training:
            loss_align = alignment_loss(feat, label)
            loss_unif = uniformity_loss(feat)
        else:
            loss_align = torch.zeros((), device=logits.device)
            loss_unif = torch.zeros((), device=logits.device)

        overall_loss = loss_cls + self.lambda_align * loss_align + self.lambda_unif * loss_unif
        return {
            'overall': overall_loss,
            'ce_loss': loss_cls,
            'align_loss': loss_align,
            'unif_loss': loss_unif,
        }

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        # metrics on the ORIGINAL batch (pred_dict['cls'] and data_dict['label'] both size N)
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
