import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train


# ==========================================
# 1. LOSS FUNCTIONS
# ==========================================
def alignment_loss(embeddings, labels, alpha=2):
    n_samples = embeddings.size(0)
    if n_samples < 2:
        return torch.tensor(0.0, device=embeddings.device)

    # Pairwise label comparison matrix (N x N), upper triangle only (exclude self-pairs)
    labels_equal_mask = (labels[:, None] == labels[None, :]).triu(diagonal=1)

    positive_indices = torch.nonzero(labels_equal_mask, as_tuple=False)
    if positive_indices.numel() == 0:
        return torch.tensor(0.0, device=embeddings.device)

    x = embeddings[positive_indices[:, 0]]
    y = embeddings[positive_indices[:, 1]]

    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity_loss(x, t=2, clip_value=1e-6):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().clamp(min=clip_value).log()


# ==========================================
# 2. GenD DETECTOR
# ==========================================
@DETECTOR.register_module(module_name='gend')
class GenDDetector(AbstractDetector):
    def __init__(self, config=None):
        super().__init__()
        self.config = config

        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(1024, 2)
        self.loss_func = self.build_loss(config)

        self.lambda_align = config.get('lambda_align', 0.1) if config else 0.1
        self.lambda_unif = config.get('lambda_unif', 0.5) if config else 0.5

    def build_backbone(self, config):
        print("Loading CLIP ViT-L/14 for GenD...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        backbone = clip_model.vision_model

        # Freeze all parameters
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze LayerNorm layers only (matches GenD.py unfreeze_layers=['layer_norm'] strategy)
        trainable_params = 0
        for name, param in backbone.named_parameters():
            if 'layer_norm' in name or 'layernorm' in name:
                param.requires_grad = True
                trainable_params += param.numel()

        print(f"GenD Backbone: trainable params (LayerNorm only) = {trainable_params}")
        return backbone

    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        return loss_class()

    def features(self, data_dict: dict) -> torch.Tensor:
        outputs = self.backbone(data_dict['image'])
        return outputs.pooler_output

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(self, data_dict: dict, inference=False) -> dict:
        raw_features = self.features(data_dict)
        norm_features = F.normalize(raw_features, p=2, dim=1)
        pred = self.classifier(norm_features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {'cls': pred, 'prob': prob, 'feat': raw_features, 'feat_norm': norm_features}

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        feat_norm = pred_dict['feat_norm']

        loss_cls = self.loss_func(pred, label)

        loss_align = torch.tensor(0.0, device=pred.device)
        loss_unif = torch.tensor(0.0, device=pred.device)

        if self.training:
            loss_align = alignment_loss(feat_norm, label)
            loss_unif = uniformity_loss(feat_norm)

        overall_loss = loss_cls + (self.lambda_align * loss_align) + (self.lambda_unif * loss_unif)

        return {
            'overall': overall_loss,
            'ce_loss': loss_cls,
            'align_loss': loss_align,
            'unif_loss': loss_unif,
        }

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
