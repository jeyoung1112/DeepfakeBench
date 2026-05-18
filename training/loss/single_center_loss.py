import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import LOSSFUNC

# @LOSSFUNC.register_module(module_name='scl')
# class SingleCenterLoss(nn.Module):
#     def __init__(self, feat_dim, ema_tau=0.99):
#         super().__init__()
#         self.register_buffer('center', torch.zeros(feat_dim))
#         self.register_buffer('initialized', torch.tensor(False))
#         self.feat_dim = feat_dim
#         self.ema_tau = ema_tau

#     def forward(self, features, labels):
#         real_mask = (labels == 0)
#         if real_mask.sum() == 0:
#             return torch.tensor(0.0, device=features.device, requires_grad=True)

#         # EMA update, no gradient
#         with torch.no_grad():
#             batch_mean = features[real_mask].mean(dim=0)
#             if not self.initialized:
#                 self.center.copy_(batch_mean)
#                 self.initialized.fill_(True)
#             else:
#                 self.center.mul_(self.ema_tau).add_(batch_mean, alpha=1 - self.ema_tau)

#         real_feats = features[real_mask]
#         distances = torch.norm(real_feats - self.center, dim=1) / (self.feat_dim ** 0.5)
#         return distances.mean()

@LOSSFUNC.register_module(module_name='scl')
class SingleCenterLoss(nn.Module):
    def __init__(self, feat_dim, ema_tau=0.99):
        super().__init__()
        self.register_buffer('center', torch.zeros(feat_dim))
        self.register_buffer('initialized', torch.tensor(False))
        self.ema_tau = ema_tau

    def forward(self, features, labels):
        real_mask = (labels == 0)
        if real_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # Normalize to unit sphere
        feats = F.normalize(features, dim=1)
        
        # EMA update on normalized space
        with torch.no_grad():
            batch_mean = F.normalize(
                feats[real_mask].mean(dim=0, keepdim=True), dim=1
            ).squeeze(0)
            if not self.initialized:
                self.center.copy_(batch_mean)
                self.initialized.fill_(True)
            else:
                self.center.mul_(self.ema_tau).add_(batch_mean, alpha=1 - self.ema_tau)
                # Re-normalize center back to sphere
                self.center.copy_(
                    F.normalize(self.center.unsqueeze(0), dim=1).squeeze(0)
                )
        
        # Cosine distance, bounded in [0, 2]
        d = 1.0 - feats[real_mask] @ self.center
        return d.mean()

# @LOSSFUNC.register_module(module_name='scl')
# class SingleCenterLoss(nn.Module):
#     def __init__(self, feat_dim, ema_tau=0.99, margin=0.3):
#         super().__init__()
#         self.register_buffer('center', torch.zeros(feat_dim))
#         self.register_buffer('initialized', torch.tensor(False))
#         self.ema_tau = ema_tau
#         self.margin = margin

#     def forward(self, features, labels):
#         """
#         Args:
#             features: [B, D] feature tensor
#             labels: [B] long tensor with 0=real, 1=fake
        
#         Returns:
#             scalar loss tensor connected to the computation graph
#         """
#         real_mask = (labels == 0)
#         fake_mask = (labels == 1)
        
#         # Edge case: no reals in batch — return zero loss with grad flow
#         if real_mask.sum() == 0:
#             return features.sum() * 0.0
        
#         # Normalize features to unit sphere for cosine geometry
#         feats = F.normalize(features, dim=1)
        
#         # EMA update of center using only real features (asymmetric)
#         with torch.no_grad():
#             batch_mean = F.normalize(
#                 feats[real_mask].mean(dim=0, keepdim=True), dim=1
#             ).squeeze(0)
            
#             if not self.initialized:
#                 self.center.copy_(batch_mean)
#                 self.initialized.fill_(True)
#             else:
#                 self.center.mul_(self.ema_tau).add_(
#                     batch_mean, alpha=1 - self.ema_tau
#                 )
#                 # Re-normalize center back to unit sphere after EMA update
#                 self.center.copy_(
#                     F.normalize(self.center.unsqueeze(0), dim=1).squeeze(0)
#                 )
        
#         # Compute cosine distances (in [0, 2])
#         d_real = 1.0 - feats[real_mask] @ self.center  # [N_real]
#         M_nat = d_real.mean()
        
#         # If no fakes in batch, return only the real-attraction term
#         if fake_mask.sum() == 0:
#             return M_nat
        
#         d_fake = 1.0 - feats[fake_mask] @ self.center  # [N_fake]
#         M_man = d_fake.mean()
        
#         # Relative hinge: penalize when M_man - M_nat < margin
#         # When M_man > M_nat + margin, hinge is 0 (sufficient separation)
#         # When M_man < M_nat + margin, hinge pushes for larger separation
#         hinge = torch.clamp(M_nat - M_man + self.margin, min=0.0)
        
#         return M_nat + hinge

# @LOSSFUNC.register_module(module_name='scl')
# class SingleCenterLoss(nn.Module):
#     """
#     From Li et al. (2021) Frequency-aware Discriminative Feature Learning Supervised by Single-Center Loss for Face Forgery Detection.
#     """
#     def __init__(self, feat_dim, margin=0.3):
#         super().__init__()
#         self.center = nn.Parameter(torch.randn(feat_dim))
#         self.margin = margin
#         self.feat_dim = feat_dim
    
#     def forward(self, features, labels):
#         # labels: 0 = real, 1 = fake
#         real_mask = (labels == 0)
#         fake_mask = (labels == 1)
        
#         if real_mask.sum() == 0 or fake_mask.sum() == 0:
#             return torch.tensor(0.0, device=features.device)
        
#         # Mean distance from real faces to center
#         real_feats = features[real_mask]
#         d_real = torch.norm(real_feats - self.center, dim=1)  # [N_real]
#         M_nat = d_real.mean()
        
#         # Mean distance from fake faces to center
#         fake_feats = features[fake_mask]
#         d_fake = torch.norm(fake_feats - self.center, dim=1)  # [N_fake]
#         M_man = d_fake.mean()
        
#         # SCL loss 
#         margin_scaled = self.margin * (self.feat_dim ** 0.5)
#         hinge = torch.clamp(M_nat - M_man + margin_scaled, min=0.0)
        
#         return M_nat + hinge

@LOSSFUNC.register_module(module_name='scl_original')
class OriginalSingleCenterLoss(nn.Module):
    """
    L_sc = M_nat + max(M_nat - M_man + m*sqrt(D), 0)
    """

    # def __init__(self, feat_dim: int, margin: float = 0.3):
    #     super().__init__()
    #     self.feat_dim = feat_dim
    #     self.margin = margin
    #     self.center = nn.Parameter(torch.randn(feat_dim))

    # def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    #     real_mask = (labels == 0)
    #     fake_mask = (labels == 1)

    #     if real_mask.sum() == 0:
    #         return features.sum() * 0.0

    #     d_real = torch.norm(features[real_mask] - self.center, dim=1).clamp(min=1e-8)
    #     M_nat = d_real.mean()

    #     if fake_mask.sum() == 0:
    #         return M_nat

    #     d_fake = torch.norm(features[fake_mask] - self.center, dim=1).clamp(min=1e-8)
    #     M_man = d_fake.mean()

    #     margin_scaled = self.margin * (self.feat_dim ** 0.5)
    #     hinge = torch.clamp(M_nat - M_man + margin_scaled, min=0.0)
    #     return M_nat + hinge
    

    def __init__(self, margin = 0.3, feat_dim = 1000, use_gpu=True):
        super(OriginalSingleCenterLoss, self).__init__()
        self.m = margin
        self.D = feat_dim
        self.margin = self.m * torch.sqrt(torch.tensor(self.D).float())
        self.l2loss = nn.MSELoss(reduction = 'none')
        self.C = nn.Parameter(torch.randn(self.D))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        eud_mat = torch.sqrt(self.l2loss(x, self.C.expand(batch_size, self.C.size(0))).sum(dim=1, keepdim=True))

        labels = labels.unsqueeze(1)

        fake_count = labels.sum()
        real_count = batch_size - fake_count

        dist_real = (eud_mat * (1 - labels.float())).clamp(min=1e-12, max=1e+12).sum()
        dist_fake = (eud_mat * labels.float()).clamp(min=1e-12, max=1e+12).sum()

        if real_count != 0:
            dist_real /= real_count

        if fake_count != 0:
            dist_fake /= fake_count

        max_margin = dist_real - dist_fake + self.margin

        if max_margin < 0:
            max_margin = 0

        loss = dist_real + max_margin

        return loss