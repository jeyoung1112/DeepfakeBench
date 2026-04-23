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
    
    # def forward(self, features, labels):
    #     # Normalize features to unit sphere
    #     feats = F.normalize(features, dim=1)
        
    #     real_mask = (labels == 0)
    #     fake_mask = (labels == 1)
    #     if real_mask.sum() == 0 or fake_mask.sum() == 0:
    #         return torch.tensor(0.0, device=feats.device, requires_grad=True)
        
    #     # EMA update on normalized features
    #     with torch.no_grad():
    #         batch_real_mean = F.normalize(
    #             feats[real_mask].mean(dim=0, keepdim=True), dim=1
    #         ).squeeze(0)
    #         if not self.initialized:
    #             self.center.copy_(batch_real_mean)
    #             self.initialized.fill_(True)
    #         else:
    #             self.center.mul_(self.ema_tau).add_(
    #                 batch_real_mean, alpha=1 - self.ema_tau
    #             )
    #             # Re-normalize after EMA so center stays on sphere
    #             self.center.copy_(
    #                 F.normalize(self.center.unsqueeze(0), dim=1).squeeze(0)
    #             )
        
    #     # Cosine distance on unit sphere, bounded in [0, 2]
    #     d = 1.0 - feats @ self.center
        
    #     M_nat = d[real_mask].mean()
    #     M_man = d[fake_mask].mean()
    #     hinge = torch.clamp(M_nat - M_man + self.margin, min=0.0)
    #     return M_nat + hinge