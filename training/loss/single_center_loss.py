import torch
import torch.nn as nn
from loss import LOSSFUNC

@LOSSFUNC.register_module(module_name='scl')
class SingleCenterLoss(nn.Module):
    """
    From Li et al. (2021) FDFL paper.
    """
    def __init__(self, feat_dim, margin=0.3):
        super().__init__()
        self.center = nn.Parameter(torch.randn(feat_dim))
        self.margin = margin
        self.feat_dim = feat_dim
    
    def forward(self, features, labels):
        # labels: 0 = real, 1 = fake
        real_mask = (labels == 0)
        fake_mask = (labels == 1)
        
        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        
        # Mean distance from real faces to center
        real_feats = features[real_mask]
        d_real = torch.norm(real_feats - self.center, dim=1)  # [N_real]
        M_nat = d_real.mean()
        
        # Mean distance from fake faces to center
        fake_feats = features[fake_mask]
        d_fake = torch.norm(fake_feats - self.center, dim=1)  # [N_fake]
        M_man = d_fake.mean()
        
        # SCL loss (Eq. 1 from FDFL paper)
        margin_scaled = self.margin * (self.feat_dim ** 0.5)
        hinge = torch.clamp(M_nat - M_man + margin_scaled, min=0.0)
        
        return M_nat + hinge