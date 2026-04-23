import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import LOSSFUNC

@LOSSFUNC.register_module(module_name='vicreg')
class VICRegLoss(nn.Module):
    def __init__(self, 
                lambda_inv=25.0,
                mu_var=25.0,
                nu_cov=1.0,
                gamma=1.0,
                eps=1e-4):

        super().__init__()
        self.lambda_inv = lambda_inv
        self.mu_var = mu_var
        self.nu_cov = nu_cov
        self.gamma = gamma
        self.eps = eps

    def invariance(self, z1, z2):
        return F.mse_loss(z1, z2)
    
    def variance(self, z):
        std = torch.sqrt(z.var(dim=0) + self.eps)
        return torch.mean(F.relu(self.gamma - std))

    def covariance(self, z):
        n, d = z.shape
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (n - 1)
        cov.fill_diagonal_(0)
        return (cov.pow(2)).sum() / d

    def forward(self, z1, z2):

        inv_loss = self.invariance(z1, z2)

        var_loss = self.variance(z1) + self.variance(z2)

        cov_loss = self.covariance(z1) + self.covariance(z2)

        total_loss = self.lambda_inv * inv_loss + self.mu_var * var_loss + self.nu_cov * cov_loss

        loss_dict = {
            'vicreg_inv': inv_loss.item(),
            'vicreg_var': var_loss.item(),
            'vicreg_cov': cov_loss.item(),
            'vicreg_total': total_loss.item(),
            'std_z1_min': z1.std(dim=0).min().item(),
            'std_z2_min': z2.std(dim=0).min().item(),
        }
 
        return total_loss, loss_dict


@LOSSFUNC.register_module(module_name='varcov')
class VarCovRegLoss(nn.Module):
    """Variance + covariance regularization"""

    def __init__(self, gamma=1.0, eps=1e-4):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def variance(self, z):
        std = torch.sqrt(z.var(dim=0) + self.eps)
        return torch.mean(F.relu(self.gamma - std))

    def covariance(self, z):
        n, d = z.shape
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (n - 1)
        cov.fill_diagonal_(0)
        return cov.pow(2).sum() / d

    def forward(self, z):
        return self.variance(z), self.covariance(z)
