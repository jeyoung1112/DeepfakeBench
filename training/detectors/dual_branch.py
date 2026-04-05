import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC


logger = logging.getLogger(__name__)


class Expander(nn.Module):

    def __init__(self, in_dim, embed_dim=4096):
        super().__init()
        self.exp = nn.Sequential([
            nn.Linear(in_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        ])

    def forward(self, x):
        return self.exp(x)

@DETECTOR.register_module(module_name='dual_branch')
class DualBranchDetector(AbstractDetector)

    def __init__(self, config):
        