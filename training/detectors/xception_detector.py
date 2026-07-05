'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the XceptionDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{rossler2019faceforensics++,
  title={Faceforensics++: Learning to detect manipulated facial images},
  author={Rossler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={1--11},
  year={2019}
}
'''

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='xception')
class XceptionDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.video_names = []
        self.correct, self.total = 0, 0

        # Feature-collapse diagnostics (logged every N training steps; 0 disables)
        self._diag_step = 0
        self._diag_log_every = config.get('diag_log_every', 500)

    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        # if donot load the pretrained weights, fail to get good results
        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model successfully!')
        return backbone
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image']) #32,3,256,256

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    @torch.no_grad()
    def _log_feature_diagnostics(self, feat: torch.Tensor, label: torch.Tensor) -> None:
        """Log feature-collapse indicators for the pooled penultimate embedding.

        `feat` must be the (B, D) pooled embedding (pred_dict['cls_feat']), NOT the
        4D backbone feature map. Metrics mirror dual_branch_scl for cross-run
        comparability: per-class L2 norm, per-class mean variance, and the
        effective rank of the real-class covariance -- a direct measure of
        dimensional collapse (eff_rank -> 1 means the features live on ~1 axis;
        0.00 means total collapse to a single point).
        """
        real_mask = label == 0
        fake_mask = label >= 1  # any non-real label counts as fake

        # Per-class L2 norms
        for cls_name, mask in (('real', real_mask), ('fake', fake_mask)):
            if mask.sum() > 0:
                mean_norm = feat[mask].norm(dim=1).mean().item()
                logger.info(f"  feat_norm/{cls_name}: {mean_norm:.4f}")

        # Per-class mean per-dimension variance (collapse -> variance -> 0)
        for cls_name, mask in (('real', real_mask), ('fake', fake_mask)):
            if mask.sum() > 1:
                var = feat[mask].float().var(dim=0).mean().item()
                logger.info(f"  {cls_name}_feat_var: {var:.4f}")

        # Effective rank of the real-class covariance: exp(H), where H is the
        # entropy of the normalised eigenvalue spectrum. Uses SVD on mean-centred
        # features for numerical stability (batch is small, so no dim truncation).
        if real_mask.sum() > 1:
            real_feats = feat[real_mask].float()
            centered = real_feats - real_feats.mean(dim=0, keepdim=True)
            try:
                _, S, _ = torch.linalg.svd(centered, full_matrices=False)
                eigvals = S.pow(2) / max(centered.size(0) - 1, 1)
                eigvals = eigvals[eigvals > 1e-12]
                if eigvals.numel() == 0:
                    # All variance gone -> features collapsed to a single point.
                    logger.info("  real_cov_eff_rank: 0.00 (collapsed)")
                else:
                    p = eigvals / eigvals.sum()
                    eff_rank = torch.exp(-(p * p.log()).sum()).item()
                    logger.info(f"  real_cov_eff_rank: {eff_rank:.2f}")
            except Exception as e:  # SVD non-convergence / NaN features
                logger.debug(f"  real_cov_eff_rank skipped: {e!r}")

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        overall_loss = loss
        loss_dict = {'overall': overall_loss, 'cls': loss,}

        # Periodically log feature-collapse diagnostics on the pooled embedding.
        self._diag_step += 1
        if self._diag_log_every > 0 and self._diag_step % self._diag_log_every == 0:
            feat = pred_dict.get('cls_feat')
            if feat is not None and feat.dim() == 2 and feat.size(0) > 1:
                logger.info(f"[diag step={self._diag_step}]")
                self._log_feature_diagnostics(feat.detach(), label.detach())
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        # we dont compute the video-level metrics for training
        self.video_names = []
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone (4D spatial map, B x C x H x W)
        features = self.features(data_dict)
        # Pooled penultimate embedding (B, C) for feature-collapse diagnostics.
        # Mirror backbone.classifier(): ReLU (skipped only for adjust_channel) then
        # global average pool. Computed BEFORE self.classifier(), whose inplace ReLU
        # would otherwise mutate `features`; under no_grad since it is diagnostics-only.
        with torch.no_grad():
            feat_map = features if getattr(self.backbone, 'mode', None) == 'adjust_channel' else F.relu(features)
            cls_feat = F.adaptive_avg_pool2d(feat_map, (1, 1)).flatten(1)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features, 'cls_feat': cls_feat}
        return pred_dict
