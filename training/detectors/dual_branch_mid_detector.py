import logging
import torch
import torch.nn as nn
from metrics.base_metrics_class import calculate_metrics_for_train

from networks.pixel_branch import PixelBranch
from networks.frequency_branch import FrequencyBranch

from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='dual_branch_mids')
class DualBranchMIDSDetector(AbstractDetector):
    """Dual-branch detector with Manipulation-Invariant Deviation Subspace
    (MIDS) regularization.

    subspace_reg: 'off'    -> original isotropic var/cov path
                  'pooled' -> uniform-weight Delta baseline (per-env-tracked)
                  'maxmin' -> max-min shared forgery subspace (Method 1)

    Requires data_dict['env'] (LongTensor [B], -1 = real, 0..M-1 =
    manipulation id in train-list order) when subspace_reg != 'off'.
    Falls back to a single fake environment with a warning if absent.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.mode = config.get('mode', 'dual')

        self.pixel_branch, self.freq_branch = self.build_backbone(config)

        pixel_dim = self.pixel_branch.out_dim if self.pixel_branch else 0
        freq_dim = self.freq_branch.out_dim if self.freq_branch else 0
        self._pixel_dim = pixel_dim

        (self.loss_func, self.scl_loss,
         self.varcov_loss, self.subspace_loss) = self.build_loss(
            config, pixel_dim + freq_dim)

        head_input_dim = pixel_dim + freq_dim
        head_hidden = config.get('head_hidden_dim', 256)
        head_dropout = config.get('head_dropout', 0.3)
        num_classes = config.get('backbone_config', {}).get('num_classes', 2)

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_classes),
        )

        # Optional auxiliary head on the k-dim shared-subspace projection:
        # encourages the encoder to make the shared coordinates themselves
        # discriminative ("route detection through S*"). Keep 0.0 for the
        # first pooled-vs-maxmin comparison; enable as a second phase.
        self.inv_head_weight = config.get('inv_head_weight', 0.0)
        self.inv_head = None
        if self.subspace_loss is not None and self.inv_head_weight > 0:
            self.inv_head = nn.Linear(config.get('subspace_k', 16), num_classes)

        self.scl_weight = config.get('scl_weight', 0.1)
        self.var_weight = config.get('var_weight', 0.04)
        self.cov_weight = config.get('cov_weight', 0.01)
        self.real_only = config.get('varcov_real_only', False)

        self._diag_step = 0
        self._diag_log_every = config.get('diag_log_every', 500)
        self._warned_no_env = False

        self._log_params()

    def _log_params(self):
        total, trainable = 0, 0
        components = {
            'pixel_branch': self.pixel_branch,
            'freq_branch': self.freq_branch,
            'head': self.head,
            'inv_head': self.inv_head,
        }
        for name, module in components.items():
            if module is not None:
                t = sum(p.numel() for p in module.parameters())
                tr = sum(p.numel() for p in module.parameters() if p.requires_grad)
                total += t
                trainable += tr
                logger.info(f"  {name}: {tr:,} trainable / {t:,} total")
        logger.info(f"DualBranchMIDS [{self.mode}]: "
                    f"{trainable:,} trainable / {total:,} total")

    def classifier(self, feat_dict):
        parts = []
        if feat_dict['y_pixel'] is not None:
            parts.append(feat_dict['y_pixel'])
        if feat_dict['y_freq'] is not None:
            parts.append(feat_dict['y_freq'])
        fused = torch.cat(parts, dim=-1)
        return self.head(fused)

    def build_backbone(self, config):
        pixel_branch = None
        freq_branch = None
        mode = config.get('mode', 'dual')
        if mode in ('dual', 'pixel_only'):
            pixel_branch = PixelBranch(config)
        if mode in ('dual', 'freq_only'):
            freq_branch = FrequencyBranch(config)
        return pixel_branch, freq_branch

    def build_loss(self, config, fused_dim):
        cls_loss_class = LOSSFUNC[config.get('loss_func', 'cross_entropy')]
        cls_loss = cls_loss_class()

        scl_loss = None
        if config.get('scl_weight', 0) > 0:
            scl_variant = config.get('scl_variant', 'scl')
            scl_loss = LOSSFUNC[scl_variant](
                feat_dim=fused_dim,
                ema_tau=config.get('scl_ema_tau', 0.99),
            )

        varcov_loss = LOSSFUNC['varcov']()

        subspace_loss = None
        subspace_mode = config.get('subspace_reg', 'off')
        if subspace_mode in ('pooled', 'maxmin'):
            subspace_loss = LOSSFUNC['maxmin_subspace'](
                feat_dim=fused_dim,
                num_envs=config.get('subspace_num_envs', 4),
                k=config.get('subspace_k', 16),
                moment_ema=config.get('subspace_moment_ema', 0.995),
                refresh_every=config.get('subspace_refresh_every', 200),
                warmup_steps=config.get('subspace_warmup_steps', 500),
                solver_iters=config.get('subspace_solver_iters', 30),
                solver_lr=config.get('subspace_solver_lr', 0.5),
                apply_to=config.get('subspace_apply_to', 'real'),
                mode=subspace_mode,
            )

        return cls_loss, scl_loss, varcov_loss, subspace_loss

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def features(self, data_dict: dict) -> dict:
        img = data_dict['image']
        feat_dict = {'y_pixel': None, 'y_freq': None}
        if self.pixel_branch is not None:
            feat_dict['y_pixel'] = self.pixel_branch(img)
        if self.freq_branch is not None:
            feat_dict['y_freq'] = self.freq_branch(img)
        return feat_dict

    @torch.no_grad()
    def _log_feature_diagnostics(self, feat: torch.Tensor,
                                 label: torch.Tensor) -> None:
        real_mask = label == 0
        fake_mask = label == 1

        for cls_name, mask in (('real', real_mask), ('fake', fake_mask)):
            if mask.sum() > 0:
                mean_norm = feat[mask].norm(dim=1).mean().item()
                logger.info(f"  feat_norm/{cls_name}: {mean_norm:.4f}")

        if real_mask.sum() > 1:
            real_feats = feat[real_mask].float()
            real_var = real_feats.var(dim=0).mean().item()
            logger.info(f"  real_feat_var: {real_var:.4f}")

            centered = real_feats - real_feats.mean(dim=0, keepdim=True)
            if centered.size(1) > 512:
                centered = centered[:, :512]
            try:
                _, S, _ = torch.linalg.svd(centered, full_matrices=False)
                eigvals = S.pow(2) / max(centered.size(0) - 1, 1)
                eigvals = eigvals[eigvals > 0]
                p = eigvals / eigvals.sum()
                eff_rank = torch.exp(-(p * p.log()).sum()).item()
                logger.info(f"  real_cov_eff_rank: {eff_rank:.2f}")
            except Exception:
                pass

            # MIDS geometry diagnostics: the variance decomposition that
            # replaces the thesis Figure-5 trajectory, plus where the shared
            # subspace lives across the two branches.
            if (self.subspace_loss is not None
                    and bool(self.subspace_loss.subspace_ready)):
                c_var, f_var = self.subspace_loss.content_forgery_variance(
                    real_feats)
                logger.info(f"  real_var_content: {c_var:.4f}  "
                            f"real_var_shared_forgery: {f_var:.4f}")
                U = self.subspace_loss.U
                if 0 < self._pixel_dim < U.size(0):
                    pm = float(U[:self._pixel_dim].pow(2).sum()
                               / U.pow(2).sum())
                    logger.info(f"  shared_subspace_pixel_mass: {pm:.3f} "
                                f"(freq mass {1 - pm:.3f})")

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        cls_loss = self.loss_func(pred_dict['cls'], label)
        overall = cls_loss
        losses = {'cls': cls_loss}

        if self.scl_loss is not None:
            scl_loss = self.scl_loss(pred_dict['feat'], label)
            losses['scl'] = scl_loss
            overall = overall + self.scl_weight * scl_loss

        feat = pred_dict['feat']

        if self.subspace_loss is not None:
            env = data_dict.get('env', None)
            if env is None:
                if not self._warned_no_env:
                    logger.warning(
                        "data_dict has no 'env'; treating all fakes as one "
                        "environment (subspace degenerates to pooled). Patch "
                        "the dataset to emit manipulation ids.")
                    self._warned_no_env = True
                env = torch.zeros_like(label)
            else:
                env = env.to(label.device).long()

            if feat.size(0) > 1 and feat.size(1) > 0:
                var_loss, cov_loss = self.subspace_loss(feat, label, env)
                losses['var'] = var_loss
                losses['cov'] = cov_loss
                overall = (overall + self.var_weight * var_loss
                           + self.cov_weight * cov_loss)

            if (self.inv_head is not None
                    and bool(self.subspace_loss.subspace_ready)):
                # Differentiable wrt feat only; mu_r and U are buffers.
                z = (feat - self.subspace_loss.mu_r.to(feat.dtype)) \
                    @ self.subspace_loss.U.to(feat.dtype)
                inv_loss = self.loss_func(self.inv_head(z), label)
                losses['inv_head'] = inv_loss
                overall = overall + self.inv_head_weight * inv_loss
        else:
            tgt = feat[label == 0] if self.real_only else feat
            if tgt.size(0) > 1 and tgt.size(1) > 0:
                var_loss, cov_loss = self.varcov_loss(tgt)
                losses['var'] = var_loss
                losses['cov'] = cov_loss
                overall = (overall + self.var_weight * var_loss
                           + self.cov_weight * cov_loss)

        losses['overall'] = overall

        self._diag_step += 1
        if self._diag_step % self._diag_log_every == 0:
            f = pred_dict['feat']
            if f.size(0) > 1 and f.size(1) > 0:
                logger.info(f"[diag step={self._diag_step}]")
                self._log_feature_diagnostics(f.detach(), label.detach())

        return losses

    def forward(self, data_dict, inference=False):
        feat_dict = self.features(data_dict)
        pred = self.classifier(feat_dict)
        prob = torch.softmax(pred, dim=1)[:, 1]

        parts = []
        if feat_dict['y_pixel'] is not None:
            parts.append(feat_dict['y_pixel'])
        if feat_dict['y_freq'] is not None:
            parts.append(feat_dict['y_freq'])
        feat_tensor = (torch.cat(parts, dim=-1) if parts
                       else pred.new_zeros(pred.size(0), 0))

        cls_feat = self.head[:-1](feat_tensor)

        return {
            'cls': pred,
            'prob': prob,
            'feat': feat_tensor,
            'cls_feat': cls_feat,
            'feat_dict': feat_dict,
        }