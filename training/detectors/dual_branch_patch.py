import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics.base_metrics_class import calculate_metrics_for_train

from networks.pixel_branch_patch import PixelBranchPatch
from networks.frequency_branch_patch import FrequencyBranchPatch

from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


class PatchRealCenterLoss(nn.Module):
    """Real-only center loss at patch granularity: cosine attraction of every
    authentic patch toward an EMA center; fakes unconstrained."""
    def __init__(self, feat_dim, ema_tau=0.99):
        super().__init__()
        self.tau = ema_tau
        self.register_buffer('center', torch.zeros(feat_dim))
        self.register_buffer('initialized', torch.tensor(False))

    @torch.no_grad()
    def _update(self, real_patches):
        m = real_patches.mean(0)
        if not self.initialized:
            self.center.copy_(m); self.initialized.fill_(True)
        else:
            self.center.mul_(self.tau).add_(m, alpha=1 - self.tau)

    def forward(self, tokens, label):                 # tokens [B,N,D], label [B]
        real = tokens[label == 0]
        if real.numel() == 0:
            return tokens.sum() * 0.0
        real = real.reshape(-1, real.size(-1))        # [Br*N, D] — all authentic
        self._update(real.detach())
        c = F.normalize(self.center, dim=0)
        return (1 - F.normalize(real, dim=1) @ c).mean()


@DETECTOR.register_module(module_name='dual_branch_patch')
class DualBranchPatch(AbstractDetector):
    """Dual-branch (LoRA-CLIP + FFT-ResNet) detector with a patch-level
    real-only center loss, per-patch var-cov, and an optional cross-manipulation
    invariance penalty. Requires dual mode (both branches present)."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.mode = config.get('mode', 'dual')

        self.pixel_branch, self.freq_branch = self.build_backbone(config)

        pixel_dim = self.pixel_branch.out_dim if self.pixel_branch else 0
        freq_dim  = self.freq_branch.out_dim  if self.freq_branch  else 0        # pooled dims
        freq_map_dim = self.freq_branch.map_dim if self.freq_branch else 0       # spatial-map channels
        fused_dim = pixel_dim + freq_dim          # pooled fusion -> CE head
        patch_dim = pixel_dim + freq_map_dim      # per-patch fusion -> geometry losses

        # pooled classification head — unchanged from the global model
        head_hidden  = config.get('head_hidden_dim', 256)
        head_dropout = config.get('head_dropout', 0.3)
        num_classes  = config.get('backbone_config', {}).get('num_classes', 2)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_classes),
        )

        # losses + weights
        self.loss_func, self.scl_loss, self.varcov_loss, self.aniso_loss = \
            self.build_loss(config, fused_dim)
        self.scl_weight = config.get('scl_weight', 0.1)
        self.var_weight = config.get('var_weight', 0.04)
        self.cov_weight = config.get('cov_weight', 0.01)
        self.real_only  = config.get('varcov_real_only', False)

        # patch path
        self.grid = config.get('clip_grid', 16)                 # clip_input_size // patch_size
        self.freq_tok_norm = nn.LayerNorm(freq_map_dim) if freq_map_dim else None
        self.patch_center = PatchRealCenterLoss(
            feat_dim=patch_dim, ema_tau=config.get('scl_ema_tau', 0.99))
        self.patch_center_weight = config.get('patch_center_weight', 0.1)

        # --- Fishr cross-manipulation invariance (toggle: 0 = baseline, >0 = Fishr on) ---
        # Matches the per-env variance of per-sample gradients of the final Linear
        # (head[-1]); domains are the FF++ manipulation ids in data_dict['env'].
        self.fishr_weight = config.get('fishr_weight', 0.0)
        self.fishr_warmup = config.get('fishr_warmup', 1500)
        self.fishr_gamma  = config.get('fishr_gamma', 0.99)
        self.fishr_envs   = list(config.get('fishr_envs', [0, 1, 2, 3]))
        self._env2idx     = {e: i for i, e in enumerate(self.fishr_envs)}
        _head_out = self.head[-1]                                  # final Linear (H -> C)
        fishr_P   = _head_out.weight.numel() + _head_out.bias.numel()
        self.register_buffer('fishr_ema', torch.zeros(len(self.fishr_envs), fishr_P))
        self.register_buffer('fishr_ema_init',
                             torch.zeros(len(self.fishr_envs), dtype=torch.bool))

        # diagnostics
        self._diag_step = 0
        self._diag_log_every = config.get('diag_log_every', 500)

        self._log_params()

    def _log_params(self):
        total, trainable = 0, 0
        components = {
            'pixel_branch': self.pixel_branch,
            'freq_branch': self.freq_branch,
            'head': self.head,
        }
        for name, module in components.items():
            if module is not None:
                t = sum(p.numel() for p in module.parameters())
                tr = sum(p.numel() for p in module.parameters() if p.requires_grad)
                total += t
                trainable += tr
                logger.info(f"  {name}: {tr:,} trainable / {t:,} total")
        logger.info(f"DualBranchPatch [{self.mode}]: {trainable:,} trainable / {total:,} total")

    def build_backbone(self, config):
        pixel_branch = None
        freq_branch = None
        mode = config.get("mode", "dual")
        if mode in ('dual', 'pixel_only'):
            pixel_branch = PixelBranchPatch(config)
        if mode in ('dual', 'freq_only'):
            freq_branch = FrequencyBranchPatch(config)
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

        aniso_loss = None
        if config.get('aniso_varcov', False):
            aniso_loss = LOSSFUNC['aniso_varcov'](
                feat_dim=fused_dim, k=config.get('aniso_k', 16),
                cov_ema=config.get('aniso_cov_ema', 0.99),
                refresh_every=config.get('aniso_refresh_every', 200),
                apply_to=config.get('aniso_apply_to', 'real'))

        return cls_loss, scl_loss, varcov_loss, aniso_loss

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def features(self, data_dict: dict) -> dict:
        # Pooled features only (abstract-interface compliance); the patch path
        # lives in forward(). Pixel branch defaults return_tokens=False.
        img = data_dict['image']
        feat_dict = {'y_pixel': None, 'y_freq': None}
        if self.pixel_branch is not None:
            feat_dict['y_pixel'] = self.pixel_branch(img)
        if self.freq_branch is not None:
            feat_dict['y_freq'] = self.freq_branch(img)
        return feat_dict

    def classifier(self, feat_dict):
        parts = []
        if feat_dict['y_pixel'] is not None:
            parts.append(feat_dict['y_pixel'])
        if feat_dict['y_freq'] is not None:
            parts.append(feat_dict['y_freq'])
        return self.head(torch.cat(parts, dim=-1))

    @torch.no_grad()
    def _log_feature_diagnostics(self, feat: torch.Tensor, label: torch.Tensor) -> None:
        real_mask = label == 0
        fake_mask = label == 1

        for cls_name, mask in (('real', real_mask), ('fake', fake_mask)):
            if mask.sum() > 0:
                logger.info(f"  feat_norm/{cls_name}: {feat[mask].norm(dim=1).mean().item():.4f}")

        if real_mask.sum() > 1:
            real_feats = feat[real_mask].float()

            if self.aniso_loss is not None and bool(self.aniso_loss.initialized):
                Q, k = self.aniso_loss.Q, self.aniso_loss.k
                z = real_feats @ Q
                logger.info(f"  real_var_forgery_subspace: {z[:, -k:].var(dim=0).mean().item():.4f}")
                logger.info(f"  real_var_content_subspace: {z[:, :-k].var(dim=0).mean().item():.4f}")

            logger.info(f"  real_feat_var: {real_feats.var(dim=0).mean().item():.4f}")

            centered = real_feats - real_feats.mean(dim=0, keepdim=True)
            if centered.size(1) > 512:
                centered = centered[:, :512]
            try:
                _, S, _ = torch.linalg.svd(centered, full_matrices=False)
                eigvals = S.pow(2) / max(centered.size(0) - 1, 1)
                eigvals = eigvals[eigvals > 0]
                p = eigvals / eigvals.sum()
                logger.info(f"  real_cov_eff_rank: {torch.exp(-(p * p.log()).sum()).item():.2f}")
            except Exception:
                pass

    @torch.no_grad()
    def _log_patch_diagnostics(self, patch_tokens: torch.Tensor, label: torch.Tensor) -> None:
        D = patch_tokens.size(-1)
        real = patch_tokens[label == 0].reshape(-1, D).float()
        fake = patch_tokens[label == 1].reshape(-1, D).float() if (label == 1).any() else None

        if real.numel() > 0:
            logger.info(f"  patch_norm/real: {real.norm(dim=1).mean().item():.4f}")
        if fake is not None and fake.numel() > 0:
            logger.info(f"  patch_norm/fake: {fake.norm(dim=1).mean().item():.4f}")
        if real.size(0) <= 1:
            return

        # (1) collapse signal: mean per-dim variance of real patches (patch-level real_feat_var)
        logger.info(f"  patch_real_var: {real.var(dim=0).mean().item():.4f}")

        # (2) effective rank of the real-patch covariance
        centered = real - real.mean(dim=0, keepdim=True)
        if centered.size(1) > 512:
            centered = centered[:, :512]
        try:
            _, S, _ = torch.linalg.svd(centered, full_matrices=False)
            eig = S.pow(2) / max(centered.size(0) - 1, 1)
            eig = eig[eig > 0]
            p = eig / eig.sum()
            logger.info(f"  patch_real_eff_rank: {torch.exp(-(p * p.log()).sum()).item():.2f}")
        except Exception:
            pass

        # (3) real-vs-fake separation in center-distance space — the low-FPR anomaly signal
        if bool(self.patch_center.initialized):
            c = F.normalize(self.patch_center.center.float(), dim=0)
            d_real = 1 - F.normalize(real, dim=1) @ c
            logger.info(f"  patch_dist/real: {d_real.mean().item():.4f}")
            if fake is not None and fake.numel() > 0:
                d_fake = 1 - F.normalize(fake, dim=1) @ c
                logger.info(f"  patch_dist/fake: {d_fake.mean().item():.4f}  "
                            f"gap: {(d_fake.mean() - d_real.mean()).item():.4f}")

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']

        self._diag_step += 1
        do_diag = (self._diag_step % self._diag_log_every == 0)

        cls_loss = self.loss_func(pred_dict['cls'], label)
        overall = cls_loss
        losses = {'cls': cls_loss}

        ptok = pred_dict['patch_tokens']                       # [B, N, D]
        B, N, D = ptok.shape

        # patch-level real-only center loss
        pc = self.patch_center(ptok, label)
        losses['patch_center'] = pc
        overall = overall + self.patch_center_weight * pc

        # per-patch variance-covariance on real patches (collapse guard, patch grain)
        real_flat = ptok[label == 0].reshape(-1, D)
        if real_flat.size(0) > 1:
            pv, pcv = self.varcov_loss(real_flat)
            losses['patch_var'], losses['patch_cov'] = pv, pcv
            overall = overall + self.var_weight * pv + self.cov_weight * pcv

        # cross-manipulation invariance via Fishr (Rame et al., ICML 2022): match the
        # per-env variance of per-sample gradients of the final Linear. Toggle with
        # fishr_weight (0 = baseline). Domains are the fake manipulation ids in
        # data_dict['env'] (real=-1, DF/F2F/FS/NT=0..3, -2=unmapped fake -> ignored).
        # The per-sample grad of CE wrt head[-1] has the closed form (p - y) (x) a, so the
        # penalty is a first-order function of network outputs -- no double-backward needed.
        if self.fishr_weight > 0 and 'env' in data_dict:
            env = data_dict['env']
            a = pred_dict['cls_feat']                              # [B,H] penultimate (shared-mask)
            p = torch.softmax(pred_dict['cls'], dim=1)             # [B,C]
            resid = p - F.one_hot(label, p.size(1)).to(p.dtype)    # d(CE)/d(logits), [B,C]
            g = torch.cat([torch.einsum('bc,bh->bhc', resid, a).reshape(a.size(0), -1),
                           resid], dim=1)                          # per-sample grad [B, H*C+C]
            ema_vals, present = [], []
            for e in self.fishr_envs:
                m = (env == e)
                if int(m.sum()) >= 2:                              # variance needs >=2 samples
                    idx = self._env2idx[e]
                    v_live = g[m].var(dim=0, unbiased=False)       # [P], carries grad -> backbone
                    if bool(self.fishr_ema_init[idx]):
                        v_ema = self.fishr_gamma * self.fishr_ema[idx] \
                                + (1 - self.fishr_gamma) * v_live
                    else:
                        v_ema = v_live                             # first obs: seed the EMA
                        self.fishr_ema_init[idx] = True
                    self.fishr_ema[idx] = v_ema.detach()           # persist running estimate
                    ema_vals.append(v_ema); present.append(e)
            # penalty added only after warmup (EMA still warms during it) and with >=2 envs
            if len(present) >= 2 and self._diag_step >= self.fishr_warmup:
                V = torch.stack(ema_vals, 0)                       # [E, P]
                fishr = ((V - V.mean(0, keepdim=True)) ** 2).sum(1).mean()
                losses['fishr'] = fishr
                overall = overall + self.fishr_weight * fishr
                if do_diag:
                    logger.info(f"  fishr/grad_var_dist: {fishr.item():.4e}  "
                                f"envs_present: {present}")

        losses['overall'] = overall

        # diagnostics (pooled + patch), on the same cadence
        if do_diag:
            logger.info(f"[diag step={self._diag_step}]")
            feat = pred_dict['feat']
            if feat.size(0) > 1 and feat.size(1) > 0:
                self._log_feature_diagnostics(feat.detach(), label.detach())
            self._log_patch_diagnostics(ptok.detach(), label.detach())

        return losses

    def forward(self, data_dict, inference=False):
        img = data_dict['image']

        pooled_px, tok_px = self.pixel_branch(img, return_tokens=True)  # [B,Dp], [B,N,Dp]
        pooled_fq, fmap   = self.freq_branch(img, return_map=True)      # [B,Df], [B,Cf,h,w]

        fmap = F.interpolate(fmap, size=(self.grid, self.grid),
                             mode='bilinear', align_corners=False)      # [B,Cf,grid,grid]
        tok_fq = self.freq_tok_norm(fmap.flatten(2).transpose(1, 2))    # [B,N,Cf]
        patch_tokens = torch.cat([tok_px, tok_fq], dim=-1)             # [B,N,Dp+Cf]

        fused = torch.cat([pooled_px, pooled_fq], dim=-1)              # [B,Dp+Df]
        cls_feat = self.head[:-1](fused)              # Linear->ReLU->Dropout (single mask)
        pred = self.head[-1](cls_feat)                # final Linear on the SAME activation
        prob = torch.softmax(pred, dim=1)[:, 1]

        return {
            'cls': pred,
            'prob': prob,
            'feat': fused,                 # pooled fused, for diagnostics/viz
            'cls_feat': cls_feat,
            'patch_tokens': patch_tokens,  # consumed by get_losses
        }