"""dual_branch_patch_v6: DualBranchPatchV5 + the environment-design axis.

Research question: "What is an environment for deepfake detection?"
This detector adds ONE config knob, fishr_env_mode, selecting how the single
FF++ training set is partitioned into Fishr environments:

  'fake_only'   (default) bit-identical to v5: envs = manipulation types
                (DF/F2F/FS/NT), reals excluded. get_losses delegates to the
                parent untouched, so v6+fake_only == v5 exactly.
  'shuffle'     negative control: env labels are randomly permuted among the
                fake samples at every step, destroying the env<->manipulation
                correspondence while preserving group sizes and the fake-only
                composition. Tests whether the STRUCTURE of the partition (and
                not merely an extra gradient-variance regularizer) drives the
                generalization gain.
  'real_shared' env e = {all reals} U {fakes of manipulation e}: every
                environment is a complete "real vs generator-e" binary task
                (the classic IRM-style environment family), and the shared
                real mass anchors each per-env variance estimate.

Implemented as a NEW detector (not an edit to dual_branch_patch / v5) so that
training and evaluation of all existing configs execute byte-identical code.
"""

import logging
import torch
import torch.nn.functional as F

from detectors import DETECTOR
from .dual_branch_patch_v5 import DualBranchPatchV5

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='dual_branch_patch_v6')
class DualBranchPatchV6(DualBranchPatchV5):

    ENV_MODES = ('fake_only', 'shuffle', 'real_shared')

    def __init__(self, config):
        super().__init__(config)
        self.fishr_env_mode = config.get('fishr_env_mode', 'fake_only')
        if self.fishr_env_mode not in self.ENV_MODES:
            raise ValueError(f"fishr_env_mode must be one of {self.ENV_MODES}, "
                             f"got {self.fishr_env_mode!r}")
        logger.info(f"DualBranchPatchV6: fishr_env_mode = {self.fishr_env_mode}")

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        mode = self.fishr_env_mode
        # Delegate whenever the parent's fake-only semantics apply verbatim:
        # fake_only mode, Fishr off, eval batches, or missing env key (the
        # parent owns the one-time warning for that case).
        if (mode == 'fake_only' or self.fishr_weight <= 0
                or not self.training or 'env' not in data_dict):
            return super().get_losses(data_dict, pred_dict)

        if mode == 'shuffle':
            env = data_dict['env']
            shuffled = env.clone()
            fk = (env >= 0).nonzero(as_tuple=True)[0]
            if fk.numel() > 1:
                perm = fk[torch.randperm(fk.numel(), device=env.device)]
                shuffled[fk] = env[perm]
            shadow = dict(data_dict)          # shallow copy: the loader's dict
            shadow['env'] = shuffled          # and its env tensor stay intact
            return super().get_losses(shadow, pred_dict)

        # ---- real_shared ----
        # Run the parent with Fishr silenced (keeps cls/center/varcov/diag and
        # the _diag_step increment identical), then add the real-shared
        # penalty. The block below mirrors dual_branch_patch.py get_losses
        # (lines ~314-370) -- keep the two in sync; the only semantic changes
        # are the env mask (reals join every env) and the requirement that an
        # env contributes at least one fake (otherwise its variance would be a
        # degenerate copy of the real-only variance, dragging v_bar).
        w = self.fishr_weight
        self.fishr_weight = 0.0
        try:
            losses = super().get_losses(data_dict, pred_dict)
        finally:
            self.fishr_weight = w

        env = data_dict['env']
        label = data_dict['label']
        a = pred_dict['cls_feat']                          # [B,H] penultimate (shared-mask)
        p = torch.softmax(pred_dict['cls'], dim=1)         # [B,C]
        y_soft = F.one_hot(label, p.size(1)).to(p.dtype)
        if self.label_smoothing > 0:
            y_soft = (1 - self.label_smoothing) * y_soft \
                     + self.label_smoothing / p.size(1)
        resid = p - y_soft                                 # [B,C]
        g = torch.cat([torch.einsum('bc,bh->bhc', resid, a).reshape(a.size(0), -1),
                       resid], dim=1)                      # per-sample grad [B, H*C+C]
        real_m = (label == 0)
        ema_vals, present = [], []
        for e in self.fishr_envs:
            fake_m = (env == e)
            m = fake_m | real_m                            # reals shared by every env
            if int(fake_m.sum()) >= 1 and int(m.sum()) >= 2:
                idx = self._env2idx[e]
                v_live = g[m].var(dim=0, unbiased=self.fishr_unbiased)  # [P], carries grad
                if bool(self.fishr_ema_init[idx]):
                    v_ema = self.fishr_gamma * self.fishr_ema[idx] \
                            + (1 - self.fishr_gamma) * v_live
                else:
                    v_ema = v_live                         # first obs: seed the EMA
                    self.fishr_ema_init[idx] = True
                self.fishr_ema[idx] = v_ema.detach()       # persist running estimate
                ema_vals.append(v_ema); present.append(e)
        if len(present) >= 2 and self._diag_step >= self.fishr_warmup:
            V = torch.stack(ema_vals, 0)                   # [E, P]
            v_bar = V.mean(0, keepdim=True)
            dev = V - v_bar
            if self.fishr_relative:
                vb = v_bar.detach()
                floor = self.fishr_rel_floor * vb.mean() + self.fishr_rel_eps
                dev = dev / (vb + floor)
            fishr = (dev ** 2).sum(1).mean()
            losses['fishr'] = fishr
            losses['overall'] = losses['overall'] + self.fishr_weight * fishr
            if self._diag_step % self._diag_log_every == 0:
                logger.info(f"  fishr/grad_var_dist: {fishr.item():.4e}  "
                            f"relative: {self.fishr_relative}  "
                            f"env_mode: real_shared  envs_present: {present}")
        return losses
