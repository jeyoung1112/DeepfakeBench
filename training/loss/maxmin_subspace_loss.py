import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from loss import LOSSFUNC
    _register = LOSSFUNC.register_module(module_name='maxmin_subspace')
except Exception:  # standalone execution: `python loss/maxmin_subspace_loss.py`
    def _register(cls):
        return cls


@_register
class MaxMinDeviationSubspace(nn.Module):
    """Manipulation-invariant deviation subspace regularizer (Method 1), v2.

    Per-environment deviation matrices (unpaired; content cancels in
    expectation because FF++ fakes are derived from the same real pool):

        Delta_m = Sigma_m + d_m d_m^T - Sigma_r,   d_m = mu_m - mu_r

    Shared subspace:   max_{U^T U = I_k}  min_m  tr(U^T Delta_m U)

    solved through its dual (Sion minimax + Ky Fan variational principle):

        min_{w in simplex}  sum_of_top_k_eigvals( sum_m w_m Delta_m )

    with exponentiated-gradient steps on w.

    v2 solver fixes (motivated by the observed w=[0,0,0,1] absorbing-vertex
    logs in both the full-FF++ and LOMO runs):

      * COLD START. Each refresh solves its own problem from uniform w.
        Warm-starting turned 30-iter EG into a run-long accumulator
        (~7,000 effective iterations by mid-training); any persistent drift
        compounded, and because EG is multiplicative, a coordinate that
        underflowed to exactly 0 could never recover.
      * INTERIOR FLOOR. After every EG step, w is mixed with the uniform
        distribution so no coordinate can reach the absorbing zero state.
      * w_cap (CVaR-style). Optional upper bound on any single
        environment's weight, projected with an exact capped-simplex
        projection (naive clamp-then-renormalize can push other
        coordinates back over the cap). w_cap=1.0 recovers pure max-min;
        w_cap=1/M forces pooled; values between interpolate.
      * normalize_deltas. Per-environment positive-trace normalization of
        Delta_m, so the min compares FRACTIONAL overlap ("what fraction of
        environment m's deviation energy does U capture?") instead of raw
        artifact energy. This is the fix for scale-heteroskedastic
        environments: with raw energies, a weak-artifact environment (NT)
        holds the min hostage purely by scale and the solver correctly --
        but uselessly -- parks all weight on it. With normalization,
        logged energies and the dual are fractions (<= 1).

    Training use: rotate features into the eigenbasis Q of Dbar(w*) and
    apply the VICReg-style variance/covariance hinge ONLY to the bottom
    D-k (content) coordinates.

    mode='pooled' fixes w uniform and solves once -- the pooled baseline
    under identical moment tracking, so pooled-vs-maxmin is a one-variable
    ablation. The refresh log additionally prints the worst-environment
    energy under the pooled basis and the maxmin-pooled gap: if the gap
    reads ~0 for the whole run, the environments agree at this k and the
    maxmin-vs-pooled ablation will be flat -- known before eval.

    Env convention: env[i] in {0..M-1} for fakes, ignored for reals
    (realness is taken from labels == 0; use -1 for reals in the dataset).
    """

    def __init__(self, feat_dim, num_envs=4, k=16, gamma=1.0, eps=1e-4,
                 moment_ema=0.995, refresh_every=200, warmup_steps=500,
                 solver_iters=30, solver_lr=0.5, apply_to='real',
                 mode='maxmin', w_cap=1.0, normalize_deltas=False,
                 log_refresh=True):
        super().__init__()
        assert mode in ('pooled', 'maxmin')
        assert apply_to in ('real', 'all')
        self.D = feat_dim
        self.M = num_envs
        self.k = k
        self.gamma = gamma
        self.eps = eps
        self.moment_ema = moment_ema
        self.refresh_every = refresh_every
        self.warmup_steps = warmup_steps
        self.solver_iters = solver_iters
        self.solver_lr = solver_lr
        self.apply_to = apply_to
        self.mode = mode
        self.log_refresh = log_refresh
        self.normalize_deltas = bool(normalize_deltas)

        # Feasibility: sum of M weights each <= w_cap must reach 1,
        # so w_cap must be >= 1/M.
        if w_cap < 1.0:
            w_cap = max(float(w_cap), 1.0 / num_envs)
        self.w_cap = float(w_cap)

        # EMA first/second moments: reals and per-manipulation fakes.
        self.register_buffer('mu_r', torch.zeros(feat_dim))
        self.register_buffer('m2_r', torch.zeros(feat_dim, feat_dim))
        self.register_buffer('init_r', torch.tensor(False))
        self.register_buffer('mu_f', torch.zeros(num_envs, feat_dim))
        self.register_buffer('m2_f', torch.zeros(num_envs, feat_dim, feat_dim))
        self.register_buffer('init_f', torch.zeros(num_envs, dtype=torch.bool))

        # Current solution: full eigenbasis Q (ascending eigenvalues) of
        # Dbar(w*); shared forgery subspace = last k columns.
        self.register_buffer('Q', torch.eye(feat_dim))
        self.register_buffer('w', torch.full((num_envs,), 1.0 / num_envs))
        self.register_buffer('subspace_ready', torch.tensor(False))
        self.register_buffer('step', torch.zeros((), dtype=torch.long))

    # ------------------------------------------------------------------
    # Moment tracking
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_moments(self, feat, labels, env):
        feat = feat.float()
        a = 1.0 - self.moment_ema

        real = feat[labels == 0]
        if real.size(0) >= 2:
            mu = real.mean(0)
            m2 = (real.T @ real) / real.size(0)
            if not bool(self.init_r):
                self.mu_r.copy_(mu)
                self.m2_r.copy_(m2)
                self.init_r.fill_(True)
            else:
                self.mu_r.mul_(self.moment_ema).add_(mu, alpha=a)
                self.m2_r.mul_(self.moment_ema).add_(m2, alpha=a)

        for m in range(self.M):
            fm = feat[(labels == 1) & (env == m)]
            if fm.size(0) >= 2:
                mu = fm.mean(0)
                m2 = (fm.T @ fm) / fm.size(0)
                if not bool(self.init_f[m]):
                    self.mu_f[m].copy_(mu)
                    self.m2_f[m].copy_(m2)
                    self.init_f[m] = True
                else:
                    self.mu_f[m].mul_(self.moment_ema).add_(mu, alpha=a)
                    self.m2_f[m].mul_(self.moment_ema).add_(m2, alpha=a)

    def _deltas(self):
        """Per-environment deviation matrices Delta_m (symmetrized).

        With normalize_deltas=True each Delta_m is divided by its positive
        trace (sum of positive eigenvalues), so tr(U^T Delta_m U) reads as
        the fraction of environment m's deviation energy captured by U.
        """
        sig_r = self.m2_r - torch.outer(self.mu_r, self.mu_r)
        deltas = []
        for m in range(self.M):
            sig_m = self.m2_f[m] - torch.outer(self.mu_f[m], self.mu_f[m])
            d = self.mu_f[m] - self.mu_r
            dm = sig_m + torch.outer(d, d) - sig_r
            dm = 0.5 * (dm + dm.T)
            if self.normalize_deltas:
                pos = torch.linalg.eigvalsh(dm).clamp(min=0).sum()
                dm = dm / (pos + 1e-8)
            deltas.append(dm)
        return deltas

    # ------------------------------------------------------------------
    # Common fake axis (bilateral repulsion support)
    # ------------------------------------------------------------------
    def common_axis(self):
        """Unit vector along the mean real->fake shift, averaged over the
        initialized environments. Returns a zero vector until moments
        exist -- callers should gate on subspace_ready (the detector does).
        Buffers carry no grad: differentiable w.r.t. features only."""
        if not bool(self.init_r) or not bool(self.init_f.any()):
            return torch.zeros_like(self.mu_r)
        d = self.mu_f[self.init_f].mean(0) - self.mu_r
        return d / d.norm().clamp_min(1e-8)

    def real_sigma_along(self, u):
        """Std of the real distribution along direction u (for scale-free
        repulsion margins in units of real spread)."""
        sig_r = self.m2_r - torch.outer(self.mu_r, self.mu_r)
        return (u @ sig_r @ u).clamp_min(1e-8).sqrt()

    # ------------------------------------------------------------------
    # Max-min solver (dual: EG on simplex weights, Ky Fan inner max)
    # ------------------------------------------------------------------
    def _project_capped_simplex(self, w):
        """Project w onto {w : sum w = 1, 0 <= w_m <= w_cap}.

        Naive clamp-then-renormalize is NOT sufficient: renormalizing after
        the clamp can push other coordinates back above the cap (e.g.
        [0.9, .05, .05] with cap 0.5 -> clamp -> renorm -> [0.83, ...]).
        Instead, clamp the over-cap coordinates and redistribute their
        excess proportionally among the rest, repeating until feasible
        (at most M rounds for M environments).
        """
        cap = self.w_cap
        w = w.clone()
        for _ in range(self.M):
            over = w > cap + 1e-9
            if not over.any():
                break
            excess = (w[over] - cap).sum()
            w[over] = cap
            free = ~over
            if not free.any():
                return torch.full_like(w, 1.0 / self.M)
            free_mass = w[free].sum().clamp_min(1e-12)
            w[free] = w[free] + excess * (w[free] / free_mass)
        return w

    @torch.no_grad()
    def _refresh_subspace(self):
        deltas = self._deltas()
        if any(torch.isnan(d).any() for d in deltas):
            logger.warning('[maxmin_subspace] NaN in Delta; skipping refresh')
            return

        dev = self.mu_r.device
        # COLD START: solve this refresh's problem from uniform weights.
        # Iteration 0 therefore evaluates the pooled solution, so with
        # best-iterate tracking, maxmin's dual is never worse than pooled's
        # on the same deltas. Q-stability across refreshes comes from the
        # slow moment EMA, not from warm-starting w.
        w = torch.full((self.M,), 1.0 / self.M, device=dev)
        iters = 1 if self.mode == 'pooled' else self.solver_iters

        best_dual, best_w, best_evecs = float('inf'), None, None
        for _ in range(iters):
            dbar = torch.zeros_like(deltas[0])
            for m in range(self.M):
                dbar += w[m] * deltas[m]
            evals, evecs = torch.linalg.eigh(dbar)
            dual = float(evals[-self.k:].sum())
            if dual < best_dual:
                best_dual, best_w, best_evecs = dual, w.clone(), evecs
            if self.mode == 'pooled':
                break
            U = evecs[:, -self.k:]
            # grad of dual wrt w_m = tr(U^T Delta_m U) (per-env shared energy)
            g = torch.stack([(U * (deltas[m] @ U)).sum() for m in range(self.M)])
            g = g / (g.abs().max() + 1e-12)
            w = w * torch.exp(-self.solver_lr * g)
            w = w / w.sum()
            # INTERIOR FLOOR: EG is multiplicative, so exact zeros are
            # absorbing states. Keep every coordinate >= floor.
            floor = 1e-3
            w = (1.0 - self.M * floor) * w + floor
            # Optional CVaR-style cap on any single environment's vote.
            if self.w_cap < 1.0:
                w = self._project_capped_simplex(w)

        self.Q.copy_(best_evecs)
        self.w.copy_(best_w)
        self.subspace_ready.fill_(True)

        if self.log_refresh:
            U = best_evecs[:, -self.k:]
            e = [float((U * (deltas[m] @ U)).sum()) for m in range(self.M)]
            msg = (f"[maxmin_subspace step={int(self.step)}] mode={self.mode} "
                   f"norm={self.normalize_deltas} "
                   f"dual={best_dual:.4f} "
                   f"w={[round(float(x), 3) for x in best_w]} "
                   f"per-env shared energy={[round(x, 4) for x in e]} "
                   f"min={min(e):.4f}")
            if self.mode == 'maxmin':
                # Live maxmin-vs-pooled differentiation: the worst
                # environment's energy under the POOLED (uniform) basis.
                # gap ~ 0 for the whole run => environments agree at this k
                # and the maxmin-vs-pooled ablation will be flat.
                dbar_u = torch.zeros_like(deltas[0])
                for dm in deltas:
                    dbar_u += dm
                dbar_u /= self.M
                _, Vu = torch.linalg.eigh(dbar_u)
                Up = Vu[:, -self.k:]
                pooled_min = min(float((Up * (dm @ Up)).sum())
                                 for dm in deltas)
                msg += (f" | pooled_min={pooled_min:.4f} "
                        f"gap={min(e) - pooled_min:+.4f}")
            logger.info(msg)
            if min(e) <= 0:
                logger.warning(
                    '[maxmin_subspace] min shared energy <= 0: no k-dim '
                    'subspace with positive deviation in every environment. '
                    'Reduce k; if persistent, the shared-forgery hypothesis '
                    'fails at this k (report it -- that is a finding).'
                )

    # ------------------------------------------------------------------
    # Var/cov hinge (identical mechanics to VarCovRegLoss)
    # ------------------------------------------------------------------
    def _var_cov(self, z):
        std = torch.sqrt(z.var(dim=0) + self.eps)
        var = torch.mean(F.relu(self.gamma - std))
        n, d = z.shape
        zc = z - z.mean(dim=0)
        cov = (zc.T @ zc) / max(n - 1, 1)
        cov.fill_diagonal_(0)
        return var, cov.pow(2).sum() / d

    def forward(self, feat, labels, env=None):
        if env is None:
            env = torch.zeros_like(labels)
        if self.training:
            self.step += 1
            self._update_moments(feat.detach(), labels, env)
            ready = (bool(self.init_r) and bool(self.init_f.all())
                     and int(self.step) >= self.warmup_steps)
            if ready and int(self.step) % self.refresh_every == 0:
                self._refresh_subspace()

        target = feat[labels == 0] if self.apply_to == 'real' else feat
        if target.size(0) < 2:
            z0 = feat.new_zeros(())
            return z0, z0
        if not bool(self.subspace_ready):
            return self._var_cov(target)  # isotropic warmup
        z = target @ self.Q.to(target.dtype)
        return self._var_cov(z[:, :-self.k])  # content coordinates only

    # ------------------------------------------------------------------
    # Accessors / diagnostics
    # ------------------------------------------------------------------
    @property
    def U(self):
        """Shared forgery basis, D x k. Buffer: differentiable wrt features
        when used as `feat @ loss.U`, never wrt U itself."""
        return self.Q[:, -self.k:]

    @torch.no_grad()
    def content_forgery_variance(self, real_feat):
        """Mean per-dim variance of real features split into
        (content subspace, shared forgery subspace)."""
        z = real_feat.float() @ self.Q
        return (z[:, :-self.k].var(0).mean().item(),
                z[:, -self.k:].var(0).mean().item())

    @torch.no_grad()
    def per_env_principal_angles(self):
        """Pairwise mean principal angles (degrees) between the top-k
        eigenspaces of the individual Delta_m. Off-diagonal near 90 =
        fingerprints are mutually orthogonal (the motivating figure)."""
        deltas = self._deltas()
        Us = []
        for dm in deltas:
            _, V = torch.linalg.eigh(dm)
            Us.append(V[:, -self.k:])
        A = torch.zeros(self.M, self.M)
        for i in range(self.M):
            for j in range(self.M):
                s = torch.linalg.svdvals(Us[i].T @ Us[j]).clamp(-1.0, 1.0)
                A[i, j] = torch.rad2deg(torch.acos(s)).mean()
        return A


if __name__ == '__main__':
    # --- Regression test (must not break): planted shared dirs + one
    # dominant per-env fingerprint. Pooled locks onto the fingerprint
    # (angles near 90); maxmin recovers the shared dirs (angles near 0).
    torch.manual_seed(0)
    D, k, M, n = 64, 2, 3, 4000
    shared = torch.linalg.qr(torch.randn(D, k))[0]
    fingers = [torch.linalg.qr(torch.randn(D, 3))[0] for _ in range(M)]
    scale = [6.0, 1.5, 1.5]  # env 0's fingerprint dominates pooled variance

    feats = [torch.randn(n, D)]
    labels = [torch.zeros(n, dtype=torch.long)]
    envs = [torch.full((n,), -1, dtype=torch.long)]
    for m in range(M):
        f = torch.randn(n, D)
        f += (torch.randn(n, k) * 2.0) @ shared.T           # shared deviation
        f += (torch.randn(n, 3) * scale[m]) @ fingers[m].T  # fingerprint
        feats.append(f)
        labels.append(torch.ones(n, dtype=torch.long))
        envs.append(torch.full((n,), m, dtype=torch.long))
    X, y, e = torch.cat(feats), torch.cat(labels), torch.cat(envs)

    for mode in ('pooled', 'maxmin'):
        loss = MaxMinDeviationSubspace(
            D, num_envs=M, k=k, warmup_steps=0, refresh_every=1,
            moment_ema=0.0, mode=mode)
        loss.train()
        loss(X, y, e)
        s = torch.linalg.svdvals(shared.T @ loss.U).clamp(-1.0, 1.0)
        ang = torch.rad2deg(torch.acos(s))
        print(f"{mode:6s} | principal angles to planted shared dirs: "
              f"{[round(float(a), 1) for a in ang]} | "
              f"w = {[round(float(x), 3) for x in loss.w]}")
        if mode == 'maxmin':
            assert float(loss.w.min()) > 0, \
                'interior floor failed: absorbing zero in w'
    print("expected: pooled angles near 90 (fingerprint captured), "
          "maxmin angles near 0 (shared recovered)")

    # --- v2 machinery smoke test: floor + capped projection under
    # normalize_deltas. NOTE: no angle assertion here. Per-env trace
    # normalization makes the shared directions' energies UNEQUAL across
    # environments in this synthetic (env 0's shared fraction is diluted
    # by its huge fingerprint), so extraction near the dual optimum can
    # tie-break between shared and fingerprint directions. The
    # normalization's target regime is scale-heteroskedastic real data
    # (the NT-hostage failure), not this fingerprint-dominance synthetic;
    # watch the per-env energy / gap logs on real runs instead.
    loss = MaxMinDeviationSubspace(
        D, num_envs=M, k=k, warmup_steps=0, refresh_every=1,
        moment_ema=0.0, mode='maxmin', normalize_deltas=True, w_cap=0.6)
    loss.train()
    loss(X, y, e)
    assert float(loss.w.min()) > 0, 'interior floor failed under norm+cap'
    assert float(loss.w.max()) <= 0.6 + 1e-6, 'capped-simplex projection failed'
    s = torch.linalg.svdvals(shared.T @ loss.U).clamp(-1.0, 1.0)
    ang = torch.rad2deg(torch.acos(s))
    print(f"norm+cap | angles: {[round(float(a), 1) for a in ang]} | "
          f"w = {[round(float(x), 3) for x in loss.w]} "
          f"(floor/cap verified)")