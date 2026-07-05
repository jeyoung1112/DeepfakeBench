"""
FreqDebias (UPDATE) detector for DeepfakeBench.  Registers as module_name='freqdebias_update'.

This is the corrected copy of freqdebias.py. The original is kept intact so the two can be
A/B'd. Differences from the original:
  1. Fo-Mixup reconstruction is clamped to the SOURCE value range, not a hard [0,1]. With
     data normalized to [-1,1] (mean=std=0.5) the old clamp(0,1) crushed the entire negative
     half of every synthesized image, corrupting the view the consistency losses train on.
  2. Diagnostics: every `debug_log_interval` training iters it logs
       - ||x_s - x_t|| / ||x_t||   (is the augmentation actually non-trivial?)
       - OHEM band-loss spread       (is band selection meaningful or ~random?)
     so a short 10-epoch run tells us *why* it does or doesn't move, not just the AUC.
  3. Scorer checkpoint is loaded with strict=False (OHEM only needs the backbone), with a
     logged report of missing/unexpected keys instead of a hard crash on architecture drift.

Everything else mirrors freqdebias.py / core_detector.py:
  features / classifier / get_losses / get_train_metrics / get_test_metrics / forward
Overall loss (Eq.13):  L = L_cls + eta*L_CAM + delta*L_att + mu*L_cls_sphere + rho*L_sphere
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# Angular/radial partition of the amplitude spectrum into T segments (Alg.1, line 3)
# ----------------------------------------------------------------------------------
def build_angular_radial_masks(h, w, n_radial, n_angular, device):
    """
    Partition the centered frequency plane into n_radial * n_angular = T segments.
    Returns a LongTensor [h, w] assigning every frequency bin to a segment id in [0, T).
    The DC term sits at the center after fftshift.
    """
    cy, cx = h // 2, w // 2
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device) - cy,
        torch.arange(w, device=device) - cx,
        indexing="ij",
    )
    radius = torch.sqrt(xx.float() ** 2 + yy.float() ** 2)
    angle = torch.atan2(yy.float(), xx.float()) % (2 * torch.pi)   # [0, 2pi)

    max_r = radius.max().clamp(min=1.0)
    r_idx = (radius / max_r * n_radial).long().clamp(max=n_radial - 1)
    a_idx = (angle / (2 * torch.pi) * n_angular).long().clamp(max=n_angular - 1)
    seg_id = r_idx * n_angular + a_idx  # [h, w] in [0, T)
    return seg_id


# ----------------------------------------------------------------------------------
# Mean log-scaled spectrum per segment (Alg.1, line 4), then K-means over segments
# into k clusters -> k binary masks (Alg.1, lines 5-6).
# ----------------------------------------------------------------------------------
def _kmeans_1d(values, k, iters=10):
    """Tiny 1-D k-means over scalar segment energies. values: [T]. Returns assignment [T]."""
    qs = torch.linspace(0, 1, k, device=values.device)
    centers = torch.quantile(values, qs)                  # init at quantiles for stability
    assign = torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
    for _ in range(iters):
        d = (values[:, None] - centers[None, :]).abs()    # [T, k]
        assign = d.argmin(dim=1)
        for c in range(k):
            sel = values[assign == c]
            if sel.numel() > 0:
                centers[c] = sel.mean()
    return assign


def cluster_segments_to_masks(amp_log, seg_id, T, k):
    """
    amp_log : [H, W] log-scaled amplitude (single channel or channel-averaged)
    seg_id  : [H, W] segment assignment
    Returns : list of k boolean masks [H, W], one per amplitude cluster.
    """
    seg_means = torch.zeros(T, device=amp_log.device)
    for z in range(T):
        m = seg_id == z
        if m.any():
            seg_means[z] = amp_log[m].mean()
    cluster_of_seg = _kmeans_1d(seg_means, k)          # [T] -> cluster id
    cluster_of_pixel = cluster_of_seg[seg_id]          # [H, W] -> cluster id
    masks = [(cluster_of_pixel == c) for c in range(k)]
    return masks


# ----------------------------------------------------------------------------------
# OHEM band selection (Alg.1, lines 7-11): build k band-isolated images, score each
# with the (pretrained) detector's loss, keep the top-t highest-loss bands, pick one.
# The high-loss bands are the ones the detector struggles to classify in isolation; the
# augmentation PRESERVES those (B) and perturbs the easy/shortcut bands (1-B).
# ----------------------------------------------------------------------------------
@torch.no_grad()
def _score_bands(amp, phase, masks, label, detector):
    """Per-band classification loss of band-isolated reconstructions. Returns [k]."""
    losses = []
    for mask in masks:
        m = mask.to(amp.dtype)[None]                    # [1,H,W] broadcast over C
        comp = (amp * m) * torch.exp(1j * phase)        # keep only this band
        filt = torch.fft.ifft2(torch.fft.ifftshift(comp, dim=(-2, -1))).real
        logit = detector(filt[None])                    # [1,2]
        losses.append(F.cross_entropy(logit, label.view(1)))
    return torch.stack(losses)                          # [k]


@torch.no_grad()
def ohem_select_mask(amp, phase, masks, label, detector, top_t):
    """Pick one boolean mask [H, W] at random from the top-t highest-loss bands."""
    losses = _score_bands(amp, phase, masks, label, detector)
    top_idx = torch.topk(losses, min(top_t, len(masks))).indices
    chosen = top_idx[torch.randint(len(top_idx), (1,), device=amp.device)].item()
    return masks[chosen]


@torch.no_grad()
def ohem_band_losses(xi, label_i, detector, *, n_radial=4, n_angular=8, k=8, **_):
    """Diagnostic: per-band OHEM losses for a single image. A near-zero spread means the
    scorer can't tell the bands apart -> band selection is effectively random.
    Extra Fo-Mixup kwargs (top_t, p_a_std, ...) are accepted and ignored."""
    H, W = xi.shape[-2], xi.shape[-1]
    Fi = torch.fft.fftshift(torch.fft.fft2(xi), dim=(-2, -1))
    amp_i, phase_i = Fi.abs(), Fi.angle()
    amp_log = torch.log1p(amp_i).mean(0)
    seg_id = build_angular_radial_masks(H, W, n_radial, n_angular, xi.device)
    masks = cluster_segments_to_masks(amp_log, seg_id, n_radial * n_angular, k)
    return _score_bands(amp_i, phase_i, masks, label_i, detector)


# ----------------------------------------------------------------------------------
# Full Fo-Mixup for one (xi, xj) pair (Alg.1, lines 12-14; Eqs. 1-2).
# ----------------------------------------------------------------------------------
def fo_mixup_pair(xi, xj, label_i, detector, *,
                  n_radial=4, n_angular=8, k=8, top_t=3, xi_interp=None, p_a_std=0.1):
    """
    xi, xj  : [C, H, W] two images of the SAME class.
    detector: eval-mode callable image[1,C,H,W] -> logits[1,2] for OHEM scoring.
    Returns : x_ij [C, H, W], the augmented image.

    Eq.1 :  A_hat = A(xi) * B  +  [(1-xi)A(xi) + xi A(xj)] * (1-B)
    Eq.2 :  x_ij = IFFT[ (p_A * A_hat) dot exp(-i·P(xi)) ]    (original phase preserved)
    """
    device = xi.device
    if xi_interp is None:
        xi_interp = torch.rand(1, device=device).item()   # ξ ~ U[0,1] per paper
    H, W = xi.shape[-2], xi.shape[-1]

    Fi = torch.fft.fftshift(torch.fft.fft2(xi), dim=(-2, -1))
    Fj = torch.fft.fftshift(torch.fft.fft2(xj), dim=(-2, -1))
    amp_i, phase_i = Fi.abs(), Fi.angle()
    amp_j = Fj.abs()

    # dominant-band identification on channel-averaged log amplitude
    amp_log = torch.log1p(amp_i).mean(0)                  # [H, W]
    seg_id = build_angular_radial_masks(H, W, n_radial, n_angular, device)
    T = n_radial * n_angular
    masks = cluster_segments_to_masks(amp_log, seg_id, T, k)
    B = ohem_select_mask(amp_i, phase_i, masks, label_i, detector, top_t)
    B = B.to(amp_i.dtype)[None]                           # [1,H,W]

    # Eq.1 amplitude mixing: keep xi's selected band, mix the rest with xj
    amp_mix = (1 - xi_interp) * amp_i + xi_interp * amp_j
    amp_hat = amp_i * B + amp_mix * (1 - B)

    # p_A multiplicative perturbation on amplitude. MUST stay non-negative: a negative
    # multiplier flips the sign of the complex coefficient, i.e. injects a pi phase shift,
    # which would create exactly the kind of structural artifact RQ1 forbids. Clamp at 0.
    p_A = (torch.randn_like(amp_hat) * p_a_std + 1.0).clamp_min(0.0)
    amp_hat = amp_hat * p_A

    comp = amp_hat * torch.exp(1j * phase_i)              # original phase preserved
    x_ij = torch.fft.ifft2(torch.fft.ifftshift(comp, dim=(-2, -1))).real
    # FIX: clamp reconstruction outliers back into the SOURCE image's value range. Data here
    # is normalized to [-1,1] (mean=std=0.5); a hard clamp(0,1) would crush the negative half
    # of x_s. Source min/max is range-agnostic ([0,1] or [-1,1]).
    return x_ij.clamp(xi.min(), xi.max())


@torch.no_grad()
def fo_mixup_batch(images, labels, detector, **kwargs):
    """
    Apply Fo-Mixup across a batch. Pairs each sample with a random other sample of the
    SAME label (so real mixes with real, fake with fake). Returns augmented batch.
    """
    out = torch.empty_like(images)
    B = images.shape[0]
    for i in range(B):
        same = (labels == labels[i]).nonzero(as_tuple=True)[0]
        same = same[same != i]
        j = same[torch.randint(len(same), (1,))].item() if len(same) else i
        out[i] = fo_mixup_pair(images[i], images[j], labels[i], detector, **kwargs)
    return out


# ----------------------------------------------------------------------------------
# Class Activation Maps + class-wise normalization (Eqs. 4-5) and JSD match (Eq.6)
# ----------------------------------------------------------------------------------
class CAMConsistency(nn.Module):
    """
    Computes CAMs M = W^T F_M from the last feature map and the classifier weights W,
    class-normalizes them, and matches x^s vs x^t CAMs with Jensen-Shannon divergence.
    """
    def __init__(self, in_channels, num_classes=2, tau=4.0):
        super().__init__()
        self.fc = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False)
        self.inst_norm = nn.InstanceNorm2d(num_classes, affine=True)
        self.tau = tau

    def cam(self, feat):
        return self.fc(feat)                              # [B, C, h, w] -> [B, n, h, w]

    @staticmethod
    def _class_normalize(M):
        # Paper Eq.5: K-means(k=2) per (batch, class) CAM map to separate discriminative
        # (high-response) from non-discriminative pixels; keep the high-response cluster.
        B, n, h, w = M.shape
        flat = M.detach().view(B * n, h * w)            # [B*n, hw] — mask only, no grad
        lo = flat.min(dim=1, keepdim=True).values       # cluster-0 center init
        hi = flat.max(dim=1, keepdim=True).values       # cluster-1 center init
        for _ in range(5):
            in_hi = ((flat - hi).abs() <= (flat - lo).abs()).float()
            in_lo = 1.0 - in_hi
            hi = (flat * in_hi).sum(1, keepdim=True) / in_hi.sum(1, keepdim=True).clamp(1)
            lo = (flat * in_lo).sum(1, keepdim=True) / in_lo.sum(1, keepdim=True).clamp(1)
        M_high = in_hi.view(B, n, h, w)
        return M * M_high

    def forward(self, feat_s, feat_t):
        Ms = self.inst_norm(self._class_normalize(self.cam(feat_s)))
        Mt = self.inst_norm(self._class_normalize(self.cam(feat_t)))
        B, n, h, w = Ms.shape
        ps = F.softmax(Ms.view(B, n, -1) / self.tau, dim=-1)
        pt = F.softmax(Mt.view(B, n, -1) / self.tau, dim=-1)
        m = 0.5 * (ps + pt)
        jsd = 0.5 * F.kl_div(m.log(), ps, reduction="batchmean") \
            + 0.5 * F.kl_div(m.log(), pt, reduction="batchmean")
        return jsd

    def cam_ce(self, feat, label):
        # L_CAM: train the CAM classifier head with CE on GAP-ed logits (Eq. text).
        logit = F.adaptive_avg_pool2d(self.cam(feat), 1).flatten(1)
        return F.cross_entropy(logit, label)


# ----------------------------------------------------------------------------------
# von Mises-Fisher classifier + distribution-matching consistency (Eqs. 7-12)
# ----------------------------------------------------------------------------------
class VMFConsistency(nn.Module):
    """
    Two vMF distributions (one per class) on the unit hypersphere with learnable mu_i and
    kappa_i. Provides cls_sphere loss (Eq.9, trains mu/kappa) and sphere loss (Eq.12,
    1 - DMS between x^s and x^t per-class vMF params).
    """
    def __init__(self, feat_dim, num_classes=2):
        super().__init__()
        self.mu = nn.Parameter(F.normalize(torch.randn(num_classes, feat_dim), dim=1))
        self.log_kappa = nn.Parameter(torch.zeros(num_classes))  # kappa = exp(.) >= 0
        self.num_classes = num_classes

    def _normalize(self, f):
        return F.normalize(f, dim=1)

    def log_vmf(self, f_unit):
        mu = F.normalize(self.mu, dim=1)
        kappa = self.log_kappa.exp().clamp(max=60.0)          # numerical guard
        return kappa[None, :] * (f_unit @ mu.t())              # [B, n]

    def posterior(self, f_unit):
        return F.softmax(self.log_vmf(f_unit), dim=1)

    def cls_sphere_loss(self, feat_cat, label):
        f = self._normalize(feat_cat)
        logits = self.log_vmf(f)                               # acts as class scores
        return F.cross_entropy(logits, label)

    def sphere_consistency(self, feat_cat_s, feat_cat_t, label):
        """
        Eq.11-12: per-class DMS between the synthesized and original vMF distributions.
        Parameters estimated per-batch via MLE (mean direction mu, concentration
        kappa=(d-1)/(2(1-r_bar))), KL via the large-kappa Bessel approximation.
        """
        d = feat_cat_s.shape[1]
        loss = feat_cat_s.new_zeros(())
        count = 0
        for c in range(self.num_classes):
            fs = self._normalize(feat_cat_s[label == c])  # [n_c, d] on unit sphere
            ft = self._normalize(feat_cat_t[label == c])
            if fs.numel() == 0 or ft.numel() == 0:
                continue

            rs_vec = fs.mean(0)                                    # MLE resultant vectors
            rt_vec = ft.mean(0)
            r_s = rs_vec.norm().clamp(1e-6, 1 - 1e-6)
            r_t = rt_vec.norm().clamp(1e-6, 1 - 1e-6)

            mu_s = F.normalize(rs_vec, dim=0)
            mu_t = F.normalize(rt_vec, dim=0)

            kap_s = ((d - 1) / (2.0 * (1.0 - r_s))).clamp(max=60.0)
            kap_t = ((d - 1) / (2.0 * (1.0 - r_t))).clamp(max=60.0)

            cos_theta = (mu_s * mu_t).sum().clamp(-1.0, 1.0)

            log_c_ratio = 0.5 * (d - 1) * (kap_s / kap_t).log() + (kap_t - kap_s)
            kl = (log_c_ratio + r_s * (kap_s - kap_t * cos_theta)).clamp(min=0.0)

            dms = 1.0 / (1.0 + kl)
            loss = loss + (1.0 - dms)
            count += 1
        return loss / max(count, 1)


@DETECTOR.register_module(module_name="freqdebias_update")
class FreqDebiasUpdateDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)

        # loss weights (paper defaults: tau=4, eta=.5, delta=.1, mu=1, rho=.1)
        lw = config.get("loss_weights", {})
        self.eta = lw.get("eta", 0.5)
        self.delta = lw.get("delta", 0.1)
        self.mu = lw.get("mu", 1.0)
        self.rho = lw.get("rho", 0.1)
        self.tau = config.get("tau", 4.0)

        feat_c = config.get("feat_channels", 512)     # ResNet-34 last stage = 512

        # Multi-stage alignment heads w_i (paper Sec.3): one per ResNet stage.
        stage_channels = config.get("stage_channels", [64, 128, 256, 512])
        cat_dim = config.get("cat_dim", 512)
        out_per_stage = cat_dim // len(stage_channels)
        actual_cat_dim = out_per_stage * len(stage_channels)

        self.align_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_per_stage, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_per_stage),
                nn.ReLU(inplace=True),
            )
            for c in stage_channels
        ])

        self.cam_cr = CAMConsistency(feat_c, num_classes=2, tau=self.tau)
        self.vmf_cr = VMFConsistency(actual_cat_dim, num_classes=2)

        # confidence sampling fraction lambda (paper default 0.5)
        self.lam = config.get("confidence_lambda", 0.5)

        # Fo-Mixup hyperparameters
        fm = config.get("fo_mixup", {})
        self.fm_kwargs = dict(
            n_radial=fm.get("n_radial", 4),
            n_angular=fm.get("n_angular", 8),
            k=fm.get("k", 8),
            top_t=fm.get("top_t", 3),
            p_a_std=fm.get("p_a_std", 0.1),
        )

        # diagnostics
        self._dbg_interval = config.get("debug_log_interval", 100)
        self._dbg_step = 0

        # frozen pretrained scorer for OHEM band selection
        scorer_ckpt = config.get("scorer_checkpoint")
        if scorer_ckpt:
            scorer_config = {k: v for k, v in config.items() if k != "scorer_checkpoint"}
            scorer = FreqDebiasUpdateDetector(scorer_config)
            state = torch.load(scorer_ckpt, map_location="cpu", weights_only=False)
            state = state.state_dict() if hasattr(state, "state_dict") else state
            missing, unexpected = scorer.load_state_dict(state, strict=False)
            if missing or unexpected:
                logger.warning(
                    "[freqdebias_update] scorer load: %d missing / %d unexpected keys "
                    "(OHEM only needs the backbone).", len(missing), len(unexpected))
            self.load_pretrained_scorer(scorer)
            logger.info("[freqdebias_update] OHEM scorer loaded from %s", scorer_ckpt)
        else:
            self.scorer = None
            logger.warning("[freqdebias_update] no scorer_checkpoint: Fo-Mixup/consistency "
                           "are DISABLED, training falls back to plain cross-entropy.")

    # ----- backbone / interface plumbing (matches core_detector.py) -----------------
    def build_backbone(self, config):
        backbone_class = BACKBONE[config["backbone_name"]]
        backbone = backbone_class(config["backbone_config"])
        if config.get("pretrained"):
            state = torch.load(config["pretrained"])
            state = {k: v for k, v in state.items() if "fc" not in k}
            backbone.load_state_dict(state, strict=False)
            logger.info("Loaded ImageNet-pretrained backbone.")
        return backbone

    def build_loss(self, config):
        return None  # losses assembled inline in get_losses (multi-term, Eq.13)

    def load_pretrained_scorer(self, scorer):
        """A frozen, eval-mode detector used only for OHEM band scoring in Fo-Mixup."""
        self.scorer = scorer
        for p in self.scorer.parameters():
            p.requires_grad_(False)
        self.scorer.eval()

    def features(self, data_dict):
        return self.backbone.features(data_dict["image"])

    def classifier(self, features):
        return self.backbone.classifier(features)

    def train(self, mode=True):
        """Keep the frozen OHEM scorer in eval mode even when the detector is set to train."""
        super().train(mode)
        if getattr(self, "scorer", None) is not None:
            self.scorer.eval()
        return self

    # ----- forward: shared network over original AND synthesized views --------------
    def forward(self, data_dict, inference=False):
        x = data_dict["image"]
        if inference or self.scorer is None or not self.training:
            feat = self.features({"image": x})
            logit = self.classifier(feat)
            return {"cls": logit, "prob": torch.softmax(logit, 1)[:, 1], "feat": feat}

        label = data_dict["label"]

        # ----- build synthesized view via Fo-Mixup (amplitude-only, phase kept)
        scorer_fn = lambda im: self.scorer.classifier(self.scorer.features({"image": im}))
        with torch.no_grad():
            x_s = fo_mixup_batch(x, label, scorer_fn, **self.fm_kwargs)
            x_s, x_t, label = self._confidence_sample(x_s, x, label, scorer_fn)
            self._log_diagnostics(x_s, x_t, label, scorer_fn)

        # ----- shared backbone forward on both views (multi-stage, single pass each)
        feat_t, cat_t = self._forward_backbone(x_t)
        feat_s, cat_s = self._forward_backbone(x_s)
        logit_t = self.classifier(feat_t)
        logit_s = self.classifier(feat_s)

        return {
            "cls": logit_t,
            "prob": torch.softmax(logit_t, 1)[:, 1],
            "feat": feat_t,
            "logit_s": logit_s, "logit_t": logit_t,
            "feat_s": feat_s, "feat_t": feat_t,
            "cat_s": cat_s, "cat_t": cat_t,
            "label_eff": label,
        }

    @torch.no_grad()
    def _log_diagnostics(self, x_s, x_t, label, scorer_fn):
        """Every dbg_interval iters: is x_s a non-trivial augmentation, and is OHEM band
        selection meaningful? Cheap (one image scored for the OHEM spread)."""
        step = self._dbg_step
        self._dbg_step += 1
        if self._dbg_interval <= 0 or step % self._dbg_interval != 0:
            return
        num = (x_s - x_t).flatten(1).norm(dim=1)
        den = x_t.flatten(1).norm(dim=1).clamp_min(1e-6)
        rel = (num / den).mean().item()
        bl = ohem_band_losses(x_t[0], label[0], scorer_fn, **self.fm_kwargs)
        logger.info(
            "[freqdebias_update][diag] step~%d  ||x_s-x_t||/||x_t||=%.4f  "
            "OHEM band-loss spread=%.4f (min=%.3f max=%.3f over %d bands)",
            step, rel, (bl.max() - bl.min()).item(), bl.min().item(), bl.max().item(), bl.numel())

    def _forward_backbone(self, x):
        """Run backbone stage-by-stage, collecting multi-scale features (paper Fig.2).
        Returns last-stage map (for CAM/classifier) and concatenated stage vectors (vMF)."""
        h = self.backbone.resnet[:4](x)   # stem: conv1 + bn + relu + maxpool
        stage_vecs = []
        for i in range(4):
            h = self.backbone.resnet[4 + i](h)
            fi = F.adaptive_avg_pool2d(self.align_heads[i](h), 1).flatten(1)
            stage_vecs.append(fi)
        return h, torch.cat(stage_vecs, dim=1)

    @torch.no_grad()
    def _confidence_sample(self, x_s, x_t, label, scorer_fn):
        """Keep top-lambda fraction of LOW-entropy (highest-confidence) synth samples."""
        logits = scorer_fn(x_s)
        p = torch.softmax(logits, 1)
        entropy = -(p * p.clamp_min(1e-9).log()).sum(1)
        keep = max(1, int(self.lam * x_s.shape[0]))
        idx = torch.topk(-entropy, keep).indices       # lowest entropy = highest conf
        return x_s[idx], x_t[idx], label[idx]

    # ----- losses (Eq.13) -----------------------------------------------------------
    def get_losses(self, data_dict, pred_dict):
        if "logit_s" not in pred_dict:                  # inference / eval path
            loss = F.cross_entropy(pred_dict["cls"], data_dict["label"])
            return {"overall": loss}

        y = pred_dict["label_eff"]
        logit_s, logit_t = pred_dict["logit_s"], pred_dict["logit_t"]
        feat_s, feat_t = pred_dict["feat_s"], pred_dict["feat_t"]
        cat_s, cat_t = pred_dict["cat_s"], pred_dict["cat_t"]

        # L_cls (Eq.3): CE(x^s)+CE(x^t)+KL(softened x^s || x^t)
        ce = F.cross_entropy(logit_s, y) + F.cross_entropy(logit_t, y)
        kl = F.kl_div(
            F.log_softmax(logit_s / self.tau, 1),
            F.softmax(logit_t / self.tau, 1),
            reduction="batchmean",
        ) * (self.tau ** 2)
        l_cls = ce + kl

        # L_CAM (train CAM head) + L_att (Eq.6, JSD CAM consistency)
        l_cam = self.cam_cr.cam_ce(feat_t, y) + self.cam_cr.cam_ce(feat_s, y)
        l_att = self.cam_cr(feat_s, feat_t)

        # L_cls_sphere (Eq.9) + L_sphere (Eq.12)
        l_cls_sphere = self.vmf_cr.cls_sphere_loss(cat_t, y) \
            + self.vmf_cr.cls_sphere_loss(cat_s, y)
        l_sphere = self.vmf_cr.sphere_consistency(cat_s, cat_t, y)

        overall = (l_cls + self.eta * l_cam + self.delta * l_att
                   + self.mu * l_cls_sphere + self.rho * l_sphere)
        return {
            "overall": overall, "cls": l_cls, "cam": l_cam,
            "att": l_att, "cls_sphere": l_cls_sphere, "sphere": l_sphere,
        }

    def get_train_metrics(self, data_dict, pred_dict):
        label = pred_dict.get("label_eff", data_dict["label"])
        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach(), pred_dict["cls"].detach())
        return {"acc": acc, "auc": auc, "eer": eer, "ap": ap}

    def get_test_metrics(self):
        pass
