"""
FreqDebias detector for DeepfakeBench.

Drop this file in training/detectors/freqdebias_detector.py and add a matching YAML
config (see freqdebias.yaml). Registers as module_name='freqdebias'.

Interface mirrors core_detector.py exactly:
  features / classifier / get_losses / get_train_metrics / get_test_metrics / forward
with data_dict carrying 'image' and 'label', and pred_dict carrying 'cls'/'prob'/'feat'.

Faithful to FreqDebias (arXiv:2509.22412). Reimplementation from equations; the authors
released no official code, so validate against Table 4 before trusting the numbers.
Overall loss (Eq.13):
    L = L_cls + eta*L_CAM + delta*L_att + mu*L_cls_sphere + rho*L_sphere

Key design points the paper is explicit about:
  - Backbone: ResNet-34, ImageNet-pretrained (ConvNeXt/ResNet-50 also work; Table 5).
  - A pretrained copy of the detector scores bands for OHEM inside Fo-Mixup.
  - Auxiliary feature-alignment branches w_i after each stage feed the vMF classifier.
  - All auxiliary heads are REMOVED at test time (no inference-time cost).
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
    # angle in [0, 2pi)
    angle = torch.atan2(yy.float(), xx.float()) % (2 * torch.pi)
 
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
    T = values.shape[0]
    # init centers at quantiles for stability
    qs = torch.linspace(0, 1, k, device=values.device)
    centers = torch.quantile(values, qs)
    assign = torch.zeros(T, dtype=torch.long, device=values.device)
    for _ in range(iters):
        d = (values[:, None] - centers[None, :]).abs()  # [T, k]
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
# These are the DOMINANT frequency components the detector is leaning on.
# ----------------------------------------------------------------------------------
@torch.no_grad()
def ohem_select_mask(x, amp, phase, masks, label, detector, top_t):
    """
    x      : [C, H, W] single image (for shape only)
    amp    : [C, H, W] amplitude of x
    phase  : [C, H, W] phase of x
    masks  : list of k boolean masks [H, W]
    label  : scalar long
    detector: callable image[1,C,H,W] -> logits[1,2], in eval mode
    Returns: a single chosen boolean mask [H, W]
    """
    C, H, W = x.shape
    losses = []
    for mask in masks:
        m = mask.to(amp.dtype)[None]                    # [1,H,W] broadcast over C
        amp_band = amp * m                              # keep only this band
        comp = amp_band * torch.exp(1j * phase)
        filt = torch.fft.ifft2(torch.fft.ifftshift(comp, dim=(-2, -1))).real
        logit = detector(filt[None])                    # [1,2]
        loss = F.cross_entropy(logit, label.view(1))
        losses.append(loss)
    losses = torch.stack(losses)                        # [k]
    top_idx = torch.topk(losses, min(top_t, len(masks))).indices
    chosen = top_idx[torch.randint(len(top_idx), (1,), device=x.device)].item()
    return masks[chosen]
 
 
# ----------------------------------------------------------------------------------
# Full Fo-Mixup for one (xi, xj) pair (Alg.1, lines 12-14; Eqs. 1-2).
# ----------------------------------------------------------------------------------
def fo_mixup_pair(xi, xj, label_i, detector, *,
                  n_radial=4, n_angular=8, k=8, top_t=3, xi_interp=None):
    """
    xi, xj  : [C, H, W] two images of the SAME class (paper mixes within forgery class;
              for your symmetric-real RQ1, call with two real images and a real label).
    detector: eval-mode callable for OHEM scoring (the pretrained detector).
    Returns : x_ij [C, H, W], the augmented image.
 
    Eq.1 :  A_hat = A(xi) * B  +  [(1-xi)A(xi) + xi A(xj)] * (1-B)
    Eq.2 :  x_ij = IFFT[ (p_A * A_hat) dot exp(-i·P(xi)) ]    (original phase preserved)
    """
    C, H, W = xi.shape
    device = xi.device
    if xi_interp is None:
        xi_interp = torch.rand(1, device=device).item()   # ξ ~ U[0,1] per paper
 
    Fi = torch.fft.fftshift(torch.fft.fft2(xi), dim=(-2, -1))
    Fj = torch.fft.fftshift(torch.fft.fft2(xj), dim=(-2, -1))
    amp_i, phase_i = Fi.abs(), Fi.angle()
    amp_j = Fj.abs()
 
    # dominant-band identification on channel-averaged log amplitude
    amp_log = torch.log1p(amp_i).mean(0)                  # [H, W]
    seg_id = build_angular_radial_masks(H, W, n_radial, n_angular, device)
    T = n_radial * n_angular
    masks = cluster_segments_to_masks(amp_log, seg_id, T, k)
    B = ohem_select_mask(xi, amp_i, phase_i, masks, label_i, detector, top_t)
    B = B.to(amp_i.dtype)[None]                           # [1,H,W]
 
    # Eq.1 amplitude mixing: keep xi's dominant band, mix the rest with xj
    amp_mix = (1 - xi_interp) * amp_i + xi_interp * amp_j
    amp_hat = amp_i * B + amp_mix * (1 - B)
 
    # p_A multiplicative perturbation on amplitude. MUST stay non-negative: a negative
    # multiplier flips the sign of the complex coefficient, i.e. injects a pi phase shift,
    # which would create exactly the kind of structural artifact RQ1 forbids. Clamp at 0.
    # (Paper writes N(1,0) in text; the variance is small. std is configurable -- verify.)
    p_A = (torch.randn_like(amp_hat) * 0.1 + 1.0).clamp_min(0.0) 
    amp_hat = amp_hat * p_A
 
    comp = amp_hat * torch.exp(1j * phase_i)              # original phase preserved
    x_ij = torch.fft.ifft2(torch.fft.ifftshift(comp, dim=(-2, -1))).real
    return x_ij.clamp(0, 1) if xi.max() <= 1.0 else x_ij
 
 
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
        # feat: [B, C, h, w] -> CAM [B, n, h, w]
        return self.fc(feat)
 
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
        # flatten spatial, softmax with temperature, JSD
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
    Mixture of two vMF distributions (one per class) on the unit hypersphere.
    Learnable orientation mu_i and concentration kappa_i. Provides:
      - cls_sphere loss (Eq.9): CE on vMF posterior (trains mu, kappa)
      - sphere loss   (Eq.12): 1 - DMS between x^s and x^t per-class vMF params
    DMS uses a KL surrogate between the fitted distributions (Eq.11).
    """
    def __init__(self, feat_dim, num_classes=2):
        super().__init__()
        self.mu = nn.Parameter(F.normalize(torch.randn(num_classes, feat_dim), dim=1))
        self.log_kappa = nn.Parameter(torch.zeros(num_classes))  # kappa = exp(.) >= 0
        self.num_classes = num_classes
 
    def _normalize(self, f):
        return F.normalize(f, dim=1)
 
    def log_vmf(self, f_unit):
        # unnormalized log density per class up to C_d(kappa): kappa * f.mu^T
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

        Parameters for each domain are estimated per-batch via MLE:
          - mean direction μ = normalize(mean(f_i))
          - concentration κ via the approximation A_d(κ) ≈ 1−(d−1)/(2κ),
            solved to κ = (d−1)/(2(1−r̄)) where r̄ = ||mean(f_i)||.

        KL(vMF(μ_s,κ_s) || vMF(μ_t,κ_t)) is computed with the large-κ Bessel
        approximation log I_v(κ) ≈ κ − ½log(2πκ), giving:
          KL ≈ (d−1)/2·log(κ_s/κ_t) + (κ_t−κ_s) + r̄_s·(κ_s − κ_t·cos θ)
        where r̄_s = A_d(κ_s) by the MLE identity.
        """
        d = feat_cat_s.shape[1]
        loss = feat_cat_s.new_zeros(())
        count = 0
        for c in range(self.num_classes):
            fs = self._normalize(feat_cat_s[label == c])  # [n_c, d] on unit sphere
            ft = self._normalize(feat_cat_t[label == c])
            if fs.numel() == 0 or ft.numel() == 0:
                continue

            # MLE resultant vectors and lengths
            rs_vec = fs.mean(0)                                    # [d]
            rt_vec = ft.mean(0)
            r_s = rs_vec.norm().clamp(1e-6, 1 - 1e-6)            # r̄_s in (0,1)
            r_t = rt_vec.norm().clamp(1e-6, 1 - 1e-6)

            mu_s = F.normalize(rs_vec, dim=0)                     # MLE mean directions
            mu_t = F.normalize(rt_vec, dim=0)

            # MLE concentrations: κ = (d−1)/(2(1−r̄))
            kap_s = ((d - 1) / (2.0 * (1.0 - r_s))).clamp(max=60.0)
            kap_t = ((d - 1) / (2.0 * (1.0 - r_t))).clamp(max=60.0)

            cos_theta = (mu_s * mu_t).sum().clamp(-1.0, 1.0)

            # log C_d(κ_s)/C_d(κ_t) ≈ (d−1)/2·log(κ_s/κ_t) + (κ_t−κ_s)
            log_c_ratio = 0.5 * (d - 1) * (kap_s / kap_t).log() + (kap_t - kap_s)
            # A_d(κ_s) = r̄_s  by the MLE property
            kl = (log_c_ratio + r_s * (kap_s - kap_t * cos_theta)).clamp(min=0.0)

            dms = 1.0 / (1.0 + kl)
            loss = loss + (1.0 - dms)
            count += 1
        return loss / max(count, 1)


@DETECTOR.register_module(module_name="freqdebias")
class FreqDebiasDetector(AbstractDetector):
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

        # Multi-stage alignment heads ω_i (paper Sec.3): one per ResNet stage.
        # Each maps stage-i spatial features → out_per_stage channels, then GAP → vector.
        # Concatenating all stages gives F_cat of dim cat_dim (paper: Conv(Concat(F_i))).
        stage_channels = config.get("stage_channels", [64, 128, 256, 512])
        cat_dim = config.get("cat_dim", 512)
        out_per_stage = cat_dim // len(stage_channels)
        actual_cat_dim = out_per_stage * len(stage_channels)  # may differ if cat_dim%4 != 0

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
        )

        # frozen pretrained scorer for OHEM band selection (filled by load_pretrained_scorer)
        scorer_ckpt = config.get("scorer_checkpoint")
        if scorer_ckpt:
            scorer_config = {k: v for k, v in config.items() if k != "scorer_checkpoint"}
            scorer = FreqDebiasDetector(scorer_config)
            scorer.load_state_dict(torch.load(scorer_ckpt, map_location="cpu", weights_only=False))
            self.load_pretrained_scorer(scorer)
        else:
            self.scorer = None

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
        return None  # losses are assembled inline in get_losses (multi-term, Eq.13)

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

    # ----- forward: shared network over original AND synthesized views --------------
    def forward(self, data_dict, inference=False):
        x = data_dict["image"]
        if inference or self.scorer is None or not self.training:
            feat = self.features({"image": x})
            logit = self.classifier(feat)
            return {"cls": logit, "prob": torch.softmax(logit, 1)[:, 1], "feat": feat}

        label = data_dict["label"]

        # ----- RQ1: build synthesized view via Fo-Mixup (amplitude-only, phase kept)
        scorer_fn = lambda im: self.scorer.classifier(self.scorer.features({"image": im}))
        with torch.no_grad():
            x_s = fo_mixup_batch(x, label, scorer_fn, **self.fm_kwargs)
            x_s, x_t, label = self._confidence_sample(x_s, x, label, scorer_fn)

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

    def _forward_backbone(self, x):
        """Run backbone stage-by-stage, collecting multi-scale features (paper Fig.2).

        Returns:
            feat     : [B, 512, h, w]  last-stage spatial map  (for CAM and classifier)
            cat_feat : [B, cat_dim]    concatenated stage vectors  (for vMF branches)
        Auxiliary alignment heads are only active during training; they add no cost at
        inference because the inference path calls self.features() / self.classifier().
        """
        h = self.backbone.resnet[:4](x)   # stem: conv1 + bn + relu + maxpool
        stage_vecs = []
        for i in range(4):
            h = self.backbone.resnet[4 + i](h)
            fi = F.adaptive_avg_pool2d(self.align_heads[i](h), 1).flatten(1)
            stage_vecs.append(fi)
        return h, torch.cat(stage_vecs, dim=1)

    @torch.no_grad()
    def _confidence_sample(self, x_s, x_t, label, scorer_fn):
        """Eq. confidence sampling: keep top-lambda fraction of LOW-entropy synth samples."""
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