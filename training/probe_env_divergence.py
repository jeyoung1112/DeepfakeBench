"""Probe: cross-environment divergence of per-sample classifier-gradient
variances at a trained optimum.

Question this answers for the paper: does the ERM optimum use a different
decision rule per manipulation environment (divergent per-env gradient
variances), and does the GAIN optimum remove that divergence in the SAME
feature space / architecture?

For one checkpoint, this script runs a fixed set of FF++ TRAINING batches
(augmentation off, fixed shuffle seed -> identical frames for every
checkpoint) through the model in eval mode, computes the closed-form
per-sample head gradient exactly as in training,

    g_n = [ (softmax(z_n) - y~_n) (outer) a_n ; softmax(z_n) - y~_n ],

with the training label smoothing, accumulates UNBIASED per-environment
variances v_e over fake environments (reals excluded, as in training), and
reports:

  rel_divergence      mean_e sum_j ((v_ej - vbar_j) / beta_j)^2 with
                      beta = vbar + rho*mean(vbar) + delta.  Scale-free
                      (degree-zero), hence comparable ACROSS checkpoints.
                      This is the paper's penalty statistic evaluated at
                      the optimum.
  vanilla_divergence  mean_e sum_j (v_ej - vbar_j)^2.  Scale-bound; printed
                      for reference only -- NOT comparable across models.

Standalone read-only probe: no training file is modified, checkpoints are
only loaded.  Run once per checkpoint with the SAME --seed / --max-batches:

  cd ~/DeepfakeBench
  python training/probe_env_divergence.py \
      --config training/config/detector/dualbranchpatch_v5_px_base.yaml \
      --weights <...>/dual_branch_patch_v5_ce/test/Celeb-DF-v2/ckpt_best.pth \
      --tag ERM --json-out logs/env_divergence.jsonl
"""
import argparse
import json
import os
import random
import yaml

import numpy as np
import torch
import torch.nn.functional as F

from detectors import DETECTOR
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset


def _worker_init(worker_id):
    # dataset __getitem__ draws frames with python/numpy RNG; seed both from
    # the torch per-worker seed so the probe is exactly reproducible
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def load_config(detector_yaml):
    # mirrors train.py: detector yaml, then train_config.yaml on top
    with open(detector_yaml, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']
    config.update(config2)
    if config.get('lmdb'):
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    # probe-specific: deterministic inputs, single process
    config['use_data_augmentation'] = False
    config['ddp'] = False
    config['local_rank'] = 0
    return config


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, help='detector yaml of the run')
    ap.add_argument('--weights', required=True, help='ckpt_best.pth path')
    ap.add_argument('--tag', default=None, help='label for the report line')
    ap.add_argument('--max-batches', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=3407,
                    help='loader shuffle seed; keep IDENTICAL across checkpoints')
    ap.add_argument('--json-out', default=None, help='append result as a JSON line')
    args = ap.parse_args()

    config = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # full determinism: sampler order AND in-__getitem__ frame draws
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # data: training set, no augmentation, fixed order across checkpoints
    dataset = DeepfakeAbstractBaseDataset(config=config, mode='train')
    gen = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=gen,
        num_workers=args.workers, collate_fn=dataset.collate_fn, drop_last=True,
        worker_init_fn=_worker_init)

    model = DETECTOR[config['model_name']](config)
    sd = torch.load(args.weights, map_location='cpu')
    if 'state_dict' in sd and isinstance(sd['state_dict'], dict):
        sd = sd['state_dict']
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # tolerate only loss-bookkeeping buffers; a missing weight is fatal
    bad = [k for k in missing if not k.startswith(('fishr_', 'patch_center.'))]
    assert not bad, f'missing weight keys: {bad[:8]}'
    if unexpected:
        print(f'[warn] ignored unexpected keys: {unexpected[:6]}'
              f'{"..." if len(unexpected) > 6 else ""}')
    model.to(device).eval()

    ls = float(config.get('label_smoothing', 0.0))
    envs = list(config.get('fishr_envs', [0, 1, 2, 3]))
    stats = {e: {'n': 0, 's': None, 'sq': None} for e in envs}   # float64 CPU

    bi = -1
    for bi, batch in enumerate(loader):
        if bi >= args.max_batches:
            break
        data_dict = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
        pred = model(data_dict, inference=True)
        z, a = pred['cls'], pred['cls_feat']            # [B,C] logits, [B,H]
        label, env = data_dict['label'], data_dict['env']
        p = torch.softmax(z.float(), dim=1)
        y = F.one_hot(label, p.size(1)).float()
        if ls > 0:
            y = (1 - ls) * y + ls / p.size(1)
        resid = p - y                                    # [B,C]
        g = torch.cat([torch.einsum('bc,bh->bhc', resid, a.float())
                       .reshape(a.size(0), -1), resid], dim=1).double().cpu()
        for e in envs:
            m = (env == e).cpu()
            if m.any():
                ge = g[m]
                st = stats[e]
                st['n'] += ge.size(0)
                st['s'] = ge.sum(0) if st['s'] is None else st['s'] + ge.sum(0)
                st['sq'] = ((ge ** 2).sum(0) if st['sq'] is None
                            else st['sq'] + (ge ** 2).sum(0))
        if (bi + 1) % 50 == 0:
            print(f'  batch {bi + 1}/{args.max_batches}  counts: '
                  + ' '.join(f'e{e}:{stats[e]["n"]}' for e in envs))

    # unbiased per-env variances (matches fishr_unbiased: true)
    vs = []
    for e in envs:
        st = stats[e]
        assert st['n'] >= 2, f'env {e}: only {st["n"]} samples; raise --max-batches'
        mean = st['s'] / st['n']
        vs.append((st['sq'] - st['n'] * mean ** 2) / (st['n'] - 1))
    V = torch.stack(vs, 0)                               # [E, P]
    vbar = V.mean(0)
    rho = float(config.get('fishr_rel_floor', 1e-3))
    delta = float(config.get('fishr_rel_eps', 1e-8))
    beta = vbar + rho * vbar.mean() + delta
    rel_e = (((V - vbar) / beta) ** 2).sum(1)
    vanilla_e = ((V - vbar) ** 2).sum(1)

    # ---- first moment: is each environment individually at rest? ----------
    # gbar_e = mean per-sample head gradient of env e.  Stationarity ratio
    #   R_e = n_e * ||gbar_e||^2 / tr(v_e)
    # is ~1 when env e is at its own optimum (mean gradient indistinguishable
    # from sampling noise) and >>1 when env e's private optimum lies
    # elsewhere.  A model stationary in AGGREGATE (R_total small) but with
    # R_e >> 1 per env sits at a cancellation point BETWEEN per-method
    # optima; R_e ~ 1 everywhere means one shared optimum (simultaneous
    # optimality).  Pairwise cosines between gbar_e expose the tug-of-war.
    ns = [stats[e]['n'] for e in envs]
    gbars = [stats[e]['s'] / stats[e]['n'] for e in envs]
    R_env = {int(e): float(n_e * gb.pow(2).sum() / max(ve.sum().item(), 1e-30))
             for e, gb, n_e, ve in zip(envs, gbars, ns, vs)}
    g_total = sum(n_e * gb for n_e, gb in zip(ns, gbars)) / sum(ns)
    R_total = float(sum(ns) * g_total.pow(2).sum()
                    / max(vbar.sum().item(), 1e-30))
    G = torch.stack(gbars, 0)
    Gn = G / G.norm(dim=1, keepdim=True).clamp_min(1e-30)
    cosM = Gn @ Gn.T
    cos_pairs = {f'{int(a)}-{int(b)}': float(cosM[i, j])
                 for i, a in enumerate(envs)
                 for j, b in enumerate(envs) if i < j}
    cos_mean = float(sum(cos_pairs.values()) / len(cos_pairs))

    out = {
        'tag': args.tag or os.path.basename(os.path.dirname(
            os.path.dirname(os.path.dirname(args.weights)))),
        'weights': args.weights,
        'batches': min(args.max_batches, bi + 1),
        'seed': args.seed,
        'counts': {int(e): stats[e]['n'] for e in envs},
        'rel_divergence': float(rel_e.mean()),
        'rel_per_env': {int(e): float(x) for e, x in zip(envs, rel_e)},
        'vanilla_divergence': float(vanilla_e.mean()),
        'grad_scale_mean_vbar': float(vbar.mean()),
        'stationarity_ratio_per_env': R_env,
        'stationarity_ratio_total': R_total,
        'grad_cosine_pairs': cos_pairs,
        'grad_cosine_mean': cos_mean,
    }
    print('\n================ env-divergence probe ================')
    print(json.dumps(out, indent=2))
    print(f"\nHEADLINE  {out['tag']}: rel_divergence = {out['rel_divergence']:.2f}"
          f"  |  stationarity R_e = "
          + '/'.join(f'{R_env[int(e)]:.1f}' for e in envs)
          + f" (total {R_total:.1f})  |  mean env-gradient cosine = {cos_mean:.2f}")
    if args.json_out:
        with open(args.json_out, 'a') as f:
            f.write(json.dumps(out) + '\n')


if __name__ == '__main__':
    main()
