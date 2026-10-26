"""Two-stage optimization of DMVC-CDL (Algorithm 1), CVCL-style training.

Stage I runs every epoch with mini-batch SGD (convex fusion of
reconstruction + assignment alignment, Eq. 15). Every T epochs the
cluster-wise weights are estimated, assignments are fused into refined
global partitions, and Stage II fine-tunes the encoders with the convex
fusion of reconstruction + prototypical contrastive loss (Eq. 16).
"""
import numpy as np
import torch

from model import DMVC_CDLNet, global_partitions, prototypical_loss, stage1_loss


def _seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batches(n, batch_size, rng):
    idx = rng.permutation(n)
    for s in range(0, n - batch_size + 1, batch_size):
        yield idx[s:s + batch_size]


def _fused_with_W(ps, W):
    fused = sum(W[v].unsqueeze(0) * ps[v] for v in range(W.shape[0]))
    return fused / fused.sum(dim=1, keepdim=True)      # Eq. 11


def train_dmvc_cdl(views, n_clusters, cfg, seed=0, device="cuda", verbose=False,
                   fine_tune=True, equal_weight=False, return_ps=False):
    _seed_all(seed)
    rng = np.random.RandomState(seed)
    xs = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in views]
    n = xs[0].shape[0]

    model = DMVC_CDLNet([v.shape[1] for v in views], n_clusters,
                        cfg.dims, cfg.high_dim, cfg.low_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    enc_params = [p for ae in model.aes for p in ae.encoder.parameters()]
    opt_enc = torch.optim.Adam(enc_params, lr=cfg.lr)

    # ---- reconstruction-only pretraining of the autoencoders ----
    for ep in range(cfg.mse_epochs):
        model.train()
        for b in _batches(n, cfg.batch_size, rng):
            xb = [x[b] for x in xs]
            _, dvs, _ = model(xb)
            loss = sum(torch.nn.functional.mse_loss(xb[v], dvs[v])
                       for v in range(len(xs)))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if verbose and (ep + 1) % 50 == 0:
            print(f"  pretrain {ep + 1}/{cfg.mse_epochs} L_R={loss.item():.4f}", flush=True)

    # ---- alternating Stage I / Stage II ----
    labels = None
    self_target = None
    for ep in range(1, cfg.con_epochs + 1):
        model.train()
        ep_loss = 0.0
        for b in _batches(n, cfg.batch_size, rng):
            xb = [x[b] for x in xs]
            loss = stage1_loss(model, xb, cfg,
                               None if self_target is None else self_target[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if verbose and (ep % 50 == 0 or ep == 1):
            print(f"  stage-I {ep}/{cfg.con_epochs} L_fuse={ep_loss:.2f}", flush=True)

        if ep % cfg.T == 0 and not cfg.baseline and fine_tune:
            labels, ps, W, _, fused = global_partitions(model, xs, cfg.sigma)
            if verbose:
                cnt = torch.bincount(labels).sort(descending=True).values
                ent = -(fused.clamp_min(1e-12).log() * fused).sum(1).mean()
                print(f"    alt @ {ep}: clusters={cnt.tolist()} "
                      f"fused_entropy={ent.item():.3f}", flush=True)
            if equal_weight:                       # ablation: uniform view weights
                W = torch.full_like(W, 1.0 / W.shape[0])
                fused = _fused_with_W(ps, W)
                labels = fused.argmax(dim=1)
            if cfg.self_label > 0:
                from model import sharpen
                self_target = sharpen(fused).detach()
            for _ in range(cfg.ft_epochs):
                with torch.no_grad():
                    zs_full = [ae.encoder(x) for ae, x in zip(model.aes, xs)]
                    K = int(labels.max().item()) + 1
                    onehot = torch.nn.functional.one_hot(labels, K).float()
                    counts = onehot.sum(dim=0, keepdim=True).t().clamp_min(1.0)
                    protos = [onehot.t() @ z / counts for z in zs_full]   # Eq. 12
                ep_loss = 0.0
                for b in _batches(n, cfg.batch_size, rng):
                    zs = [ae.encoder(x[b]) for ae, x in zip(model.aes, xs)]
                    loss_f = cfg.lmd2 * prototypical_loss(zs, labels[b],
                                                          cfg.proto_tau, protos)
                    if cfg.lmd2 < 1:
                        loss_f = loss_f + (1 - cfg.lmd2) * sum(
                            torch.nn.functional.mse_loss(ae.decoder(z), x[b])
                            for ae, z, x in zip(model.aes, zs, xs))
                    opt_enc.zero_grad()
                    loss_f.backward()
                    opt_enc.step()
                    ep_loss += loss_f.item()
                loss_f = torch.tensor(ep_loss)
            if verbose:
                print(f"    alternation @ {ep}: L_F={loss_f.item():.4f}", flush=True)

    # ---- final partitions ----
    ps = None
    if cfg.baseline:
        # CVCL-style inference: average the raw per-view softmax assignments
        model.eval()
        with torch.no_grad():
            lbps, _, _ = model(xs)
            labels = torch.stack(lbps).mean(dim=0).argmax(dim=1)
        W = None
    else:
        labels, ps, W, _, _ = global_partitions(model, xs, cfg.sigma)
        if equal_weight:
            W = torch.full_like(W, 1.0 / W.shape[0])
            fused = _fused_with_W(ps, W)
            labels = fused.argmax(dim=1)
    model.train()
    if return_ps:
        return labels.cpu().numpy(), W, ps
    return labels.cpu().numpy(), W
