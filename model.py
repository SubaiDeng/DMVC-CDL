"""DMVC-CDL model and losses.

Stage-I components follow the official CVCL implementation (the base method
referenced by the DMVC-CDL paper): CVCL-style autoencoders, shared assignment
mapping with Eq. (2) sharpening, NT-Xent cross-view alignment loss over
cluster distribution vectors (Eq. 3-4) and the balance regularization (Eq. 5).

DMVC-CDL additions on top of CVCL:
- Cluster-wise discriminability from Shannon entropy (Eq. 8-10) and weighted
  fusion of assignments (Eq. 11).
- Multi-view prototypical contrastive fine-tuning with the fused partitions
  as pseudo supervision (Eq. 12-14), alternated with Stage I (Algorithm 1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    """CVCL-style MLP autoencoder; ReLU after every layer (incl. the last)."""

    def __init__(self, input_dim, feature_dim, dims):
        super().__init__()
        enc, prev = [], input_dim
        for d in dims:
            enc += [nn.Linear(prev, d), nn.ReLU()]
            prev = d
        enc += [nn.Linear(prev, feature_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)

        dec, prev = [], feature_dim
        for d in reversed(dims):
            dec += [nn.Linear(prev, d), nn.ReLU()]
            prev = d
        dec += [nn.Linear(prev, input_dim), nn.ReLU()]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


class DMVC_CDLNet(nn.Module):
    def __init__(self, input_sizes, n_clusters, dims, high_dim, low_dim):
        super().__init__()
        self.n_views = len(input_sizes)
        self.aes = nn.ModuleList([AutoEncoder(d, high_dim, dims) for d in input_sizes])
        # shared mapping f_p: two linear layers + softmax (no activation between)
        self.label_module = nn.Sequential(
            nn.Linear(high_dim, low_dim),
            nn.Linear(low_dim, n_clusters),
            nn.Softmax(dim=1),
        )

    def forward(self, xs):
        lbps, dvs, zs = [], [], []
        for v, (ae, x) in enumerate(zip(self.aes, xs)):
            z, x_hat = ae(x)
            zs.append(z)
            lbps.append(self.label_module(z))
            dvs.append(x_hat)
        return lbps, dvs, zs


def sharpen(q):
    """Sharpened normalization of assignments, Eq. (2)."""
    weight = (q ** 2.0) / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()


def alignment_loss(p_i, p_j, tau, sim="cos"):
    """Cross-view contrastive loss on cluster distribution vectors (Eq. 3-4),
    NT-Xent style over the 2K cluster distributions, following CVCL."""
    q_i = sharpen(p_i).t()                      # K x B
    q_j = sharpen(p_j).t()
    K = q_i.shape[0]
    q = torch.cat((q_i, q_j), dim=0)            # 2K x B
    if sim == "cos":
        qn = F.normalize(q, dim=1)
        sim_mat = qn @ qn.t() / tau
    else:
        sim_mat = q @ q.t() / tau
    N = 2 * K
    pos = torch.cat((torch.diag(sim_mat, K), torch.diag(sim_mat, -K)), 0).reshape(N, 1)
    mask = torch.ones((N, N), dtype=torch.bool, device=q.device)
    mask.fill_diagonal_(False)
    for i in range(K):
        mask[i, K + i] = False
        mask[K + i, i] = False
    neg = sim_mat[mask].reshape(N, -1)
    logits = torch.cat((pos, neg), dim=1)
    target = torch.zeros(N, dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, target, reduction="sum") / N


def balance_loss(p):
    """Regularization toward balanced cluster sizes, Eq. (5)."""
    s = p.sum(dim=0)
    s = s / s.sum()
    return (s * torch.log(s.clamp_min(1e-8))).sum()


def stage1_loss(model, xs, cfg, self_target=None):
    """L_fuse = (1-lmd)*L_R + lmd*L_A (Eq. 15), convex weighting:
    (1-lmd)*MSE + lmd*l_align + beta*l_norm over all view pairs.
    If `self_target` (detached sharpened fused assignments) is given, a
    per-view consistency term toward the refined partitions is added."""
    lbps, dvs, _ = model(xs)
    loss = (1.0 - cfg.lmd) * sum(F.mse_loss(xs[v], dvs[v])
                                 for v in range(len(xs)))
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            loss = loss + cfg.lmd * alignment_loss(lbps[i], lbps[j], cfg.tau, cfg.sim)
            loss = loss + cfg.beta * (balance_loss(lbps[i]) + balance_loss(lbps[j]))
    if self_target is not None and cfg.self_label > 0:
        loss = loss + cfg.self_label * sum(
            -(self_target * torch.log(lbps[v].clamp_min(1e-8))).sum(dim=1).mean()
            for v in range(len(xs)))
    return loss


@torch.no_grad()
def global_partitions(model, xs, sigma):
    """Estimate cluster-wise weights (Eq. 8-10), fuse assignments (Eq. 11).

    Returns (pseudo labels, sharpened per-view assignments, weight matrix W,
    view-specific representations, fused assignments)."""
    model.eval()
    lbps, _, zs = model(xs)
    ps = [sharpen(l) for l in lbps]                      # N x K per view
    V, N, K = len(ps), ps[0].shape[0], ps[0].shape[1]
    ent = [-(p * torch.log(p.clamp_min(1e-8))).sum(dim=1) for p in ps]   # Eq. 8
    hard = [p.argmax(dim=1) for p in ps]
    e = torch.zeros(V, K, device=ps[0].device)
    for v in range(V):
        for k in range(K):
            mask = hard[v] == k
            if mask.any():
                e[v, k] = ent[v][mask].mean()                        # Eq. 9
    w = torch.exp(-e / sigma)
    w = w / w.sum(dim=0, keepdim=True)                               # Eq. 10
    fused = sum(w[v].unsqueeze(0) * ps[v] for v in range(V))
    fused = fused / fused.sum(dim=1, keepdim=True)                   # Eq. 11
    labels = fused.argmax(dim=1)
    return labels, ps, w, zs, fused


def prototypical_loss(zs, labels, tau, protos=None):
    """L_F (Eq. 13-14): pull embeddings toward their cluster prototype and
    push away from other prototypes, using the fused partitions as
    pseudo supervision. Prototypes follow Eq. (12); if `protos` is given
    (precomputed from the full data) it is used as-is, otherwise gradients
    flow through the prototype computation."""
    loss = zs[0].new_zeros(())
    if protos is None:
        K = int(labels.max().item()) + 1
        protos = []
        for z in zs:
            onehot = F.one_hot(labels, K).float()
            counts = onehot.sum(dim=0, keepdim=True).t().clamp_min(1.0)
            protos.append(onehot.t() @ z / counts)                         # Eq. 12
    for z, pr in zip(zs, protos):
        s = F.cosine_similarity(z.unsqueeze(1), pr.unsqueeze(0), dim=2) / tau
        loss = loss + F.cross_entropy(s, labels)
    return loss
