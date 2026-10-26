"""Compare the learned view-cluster weight matrix W (Eq. 10) with a
ground-truth reference W_gt built from the true labels.

W_gt uses the model's own sharpened assignments: per-sample assignment
entropy (Eq. 8) is pooled per ground-truth class instead of per predicted
cluster (Eq. 9), then mapped through the same Eq. 10 normalization (same
sigma). Agreement is measured with Spearman's rank correlation (+p-value)
over the flattened V x K entries, both raw and after Hungarian alignment
of the learned cluster columns to the true classes.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from configs import CONFIGS
from data import load_dataset
from metrics import clustering_metrics
from train import train_dmvc_cdl


def gt_cluster_weights(ps, y, sigma):
    """Ground-truth view-cluster weights: Eq. 8-9 with pooling by true
    label, Eq. 10 mapping with the same sigma. Returns V x K numpy."""
    V, K = len(ps), ps[0].shape[1]
    classes, y_idx = np.unique(y, return_inverse=True)
    ent = [-(p * torch.log(p.clamp_min(1e-8))).sum(dim=1) for p in ps]   # Eq. 8
    e = torch.zeros(V, K, device=ps[0].device)
    for c in range(len(classes)):
        mask = torch.as_tensor(y_idx == c, device=ps[0].device)
        for v in range(V):
            e[v, c] = ent[v][mask].mean()                                # Eq. 9
    w = torch.exp(-e / sigma)
    return (w / w.sum(dim=0, keepdim=True)).cpu().numpy()                # Eq. 10


def align_columns(W, labels, y):
    """Reorder columns of W so that the learned cluster matched to true
    class c becomes column c (Hungarian on the pred/true confusion)."""
    labels = np.asarray(labels).astype(np.int64)
    classes, y_idx = np.unique(y, return_inverse=True)
    K = len(classes)
    conf = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_idx, labels):
        conf[p, t] += 1
    row, col = linear_sum_assignment(conf.max() - conf)
    W_al = np.zeros_like(W)
    for p, t in zip(row, col):
        W_al[:, t] = W[:, p]
    return W_al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["Scene15", "YTF-10", "uci-digit"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/w_spearman.csv")
    args = ap.parse_args()

    out_csv = Path(__file__).parent / args.out
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["dataset", "run", "ACC", "NMI",
              "rho_raw", "p_raw", "rho_aligned", "p_aligned"]
    if not out_csv.exists():
        with open(out_csv, "w", newline="") as f:
            csv.writer(f).writerow(header)

    all_rows = []
    for name in args.datasets:
        cfg = CONFIGS[name]
        views, y = load_dataset(name)
        K = len(np.unique(y))
        print(f"\n=== {name}: {len(views)} views, N={len(y)}, K={K} ===", flush=True)
        for r in range(args.runs):
            seed = args.seed0 + r
            t0 = time.time()
            labels, W, ps = train_dmvc_cdl(views, K, cfg, seed=seed,
                                           device=args.device, return_ps=True)
            W = W.cpu().numpy()
            W_gt = gt_cluster_weights(ps, y, cfg.sigma)
            rho_raw, p_raw = spearmanr(W.ravel(), W_gt.ravel())
            rho_al, p_al = spearmanr(align_columns(W, labels, y).ravel(),
                                     W_gt.ravel())
            m = clustering_metrics(y, labels)
            row = {"dataset": name, "run": seed, "ACC": round(m["ACC"], 4),
                   "NMI": round(m["NMI"], 4),
                   "rho_raw": round(rho_raw, 4), "p_raw": p_raw,
                   "rho_aligned": round(rho_al, 4), "p_aligned": p_al}
            all_rows.append(row)
            with open(out_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=header).writerow(row)
            print(f"  run {seed}: ACC={m['ACC']:.4f} NMI={m['NMI']:.4f} "
                  f"spearman raw={rho_raw:.4f} (p={p_raw:.2e}) "
                  f"aligned={rho_al:.4f} (p={p_al:.2e}) "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    print("\n===== Summary: Spearman(learned W, ground-truth W) =====")
    print(f"{'Dataset':<12} {'rho_raw':>16} {'rho_aligned':>16}")
    for name in args.datasets:
        rows = [r for r in all_rows if r["dataset"] == name]
        if not rows:
            continue
        for key in ("rho_raw", "rho_aligned"):
            vals = [r[key] for r in rows]
            line = f"{name:<12}" if key == "rho_raw" else f"{'':<12}"
            line += f" {np.mean(vals):6.3f}+-{np.std(vals):5.3f}" if key == "rho_raw" \
                else f" {'':>8}{np.mean(vals):6.3f}+-{np.std(vals):5.3f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
