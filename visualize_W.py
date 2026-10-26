"""Heatmap visualization of the view-cluster weight deviation W - 1/V.

Equal weighting corresponds to every entry being the average value 1/V,
so we visualize the deviation W - 1/V with a diverging colormap centered
at zero: red = above-average (discriminative) contribution of a view for
a cluster, blue = below-average. Cell values are annotated.

W is averaged over multiple seeds for stability. The averaged W is cached
to results/figures/W_mean_{name}.npy so the figure can be redrawn without
retraining.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from configs import CONFIGS
from data import load_dataset
from train import train_dmvc_cdl

OUT = Path(__file__).parent / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["Scene15", "uci-digit", "YTF-10"]
N_SEEDS = 10


def get_W_mean(name, views, K, fine_tune=True):
    tag = "" if fine_tune else "_noft"
    cache = OUT / f"W_mean_{name.replace('-', '_')}{tag}.npy"
    if cache.exists():
        return np.load(cache)
    cfg = CONFIGS[name]
    Ws = []
    for seed in range(N_SEEDS):
        _, W = train_dmvc_cdl(views, K, cfg, seed=seed, device="cuda",
                              fine_tune=fine_tune)
        Ws.append(np.asarray(W.detach().cpu() if hasattr(W, "detach") else W,
                             dtype=np.float64))
    W_mean = np.mean(Ws, axis=0)          # V x K
    np.save(cache, W_mean)
    return W_mean


def visualize(name, fine_tune=True, vmax=None):
    views, y = load_dataset(name)
    K = len(np.unique(y))
    W_mean = get_W_mean(name, views, K, fine_tune)
    V = W_mean.shape[0]
    D = (W_mean - 1.0 / V).T * 100.0     # K x V, in percent

    if vmax is None:
        vmax = np.abs(D).max() * 1.05
    fig, ax = plt.subplots(figsize=(15, 14))
    im = ax.imshow(D, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    for k in range(K):
        for v in range(V):
            ax.text(v, k, f"{D[k, v]:+.2f}", ha="center", va="center",
                    fontsize=40, color="black")
    ax.set_xticks(range(V), [f"view {v+1}" for v in range(V)], fontsize=30)
    ax.set_yticks(range(K), [f"C{k+1}" for k in range(K)], fontsize=30)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=30)
    fig.tight_layout()
    tag = "" if fine_tune else "_noft"
    path = OUT / f"W_heatmap_{name.replace('-', '_')}{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    mode = "with fine-tuning" if fine_tune else "without fine-tuning"
    print(f"{name} ({mode}): W mean per view={W_mean.mean(axis=1).round(3)}, "
          f"saved {path}", flush=True)
    return W_mean, D


if __name__ == "__main__":
    # shared color scale per dataset (FT and no-FT figures of the same
    # dataset use the same vmax; different datasets may differ)
    for name in DATASETS:
        views, y = load_dataset(name)
        V = len(views)
        vmax = 0.0
        for tag in ("", "_noft"):
            cache = OUT / f"W_mean_{name.replace('-', '_')}{tag}.npy"
            if cache.exists():
                W = np.load(cache)
                vmax = max(vmax, np.abs(W - 1.0 / W.shape[0]).max() * 100.0)
        vmax *= 1.05
        print(f"{name}: shared color scale +/-{vmax:.2f}", flush=True)
        visualize(name, fine_tune=True, vmax=vmax)
        visualize(name, fine_tune=False, vmax=vmax)
