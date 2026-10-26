"""3D bar charts of the lmd (lambda_1) x lmd2 (lambda_2) parameter analysis,
ACC only. One SEPARATE figure file per dataset (Scene15 / YTF-10 / COIL20 /
uci-digit): x = lambda_1 (Stage-I fusion weight), y = lambda_2 (Stage-II
fusion weight), z = ACC (%) averaged over seeds.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

RES = Path(__file__).parent / "results"
FIGDIR = RES / "figures"
DATASETS = ["Scene15", "YTF-10", "COIL20", "uci-digit"]

rows = list(csv.DictReader(open(RES / "param_analysis_l2.csv")))
lmds = sorted({float(r["lmd"]) for r in rows})
lmd2s = sorted({float(r["lmd2"]) for r in rows})

# mean ACC per (dataset, lmd, lmd2)
acc = {}
for r in rows:
    k = (r["dataset"], float(r["lmd"]), float(r["lmd2"]))
    acc.setdefault(k, []).append(float(r["ACC"]) * 100)

FIGDIR.mkdir(parents=True, exist_ok=True)
for name in DATASETS:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_axes([0.03, 0.03, 0.80, 0.94], projection="3d")
    dz = np.array([[np.mean(acc.get((name, l, s), [np.nan]))
                    for s in lmd2s] for l in lmds])
    X, Y = np.meshgrid(np.arange(len(lmds)), np.arange(len(lmd2s)),
                       indexing="ij")
    dzn = np.nan_to_num(dz)
    colors = plt.cm.viridis((dzn - dzn.min()) / max(dzn.max() - dzn.min(), 1e-9))
    ax.bar3d(X.ravel(), Y.ravel(), np.zeros(dzn.size), 0.6, 0.6,
             dzn.ravel(), color=colors.reshape(-1, 4), shade=True)
    ax.set_xticks(range(len(lmds)), [f"{v:g}" for v in lmds], fontsize=22)
    ax.set_yticks(range(len(lmd2s)), [f"{v:g}" for v in lmd2s], fontsize=22)
    ax.set_zlabel("ACC (%)", fontsize=22)
    ax.tick_params(axis="z", labelsize=18)
    ax.set_xlabel(r"$\lambda_1$ (Stage I)", fontsize=24, labelpad=18)
    ax.set_ylabel(r"$\lambda_2$ (Stage II)", fontsize=24, labelpad=18)
    out = FIGDIR / f"param_analysis_l2_{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    # autocrop uniform white margins (bbox_inches="tight" mis-measures the
    # rotated 3D z-label and clips it, so crop with PIL instead)
    from PIL import Image, ImageChops
    im = Image.open(out).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        pad = 12
        box = (max(box[0] - pad, 0), max(box[1] - pad, 0),
               min(box[2] + pad, im.width), min(box[3] + pad, im.height))
        im.crop(box).save(out)
    print(f"saved {out}")

# print the ACC grid per dataset
for name in DATASETS:
    print(f"\n{name} ACC (rows=lambda_1, cols=lambda_2):")
    print("         " + "".join(f"{s:>8.3g}" for s in lmd2s))
    for l in lmds:
        print(f"{l:>8.3g} " + "".join(
            f"{np.mean(acc.get((name, l, s), [float('nan')])):8.2f}"
            for s in lmd2s))
