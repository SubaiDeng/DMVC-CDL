"""Run DMVC-CDL on the six benchmark datasets, 10 runs each, report mean/std."""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from configs import CONFIGS
from data import DATASETS, load_dataset
from metrics import clustering_metrics
from train import train_dmvc_cdl


def run_dataset(name, n_runs, device, out_csv, baseline=False,
                fine_tune=True, equal_weight=False, overrides=None, seed0=0):
    cfg = CONFIGS[name]
    if baseline:
        cfg = type(cfg)(**{**cfg.__dict__, "baseline": True})
    if overrides:
        cfg = type(cfg)(**{**cfg.__dict__, **overrides})
    views, y = load_dataset(name)
    n_clusters = len(np.unique(y))
    print(f"\n=== {name}: {len(views)} views, N={len(y)}, K={n_clusters} ===", flush=True)
    print(f"    cfg: {cfg}", flush=True)
    rows = []
    for r in range(n_runs):
        t0 = time.time()
        seed = seed0 + r
        labels, W = train_dmvc_cdl(views, n_clusters, cfg, seed=seed, device=device,
                                   fine_tune=fine_tune, equal_weight=equal_weight)
        m = clustering_metrics(y, labels)
        m.update({"dataset": name, "run": seed, "time_s": round(time.time() - t0, 1)})
        extra = [k for k in ("lmd", "lmd2", "sigma") if overrides and k in overrides]
        if extra:
            m.update({k: overrides[k] for k in extra})
        rows.append(m)
        print(f"  run {seed}: ACC={m['ACC']:.4f} NMI={m['NMI']:.4f} "
              f"PUR={m['PUR']:.4f} ARI={m['ARI']:.4f} ({m['time_s']}s)", flush=True)
        with open(out_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=["dataset"] + extra +
                           ["run", "ACC", "NMI", "PUR", "ARI", "time_s"],
                           extrasaction="ignore").writerow(m)
    return rows


def summarize(all_rows):
    print("\n===== Summary (mean +/- std over runs) =====")
    print(f"{'Dataset':<12} {'ACC':>16} {'NMI':>16} {'PUR':>16} {'ARI':>16}")
    summary = {}
    for name in DATASETS:
        rows = [r for r in all_rows if r["dataset"] == name]
        if not rows:
            continue
        vals = {k: [r[k] for r in rows] for k in ["ACC", "NMI", "PUR", "ARI"]}
        line = f"{name:<12}"
        summary[name] = {}
        for k in ["ACC", "NMI", "PUR", "ARI"]:
            mean, std = np.mean(vals[k]), np.std(vals[k])
            summary[name][k] = (mean, std)
            line += f" {mean * 100:6.2f}+-{std * 100:5.2f}"
        print(line, flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/results_v2.csv")
    ap.add_argument("--baseline", action="store_true",
                    help="run the CVCL ablation (no Stage II / no weighting)")
    ap.add_argument("--noft", action="store_true",
                    help="ablation: skip Stage II fine-tuning (keep weighting)")
    ap.add_argument("--equalw", action="store_true",
                    help="ablation: uniform view weights (keep fine-tuning)")
    ap.add_argument("--gamma_rec", type=float, default=None,
                    help="legacy alias: --lmd2 1/(1+gamma_rec)")
    ap.add_argument("--lmd2", type=float, default=None,
                    help="Stage II mixing weight lambda_2")
    ap.add_argument("--proto_tau", type=float, default=None)
    ap.add_argument("--ft_epochs", type=int, default=None)
    ap.add_argument("--lmd", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--seed0", type=int, default=0,
                    help="first random seed (seeds used: seed0 .. seed0+runs-1)")
    args = ap.parse_args()

    overrides = {k: v for k, v in [
        ("proto_tau", args.proto_tau),
        ("ft_epochs", args.ft_epochs), ("lmd", args.lmd),
        ("sigma", args.sigma), ("lmd2", args.lmd2)] if v is not None}
    if args.gamma_rec is not None:
        overrides["lmd2"] = 1.0 / (1.0 + args.gamma_rec)

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["dataset"] + [k for k in ("lmd", "lmd2", "sigma") if k in overrides] + \
        ["run", "ACC", "NMI", "PUR", "ARI", "time_s"]
    if not out_csv.exists():
        with open(out_csv, "w", newline="") as f:
            csv.writer(f).writerow(header)

    all_rows = []
    for name in args.datasets:
        all_rows += run_dataset(name, args.runs, args.device, out_csv, args.baseline,
                                fine_tune=not args.noft, equal_weight=args.equalw,
                                overrides=overrides, seed0=args.seed0)
    summarize(all_rows)


if __name__ == "__main__":
    main()
