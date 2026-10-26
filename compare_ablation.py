"""Compare ablation cases against the full (tuned) DMVC-CDL (case 3).

case 1: no fine-tuning, with view-cluster weighting   (results_ablation_noft.csv)
case 2: equal view weighting, with fine-tuning        (results_ablation_equalw.csv)
case 3: full DMVC-CDL with the per-dataset Stage-II configs selected by the
        ablation study (results_v3*.csv + stage2_*.csv)
"""
import csv
from pathlib import Path

import numpy as np

RES = Path(__file__).parent / "results"
CASE1 = RES / "results_ablation_noft.csv"
CASE2 = RES / "results_ablation_equalw.csv"
# per-dataset source for the tuned full model
CASE3 = {
    "MSRC-v1": RES / "results_v3.csv",
    "COIL20": RES / "stage2_gr0.csv",        # gamma_rec=0
    "Scene15": RES / "stage2_gr0.5.csv",     # gamma_rec=0.5
    "uci-digit": RES / "stage2_gr0.5.csv",   # gamma_rec=0.5
    "YTF-10": RES / "results_v3.csv",
    "noisyMNIST": RES / "stage2_gr0.csv",    # gamma_rec=0
}
METRICS = ["ACC", "NMI", "PUR", "ARI"]
DATASETS = ["MSRC-v1", "COIL20", "Scene15", "uci-digit", "YTF-10", "noisyMNIST"]


def load(path):
    rows = {}
    if path.exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.setdefault(r["dataset"], []).append(r)
    return rows


def mean_of(rows, key):
    return np.mean([float(r[key]) for r in rows]) * 100


c1, c2 = load(CASE1), load(CASE2)
c3 = {ds: load(p).get(ds, []) for ds, p in CASE3.items()}

print(f"{'Dataset':<11}|" + "|".join(f" {m:^31}" for m in METRICS))
print("-" * (11 + 33 * 4))
wins = {m: [0, 0, 0] for m in METRICS}
for name in DATASETS:
    line = f"{name:<11}"
    for m in METRICS:
        vals = [mean_of(c1[name], m) if c1.get(name) else None,
                mean_of(c2[name], m) if c2.get(name) else None,
                mean_of(c3[name], m) if c3.get(name) else None]
        best = max(v for v in vals if v is not None)
        marks = ["*" if (v is not None and abs(v - best) < 1e-9) else " "
                 for v in vals]
        for i, v in enumerate(vals):
            if v is not None and abs(v - best) < 1e-9:
                wins[m][i] += 1
        fmt = lambda v: f"{v:5.2f}" if v is not None else "  -- "
        line += f"|{fmt(vals[0])}{marks[0]}/{fmt(vals[1])}{marks[1]}/{fmt(vals[2])}{marks[2]} "
    print(line)
print("\n(* = best;  order: case1(no-FT)/case2(equal-w)/case3(full))")
for m in METRICS:
    print(f"  {m}: case3 best on {wins[m][2]}/6 datasets "
          f"(case1 {wins[m][0]}/6, case2 {wins[m][1]}/6)")
