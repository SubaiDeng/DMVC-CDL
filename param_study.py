"""Parameter analysis on YTF-10: top up every (lmd, lmd2) cell to 5 seeds.

Existing runs in results/param_analysis_l2.csv are kept; only the missing
seeds of each cell are run (contiguous blocks, --seed0/--runs).
"""
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DATASETS = ["YTF-10"]
TARGET = 5
LMDS = [0.001, 0.005, 0.01, 0.05, 0.1]
LMD2S = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
OUT = Path(__file__).parent / "results" / "param_analysis_l2.csv"

have = {}
if OUT.exists():
    with open(OUT) as f:
        for r in csv.DictReader(f):
            k = (r["dataset"], float(r["lmd"]), float(r["lmd2"]))
            have.setdefault(k, set()).add(int(r["run"]))

for name in DATASETS:
    for lmd in LMDS:
        for lmd2 in LMD2S:
            done = have.get((name, lmd, lmd2), set())
            missing = [s for s in range(100) if s not in done][:max(0, TARGET - len(done))]
            if not missing:
                print(f"--- skip {name} lmd={lmd} lmd2={lmd2} "
                      f"({len(done)} runs)", flush=True)
                continue
            # group contiguous missing seeds into blocks
            blocks, start, prev = [], missing[0], missing[0]
            for s in missing[1:]:
                if s == prev + 1:
                    prev = s
                else:
                    blocks.append((start, prev - start + 1))
                    start = prev = s
            blocks.append((start, prev - start + 1))
            for seed0, n in blocks:
                cmd = [sys.executable, "run.py", "--datasets", name,
                       "--runs", str(n), "--seed0", str(seed0),
                       "--out", str(OUT),
                       "--lmd", str(lmd), "--lmd2", str(lmd2)]
                print(f">>> {name} lmd={lmd} lmd2={lmd2} "
                      f"seeds {seed0}..{seed0 + n - 1}", flush=True)
                subprocess.run(cmd, check=True,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.STDOUT)
print("done", flush=True)
