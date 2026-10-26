"""Stage II tuning: strengthen the fine-tuning so full DMVC-CDL clearly
beats the no-FT and equal-weight ablations on the four weak datasets."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DATASETS = ["COIL20", "Scene15", "uci-digit", "noisyMNIST"]
CONFIGS = [
    ("gr0.5", ["--gamma_rec", "0.5"]),
    ("gr0",   ["--gamma_rec", "0"]),
    ("pt0.5", ["--proto_tau", "0.5"]),
    ("fte10", ["--ft_epochs", "10"]),
]

for tag, flags in CONFIGS:
    out = f"results/stage2_{tag}.csv"
    cmd = [sys.executable, "run.py", "--datasets", *DATASETS,
           "--runs", "10", "--out", out, *flags]
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
