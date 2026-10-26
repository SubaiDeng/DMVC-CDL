"""Diagnose the YTF-10 collapse at lambda1=0.1: run seed 12 with verbose
cluster-size logging for lambda2 in {0.1, 0.3, 0.5} to see where/how the
degenerate partition develops."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from configs import CONFIGS
from data import load_dataset
from metrics import clustering_metrics
from train import train_dmvc_cdl

views, y = load_dataset("YTF-10")
K = len(set(y.tolist()))
for lmd2 in [0.1, 0.3, 0.5]:
    cfg = type(CONFIGS["YTF-10"])(**{**CONFIGS["YTF-10"].__dict__,
                                   "lmd": 0.1, "lmd2": lmd2})
    print(f"\n########## seed=12 lmd=0.1 lmd2={lmd2} ##########", flush=True)
    labels, _ = train_dmvc_cdl(views, K, cfg, seed=12, device="cuda",
                               verbose=True)
    m = clustering_metrics(y, labels)
    print(f"FINAL ACC={m['ACC']:.4f} NMI={m['NMI']:.4f}", flush=True)
