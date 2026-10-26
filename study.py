"""Focused hyperparameter / seed study for Scene15."""
import sys
from copy import deepcopy

import numpy as np

from configs import CONFIGS
from data import load_dataset
from metrics import clustering_metrics
from train import train_dmvc_cdl

views, y = load_dataset("Scene15")
K = 15
S3 = [0, 1, 2]


def run(tag, seeds, **over):
    cfg = deepcopy(CONFIGS["Scene15"])
    for k, v in over.items():
        setattr(cfg, k, v)
    accs = []
    for s in seeds:
        labels, _ = train_dmvc_cdl(views, K, cfg, seed=s, device="cuda")
        accs.append(clustering_metrics(y, labels)["ACC"])
    print(f"{tag:32s}: ACC={np.mean(accs) * 100:6.2f} +- {np.std(accs) * 100:5.2f}  "
          f"{[round(a, 3) for a in accs]}", flush=True)


GROUPS = {
    # CVCL's favorite seeds on the current config (is 44.59 just seed luck?)
    "seeds": lambda: [run(f"seed={s}", [s]) for s in [10, 20, 50]],
    "epochs": lambda: [
        run("con_epochs=200", S3, con_epochs=200),
        run("con_epochs=400", S3, con_epochs=400),
        run("mse_epochs=400", S3, mse_epochs=400),
        run("lr=1e-3", S3, lr=1e-3),
        run("lr=2e-4", S3, lr=2e-4),
    ],
    "lmd": lambda: [
        run("lmd=0.05", S3, lmd=0.05),
        run("lmd=0.1", S3, lmd=0.1),
        run("lmd=0.005", S3, lmd=0.005),
        run("lmd=0.02", S3, lmd=0.02),
    ],
    "beta": lambda: [
        run("beta=0.01", S3, beta=0.01),
        run("beta=0.1", S3, beta=0.1),
        run("beta=0.005", S3, beta=0.005),
        run("beta=0.02", S3, beta=0.02),
    ],
    "arch": lambda: [
        run("high=512,dims=[256,512]", S3, high_dim=512, dims=[256, 512]),
        run("high=1024,dims=[256,512,1024]", S3, high_dim=1024, dims=[256, 512, 1024]),
        run("high=2000,dims=[256,512]", S3, high_dim=2000, dims=[256, 512]),
        run("low=512", S3, low_dim=512),
    ],
    "dmvc": lambda: [
        run("sigma=0.5", S3, sigma=0.5),
        run("sigma=2", S3, sigma=2),
        run("T=5,ft=2", S3, T=5, ft_epochs=2),
        run("proto_tau=0.5", S3, proto_tau=0.5),
        run("lmd2=1.0", S3, lmd2=1.0),
        run("lmd2=0.9", S3, lmd2=0.9),
    ],
}

if __name__ == "__main__":
    GROUPS[sys.argv[1]]()
