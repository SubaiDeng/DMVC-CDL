"""Dataset loading utilities for DMVC-CDL experiments.

Datasets live in ~/workplace/DMVC-CDL/datasets. Each loader returns
(views, labels), where views is a list of N x d_v float32 arrays (each view
standardized to zero mean / unit variance per feature) and labels is an
int array of length N.
"""
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATA_DIR = Path.home() / "workplace" / "DMVC-CDL" / "datasets"


def _standardize(views, scaler="minmax"):
    cls = {"minmax": MinMaxScaler, "standard": StandardScaler}[scaler]
    return [cls().fit_transform(v.astype(np.float32)) for v in views]


def _load_mat(path):
    d = sio.loadmat(path)
    cells = d["X"].ravel()
    views = [cells[i] for i in range(len(cells))]
    y = d["Y"].ravel().astype(np.int64)
    return views, y


def _load_npy_dict(path):
    d = np.load(path, allow_pickle=True).item()
    n_views = len(d) - 1
    views = [d[i] for i in range(n_views)]
    y = np.asarray(d[n_views]).ravel().astype(np.int64)
    return views, y


def load_dataset(name, scaler="minmax"):
    name = name.lower().replace("-", "").replace("_", "")
    if name == "msrcv1":
        views, y = _load_mat(DATA_DIR / "MSRCv1.mat")
    elif name == "coil20":
        views, y = _load_mat(DATA_DIR / "COIL20" / "COIL20.mat")
    elif name == "scene15":
        views, y = _load_mat(DATA_DIR / "Scene15" / "Scene15.mat")
    elif name == "ucidigit":
        views, y = _load_npy_dict(DATA_DIR / "ucidigit.npy")
    elif name == "ytf10":
        views, y = _load_npy_dict(DATA_DIR / "YTF10.npy")
    elif name == "noisymnist":
        views, y = _load_npy_dict(DATA_DIR / "noisyMNIST.npy")
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return _standardize(views, scaler), y


DATASETS = ["MSRC-v1", "COIL20", "Scene15", "uci-digit", "YTF-10", "noisyMNIST"]
