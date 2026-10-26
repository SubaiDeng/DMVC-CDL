"""Clustering evaluation metrics: ACC, NMI, PUR, ARI."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _best_map(y_true, y_pred):
    """Hungarian matching of predicted clusters to true labels."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_true)):
        w[y_pred[i], y_true[i]] += 1
    row, col = linear_sum_assignment(w.max() - w)
    mapping = {r: c for r, c in zip(row, col)}
    return np.array([mapping[p] for p in y_pred])


def clustering_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    mapped = _best_map(y_true, y_pred)
    acc = np.mean(mapped == y_true)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    # Purity
    w = {}
    for t, p in zip(y_true, y_pred):
        w[(p, t)] = w.get((p, t), 0) + 1
    cluster_sum = {}
    for (p, t), c in w.items():
        cluster_sum[p] = max(cluster_sum.get(p, 0), c)
    pur = sum(cluster_sum.values()) / len(y_true)
    return {"ACC": acc, "NMI": nmi, "PUR": pur, "ARI": ari}
