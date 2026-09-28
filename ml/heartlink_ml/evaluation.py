"""評估工具。

分群（沒有正解）：silhouette / Davies-Bouldin / Calinski-Harabasz + Hopkins + null model + 穩定度。
排序／分類（有正解）：AUC、PR-AUC、precision@k、NDCG@k、曝光 Gini。
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score, average_precision_score, calinski_harabasz_score,
    davies_bouldin_score, ndcg_score, roc_auc_score, silhouette_score,
)

from .config import SEED


def cluster_quality(D: np.ndarray, Xfeat: np.ndarray, labels) -> dict:
    """D：Jaccard 距離矩陣（precomputed）；Xfeat：算 DB/CH 用的特徵空間；labels 中 -1 視為雜訊不計。"""
    labels = np.asarray(labels)
    mask = labels >= 0
    uniq = np.unique(labels[mask])
    out = {
        "n_clusters": int(len(uniq)),
        "n_noise": int((~mask).sum()),
        "noise_ratio": float((~mask).mean()),
        "cluster_sizes": sorted((int((labels == u).sum()) for u in uniq), reverse=True),
        "silhouette_jaccard": None, "davies_bouldin": None, "calinski_harabasz": None,
    }
    if len(uniq) >= 2 and mask.sum() > len(uniq):
        sub = np.ix_(mask, mask)
        out["silhouette_jaccard"] = float(silhouette_score(D[sub], labels[mask], metric="precomputed"))
        out["davies_bouldin"] = float(davies_bouldin_score(Xfeat[mask], labels[mask]))
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xfeat[mask], labels[mask]))
    return out


def hopkins_binary(X: np.ndarray, m=500, seed=SEED) -> float:
    """Hopkins 統計量的二元版本：H≈0.5 表示與「各標籤獨立隨機」無法區分，H→1 表示有群結構。
    對照點依每個標籤的實際流行度獨立抽樣，正好對應我們要否證的虛無假設。"""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    p = X.mean(0)
    idx = rng.choice(n, size=min(m, n // 2), replace=False)
    A = X.astype(np.float32)
    s = A.sum(1)

    def nearest(Q, exclude=None):
        inter = Q @ A.T
        union = Q.sum(1)[:, None] + s[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            Dm = 1.0 - np.where(union > 0, inter / union, 1.0)
        if exclude is not None:
            Dm[np.arange(len(Q)), exclude] = np.inf
        return Dm.min(1)

    w = nearest(A[idx], idx)
    fake = (rng.random((len(idx), d)) < p).astype(np.float32)
    u = nearest(fake)
    return float(u.sum() / (u.sum() + w.sum()))


def column_permutation_null(X: np.ndarray, seed=SEED) -> np.ndarray:
    """每一欄獨立打散：保留每個標籤的流行度，破壞標籤之間的共現關係。
    用同一個演算法跑這份資料，得到的指標就是「沒有結構時會看到的數字」。"""
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    for j in range(Xn.shape[1]):
        rng.shuffle(Xn[:, j])
    return Xn


def stability_ari(fit_predict, n_runs=5, seed=SEED):
    """fit_predict(seed) -> labels。回傳不同種子間兩兩 ARI 的平均。
    警告：ARI 高只代表最佳化收斂穩定，不代表群真的存在，一定要配合 silhouette 與 null model 一起看。"""
    labs = [np.asarray(fit_predict(seed + i)) for i in range(n_runs)]
    pairs = [(i, j) for i in range(n_runs) for j in range(i + 1, n_runs)]
    aris = [adjusted_rand_score(labs[i], labs[j]) for i, j in pairs]
    return float(np.mean(aris)) if aris else None


def ranking_metrics(groups, y_true, scores, k=10) -> dict:
    """groups：每筆樣本的 query id（例如 seeker）；每個 group 內算 precision@k / NDCG@k 再平均。"""
    groups, y_true, scores = (np.asarray(a) for a in (groups, y_true, scores))
    precs, ndcgs = [], []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 2 or y_true[m].sum() == 0:
            continue
        order = np.argsort(-scores[m])
        precs.append(float(y_true[m][order][:k].mean()))
        ndcgs.append(float(ndcg_score(y_true[m][None, :], scores[m][None, :], k=k)))
    return {"k": k, "n_groups": len(precs),
            "precision_at_k": float(np.mean(precs)) if precs else None,
            "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else None}


def classification_metrics(y_true, y_score) -> dict:
    return {"roc_auc": float(roc_auc_score(y_true, y_score)),
            "pr_auc": float(average_precision_score(y_true, y_score))}


def gini(x) -> float:
    """曝光集中度：0 = 完全平均，1 = 極度集中。"""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
