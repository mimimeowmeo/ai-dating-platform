"""文字推薦的共用評估工具。

目的：讓「TF-IDF / 關鍵字 / 主題模型 / 句子嵌入 / LLM 抽取」幾種把 bio 變成向量的方法，
用**同一把尺**比較。核心問題只有兩個：
  1. 冷啟動：新使用者只寫了 bio、還沒勾標籤時，光靠文字能不能找到「他勾了標籤之後會被推的那群人」？
     → neighbor_recovery()：以「標籤相似度的前 k 位鄰居」當正解，算文字相似度的 recall@k。
  2. 互補：文字相似度跟標籤相似度重疊多少？剩下的部分是雜訊還是標籤沒抓到的東西？
     → pair_rank_correlation()：隨機抽配對，算兩種相似度的 Spearman 相關。

記憶體提醒：n×n 的 float32 相似度矩陣在 n=3000 約 36 MB、n=10000 約 400 MB；評估請用 features.subsample(n, 3000)。
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

from .config import SEED
from .evaluation import gini


def cosine_sim(A, B=None) -> np.ndarray:
    """稀疏或稠密矩陣都可；回傳 float32 稠密相似度矩陣。"""
    return cosine_similarity(A, B, dense_output=True).astype(np.float32)


def zscore(S: np.ndarray) -> np.ndarray:
    """把相似度矩陣標準化，讓不同來源（標籤 vs 文字）的分數能加權相加。"""
    return ((S - S.mean()) / (S.std() + 1e-9)).astype(np.float32)


def blend(S_a: np.ndarray, S_b: np.ndarray, w_a: float) -> np.ndarray:
    """混合分數 = w_a·z(S_a) + (1-w_a)·z(S_b)。"""
    return (w_a * zscore(S_a) + (1.0 - w_a) * zscore(S_b)).astype(np.float32)


def _topk_rows(S_rows: np.ndarray, row_ids: np.ndarray, k: int) -> np.ndarray:
    """對每一列取前 k 大的欄索引，並排除「自己」那一欄。S_rows 的第 i 列對應使用者 row_ids[i]。"""
    S = S_rows.astype(np.float32, copy=True)
    S[np.arange(len(row_ids)), row_ids] = -np.inf
    return np.argpartition(-S, k, axis=1)[:, :k]


def neighbor_recovery(S_text: np.ndarray, S_tag: np.ndarray, k: int = 50, seekers=None) -> dict:
    """冷啟動評估：文字相似度找到的前 k 位，有多少也在標籤相似度的前 k 位裡（recall@k）。

    S_text / S_tag：同一批使用者的 n×n 相似度矩陣。seekers：要評估的列索引（預設全部）。
    隨機基準 = k / (n-1)；lift_over_random = recall / 隨機基準。
    """
    n = S_text.shape[0]
    seekers = np.arange(n) if seekers is None else np.asarray(seekers)
    truth = _topk_rows(S_tag[seekers], seekers, k)
    pred = _topk_rows(S_text[seekers], seekers, k)
    recall = np.array([len(set(t) & set(p)) / k for t, p in zip(truth, pred)])
    base = k / (n - 1)
    return {
        "k": k, "n_seekers": int(len(seekers)),
        "recall_at_k": float(recall.mean()), "recall_std": float(recall.std()),
        "random_baseline": float(base), "lift_over_random": float(recall.mean() / base),
    }


def pair_rank_correlation(S_a: np.ndarray, S_b: np.ndarray, n_pairs: int = 200_000, seed: int = SEED) -> dict:
    """隨機抽 n_pairs 組 (i, j)，算兩種相似度的 Spearman 相關。接近 0 = 兩者幾乎獨立。"""
    rng = np.random.default_rng(seed)
    n = S_a.shape[0]
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    m = i != j
    rho, p = spearmanr(S_a[i[m], j[m]], S_b[i[m], j[m]])
    return {"spearman": float(rho), "p_value": float(p), "n_pairs": int(m.sum())}


def exposure_counts(S: np.ndarray, k: int = 10, seekers=None) -> np.ndarray:
    """每位候選出現在所有 seeker 前 k 名的次數（曝光集中度用；配合 evaluation.gini）。"""
    n = S.shape[0]
    seekers = np.arange(n) if seekers is None else np.asarray(seekers)
    top = _topk_rows(S[seekers], seekers, k)
    return np.bincount(top.ravel(), minlength=n)


def exposure_gini(S: np.ndarray, k: int = 10, seekers=None) -> float:
    return gini(exposure_counts(S, k, seekers))
