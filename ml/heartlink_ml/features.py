"""把 DataFrame 轉成模型吃的矩陣：二元標籤、IDF 加權、Jaccard 距離、SVD、bio TF-IDF。"""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import CATEGORY_WEIGHTS, SEED
from .data import TRUTHY, column_category


def binary_matrix(df, cols) -> np.ndarray:
    """回傳 (n, len(cols)) 的 bool 矩陣。CSV 裡的 0/1 可能是數字或字串，一律照字面判斷。"""
    return df[cols].astype(str).isin(TRUTHY).to_numpy(dtype=bool)


def idf_weights(X: np.ndarray) -> np.ndarray:
    """越少人有的標籤越有辨識度：idf = ln(N / df)。"""
    n = X.shape[0]
    return np.log(n / np.maximum(X.sum(0), 1)).astype(np.float32)


def category_weights(cols, weights=CATEGORY_WEIGHTS) -> np.ndarray:
    return np.array([weights.get(column_category(c), 1.0) for c in cols], dtype=np.float32)


def weighted_matrix(X: np.ndarray, cols, use_idf=True, use_category=True):
    """回傳 (加權後的 float32 矩陣, 每欄權重)。"""
    w = np.ones(X.shape[1], dtype=np.float32)
    if use_idf:
        w *= idf_weights(X)
    if use_category:
        w *= category_weights(cols)
    return X.astype(np.float32) * w, w


def jaccard_distance_matrix(X: np.ndarray) -> np.ndarray:
    """向量化的 Jaccard 距離矩陣（n×n float32）。n=3000 約 36 MB；n=10000 約 400 MB，請先抽樣。"""
    A = X.astype(np.float32)
    inter = A @ A.T
    s = A.sum(1)
    union = s[:, None] + s[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        D = 1.0 - np.where(union > 0, inter / union, 1.0)
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, 1.0).astype(np.float32)


def svd_embed(Xw: np.ndarray, n_components=16, seed=SEED):
    """TruncatedSVD 降維：給 K-means / GMM 一個比原始 0/1 更合理的歐氏空間，也用來畫 2D 圖。"""
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    return svd.fit_transform(Xw), svd


def bio_tfidf(texts, ngram_range=(1, 2), min_df=3, max_features=20000):
    """中文 bio 用字元 n-gram 做 TF-IDF，不需要斷詞器。"""
    vec = TfidfVectorizer(
        analyzer="char", ngram_range=ngram_range, min_df=min_df,
        max_features=max_features, sublinear_tf=True,
    )
    return vec.fit_transform(texts), vec


def subsample(n: int, size: int, seed=SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(size, n), replace=False))
