"""共識分群（Cluster Ensemble）：證據累積 Evidence Accumulation Clustering（Fred & Jain, 2005）。

=== 這個方法是什麼 ===
監督式學習的 ensemble（bagging / random forest）是「很多個弱模型投票」。分群也可以投票，
只是投的不是「這筆屬於第幾群」（不同分群結果的群編號彼此對不上），而是投
「這兩個人是不是同一群」。做法只有三步：

  1. 產生 M 個彼此不同的基礎分群結果（本腳本 M = 30）。
  2. 建 co-association 矩陣 C：C[i, j] = 第 i 人和第 j 人被分到同群的次數 / M。
     C 接近 1 表示「不管用哪種演算法、哪個 k、哪個 seed，這兩人都在一起」。
  3. 把 1 − C 當距離，做一次 average-linkage 階層式分群（HAC），切出最終的共識群。

=== 為什麼「多樣性」很重要 ===
Ensemble 只能平均掉「隨機誤差」，平均不掉「系統性偏誤」。若 30 個成員都是 K-means 只換 seed，
它們會犯同一種錯（永遠切出凸的、大小相近的球狀群），C 矩陣只是把同一個偏見重複 30 次。
更極端的例子：K-modes 用 init="Cao" 是確定性的初始化，換 seed 結果一模一樣
（本腳本每次執行都會實際驗證並寫進 metrics.json 的 kmodes_seed_sensitivity：同 k 換 seed 的 ARI = 1.0；
對照 init="Huang" 則會隨 seed 變）——「換 seed」在這裡完全沒有多樣性。
所以我們刻意混三個家族、三種距離觀點：
  - K-means     ：IDF × 類別加權 → SVD-16 歐氏空間，k ∈ [3, 10] 隨機、seed 各異、n_init=1
  - K-modes     ：原始 0/1，漢明（matching）距離，k ∈ [3, 8] 隨機，init="Cao"
  - HAC-average ：Jaccard 距離（precomputed），k ∈ [3, 10] 隨機
K-modes / HAC 是確定性的，同家族內的多樣性只來自 k；所以抽 k 時先讓每個 k 各出現一次再隨機補，
減少抽到完全相同的成員（完全相同的成員只是把同一票投兩次）。但 K-modes 的 k 只有 6 個值卻要 10 個成員，
重複無法完全避免；metrics.json 的 member_diversity 會如實回報 n_identical_pairs 與去重後的有效成員數。
另外 HAC 在 k 較大時會先把 Jaccard 離群點切成幾人的小群（退化切法），這些成員對 C 幾乎沒有貢獻；
每個成員的 degenerate 旗標也一併記錄。
反面教材也會直接出現在結果裡：HAC 的 10 個成員是同一棵樹的不同切法，彼此 ARI 很高，
它們會像一個「加權 10 倍的成員」一樣主導共識——所以一定要看「共識 vs 各家族的 ARI」。

=== 為什麼適合／不適合這份資料 ===
適合：68 個 0/1 標籤沒有唯一「正確的距離」，Jaccard、漢明、加權歐氏各有道理；
ensemble 讓我們不必先押寶其中一種、也不必先猜 k。
不適合／必須誠實：ensemble 降低的是變異（variance），不是偏差；它不能無中生有。
主控分析已證明這份 seed 資料的標籤是獨立隨機生成的（Hopkins ≈ 0.51、silhouette ≈ 0.01）。
所以本腳本的重點不是「找到幾群」，而是示範正確的檢查流程：同一套 ensemble 跑在兩種 null 資料上。

=== 兩種 null（對照組）===
  1. 逐欄打散（evaluation.column_permutation_null）：每一欄獨立打散。保留每個標籤的流行度、
     破壞所有共現。但它連「每人剛好 1 個 lifestyle、1–2 個 dating_goal」這種資料產生器的單選限制
     都破壞了：null 的每人標籤數變異變大、出現只有 5 個標籤的人，這些人在 Jaccard 空間是離群點，
     average-linkage 會把他們切成 2–5 人的小群（退化分割），silhouette 因此被灌水。
     ⇒ 真實 vs 這個 null 的差異，有一部分只是「每人標籤數比較整齊」，不是使用者分群。
  2. 類別區塊打散（本腳本自己實作 category_block_permutation_null）：同一類別的所有欄位用同一個
     列排列一起打散。保留每人在每個類別內的標籤數與類別內互斥關係，只破壞跨類別的共現。
     這才是「有沒有跨類別的使用者分群」的公平對照；真實資料若和它幾乎一樣，就是沒有結構。

=== 怎麼跑 ===
  cd ml && .venv/bin/python experiments/clustering_ensemble.py        （筆電約 2 分鐘）
只讀 seed CSV、不連資料庫、不改任何既有檔案。輸出：
  outputs/clustering_ensemble/metrics.json          所有數字（真實 vs 兩種 null 並列）
  outputs/clustering_ensemble/consensus_labels.npy  3000 筆抽樣的共識標籤（0..k-1，沒有雜訊 -1）
  outputs/clustering_ensemble/subsample_idx.npy     這 3000 筆在原始 10,000 筆中的列索引
  outputs/models/clustering_ensemble_members.joblib 30 個基礎器的標籤、設定與 SVD 模型
  outputs/figures/clustering_ensemble_*.png         熱圖、直方圖、silhouette-vs-k、穩定度-vs-k、成員一致性

=== 怎麼讀結果 ===
- silhouette_coassoc（以 1 − C 為距離）：共識標籤「對 ensemble 自己」有多自洽。它一定偏高，
  因為共識分群就是直接在 1 − C 上最佳化的；null 資料也會是正數。只能拿來比較不同 k，不能拿來宣稱有群。
- silhouette_jaccard（以原始 Jaccard 距離）：共識標籤在「真實特徵空間」是否緊密分離。這才是誠實的指標，
  一定要和 null 的同一個數字並列看。≈ 0 且與 null 無差 ⇒ 沒有結構。最終 k 用這個指標挑。
- 退化分割（表中標 *）：最小群 < 1%（30 人）的分割只是把離群點切出來，不是分群；它的 silhouette 沒有意義。
  挑 k 時優先選非退化的分割。
- 對基礎器平均 ARI（分家族）：共識比較像哪一派。跨家族兩兩 ARI 若接近 0，表示三種距離觀點
  看到的是完全不同的東西——有真結構時，不同演算法應該會找到大致相同的群。
- 「共識群是不是只是某一欄？」：每個標籤欄位對共識標籤的 ARI。若某一欄 ARI 很高，
  共識只是把那一個欄位（例如單選的 lifestyle）抄了一遍，不是發現使用者分群。
- 兩次共識 ARI vs 單一 K-means 換 seed 的 ARI：ensemble 是否比單一模型穩定。
  注意：20/30 個成員（K-modes、HAC）是確定性的，所以共識的穩定度天生偏高，null 上也一樣高
  ——穩定不代表群存在。
- C 值直方圖（真實 vs 兩種 null）：有結構時 C 會雙峰（接近 0 和接近 1）；沒結構時分佈重疊。
  用 Jensen–Shannon 距離量化（0 = 完全相同）。看「真實 vs 類別區塊 null」那一組才公平。
- 熱圖：依共識標籤排序後，有結構時對角線會出現清楚的亮方塊；沒結構時真實和 null 長得一樣。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from kmodes.kmodes import KModes  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402
from sklearn.cluster import AgglomerativeClustering, KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402

from heartlink_ml import data, evaluation, features, plotting  # noqa: E402
from heartlink_ml.config import FIG_DIR, MODEL_DIR, OUTPUT_DIR, SEED, ensure_dirs  # noqa: E402

# plotting 已經設定 Agg 後端與中文字型，之後才 import pyplot 畫自訂圖。
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

MODULE = "clustering_ensemble"
OUT_DIR = OUTPUT_DIR / MODULE
N_SUB = 3000                       # O(n²) 方法一律抽樣 3000（與其他模組同 seed，可互相比較）
M_PER_FAMILY = 10                  # 每個家族 10 個成員，共 M = 30
K_CONSENSUS = list(range(2, 11))   # 共識分群要掃的 k
SEED_B = SEED + 1000               # 第二次共識用的 ensemble seed（抽樣不變，只換基礎器的隨機性）
FAMILY_LABEL = {"kmeans": "K-means(SVD-16)", "kmodes": "K-modes(0/1)", "hac": "HAC-avg(Jaccard)"}
DATASETS = {"real": "真實", "null_col": "null 逐欄打散", "null_block": "null 類別區塊打散"}


# ---------------------------------------------------------------------------
# 0. 第二種 null（基礎層沒有，寫在這裡）
# ---------------------------------------------------------------------------
def category_block_permutation_null(X: np.ndarray, cols, seed=SEED) -> np.ndarray:
    """類別區塊打散：同一類別的所有欄位用同一個列排列一起打散。
    保留：每欄流行度、每人在每個類別內的標籤數（lifestyle 剛好 1 個、dating_goal 1–2 個…）、類別內互斥。
    破壞：跨類別的共現。"""
    rng = np.random.default_rng(seed)
    cat = np.array([data.column_category(c) for c in cols])
    Xn = X.copy()
    for c in np.unique(cat):
        m = np.where(cat == c)[0]
        Xn[:, m] = X[np.ix_(rng.permutation(X.shape[0]), m)]
    return Xn


# ---------------------------------------------------------------------------
# 1. 基礎分群器集合
# ---------------------------------------------------------------------------
def draw_ks(rng: np.random.Generator, lo: int, hi: int, m: int) -> list[int]:
    """在 [lo, hi] 抽 m 個 k：先把每個值各抽一次並打亂，不夠再隨機補。
    這樣確定性的演算法（K-modes Cao、HAC）不會因為抽到同一個 k 而產生完全相同的成員。"""
    base = rng.permutation(np.arange(lo, hi + 1))
    extra = rng.integers(lo, hi + 1, size=max(0, m - len(base)))
    return np.concatenate([base, extra])[:m].astype(int).tolist()


def build_members(X: np.ndarray, cols, rng: np.random.Generator):
    """產生 M = 30 個基礎分群結果。

    回傳 (L, configs, emb, D, svd)：
      L        (M, n) int  每個成員的標籤
      configs  list[dict]  每個成員的家族 / k / seed
      emb      (n, 16)     K-means 用的 SVD-16 空間（之後算 DB / CH 也用它）
      D        (n, n)      Jaccard 距離矩陣（HAC 成員與 silhouette_jaccard 共用）
    """
    Xw, _ = features.weighted_matrix(X, cols)             # IDF × 類別權重
    emb, svd = features.svd_embed(Xw, n_components=16, seed=SEED)
    D = features.jaccard_distance_matrix(X)
    X_u8 = X.astype(np.uint8)

    labels, configs = [], []
    # (a) K-means：歐氏空間、隨機 k、每個成員一次隨機初始化（n_init=1 才有多樣性）
    for k in draw_ks(rng, 3, 10, M_PER_FAMILY):
        s = int(rng.integers(0, 2**31 - 1))
        labels.append(KMeans(n_clusters=k, n_init=1, random_state=s).fit_predict(emb))
        configs.append({"family": "kmeans", "k": k, "seed": s})
    # (b) K-modes：原始 0/1、漢明距離、Cao 初始化（確定性，多樣性只來自 k）
    for k in draw_ks(rng, 3, 8, M_PER_FAMILY):
        s = int(rng.integers(0, 2**31 - 1))
        km = KModes(n_clusters=k, init="Cao", n_init=1, random_state=s, verbose=0)
        labels.append(km.fit_predict(X_u8))
        configs.append({"family": "kmodes", "k": k, "seed": s})
    # (c) HAC-average：Jaccard precomputed、隨機 k（確定性，成員是同一棵樹的不同切法）
    for k in draw_ks(rng, 3, 10, M_PER_FAMILY):
        hac = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels.append(hac.fit_predict(D))
        configs.append({"family": "hac", "k": k, "seed": None})

    L = np.vstack(labels).astype(np.int16)
    for c, lab in zip(configs, L):
        c["n_clusters_found"] = int(len(np.unique(lab)))
        c["smallest_cluster"] = int(np.bincount(lab).min())
        c["degenerate"] = is_degenerate(lab)                  # 最小群 < 1%：只是把離群點切出來
    return L, configs, emb, D, svd


# ---------------------------------------------------------------------------
# 2. 證據累積：co-association 矩陣與共識分群
# ---------------------------------------------------------------------------
def coassociation(L: np.ndarray) -> np.ndarray:
    """C[i, j] = 兩人被分到同群的成員數 / M。用 one-hot 矩陣乘法一次算完，不用雙迴圈。"""
    M, n = L.shape
    C = np.zeros((n, n), dtype=np.float32)
    for lab in L:
        B = (lab[:, None] == np.unique(lab)[None, :]).astype(np.float32)   # (n, k) one-hot
        C += B @ B.T
    C /= M
    np.fill_diagonal(C, 1.0)
    return C


def is_degenerate(labels: np.ndarray) -> bool:
    """最小群 < 1%（且 < 10 人視為一定退化）：只是把離群點切出來，不算分群。"""
    n = len(labels)
    return int(np.bincount(labels).min()) < max(10, int(0.01 * n))


def consensus_sweep(C: np.ndarray, D: np.ndarray, emb: np.ndarray, ks) -> dict:
    """對 1 − C 做 average-linkage HAC，k 從 ks 掃一遍；每個 k 回報兩種 silhouette 與 cluster_quality。"""
    Dc = np.clip(1.0 - C, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(Dc, 0.0)
    out = {}
    for k in ks:
        hac = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        lab = hac.fit_predict(Dc)
        q = evaluation.cluster_quality(D, emb, lab)                       # silhouette_jaccard / DB / CH
        q["silhouette_coassoc"] = float(silhouette_score(Dc, lab, metric="precomputed"))
        q["degenerate"] = is_degenerate(lab)
        q["labels"] = lab
        out[k] = q
    return out


def pick_best_k(sweep: dict, key: str) -> int:
    """優先在非退化分割中挑 key 最大的 k；全部退化才在全部裡挑。"""
    cands = [k for k in sweep if not sweep[k]["degenerate"]] or list(sweep)
    return max(cands, key=lambda k: (sweep[k][key] if sweep[k][key] is not None else -np.inf))


def run_pipeline(X: np.ndarray, cols, ens_seed: int) -> dict:
    """整套流程：基礎器 → C → 共識掃 k。真實、兩種 null、不同 seed 都呼叫這一個函式，保證流程一致。"""
    rng = np.random.default_rng(ens_seed)
    L, configs, emb, D, svd = build_members(X, cols, rng)
    C = coassociation(L)
    sweep = consensus_sweep(C, D, emb, K_CONSENSUS)
    return {"L": L, "configs": configs, "emb": emb, "D": D, "svd": svd, "C": C, "sweep": sweep}


# ---------------------------------------------------------------------------
# 3. 評估用的小工具
# ---------------------------------------------------------------------------
def member_agreement(consensus: np.ndarray, L: np.ndarray, configs) -> dict:
    """共識標籤 vs 每個基礎器的 ARI：平均、分佈、分家族。"""
    aris = np.array([adjusted_rand_score(consensus, lab) for lab in L])
    fam = np.array([c["family"] for c in configs])
    return {
        "mean": float(aris.mean()), "std": float(aris.std()),
        "min": float(aris.min()), "max": float(aris.max()),
        "quantiles_25_50_75": [float(v) for v in np.percentile(aris, [25, 50, 75])],
        "by_family": {f: float(aris[fam == f].mean()) for f in FAMILY_LABEL},
        "per_member": aris.round(4).tolist(),
    }


def member_diversity(L: np.ndarray, configs) -> dict:
    """成員之間兩兩 ARI：越低越多樣。分「同家族內」與「跨家族」看，佐證 docstring 的多樣性論點。"""
    fam = np.array([c["family"] for c in configs])
    M = len(L)
    within, across, dup = [], [], 0
    is_dup = np.zeros(M, dtype=bool)                  # 第 j 個成員是否和更早的某個成員完全相同
    per_family = {f: [] for f in FAMILY_LABEL}
    for i in range(M):
        for j in range(i + 1, M):
            a = adjusted_rand_score(L[i], L[j])
            if a > 0.9999:
                dup += 1
                is_dup[j] = True
            if fam[i] == fam[j]:
                within.append(a)
                per_family[fam[i]].append(a)
            else:
                across.append(a)
    return {
        "mean_pairwise_ari_all": float(np.mean(within + across)),
        "mean_pairwise_ari_within_family": float(np.mean(within)),
        "mean_pairwise_ari_across_family": float(np.mean(across)),
        "per_family_within": {f: float(np.mean(v)) for f, v in per_family.items()},
        "n_identical_pairs": int(dup),
        "n_unique_members": int(M - is_dup.sum()),   # 去重後的有效成員數（重複成員 = 同一票投兩次）
        "n_degenerate_members_by_family": {f: int(sum(bool(c.get("degenerate", False)) for c in configs if c["family"] == f))
                                           for f in FAMILY_LABEL},
    }


def kmodes_seed_sensitivity(X: np.ndarray, k: int = 5) -> dict:
    """驗證 docstring 的說法：K-modes init="Cao" 是確定性的，換 seed 結果一模一樣（ARI = 1）；
    對照 init="Huang"（隨機初始化）才會隨 seed 改變。只跑一組 k=5、兩個 seed，幾秒鐘。"""
    X_u8 = X.astype(np.uint8)
    out = {}
    for init in ("Cao", "Huang"):
        a = KModes(n_clusters=k, init=init, n_init=1, random_state=SEED, verbose=0).fit_predict(X_u8)
        b = KModes(n_clusters=k, init=init, n_init=1, random_state=SEED_B, verbose=0).fit_predict(X_u8)
        out[init] = float(adjusted_rand_score(a, b))
    return {"k": k, "seeds": [SEED, SEED_B], "two_seed_ari_by_init": out}


def coassoc_stats(C: np.ndarray, M: int) -> dict:
    """C 的上三角數值分佈。C 只會是 j/M（j=0..M），直方圖的 bin 直接對齊這些離散值。"""
    n = C.shape[0]
    v = C[np.triu_indices(n, k=1)]
    edges = (np.arange(M + 2) - 0.5) / M
    hist, _ = np.histogram(v, bins=edges)
    return {
        "mean": float(v.mean()), "std": float(v.std()),
        "quantiles_5_25_50_75_95": [float(q) for q in np.percentile(v, [5, 25, 50, 75, 95])],
        "frac_le_0.1": float((v <= 0.1).mean()),
        "frac_ge_0.9": float((v >= 0.9).mean()),
        "frac_confident(<=0.1 or >=0.9)": float(((v <= 0.1) | (v >= 0.9)).mean()),
        "frac_middle(0.2..0.8)": float(((v > 0.2) & (v < 0.8)).mean()),
        "hist_centers": (np.arange(M + 1) / M).round(4).tolist(),
        "hist_counts": hist.tolist(),
    }


def js_distance(st_a: dict, st_b: dict) -> float:
    p = np.array(st_a["hist_counts"], float) + 1e-9
    q = np.array(st_b["hist_counts"], float) + 1e-9
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2))


def single_baselines(emb: np.ndarray, D: np.ndarray, ks, seed: int, with_stability: bool) -> dict:
    """單一模型的對照組：單次 K-means（n_init=1 與 n_init=10 各一個，兩者是不同的模型，分開報）
    與單次 HAC-average，同樣掃 k。
    with_stability=True 時再算單次 K-means 換 seed 的穩定度（只對真實資料算，省時間）。"""
    out = {}
    for k in ks:
        km1_lab = KMeans(n_clusters=k, n_init=1, random_state=seed).fit_predict(emb)
        km_lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(emb)
        hac_lab = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(D)
        row = {
            "kmeans1_labels": km1_lab,
            "kmeans1_silhouette_jaccard": float(silhouette_score(D, km1_lab, metric="precomputed")),
            "kmeans1_degenerate": is_degenerate(km1_lab),
            "kmeans_labels": km_lab,
            "kmeans_silhouette_jaccard": float(silhouette_score(D, km_lab, metric="precomputed")),
            "kmeans_degenerate": is_degenerate(km_lab),
            "hac_labels": hac_lab,
            "hac_silhouette_jaccard": float(silhouette_score(D, hac_lab, metric="precomputed")),
            "hac_degenerate": is_degenerate(hac_lab),
        }
        if with_stability:
            # 單次 K-means 換 seed 的穩定度（5 個 seed 兩兩 ARI 平均）；n_init=1 是「真正的單次」
            row["kmeans_stability_ari_ninit1"] = evaluation.stability_ari(
                lambda s, k=k: KMeans(n_clusters=k, n_init=1, random_state=s).fit_predict(emb), n_runs=5, seed=seed)
            row["kmeans_stability_ari_ninit10"] = evaluation.stability_ari(
                lambda s, k=k: KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(emb), n_runs=5, seed=seed)
        out[k] = row
    return out


def single_column_explanation(labels: np.ndarray, X: np.ndarray, cols, top=5) -> dict:
    """「共識群是不是只是某一欄？」：每個標籤欄位（0/1 二分）對共識標籤的 ARI，取前幾名；
    另外對「每人剛好一個」的單選類別，算類別 argmax 分群對共識的 ARI。"""
    per_col = sorted(((cols[j], float(adjusted_rand_score(labels, X[:, j].astype(int)))) for j in range(X.shape[1])),
                     key=lambda t: -t[1])
    cat = np.array([data.column_category(c) for c in cols])
    single_choice = {}
    for c in np.unique(cat):
        m = np.where(cat == c)[0]
        if np.all(X[:, m].sum(1) == 1):                       # 單選類別
            single_choice[c] = float(adjusted_rand_score(labels, X[:, m].argmax(1)))
    return {"top_columns_by_ari": [{"column": c, "ari": a} for c, a in per_col[:top]],
            "single_choice_category_ari": single_choice}


def cluster_profiles(labels: np.ndarray, X: np.ndarray, cols, top=3) -> list:
    """每個共識群最有代表性的標籤（lift = 群內比例 / 全體比例）。"""
    p_all = np.maximum(X.mean(0), 1e-9)
    out = []
    for u in np.unique(labels):
        m = labels == u
        lift = X[m].mean(0) / p_all
        o = np.argsort(-lift)[:top]
        out.append({"cluster": int(u), "size": int(m.sum()),
                    "top_tags": [{"tag": cols[j], "share": float(X[m][:, j].mean()), "lift": float(lift[j])} for j in o]})
    return out


def strip_labels(sweep: dict) -> dict:
    """metrics.json 不放整條標籤向量，只留數字。"""
    return {int(k): {kk: vv for kk, vv in v.items() if not kk.endswith("labels")} for k, v in sweep.items()}


def row_sum_stats(X: np.ndarray) -> dict:
    s = X.sum(1)
    return {"mean": float(s.mean()), "std": float(s.std()), "min": int(s.min()), "max": int(s.max())}


# ---------------------------------------------------------------------------
# 4. 圖
# ---------------------------------------------------------------------------
def plot_heatmaps(panels, path):
    """panels: list of (C, labels, title)。"""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.6))
    for ax, (C, lab, title) in zip(np.atleast_1d(axes), panels):
        order = np.argsort(lab, kind="stable")
        im = ax.imshow(C[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1, interpolation="antialiased")
        for b in np.cumsum(np.bincount(lab))[:-1]:            # 畫群邊界
            ax.axhline(b, color="white", lw=0.6, alpha=0.7)
            ax.axvline(b, color="white", lw=0.6, alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), fraction=0.02, pad=0.02, label="C[i,j]：30 個基礎器中被分到同群的比例")
    fig.suptitle("Co-association 矩陣（依各自的共識標籤排序；有結構時對角線會出現亮方塊）")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_coassoc_hist(stats: dict, M: int, js_col: float, js_block: float, path):
    centers = np.array(stats["real"]["hist_centers"])
    w = 1.0 / M
    names = list(stats)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, name in enumerate(names):
        h = np.array(stats[name]["hist_counts"], float)
        ax.bar(centers + (i - 1) * w / 3.2, h / h.sum(), width=w / 3.4, label=DATASETS[name], alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("C[i,j]（每一對使用者被分到同群的比例）")
    ax.set_ylabel("配對比例（對數座標）")
    ax.set_title(f"C 值分佈：真實 vs 兩種 null（JS 距離：vs 逐欄 {js_col:.3f}、vs 類別區塊 {js_block:.3f}；0 = 相同）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_member_ari(aris, configs, path):
    colors = {"kmeans": "C0", "kmodes": "C1", "hac": "C2"}
    x = np.arange(len(aris))
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.bar(x, aris, color=[colors[c["family"]] for c in configs])
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={c['k']}" for c in configs], rotation=90, fontsize=7)
    ax.set_ylabel("ARI（共識 vs 該成員）")
    ax.set_title("共識標籤與 30 個基礎器的一致性（依家族著色）")
    ax.legend(handles=[Patch(color=colors[f], label=FAMILY_LABEL[f]) for f in FAMILY_LABEL], fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. 主程式
# ---------------------------------------------------------------------------
def fmt(v, nd=3, star=False):
    if v is None:
        return "—"
    return f"{v:.{nd}f}" + ("*" if star else "")


def main():
    t0 = time.time()
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 資料與抽樣（與其他模組相同 seed，讓 subsample_idx 可互相對照）
    df = data.load_seed()
    cols = data.trait_columns(df)
    X_full = features.binary_matrix(df, cols)
    idx = features.subsample(len(df), N_SUB, SEED)
    Xs = {
        "real": X_full[idx],
        "null_col": evaluation.column_permutation_null(X_full[idx], seed=SEED),
        "null_block": category_block_permutation_null(X_full[idx], cols, seed=SEED),
    }
    X = Xs["real"]
    row_sums = {d: row_sum_stats(Xd) for d, Xd in Xs.items()}
    print(f"資料：{X_full.shape} → 抽樣 {X.shape}")
    for d, rs in row_sums.items():
        print(f"  每人標籤數 {DATASETS[d]:12s}: mean {rs['mean']:.2f} std {rs['std']:.2f} 範圍 [{rs['min']}, {rs['max']}]")
    hopkins = {d: evaluation.hopkins_binary(Xd, seed=SEED) for d, Xd in Xs.items()}
    print("Hopkins（0.5 = 與獨立隨機無異）：" + "｜".join(f"{DATASETS[d]} {h:.3f}" for d, h in hopkins.items()))
    # docstring 的說法要有數字支持：K-modes Cao 換 seed 是否真的一模一樣
    kmodes_seed = kmodes_seed_sensitivity(X)
    ks_ari = kmodes_seed["two_seed_ari_by_init"]
    print(f"K-modes 換 seed 的 ARI（k={kmodes_seed['k']}）：init=Cao {ks_ari['Cao']:.3f}（確定性）｜init=Huang {ks_ari['Huang']:.3f}")

    # ---- 六次完整流程：三種資料 × 兩個 ensemble seed
    runs = {}
    for d in Xs:
        for tag, s in (("A", SEED), ("B", SEED_B)):
            t = time.time()
            runs[(d, tag)] = run_pipeline(Xs[d], cols, s)
            print(f"  {DATASETS[d]} seed {tag}: 30 個基礎器 + 共識 k=2..10 完成（{time.time() - t:.1f}s）")
    A = runs[("real", "A")]
    M = len(A["L"])

    # ---- 挑最終 k：用誠實的 silhouette_jaccard、優先非退化；1−C 的 silhouette 另外報
    k_best = pick_best_k(A["sweep"], "silhouette_jaccard")
    k_best_coassoc = pick_best_k(A["sweep"], "silhouette_coassoc")
    consensus = A["sweep"][k_best]["labels"]
    best_k_null = {d: pick_best_k(runs[(d, "A")]["sweep"], "silhouette_jaccard") for d in ("null_col", "null_block")}

    # ---- (a) 共識品質、(b) 一致性、多樣性、單欄解釋
    quality = {k: v for k, v in A["sweep"][k_best].items() if k != "labels"}
    agreement = {d: member_agreement(runs[(d, "A")]["sweep"][k_best]["labels"], runs[(d, "A")]["L"], runs[(d, "A")]["configs"])
                 for d in Xs}
    diversity = {d: member_diversity(runs[(d, "A")]["L"], runs[(d, "A")]["configs"]) for d in Xs}
    explain = single_column_explanation(consensus, X, cols)
    profiles = cluster_profiles(consensus, X, cols)
    family_vs_single_choice = {}                                        # 每個家族的成員平均對單選類別的 ARI
    cat = np.array([data.column_category(c) for c in cols])
    for c in explain["single_choice_category_ari"]:
        m = np.where(cat == c)[0]
        cat_lab = X[:, m].argmax(1)
        fam = np.array([cfg["family"] for cfg in A["configs"]])
        family_vs_single_choice[c] = {f: float(np.mean([adjusted_rand_score(l, cat_lab) for l in A["L"][fam == f]]))
                                      for f in FAMILY_LABEL}

    # ---- (c) 穩定度：兩次共識 ARI（每個 k、每種資料）vs 單一 K-means 換 seed
    baselines = {d: single_baselines(runs[(d, "A")]["emb"], runs[(d, "A")]["D"], K_CONSENSUS, SEED, with_stability=(d == "real"))
                 for d in Xs}
    stability = {}
    for k in K_CONSENSUS:
        row = {f"consensus_two_seed_ari_{d}": float(adjusted_rand_score(runs[(d, "A")]["sweep"][k]["labels"],
                                                                        runs[(d, "B")]["sweep"][k]["labels"])) for d in Xs}
        row["single_kmeans_ari_ninit1"] = baselines["real"][k]["kmeans_stability_ari_ninit1"]
        row["single_kmeans_ari_ninit10"] = baselines["real"][k]["kmeans_stability_ari_ninit10"]
        stability[int(k)] = row

    # 單一模型在 k_best 的對照列（跟共識用同一把尺：silhouette_jaccard、對基礎器 ARI）
    def single_row(model):
        b = baselines["real"][k_best]
        lab = b[f"{model}_labels"]
        return {
            "silhouette_jaccard": b[f"{model}_silhouette_jaccard"], "degenerate": b[f"{model}_degenerate"],
            "mean_ari_vs_members": float(np.mean([adjusted_rand_score(lab, l) for l in A["L"]])),
            "null_col_silhouette_jaccard": baselines["null_col"][k_best][f"{model}_silhouette_jaccard"],
            "null_col_degenerate": baselines["null_col"][k_best][f"{model}_degenerate"],
            "null_block_silhouette_jaccard": baselines["null_block"][k_best][f"{model}_silhouette_jaccard"],
            "null_block_degenerate": baselines["null_block"][k_best][f"{model}_degenerate"],
        }
    # n_init=1 與 n_init=10 是兩個不同的模型：各自的 silhouette / 對基礎器 ARI 分開算，不共用同一組標籤
    single_rows = {"kmeans_ninit1": single_row("kmeans1"), "kmeans_ninit10": single_row("kmeans"), "hac": single_row("hac")}
    single_rows["kmeans_ninit1"]["stability_ari"] = baselines["real"][k_best]["kmeans_stability_ari_ninit1"]
    single_rows["kmeans_ninit10"]["stability_ari"] = baselines["real"][k_best]["kmeans_stability_ari_ninit10"]
    single_rows["hac"]["stability_ari"] = 1.0                            # 確定性演算法，換 seed 不變

    # ---- (d) 真實 vs null：C 的數值分佈
    cstats = {d: coassoc_stats(runs[(d, "A")]["C"], M) for d in Xs}
    js_col, js_block = js_distance(cstats["real"], cstats["null_col"]), js_distance(cstats["real"], cstats["null_block"])

    # ---- 結論（由數字決定，不預設立場）
    sil_r = quality["silhouette_jaccard"]
    sil_col = runs[("null_col", "A")]["sweep"][k_best]["silhouette_jaccard"]
    sil_block = runs[("null_block", "A")]["sweep"][k_best]["silhouette_jaccard"]
    cross_fam = diversity["real"]["mean_pairwise_ari_across_family"]
    top_col = explain["top_columns_by_ari"][0]
    top_single = max(explain["single_choice_category_ari"].items(), key=lambda t: t[1], default=(None, 0.0))
    notes = []
    if row_sums["null_col"]["std"] > 1.3 * row_sums["real"]["std"]:
        notes.append(f"逐欄打散 null 的每人標籤數 std 從 {row_sums['real']['std']:.2f} 變成 {row_sums['null_col']['std']:.2f}"
                     "（單選限制被破壞），產生 Jaccard 離群點與退化分割；真實 vs 逐欄 null 的差異不能直接讀成「有結構」，"
                     "請看類別區塊 null。")
    if top_single[0] is not None and top_single[1] >= 0.10:
        notes.append(f"共識群主要對應單選類別「{top_single[0]}」（ARI={top_single[1]:.3f}；最相關單欄 {top_col['column']} "
                     f"ARI={top_col['ari']:.3f}）：這是資料產生器「每人剛好一個」的限制被 Jaccard/HAC 抄了一遍，不是使用者分群。")
    if sil_r >= 0.10 and (sil_r - sil_block) >= 0.05 and cross_fam >= 0.20:
        verdict = "found_structure"
        conclusion = (f"共識分群在 k={k_best} 的 silhouette_jaccard={sil_r:.3f}，比類別區塊 null 高 {sil_r - sil_block:.3f}，"
                      f"且三個家族彼此同意（跨家族 ARI={cross_fam:.3f}）：資料有 null 上看不到的結構。")
    elif (sil_r - sil_block) >= 0.02 or cross_fam >= 0.10:
        verdict = "weak_signal"
        conclusion = (f"共識分群只有微弱訊號：silhouette_jaccard={sil_r:.3f}（類別區塊 null {sil_block:.3f}）、"
                      f"跨家族 ARI={cross_fam:.3f}。不足以宣稱有群。")
    else:
        verdict = "no_structure"
        conclusion = (f"共識分群找不到使用者分群：k={k_best} 的 silhouette_jaccard={sil_r:.3f}，與類別區塊 null 的 {sil_block:.3f} "
                      f"幾乎相同（C 分佈 JS={js_block:.3f}）；三個家族彼此幾乎完全不同意（跨家族 ARI={cross_fam:.3f}）。"
                      "這與「標籤獨立隨機生成」的實況一致：ensemble 能降低變異（共識比單次 K-means 穩定），但不能無中生有。")

    # ---- 圖
    plot_heatmaps([
        (A["C"], consensus, f"真實（共識 k={k_best}）"),
        (runs[("null_col", "A")]["C"], runs[("null_col", "A")]["sweep"][best_k_null["null_col"]]["labels"],
         f"null 逐欄打散（共識 k={best_k_null['null_col']}）"),
        (runs[("null_block", "A")]["C"], runs[("null_block", "A")]["sweep"][best_k_null["null_block"]]["labels"],
         f"null 類別區塊打散（共識 k={best_k_null['null_block']}）"),
    ], FIG_DIR / f"{MODULE}_coassoc_heatmap.png")
    plot_coassoc_hist(cstats, M, js_col, js_block, FIG_DIR / f"{MODULE}_coassoc_hist.png")
    sw = {d: runs[(d, "A")]["sweep"] for d in Xs}
    plotting.line_plot(
        K_CONSENSUS,
        {"共識分群（真實）": [sw["real"][k]["silhouette_jaccard"] for k in K_CONSENSUS],
         "共識分群（null 逐欄打散，含退化分割）": [sw["null_col"][k]["silhouette_jaccard"] for k in K_CONSENSUS],
         "共識分群（null 類別區塊打散）": [sw["null_block"][k]["silhouette_jaccard"] for k in K_CONSENSUS],
         "單一 K-means n_init=10（真實）": [baselines["real"][k]["kmeans_silhouette_jaccard"] for k in K_CONSENSUS],
         "單一 HAC-average（真實）": [baselines["real"][k]["hac_silhouette_jaccard"] for k in K_CONSENSUS]},
        "silhouette_jaccard vs k（誠實指標：原始 Jaccard 距離）", "k", "silhouette（Jaccard）",
        FIG_DIR / f"{MODULE}_silhouette_jaccard_vs_k.png")
    plotting.line_plot(
        K_CONSENSUS,
        {f"共識分群（{DATASETS[d]}）": [sw[d][k]["silhouette_coassoc"] for k in K_CONSENSUS] for d in Xs},
        "silhouette 以 1−C 為距離 vs k（對 ensemble 自己的自洽度，null 也是正數）", "k", "silhouette（1−C）",
        FIG_DIR / f"{MODULE}_silhouette_coassoc_vs_k.png")
    plotting.line_plot(
        K_CONSENSUS,
        {**{f"共識：兩次 seed 的 ARI（{DATASETS[d]}）": [stability[k][f"consensus_two_seed_ari_{d}"] for k in K_CONSENSUS] for d in Xs},
         "單一 K-means n_init=1：5 seeds 兩兩 ARI": [stability[k]["single_kmeans_ari_ninit1"] for k in K_CONSENSUS],
         "單一 K-means n_init=10：5 seeds 兩兩 ARI": [stability[k]["single_kmeans_ari_ninit10"] for k in K_CONSENSUS]},
        "穩定度 vs k（穩定 ≠ 有群：null 的共識一樣穩定）", "k", "ARI",
        FIG_DIR / f"{MODULE}_stability_vs_k.png")
    plot_member_ari(agreement["real"]["per_member"], A["configs"], FIG_DIR / f"{MODULE}_member_ari.png")

    # ---- 存檔
    np.save(OUT_DIR / "consensus_labels.npy", consensus.astype(np.int16))
    np.save(OUT_DIR / "subsample_idx.npy", idx)
    joblib.dump({
        "svd": A["svd"], "member_labels": A["L"], "member_configs": A["configs"],
        "consensus_labels": consensus, "k_best": k_best, "subsample_idx": idx, "trait_columns": cols,
        "note": "co-association 矩陣 C 可由 member_labels 用 coassociation() 重建，不另外存 36 MB 的矩陣。",
    }, MODEL_DIR / f"{MODULE}_members.joblib")

    metrics = {
        "module": MODULE, "n_subsample": int(N_SUB), "seed": SEED, "seed_B": SEED_B, "n_members": int(M),
        "members": A["configs"],
        "row_sum_stats": row_sums,
        "hopkins_binary": hopkins,
        "kmodes_seed_sensitivity": kmodes_seed,
        "k_best_by_silhouette_jaccard": int(k_best),
        "k_best_by_silhouette_coassoc": int(k_best_coassoc),
        "k_best_null_by_silhouette_jaccard": best_k_null,
        "consensus_quality_at_k_best": quality,
        "consensus_quality_null_at_same_k": {d: {k: v for k, v in runs[(d, "A")]["sweep"][k_best].items() if k != "labels"}
                                             for d in ("null_col", "null_block")},
        "consensus_sweep": {d: strip_labels(sw[d]) for d in Xs},
        "agreement_consensus_vs_members": agreement,
        "member_diversity": diversity,
        "consensus_explained_by_single_column": explain,
        "family_vs_single_choice_category_ari": family_vs_single_choice,
        "consensus_cluster_profiles": profiles,
        "stability_by_k": stability,
        "single_model_rows_at_k_best": single_rows,
        "single_model_silhouette_jaccard_by_k": {
            d: {int(k): {"kmeans_ninit1": baselines[d][k]["kmeans1_silhouette_jaccard"],
                         "kmeans_ninit10": baselines[d][k]["kmeans_silhouette_jaccard"],
                         "hac": baselines[d][k]["hac_silhouette_jaccard"]}
                for k in K_CONSENSUS} for d in Xs},
        "coassociation_distribution": {**cstats, "jensen_shannon_real_vs_null_col": js_col,
                                       "jensen_shannon_real_vs_null_block": js_block},
        "verdict": verdict, "conclusion": conclusion, "notes": notes,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    evaluation.save_json(metrics, OUT_DIR / "metrics.json")

    # ---- 摘要表
    swc, swb = sw["null_col"][k_best], sw["null_block"][k_best]
    st = stability[k_best]
    print(f"\n### 共識分群（Evidence Accumulation，M=30）摘要｜抽樣 3000｜最終 k={k_best}（silhouette_jaccard、非退化優先）\n")
    print("| 方法 | k | silhouette_jaccard | 對基礎器平均ARI | 兩次共識ARI | null 逐欄 silhouette | null 類別區塊 silhouette |")
    print("|---|---|---|---|---|---|---|")
    print(f"| 共識分群 EAC（K-means+K-modes+HAC 各 10）| {k_best} | {fmt(sil_r, star=quality['degenerate'])} | "
          f"{fmt(agreement['real']['mean'])} | {fmt(st['consensus_two_seed_ari_real'])}"
          f"（null：{fmt(st['consensus_two_seed_ari_null_col'])} / {fmt(st['consensus_two_seed_ari_null_block'])}）| "
          f"{fmt(sil_col, star=swc['degenerate'])} | {fmt(sil_block, star=swb['degenerate'])} |")
    r = single_rows["kmeans_ninit1"]
    print(f"| 單一 K-means（SVD-16, n_init=1）| {k_best} | {fmt(r['silhouette_jaccard'], star=r['degenerate'])} | "
          f"{fmt(r['mean_ari_vs_members'])} | {fmt(r['stability_ari'])}（5 seeds）| "
          f"{fmt(r['null_col_silhouette_jaccard'], star=r['null_col_degenerate'])} | "
          f"{fmt(r['null_block_silhouette_jaccard'], star=r['null_block_degenerate'])} |")
    r = single_rows["kmeans_ninit10"]
    print(f"| 單一 K-means（SVD-16, n_init=10）| {k_best} | {fmt(r['silhouette_jaccard'], star=r['degenerate'])} | "
          f"{fmt(r['mean_ari_vs_members'])} | {fmt(r['stability_ari'])}（5 seeds）| "
          f"{fmt(r['null_col_silhouette_jaccard'], star=r['null_col_degenerate'])} | "
          f"{fmt(r['null_block_silhouette_jaccard'], star=r['null_block_degenerate'])} |")
    r = single_rows["hac"]
    print(f"| 單一 HAC-average（Jaccard）| {k_best} | {fmt(r['silhouette_jaccard'], star=r['degenerate'])} | "
          f"{fmt(r['mean_ari_vs_members'])} | 1.000（確定性）| "
          f"{fmt(r['null_col_silhouette_jaccard'], star=r['null_col_degenerate'])} | "
          f"{fmt(r['null_block_silhouette_jaccard'], star=r['null_block_degenerate'])} |")
    print("\n`*` = 退化分割（最小群 < 1%，只是把離群點切出來，silhouette 沒有意義）")

    print("\n| 診斷 | 真實 | null 逐欄打散 | null 類別區塊打散 |")
    print("|---|---|---|---|")
    print(f"| Hopkins（0.5 = 隨機）| {fmt(hopkins['real'])} | {fmt(hopkins['null_col'])} | {fmt(hopkins['null_block'])} |")
    print(f"| 每人標籤數 std | {row_sums['real']['std']:.2f} | {row_sums['null_col']['std']:.2f} | {row_sums['null_block']['std']:.2f} |")
    print(f"| silhouette 以 1−C 為距離（k={k_best}）| {fmt(quality['silhouette_coassoc'])} | "
          f"{fmt(swc['silhouette_coassoc'])} | {fmt(swb['silhouette_coassoc'])} |")
    print(f"| 最佳 k（silhouette_jaccard / 1−C）| {k_best} / {k_best_coassoc} | "
          f"{best_k_null['null_col']} / {pick_best_k(sw['null_col'], 'silhouette_coassoc')} | "
          f"{best_k_null['null_block']} / {pick_best_k(sw['null_block'], 'silhouette_coassoc')} |")
    print(f"| 共識群大小（k={k_best}）| {quality['cluster_sizes']} | {swc['cluster_sizes']} | {swb['cluster_sizes']} |")
    print("| C 值平均 ± 標準差 | " + " | ".join(f"{cstats[d]['mean']:.3f} ± {cstats[d]['std']:.3f}" for d in Xs) + " |")
    print("| C 落在 0.2–0.8（不確定配對）比例 | " + " | ".join(f"{cstats[d]['frac_middle(0.2..0.8)']:.3f}" for d in Xs) + " |")
    print("| C ≥ 0.9（強共識配對）比例 | " + " | ".join(f"{cstats[d]['frac_ge_0.9']:.4f}" for d in Xs) + " |")
    print(f"| C 分佈 JS 距離（vs 真實）| — | {js_col:.3f} | {js_block:.3f} |")
    print("| 共識 vs 各家族平均 ARI（K-means / K-modes / HAC）| " + " | ".join(
        " / ".join(f"{agreement[d]['by_family'][f]:.3f}" for f in FAMILY_LABEL) for d in Xs) + " |")
    print("| 成員兩兩 ARI：同家族內 / 跨家族 | " + " | ".join(
        f"{diversity[d]['mean_pairwise_ari_within_family']:.3f} / {diversity[d]['mean_pairwise_ari_across_family']:.3f}" for d in Xs) + " |")
    print("| 完全相同的成員配對數 / 去重後有效成員數（M=30）| " + " | ".join(
        f"{diversity[d]['n_identical_pairs']} / {diversity[d]['n_unique_members']}" for d in Xs) + " |")
    print("| 退化成員數（K-means / K-modes / HAC，最小群 < 1%）| " + " | ".join(
        " / ".join(str(diversity[d]["n_degenerate_members_by_family"][f]) for f in FAMILY_LABEL) for d in Xs) + " |")
    print(f"| K-modes 換 seed ARI（k={kmodes_seed['k']}；Cao / Huang）| "
          f"{ks_ari['Cao']:.3f} / {ks_ari['Huang']:.3f} | — | — |")

    print("\n共識群是不是只是某一欄？（真實資料，k=%d）" % k_best)
    print("  最相關單欄 ARI：" + "、".join(f"{t['column']}={t['ari']:.3f}" for t in explain["top_columns_by_ari"][:3]))
    for c, a in explain["single_choice_category_ari"].items():
        fam_txt = "、".join(f"{FAMILY_LABEL[f]} {v:.3f}" for f, v in family_vs_single_choice[c].items())
        print(f"  單選類別 {c}：共識 ARI={a:.3f}；各家族成員平均 ARI：{fam_txt}")
    for p in profiles:
        print(f"  群{p['cluster']}（{p['size']} 人）：" + "、".join(f"{t['tag']} {t['share']:.0%}（lift {t['lift']:.1f}）" for t in p["top_tags"]))
    for n_ in notes:
        print(f"\n注意：{n_}")
    print(f"\n結論：{conclusion}")
    print(f"\n輸出：{OUT_DIR}/metrics.json、consensus_labels.npy、subsample_idx.npy；圖在 {FIG_DIR}/{MODULE}_*.png；"
          f"總耗時 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
