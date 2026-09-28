"""監督式排序 + ensemble + 互惠推薦：模擬研究（supervised_reciprocal_ranker）

【開宗明義：這是模擬研究，證明的是 pipeline 正確，不是真實推薦品質】
我們手上沒有可用的真實互動資料（資料庫裡的 1,263 筆 likes 全來自 33 個測試帳號的批次腳本），
所以 like 標籤是由一個「透明、完整寫在這支程式裡」的偏好模擬器產生的（見 SIM 常數與
simulate_logit()）。因此這個實驗能證明的只有兩件事：
  1. 整條 pipeline（配對特徵 → 分群特徵 → 以 seeker 分組的訓練/測試切分 → ensemble →
     排序指標 → 互惠融合）與評估方法是正確、無洩漏的；
  2. 監督式模型「能不能學回」模擬器裡已知的生成規則（能 → pipeline 沒問題；不能 → 有 bug）。
它 **不能** 證明真實使用者會喜歡這個推薦。哪天接上真實 like/pass 資料，只要把 make_pairs()
的標籤來源換掉，後面所有東西都可以直接沿用。

【這個方法是什麼】
把「推薦」當成二元分類 + 排序：對 (seeker a, 候選 b) 這一對算一組可解釋的配對特徵，訓練分類器
預測 P(a like b)，再依分數對每位 seeker 的候選排序，用 precision@10 / NDCG@10 評估。
特徵裡刻意加入非監督分群的輸出（K-means、K-modes 的群 id），這就是「把分群當特徵餵進 ensemble」。
Ensemble 用 scikit-learn 內建的 VotingClassifier（軟投票）與 StackingClassifier
（LR + RF + HGB → LogisticRegression；stacking 的內部交叉驗證也以 seeker 分組，避免洩漏）。

【為什麼分群特徵預期沒有用】
主控 agent 已實測：這份資料的 68 個 traits 是隨機獨立生成的（Hopkins≈0.51、silhouette≈0.01）。
群不存在，群 id 就只是雜訊；模型忽略它們是「正確」的行為，不是模型壞掉。本腳本會把分群特徵的
重要度排名明確印出來，並附上「真實資料 vs. 逐欄打散 null」的分群診斷對照，證明群本來就不存在。

【互惠推薦】
交友是雙向的：a 喜歡 b 不代表 b 喜歡 a。用同一個模型分別算 s(a→b) 與 s(b→a)（後者把 a、b 互換
後重算特徵），再用調和平均 HM = 2·s1·s2/(s1+s2) 融合——HM 會懲罰「一方很高、另一方很低」的配對。
評估目標換成「雙向都 like」(mutual)，比較 單向 / 算術平均 / HM 的 mutual precision@10 與曝光 Gini。
注意：本模擬器的可觀測項幾乎是對稱的（只有 overlap 的分母不同），所以三種融合的差距預期很小，
這是模擬器設計使然，不代表融合方法沒用；真實資料裡的不對稱偏好才是融合能發揮的地方。

【怎麼跑】
  cd ml && .venv/bin/python experiments/supervised_reciprocal_ranker.py
筆電約 40–50 秒（其中分群診斷約 20 秒、Stacking 約 10 秒）。不連資料庫、不需要額外套件。

【怎麼讀結果】
  outputs/supervised_reciprocal_ranker/metrics.json                    模擬器參數、所有指標、特徵重要度
  outputs/figures/supervised_reciprocal_ranker_model_metrics.png       各模型 AUC 與 NDCG@10
  outputs/figures/supervised_reciprocal_ranker_feature_importance.png  特徵重要度（分群特徵標色）
  outputs/figures/supervised_reciprocal_ranker_reciprocal.png          單向 vs 算術 vs HM
  outputs/models/supervised_reciprocal_ranker_best.joblib              測試 AUC 最高的模型
- 各模型 AUC 應接近「可觀測上限」（用模擬器真實係數、但不含 pop 與雜訊算出的分數）；差距越小
  代表模型越能學回規則。「完整 P 上限」（含 pop 與當次抽到的雜訊）永遠追不到，因為那兩項都是
  不可觀測的。
- 分群特徵的重要度：看 HGB 的 permutation importance 與「拿掉分群特徵重訓」的消融差距，兩者都應接近 0。
  摘要表除了名次也印出「名次最好的分群特徵」的重要度值——名次是相對的，值才能看出它其實是 0。
  RF 的 feature_importances_ 是 impurity 法，會把雜訊欄位的隨機切分也記成貢獻，14 個分群欄位加總
  看起來不小是已知偏誤，教學上正好拿來對照。
- 分群診斷的 stability_ari：K-means 換 seed 重跑（含 SVD）；K-modes 改用 init="Huang" 隨機初始化
  換 seed 重跑（init="Cao" 是確定性的，換 seed 結果完全相同、ARI 恆為 1，拿來當穩定度會誤導）。
- 互惠：看 HM 的 mutual P@10 是否 ≥ 單向，以及 Gini 是否下降（曝光更平均）。Gini 有結構性下限
  （名額數 / 候選池人數 決定），metrics.json 的 exposure_gini_floor 有算出來，比較時要一起看。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time  # noqa: E402

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from kmodes.kmodes import KModes  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier,
)
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.model_selection import GroupKFold, GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from heartlink_ml import plotting  # noqa: E402,F401  匯入即設定中文字型
from heartlink_ml.config import FIG_DIR, MODEL_DIR, OUTPUT_DIR, SEED, TRAIT_CATEGORIES, ensure_dirs  # noqa: E402
from heartlink_ml.data import column_category, load_seed, trait_columns  # noqa: E402
from heartlink_ml.evaluation import (  # noqa: E402
    classification_metrics, column_permutation_null, gini, hopkins_binary, ranking_metrics,
    save_json, stability_ari,
)
from heartlink_ml.features import (  # noqa: E402
    binary_matrix, idf_weights, jaccard_distance_matrix, subsample, svd_embed, weighted_matrix,
)

MODULE = "supervised_reciprocal_ranker"

# ---------------------------------------------------------------------------
# 實驗常數
# ---------------------------------------------------------------------------
N_SEEKERS = 600        # 抽幾位 seeker
N_CANDS = 60           # 每位 seeker 幾位隨機候選 → 36,000 對
K_CLUSTERS = 6         # 分群特徵用的群數
SVD_DIM = 16           # K-means 用的 SVD 維度
TEST_SIZE = 0.25       # 以 seeker 分組的測試集比例
TOP_K = 10             # precision@k / NDCG@k 的 k
TARGET_POS_RATE = 0.20 # 調 b0 讓正例率落在這裡（規格：15–25%）

# 偏好模擬器的所有參數（b0 由程式自動調整，其餘固定）。
SIM = {
    "b1_overlap": 5.0,        # × (IDF 加權共同標籤權重 / seeker 的總權重)，值域 0–1
    "b2_same_lifestyle": 0.8, # × [兩人 lifestyle 相同]
    "b3_goal_intersect": 0.8, # × [dating_goal 有交集]
    "b4_age": 1.0,            # × (−|年齡差| / 10)
    "b5_gender": 1.5,         # × [異性配對：a 為 Male 且 b 為 Female，或反之]
    "g_pop": 1.0,             # × 候選的潛在受歡迎度 pop_b ~ N(0,1)（不可觀測）
    "noise_sd": 0.5,          # 加性雜訊 N(0, 0.5)
}

# 圖用色（固定順序、經過色盲檢驗的三色）：藍＝主要、橘＝對照／分群特徵、青＝第三系列。
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# 1. 個人層資料 + 分群特徵
# ---------------------------------------------------------------------------
class People:
    """把 10,000 人的資料整理成配對特徵需要的陣列。"""

    def __init__(self, df: pd.DataFrame):
        self.cols = trait_columns(df)
        self.X = binary_matrix(df, self.cols)                       # (n, 68) bool
        self.idf = idf_weights(self.X)                              # (68,)
        self.cat_masks = {c: np.array([column_category(x) == c for x in self.cols]) for c in TRAIT_CATEGORIES}
        self.age = df["age"].astype(float).to_numpy()
        self.height = df["height_cm"].astype(float).to_numpy()
        self.is_male = (df["gender"].to_numpy() == "Male")
        self.is_female = (df["gender"].to_numpy() == "Female")
        self.n = len(df)


def fit_clusters(X: np.ndarray, cols, seed=SEED):
    """用全部 10,000 人 fit K-means(k=6, SVD-16) 與 K-modes(k=6)，回傳兩組群標籤。"""
    Xw, _ = weighted_matrix(X, cols)
    emb, _ = svd_embed(Xw, n_components=SVD_DIM, seed=seed)
    km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=seed).fit(emb)
    kmo = KModes(n_clusters=K_CLUSTERS, init="Cao", n_init=1, random_state=seed)
    kmo_labels = kmo.fit_predict(X.astype(np.uint8))
    return km.labels_.astype(int), kmo_labels.astype(int)


def cluster_diagnostics(X: np.ndarray, cols, km_labels, kmo_labels) -> dict:
    """分群到底有沒有結構？回報 Hopkins、Jaccard silhouette（3000 抽樣）、K-means / K-modes 換 seed 的 ARI。
    真實資料與 null 都走這一個函式、用同一組 subsample 索引，才能並列比較。"""
    sub = subsample(len(X), 3000, SEED)
    D = jaccard_distance_matrix(X[sub])
    X_u8 = X.astype(np.uint8)

    def km_fit_predict(seed):
        Xw, _ = weighted_matrix(X, cols)
        emb, _ = svd_embed(Xw, n_components=SVD_DIM, seed=seed)
        return KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=seed).fit_predict(emb)

    def kmo_fit_predict(seed):
        # 穩定度要用「隨機初始化」才有意義：Cao 初始化是確定性的，換 seed 結果一樣、ARI 恆為 1。
        return KModes(n_clusters=K_CLUSTERS, init="Huang", n_init=1, random_state=seed).fit_predict(X_u8)

    return {
        "hopkins": hopkins_binary(X, seed=SEED),
        "silhouette_jaccard_kmeans": float(silhouette_score(D, km_labels[sub], metric="precomputed")),
        "silhouette_jaccard_kmodes": float(silhouette_score(D, kmo_labels[sub], metric="precomputed")),
        "stability_ari_kmeans_3runs": stability_ari(km_fit_predict, n_runs=3, seed=SEED),
        "stability_ari_kmodes_huang_3runs": stability_ari(kmo_fit_predict, n_runs=3, seed=SEED),
        "kmeans_sizes": np.bincount(km_labels).tolist(),
        "kmodes_sizes": np.bincount(kmo_labels).tolist(),
    }


# ---------------------------------------------------------------------------
# 2. 配對特徵（全部確定性、可解釋；絕不碰 pop 或 P）
# ---------------------------------------------------------------------------
def pair_features(P: People, a: np.ndarray, b: np.ndarray, km_labels, kmo_labels) -> pd.DataFrame:
    """a、b 是使用者索引陣列（等長）。回傳一列一對的特徵表；把 a、b 互換就是反向特徵。"""
    Xa, Xb = P.X[a], P.X[b]
    inter = Xa & Xb
    union = Xa | Xb
    F = {}
    F["jaccard"] = inter.sum(1) / np.maximum(union.sum(1), 1)
    # IDF 加權 overlap 依 6 個類別各一欄；分母統一用 a 的總 IDF 權重，
    # 所以 6 欄相加 == 模擬器用的 overlap 總量（LR 用相同係數就能完整學回）。
    a_total = np.maximum((Xa * P.idf).sum(1), 1e-9)
    for c, m in P.cat_masks.items():
        F[f"overlap_idf_{c}"] = (inter[:, m] * P.idf[m]).sum(1) / a_total
    F["same_lifestyle"] = inter[:, P.cat_masks["lifestyle"]].any(1).astype(float)  # 每人剛好一個 lifestyle
    F["goal_intersect"] = inter[:, P.cat_masks["dating_goal"]].sum(1).astype(float)
    F["age_diff"] = np.abs(P.age[a] - P.age[b])
    F["height_diff"] = np.abs(P.height[a] - P.height[b])
    F["gender_pair"] = ((P.is_male[a] & P.is_female[b]) | (P.is_female[a] & P.is_male[b])).astype(float)
    # 分群特徵：把非監督分群的輸出餵進監督式模型。
    F["cluster_kmeans_same"] = (km_labels[a] == km_labels[b]).astype(float)
    F["cluster_kmodes_same"] = (kmo_labels[a] == kmo_labels[b]).astype(float)
    for k in range(K_CLUSTERS):
        F[f"cluster_kmeans_seeker_{k}"] = (km_labels[a] == k).astype(float)
    for k in range(K_CLUSTERS):
        F[f"cluster_kmodes_seeker_{k}"] = (kmo_labels[a] == k).astype(float)
    return pd.DataFrame(F)


def overlap_total(F: pd.DataFrame) -> np.ndarray:
    """6 個類別的 IDF 加權 overlap 相加 = 模擬器的 overlap 項，也是零訓練基準的分數。"""
    return F[[f"overlap_idf_{c}" for c in TRAIT_CATEGORIES]].sum(axis=1).to_numpy()


# ---------------------------------------------------------------------------
# 3. 偏好模擬器
# ---------------------------------------------------------------------------
def observable_logit(F: pd.DataFrame) -> np.ndarray:
    """模擬器裡「用可觀測特徵就算得出來」的部分（不含 b0、pop、雜訊）。"""
    return (SIM["b1_overlap"] * overlap_total(F)
            + SIM["b2_same_lifestyle"] * F["same_lifestyle"].to_numpy()
            + SIM["b3_goal_intersect"] * (F["goal_intersect"].to_numpy() > 0)
            + SIM["b4_age"] * (-F["age_diff"].to_numpy() / 10.0)
            + SIM["b5_gender"] * F["gender_pair"].to_numpy())


def tune_b0(z: np.ndarray, target: float) -> float:
    """二分法找 b0 讓 E[sigmoid(b0 + z)] = target。"""
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if sigmoid(mid + z).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def make_pairs(P: People, km_labels, kmo_labels, seed=SEED):
    """抽 seeker/候選、算正反向特徵、用模擬器產生 y_fwd (a→b)、y_rev (b→a)。"""
    rng = np.random.default_rng(seed)
    seekers = rng.choice(P.n, size=N_SEEKERS, replace=False)
    a = np.repeat(seekers, N_CANDS)
    # 每位 seeker 抽 N_CANDS 位不含自己的隨機候選：先在 n-1 裡抽，>= 自己的索引往後移一格。
    b = np.empty_like(a)
    for i, s in enumerate(seekers):
        c = rng.choice(P.n - 1, size=N_CANDS, replace=False)
        b[i * N_CANDS:(i + 1) * N_CANDS] = c + (c >= s)

    F_fwd = pair_features(P, a, b, km_labels, kmo_labels)
    F_rev = pair_features(P, b, a, km_labels, kmo_labels)

    pop = rng.normal(0.0, 1.0, size=P.n)                     # 每人一個潛在受歡迎度（不可觀測）
    z_fwd = observable_logit(F_fwd) + SIM["g_pop"] * pop[b] + rng.normal(0, SIM["noise_sd"], len(a))
    z_rev = observable_logit(F_rev) + SIM["g_pop"] * pop[a] + rng.normal(0, SIM["noise_sd"], len(a))
    b0 = tune_b0(z_fwd, TARGET_POS_RATE)
    p_fwd, p_rev = sigmoid(b0 + z_fwd), sigmoid(b0 + z_rev)
    y_fwd = (rng.random(len(a)) < p_fwd).astype(int)
    y_rev = (rng.random(len(a)) < p_rev).astype(int)
    sim_info = {
        **SIM, "b0": float(b0), "target_pos_rate": TARGET_POS_RATE,
        "pos_rate_fwd": float(y_fwd.mean()), "pos_rate_rev": float(y_rev.mean()),
        "mutual_rate": float((y_fwd & y_rev).mean()),
        "n_seekers": N_SEEKERS, "n_cands_per_seeker": N_CANDS, "n_pairs": int(len(a)),
    }
    # 只有評估「上限」時才會用到 p_fwd / observable_logit，絕不進特徵。
    oracle = {"p_true_fwd": p_fwd, "observable_fwd": observable_logit(F_fwd)}
    return a, b, F_fwd, F_rev, y_fwd, y_rev, sim_info, oracle


# ---------------------------------------------------------------------------
# 4. 模型
# ---------------------------------------------------------------------------
def build_models(group_folds):
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED)
    hgb = HistGradientBoostingClassifier(random_state=SEED)
    base = [("lr", clone(lr)), ("rf", clone(rf)), ("hgb", clone(hgb))]
    return {
        "LogisticRegression": lr,
        "RandomForest(300)": rf,
        "HistGradientBoosting": hgb,
        "Voting(soft: LR+RF+HGB)": VotingClassifier(base, voting="soft"),
        # cv 傳入以 seeker 分組的折，stacking 的 out-of-fold 預測才不會在同一位 seeker 內洩漏。
        "Stacking(LR+RF+HGB→LR)": StackingClassifier(
            [(n, clone(m)) for n, m in base],
            final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
            cv=group_folds),
    }


def evaluate_scores(groups, y, scores) -> dict:
    out = classification_metrics(y, scores)
    out.update(ranking_metrics(groups, y, scores, k=TOP_K))
    return out


def importance_report(names, imp: np.ndarray) -> dict:
    """把重要度排序，並特別回報分群特徵（名稱以 cluster_ 開頭）的名次與占比。"""
    imp = np.asarray(imp, dtype=float)
    order = np.argsort(-imp)
    rank = np.empty(len(imp), dtype=int)
    rank[order] = np.arange(1, len(imp) + 1)
    is_cluster = np.array([n.startswith("cluster_") for n in names])
    pos = np.clip(imp, 0, None)
    best_i = int(np.where(is_cluster)[0][np.argmax(imp[is_cluster])])  # 名次最好的那個分群特徵
    return {
        "ranked": [{"feature": names[i], "importance": float(imp[i]), "rank": int(rank[i])} for i in order],
        "cluster_feature_ranks": {names[i]: int(rank[i]) for i in np.where(is_cluster)[0]},
        "best_cluster_rank": int(rank[best_i]),
        "best_cluster_feature": names[best_i],
        "best_cluster_importance": float(imp[best_i]),   # 名次只是相對的，這個值才看得出它其實≈0
        "top_importance": float(imp[order[0]]),          # 第 1 名的值，拿來對照尺度
        "n_features": int(len(names)),
        "cluster_importance_share": float(pos[is_cluster].sum() / max(pos.sum(), 1e-12)),
    }


# ---------------------------------------------------------------------------
# 5. 互惠融合
# ---------------------------------------------------------------------------
def harmonic_mean(s1, s2):
    return 2 * s1 * s2 / np.maximum(s1 + s2, 1e-12)


def top_k_exposure(groups, cand, scores, k) -> np.ndarray:
    """每位候選（測試集裡出現過的候選池）在所有 seeker top-k 裡出現的次數。"""
    counts = np.zeros(int(cand.max()) + 1)
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        top = m[np.argsort(-scores[m])[:k]]
        np.add.at(counts, cand[top], 1)
    return counts[np.unique(cand)]


# ---------------------------------------------------------------------------
# 6. 畫圖 helper（plotting.bar_plot 只能畫單一系列，這裡需要分組／標色）
# ---------------------------------------------------------------------------
def plot_model_metrics(results: dict, path):
    names = list(results)
    auc = [results[n]["roc_auc"] for n in names]
    ndcg = [results[n]["ndcg_at_k"] for n in names]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w / 2, auc, w, label="AUC", color=C_BLUE)
    b2 = ax.bar(x + w / 2, ndcg, w, label=f"NDCG@{TOP_K}", color=C_ORANGE)
    ax.bar_label(b1, fmt="%.3f", fontsize=7, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x, names, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("分數（越高越好）")
    ax.set_title("各模型在測試集（以 seeker 分組切分）的 AUC 與 NDCG@10 — 模擬標籤")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importances(rf_rep: dict, hgb_rep: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, rep, title in zip(axes, (rf_rep, hgb_rep),
                              ("RandomForest feature_importances_", "HGB permutation_importance（測試集，AUC 下降量）")):
        rows = rep["ranked"][::-1]  # 由小到大，畫水平長條時最重要的在最上面
        names = [r["feature"] for r in rows]
        vals = [r["importance"] for r in rows]
        colors = [C_ORANGE if n.startswith("cluster_") else C_BLUE for n in names]
        ax.barh(names, vals, color=colors)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", alpha=0.3)
        ax.axvline(0, color="gray", lw=0.8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_BLUE), plt.Rectangle((0, 0), 1, 1, color=C_ORANGE)]
    fig.legend(handles, ["配對特徵", "分群特徵（K-means / K-modes 輸出）"], loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("特徵重要度：分群特徵（橘）預期墊底，因為標籤沒有群結構", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reciprocal(recip: dict, path, gini_floor=None):
    names = list(recip)
    p_mutual = [recip[n]["mutual_precision_at_k"] for n in names]
    g = [recip[n]["exposure_gini"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, vals, ylabel, title in zip(
            axes, (p_mutual, g), (f"mutual precision@{TOP_K}", "曝光 Gini（越低越平均）"),
            (f"雙向都 like 的 precision@{TOP_K}", "候選在所有 seeker top-10 的曝光集中度")):
        bars = ax.bar(names, vals, color=[C_BLUE, C_AQUA, C_ORANGE])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
        ax.grid(axis="y", alpha=0.3)
    # Gini 不可能低於結構性下限（名額數 / 候選池人數決定），畫出來才不會把 0.78 誤讀成「很集中」。
    if gini_floor is not None:
        axes[1].axhline(gini_floor, ls="--", color="gray", lw=1,
                        label=f"結構性下限 {gini_floor:.3f}（每個名額都給不同人）")
        axes[1].legend(loc="upper center", fontsize=8, frameon=False)
    fig.suptitle("互惠融合：單向 s(a→b) vs 算術平均 vs 調和平均 HM", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------
def main():
    t_all = time.perf_counter()
    ensure_dirs()
    out_dir = OUTPUT_DIR / MODULE
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_seed()
    P = People(df)
    print(f"載入 {P.n} 人、{len(P.cols)} 個 traits")

    # --- 分群特徵：真實資料 vs. 逐欄打散 null 的診斷 ------------------------------------
    t0 = time.perf_counter()
    km_labels, kmo_labels = fit_clusters(P.X, P.cols)
    diag_real = cluster_diagnostics(P.X, P.cols, km_labels, kmo_labels)
    X_null = column_permutation_null(P.X, seed=SEED)
    km_null, kmo_null = fit_clusters(X_null, P.cols)
    diag_null = cluster_diagnostics(X_null, P.cols, km_null, kmo_null)
    print(f"分群特徵與診斷完成（{time.perf_counter() - t0:.1f}s）")
    print(f"  Hopkins 真實={diag_real['hopkins']:.3f} / null={diag_null['hopkins']:.3f}；"
          f"silhouette(K-means) 真實={diag_real['silhouette_jaccard_kmeans']:.3f} / null={diag_null['silhouette_jaccard_kmeans']:.3f}；"
          f"silhouette(K-modes) 真實={diag_real['silhouette_jaccard_kmodes']:.3f} / null={diag_null['silhouette_jaccard_kmodes']:.3f}")

    # --- 模擬器產生配對與標籤 --------------------------------------------------------------
    a, b, F_fwd, F_rev, y_fwd, y_rev, sim_info, oracle = make_pairs(P, km_labels, kmo_labels)
    print("模擬器參數：", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sim_info.items()})
    if not (0.15 <= sim_info["pos_rate_fwd"] <= 0.25):
        print("  警告：正例率不在 15–25% 之間，請調整 TARGET_POS_RATE 或 SIM 係數")
    feature_names = list(F_fwd.columns)
    X_fwd, X_rev = F_fwd.to_numpy(dtype=np.float32), F_rev.to_numpy(dtype=np.float32)

    # --- 切分：以 seeker 為 group，測試集裡的 seeker 完全沒在訓練出現 ------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr, te = next(gss.split(X_fwd, y_fwd, groups=a))
    assert not set(a[tr]) & set(a[te]), "訓練／測試的 seeker 有重疊，洩漏！"
    groups_te = a[te]
    group_folds = list(GroupKFold(n_splits=5).split(X_fwd[tr], y_fwd[tr], groups=a[tr]))
    print(f"切分：訓練 {len(tr)} 對（{len(set(a[tr]))} 位 seeker）／測試 {len(te)} 對（{len(set(groups_te))} 位 seeker）")

    # --- 基準與上限（不用訓練） ------------------------------------------------------------
    results = {}
    results["零訓練基準（IDF overlap）"] = {**evaluate_scores(groups_te, y_fwd[te], overlap_total(F_fwd)[te]), "fit_seconds": 0.0}
    upper = {
        "上限：可觀測部分（真實係數，不含 pop）": evaluate_scores(groups_te, y_fwd[te], oracle["observable_fwd"][te]),
        "上限：完整 P（含不可觀測的 pop 與雜訊）": evaluate_scores(groups_te, y_fwd[te], oracle["p_true_fwd"][te]),
    }

    # --- 訓練 5 個模型 ----------------------------------------------------------------------
    models = build_models(group_folds)
    fitted = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_fwd[tr], y_fwd[tr])
        secs = time.perf_counter() - t0
        s = model.predict_proba(X_fwd[te])[:, 1]
        results[name] = {**evaluate_scores(groups_te, y_fwd[te], s), "fit_seconds": float(secs)}
        fitted[name] = model
        print(f"  {name:28s} AUC={results[name]['roc_auc']:.4f} PR-AUC={results[name]['pr_auc']:.4f} "
              f"P@{TOP_K}={results[name]['precision_at_k']:.4f} NDCG@{TOP_K}={results[name]['ndcg_at_k']:.4f} ({secs:.1f}s)")

    # --- 特徵重要度 -------------------------------------------------------------------------
    rf_rep = importance_report(feature_names, fitted["RandomForest(300)"].feature_importances_)
    t0 = time.perf_counter()
    perm = permutation_importance(fitted["HistGradientBoosting"], X_fwd[te], y_fwd[te],
                                  scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=-1)
    hgb_rep = importance_report(feature_names, perm.importances_mean)
    print(f"特徵重要度完成（permutation {time.perf_counter() - t0:.1f}s）")
    print(f"  RF：分群特徵最佳名次 {rf_rep['best_cluster_rank']}/{rf_rep['n_features']}，占比 {rf_rep['cluster_importance_share']:.3%}")
    print(f"  HGB permutation：分群特徵最佳名次 {hgb_rep['best_cluster_rank']}/{hgb_rep['n_features']}，占比 {hgb_rep['cluster_importance_share']:.3%}")
    # LR 係數（標準化後）也附上，方便對照模擬器係數的方向。
    lr_pipe = fitted["LogisticRegression"]
    lr_coef = {n: float(c) for n, c in zip(feature_names, lr_pipe[-1].coef_[0])}

    # --- 消融：拿掉分群特徵重訓 LR 與 HGB，AUC 掉多少就是分群特徵的真實貢獻 ---------------
    # （RF 的 impurity importance 會把「隨機切分」也記成貢獻，14 個雜訊欄位加起來看似不小，
    #   所以要用 permutation importance 與這個消融對照才算數。）
    keep = np.array([not n.startswith("cluster_") for n in feature_names])
    ablation = {}
    for name in ("LogisticRegression", "HistGradientBoosting"):
        m = clone(models[name]).fit(X_fwd[tr][:, keep], y_fwd[tr])
        r = evaluate_scores(groups_te, y_fwd[te], m.predict_proba(X_fwd[te][:, keep])[:, 1])
        ablation[name] = {"auc_with_cluster": results[name]["roc_auc"], "auc_without_cluster": r["roc_auc"],
                          "auc_delta": results[name]["roc_auc"] - r["roc_auc"],
                          "ndcg_with_cluster": results[name]["ndcg_at_k"], "ndcg_without_cluster": r["ndcg_at_k"]}
        print(f"  消融 {name}: AUC 有分群特徵 {results[name]['roc_auc']:.4f} / 無 {r['roc_auc']:.4f}（差 {ablation[name]['auc_delta']:+.4f}）")

    # --- 互惠融合（用測試 AUC 最高的模型） -------------------------------------------------
    trained_names = list(models)
    best_name = max(trained_names, key=lambda n: results[n]["roc_auc"])
    best = fitted[best_name]
    s1 = best.predict_proba(X_fwd[te])[:, 1]   # s(a→b)
    s2 = best.predict_proba(X_rev[te])[:, 1]   # s(b→a)：反向特徵
    mutual = (y_fwd[te] & y_rev[te]).astype(int)
    fusions = {"單向 s(a→b)": s1, "算術平均": (s1 + s2) / 2, "調和平均 HM": harmonic_mean(s1, s2)}
    recip = {}
    for name, sc in fusions.items():
        rm = ranking_metrics(groups_te, mutual, sc, k=TOP_K)
        one_way = ranking_metrics(groups_te, y_fwd[te], sc, k=TOP_K)
        recip[name] = {
            "mutual_precision_at_k": rm["precision_at_k"], "mutual_ndcg_at_k": rm["ndcg_at_k"],
            "mutual_n_groups": rm["n_groups"], "mutual_auc": classification_metrics(mutual, sc)["roc_auc"],
            "oneway_precision_at_k": one_way["precision_at_k"],
            "exposure_gini": gini(top_k_exposure(groups_te, b[te], sc, TOP_K)),
        }
    # Gini 的結構性下限：n_seekers×k 個 top-k 名額分給候選池，就算每個名額都給不同人，
    # 沒被抽到的候選仍是 0 曝光，所以 Gini 不可能低於 1 − 名額數/候選池人數。
    n_pool = int(len(np.unique(b[te])))
    n_slots = len(np.unique(groups_te)) * TOP_K
    recip_extra = {
        "best_model": best_name, "corr_s1_s2": float(np.corrcoef(s1, s2)[0, 1]),
        "mutual_rate_test": float(mutual.mean()), "n_candidates_in_test_pool": n_pool,
        "n_top_k_slots": int(n_slots), "exposure_gini_floor": float(max(0.0, 1 - n_slots / n_pool)),
        "reverse_score_auc_on_y_rev": classification_metrics(y_rev[te], s2)["roc_auc"],
    }
    print(f"互惠融合（最佳模型 {best_name}；corr(s1,s2)={recip_extra['corr_s1_s2']:.3f}；"
          f"s(b→a) 對 y_rev 的 AUC={recip_extra['reverse_score_auc_on_y_rev']:.4f}）")

    # --- 圖與模型 ---------------------------------------------------------------------------
    plot_model_metrics(results, FIG_DIR / f"{MODULE}_model_metrics.png")
    plot_importances(rf_rep, hgb_rep, FIG_DIR / f"{MODULE}_feature_importance.png")
    plot_reciprocal(recip, FIG_DIR / f"{MODULE}_reciprocal.png", gini_floor=recip_extra["exposure_gini_floor"])
    model_path = MODEL_DIR / f"{MODULE}_best.joblib"
    joblib.dump({"model": best, "model_name": best_name, "feature_names": feature_names,
                 "simulator": sim_info, "note": "標籤來自模擬器，不是真實使用者行為"}, model_path)

    # --- 結論（由數字自動產生，避免文字與數字打架） -------------------------------------
    obs_auc = upper["上限：可觀測部分（真實係數，不含 pop）"]["roc_auc"]
    gap = obs_auc - results[best_name]["roc_auc"]
    hm, one = recip["調和平均 HM"], recip["單向 s(a→b)"]
    lr_auc = results["LogisticRegression"]["roc_auc"]
    best_auc = results[best_name]["roc_auc"]
    max_abl = max(abs(v["auc_delta"]) for v in ablation.values())
    conclusions = [
        "標籤由模擬器產生：本實驗只證明 pipeline 與評估正確、模型能學回已知規則，不代表真實推薦品質。",
        f"最佳模型 {best_name} 測試 AUC={best_auc:.3f}，與可觀測上限 {obs_auc:.3f} 差 {gap:.3f}"
        + ("（幾乎學回全部可觀測規則）" if gap < 0.01 else "（仍有一段距離）")
        + f"；完整 P 上限 {upper['上限：完整 P（含不可觀測的 pop 與雜訊）']['roc_auc']:.3f} 因含不可觀測的 pop 與雜訊，本來就追不到。"
        + (f" 模擬器本身就是線性 logit，所以 LR（{lr_auc:.3f}）是「模型形式正確」的那一個，ensemble 追平但贏不了它是合理的；"
           "真實資料若有非線性互動，ensemble 才有機會拉開差距。" if best_auc - lr_auc < 0.005 else ""),
        f"分群特徵：消融後 AUC 變化最大只有 {max_abl:.4f}（LR {ablation['LogisticRegression']['auc_delta']:+.4f}、"
        f"HGB {ablation['HistGradientBoosting']['auc_delta']:+.4f}），HGB permutation importance 占比 {hgb_rep['cluster_importance_share']:.1%}"
        f"（名次最好的分群特徵排 {hgb_rep['best_cluster_rank']}/{hgb_rep['n_features']}，但它的值只有 {hgb_rep['best_cluster_importance']:.4f}，"
        f"第 1 名是 {hgb_rep['top_importance']:.4f}，實質為 0）。RF impurity importance 顯示 {rf_rep['cluster_importance_share']:.1%}"
        f"（最佳名次 {rf_rep['best_cluster_rank']}/{rf_rep['n_features']}）是 impurity 法的已知偏誤：14 個雜訊欄位隨機切分也會累積分數，不能當真。"
        f" 標籤沒有群結構（Hopkins 真實 {diag_real['hopkins']:.3f} vs null {diag_null['hopkins']:.3f}、"
        f"silhouette 真實 {diag_real['silhouette_jaccard_kmeans']:.3f} vs null {diag_null['silhouette_jaccard_kmeans']:.3f}），群 id 沒有資訊量是正確結果，不是模型的錯。",
        f"互惠：HM 的 mutual P@{TOP_K}={hm['mutual_precision_at_k']:.3f} vs 單向 {one['mutual_precision_at_k']:.3f}"
        f"（差 {hm['mutual_precision_at_k'] - one['mutual_precision_at_k']:+.3f}）；曝光 Gini HM={hm['exposure_gini']:.3f} vs 單向 {one['exposure_gini']:.3f}"
        f"（差 {hm['exposure_gini'] - one['exposure_gini']:+.3f}；結構性下限 {recip_extra['exposure_gini_floor']:.3f}，因為 {recip_extra['n_top_k_slots']} 個名額分給 "
        f"{recip_extra['n_candidates_in_test_pool']} 位候選）。corr(s1,s2)={recip_extra['corr_s1_s2']:.2f}：模擬器的可觀測項近乎對稱，融合差距小是設計使然。",
    ]

    metrics = {
        "module": MODULE, "seed": SEED, "note": "like 標籤來自程式內的偏好模擬器，非真實互動資料",
        "simulator": sim_info,
        "split": {"method": "GroupShuffleSplit(by seeker)", "test_size": TEST_SIZE,
                  "n_train_pairs": int(len(tr)), "n_test_pairs": int(len(te)),
                  "n_train_seekers": int(len(set(a[tr]))), "n_test_seekers": int(len(set(groups_te)))},
        "cluster_feature_diagnostics": {"real": diag_real, "column_permutation_null": diag_null},
        "features": feature_names,
        "models": results, "upper_bounds": upper,
        "feature_importance": {"random_forest": rf_rep, "hgb_permutation": hgb_rep,
                               "logistic_regression_coef_standardized": lr_coef},
        "cluster_feature_ablation": ablation,
        "reciprocal": {**recip_extra, "fusions": recip},
        "outputs": {"figures": [str(FIG_DIR / f"{MODULE}_{s}.png") for s in ("model_metrics", "feature_importance", "reciprocal")],
                    "model": str(model_path)},
        "conclusions": conclusions,
        "runtime_seconds": float(time.perf_counter() - t_all),
    }
    save_json(metrics, out_dir / "metrics.json")

    # --- 摘要表 -----------------------------------------------------------------------------
    print("\n## 摘要（標籤來自模擬器）\n")
    print(f"| 模型 | AUC | PR-AUC | P@{TOP_K} | NDCG@{TOP_K} | 訓練秒數 |")
    print("|---|---|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['precision_at_k']:.4f} | {r['ndcg_at_k']:.4f} | {r['fit_seconds']:.1f} |")
    for name, r in upper.items():
        print(f"| {name} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['precision_at_k']:.4f} | {r['ndcg_at_k']:.4f} | – |")
    print(f"\n| 融合（最佳模型：{best_name}） | mutual P@{TOP_K} | mutual NDCG@{TOP_K} | 曝光 Gini |")
    print("|---|---|---|---|")
    for name, r in recip.items():
        print(f"| {name} | {r['mutual_precision_at_k']:.4f} | {r['mutual_ndcg_at_k']:.4f} | {r['exposure_gini']:.4f} |")
    print("\n| 分群診斷 | 真實 | null（逐欄打散） |")
    print("|---|---|---|")
    for key in ("hopkins", "silhouette_jaccard_kmeans", "silhouette_jaccard_kmodes",
                "stability_ari_kmeans_3runs", "stability_ari_kmodes_huang_3runs"):
        print(f"| {key} | {diag_real[key]:.3f} | {diag_null[key]:.3f} |")
    print("\n| 分群特徵重要度 | 最佳名次 / 特徵數 | 該名次的值（第 1 名的值） | 分群特徵占比 |\n|---|---|---|---|")
    for label, rep in (("RF feature_importances_（impurity，有偏誤）", rf_rep),
                       ("HGB permutation_importance（測試集，AUC 下降量）", hgb_rep)):
        print(f"| {label} | {rep['best_cluster_rank']} / {rep['n_features']} | "
              f"{rep['best_cluster_importance']:.4f}（{rep['top_importance']:.4f}） | {rep['cluster_importance_share']:.2%} |")
    print("\n| 消融：拿掉 14 個分群特徵 | AUC 有 | AUC 無 | 差 |\n|---|---|---|---|")
    for name, r in ablation.items():
        print(f"| {name} | {r['auc_with_cluster']:.4f} | {r['auc_without_cluster']:.4f} | {r['auc_delta']:+.4f} |")
    print("\n結論：")
    for c in conclusions:
        print(f"- {c}")
    print(f"\n總耗時 {metrics['runtime_seconds']:.1f} 秒；metrics → {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
