"""分割式分群實驗：K-means / MiniBatchKMeans / K-modes / GaussianMixture。

================================================================================
這個實驗在做什麼
================================================================================
把 10,000 位使用者依「68 個 0/1 標籤（traits）」分成 k 群，看資料裡有沒有
「天然的族群」（例如「戶外運動咖」「宅系文青」）。這是最經典的非監督式分群題目，
也是學生專案最常見的起手式。

四個方法各自的想法：

1. K-means（sklearn.cluster.KMeans）
   反覆做兩件事：把每個點分給最近的質心 → 把質心移到該群的平均位置。
   它需要「歐氏距離」和「平均值」都有意義的空間，所以我們**不直接在 0/1 上跑**，
   原因如下：
   - 兩個 0/1 向量的歐氏距離平方 = 不一樣的位數 = Hamming 距離。
     Hamming 會把「兩人都沒勾」和「兩人都有勾」算成一樣的相似，但在標籤資料裡
     「都沒勾露營」不代表兩人相像（大多數人本來就沒勾）。Jaccard 距離只看
     「至少一人有勾」的位置，才是二元標籤該用的距離。
   - 質心是群內的「平均」，平均完會變成 0.37、0.62 這種小數，不再是合法的
     標籤向量，很難解釋「這群人的代表是誰」。
   - 每個標籤的流行度差很多（lifestyle 每人剛好 1 個、interest 每人 4–8 個），
     直接用歐氏距離會被高流行度的標籤主導。
   所以做法是：features.weighted_matrix（IDF × 類別權重）→ features.svd_embed
   降到 16 維，在這個連續的歐氏空間跑 K-means。SVD 空間裡「平均」是合理的。

2. MiniBatchKMeans
   K-means 的省記憶體版本，每次只拿一小批（batch_size=1024）資料更新質心。
   10,000 筆其實用不到它，但真實平台使用者可能到百萬級，先放進來比較
   「它的結果跟完整 K-means 差多少」，未來要擴展時心裡有數。

3. K-modes（kmodes.kmodes.KModes）
   專為類別型／二元資料設計的 K-means 變體：
   - 距離用 Hamming（不一樣的位數），質心用**眾數**（每一欄取群內最常見的值）。
   - 眾數質心永遠是合法的 0/1 向量，可以直接讀成「這群人的典型標籤組合」，
     這是它對二元資料最大的優點。
   - 缺點：Hamming 一樣有「都沒勾也算像」的問題；而且在稀疏資料上眾數常常是
     全 0，容易出現一個超大群 + 幾個小群。
   直接在原始 68 維 0/1 上跑，init="Cao"（用密度挑初始質心，結果可重現）。

4. GaussianMixture（sklearn.mixture.GaussianMixture，covariance_type="diag"）
   K-means 的機率版：假設資料由 k 個高斯分佈混合而成，每個點屬於每群都有
   一個機率（軟分群）。用 BIC（越小越好）自動選 n_components，不必看肘部圖。
   一樣在 SVD-16 空間跑（高斯假設在 0/1 上完全不成立）。diag 表示每群的
   共變異矩陣只有對角線，參數少、不容易過擬合。

================================================================================
為什麼這份資料「找不到群」是正確答案
================================================================================
主控 agent 已實測：這份 seed 資料的 68 個標籤是**獨立隨機生成**的
（Hopkins ≈ 0.51、跨類別共現 |z| ≈ 1.2、標籤與性別／城市無關）。
獨立隨機的資料本來就沒有族群，任何分群演算法都「一定會」切出 k 群，
但那只是把一團均勻的雲硬切成 k 塊，不是找到結構。

所以這個實驗的重點不是「切出幾群」，而是**有沒有辦法分辨「真的有群」和
「硬切出來的群」**。我們用三道保險：
  (a) Hopkins 統計量：≈0.5 代表跟隨機無異；→1 代表有群。
  (b) Null model：evaluation.column_permutation_null 把每一欄獨立打散
      （保留每個標籤的流行度、破壞標籤之間的共現）。同一套流程跑在 null 上，
      得到的 silhouette 就是「沒有結構時會看到的數字」。真實 ≈ null 就是沒結構。
  (c) 穩定度 stability_ari：換 5 個隨機種子重跑，看分群結果有多一致。
      **注意**：ARI 高只代表最佳化收斂很穩，不代表群存在（K-means 在這份資料
      上 ARI 可到 0.8 但 silhouette ≈ 0）。一定要配合 (a)(b) 一起看。
      K-modes 的 init="Cao" 是確定性初始化，換 seed 結果完全一樣（ARI 恆為 1，
      測不到東西），所以穩定度改用隨機化的 init="Huang" 算；掃 k 與存檔仍用 Cao。

一個要小心的細節：column_permutation_null 逐欄打散時，會順便打破產生器的
「每類別配額」——真實資料每人 value 剛好 2–3 個（5 選）、diet 1–3 個（5 選）、
lifestyle 剛好 1 個、dating_goal 1–2 個、interest 4–8 個。配額固定代表
**同一類別內的標籤互斥**（value 5 個標籤兩兩相關係數 ≈ -0.20，diet ≈ -0.24）：
勾了「陪伴」就比較不會勾「信任」。這是類別內的規則，不是使用者族群，但逐欄
打散會把它抹掉，所以 K-modes 在真實資料上的 silhouette 會比 null 略高一點點
（k=2：0.020 vs 0.004）。實際去看 K-modes k=2 的切法，就是把人依 value 的兩個
互補子集（陪伴+溝通 vs 信任+分享+空間）分開，兩群的總標籤數幾乎一樣（15.5 vs 15.7），
所以差距**不是**來自「總共勾了幾個」。

為了證明這一點，本腳本另外加一個「區塊置換 null」（block_permutation_null）：
把每個類別的整塊欄位一起隨機重排列（每個類別用不同的排列）。這會保留類別內
的配額與互斥、只破壞跨類別的關聯；如果真實 ≈ 區塊 null，就代表真實資料唯一多
出來的「結構」只有類別內配額。基礎層沒有這個 null，寫在本檔內。判斷有沒有群
還是看絕對值有沒有到 0.25 這種等級。

================================================================================
怎麼跑
================================================================================
    cd ml && .venv/bin/python experiments/clustering_partitional.py
不需要資料庫、不需要網路，約 2–3 分鐘（K-modes 佔大半）。輸出：
    outputs/clustering_partitional/metrics.json        所有數字（含真實 vs null vs 區塊 null）
    outputs/figures/clustering_partitional_elbow.png            肘部圖（inertia / cost / BIC）
    outputs/figures/clustering_partitional_silhouette_vs_k.png  silhouette-vs-k（真實實線、null 虛線、區塊 null 點線）
    outputs/figures/clustering_partitional_scatter_<方法>.png   最佳 k 的 SVD 2D 散點
    outputs/models/clustering_partitional_<方法>.joblib        最佳模型（含前處理）

================================================================================
怎麼讀結果
================================================================================
- silhouette_jaccard：-1 ~ 1，>0.25 才勉強算有群；≈0 表示點到「自己群」和
  「隔壁群」一樣遠。所有方法共用同一份 3,000 筆抽樣的 Jaccard 距離矩陣算，
  所以可以直接互相比較。
- davies_bouldin（越小越好）、calinski_harabasz（越大越好）：都在 SVD-16
  空間算，四個方法用同一個空間當共同尺規；K-modes 另外附一組在原始 0/1 上
  算的版本（_raw01）供參考。
- 真實 vs null 差不多 → 沒結構。真實明顯高於 null → 有結構，才值得解讀各群。
  真實 > null 但真實 ≈ 區塊 null → 多出來的只是類別內配額，仍然沒有族群。
- 最佳 k：K-means / MiniBatch / K-modes 取 silhouette 最高的 k；GMM 取 BIC 最低。
  當所有 k 的 silhouette 都 ≈ 0 時，「最佳 k」只是雜訊裡挑一個，不要過度解讀。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from kmodes.kmodes import KModes  # noqa: E402
from sklearn.cluster import KMeans, MiniBatchKMeans  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from heartlink_ml import data, evaluation, features, plotting  # noqa: E402
from heartlink_ml.config import FIG_DIR, MODEL_DIR, OUTPUT_DIR, SEED, TRAIT_CATEGORIES, ensure_dirs  # noqa: E402

# plotting 模組已經設定好 Agg 後端與中文字型，這裡直接借用 pyplot 畫基礎層沒有的
# 「實線 vs 虛線」與多子圖。
import matplotlib.pyplot as plt  # noqa: E402

MODULE = "clustering_partitional"
SUBSAMPLE = 3000                 # O(n²) 的 silhouette 只用這麼多筆
SVD_DIM = 16
K_RANGE_KMEANS = range(2, 13)    # 2..12
K_RANGE_KMODES = range(2, 11)    # 2..10
K_RANGE_GMM = range(2, 9)        # 2..8
N_INIT = 5                       # K-means / MiniBatch 每個 k 重跑幾次取最好
KMODES_N_INIT = 3
STABILITY_RUNS = 5
# 三份資料跑同一套流程：真實、逐欄打散 null（硬規則要求）、區塊置換 null（本檔補充）
KINDS = ("real", "null", "block_null")
KIND_LABEL = {"real": "真實", "null": "null", "block_null": "區塊null"}
KIND_STYLE = {"real": "-", "null": "--", "block_null": ":"}


def _num(x) -> float:
    """指標可能是 None（群數 < 2 時 silhouette 沒定義），畫圖與格式化時當 nan。"""
    return float("nan") if x is None else float(x)


# ------------------------------------------------------------------------------
# 區塊置換 null（基礎層沒有）：保留類別內配額與互斥，只破壞跨類別關聯
# ------------------------------------------------------------------------------
def block_permutation_null(X: np.ndarray, cols, seed=SEED) -> np.ndarray:
    """對每個標籤類別（dating_goal / interest / ...）把整塊欄位「整列一起」隨機重排，
    每個類別用不同的排列。每個人在每個類別內勾了哪幾個、勾幾個都原封不動地
    搬給另一個人，所以類別內的配額與互斥完全保留；但同一個人的 value 塊和
    interest 塊來自不同人，跨類別的關聯被打掉。"""
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    for cat in TRAIT_CATEGORIES:
        jj = [j for j, c in enumerate(cols) if data.column_category(c) == cat]
        perm = rng.permutation(X.shape[0])
        Xn[:, jj] = X[perm][:, jj]
    return Xn


# ------------------------------------------------------------------------------
# 前處理：一份資料（真實或 null）→ 所有方法需要的東西
# ------------------------------------------------------------------------------
def prepare(X: np.ndarray, cols, idx: np.ndarray) -> dict:
    """X 是 (n, 68) bool。回傳 SVD 嵌入、抽樣的 Jaccard 距離矩陣等，真實與 null 各做一次。"""
    Xw, w = features.weighted_matrix(X, cols)
    emb, svd = features.svd_embed(Xw, n_components=SVD_DIM, seed=SEED)
    return {
        "X": X,
        "X_int": X.astype(np.uint8),           # K-modes 用：kmodes 0.12.2 在 numpy 2.x 上 bool/uint8 都能跑，統一轉 uint8 最保險
        "emb": emb, "svd": svd, "weights": w,
        "D": features.jaccard_distance_matrix(X[idx]),   # 抽樣的 Jaccard 矩陣（所有方法共用）
        "emb_sub": emb[idx],                              # DB / CH 用的共同尺規
        "X_sub_raw": X[idx].astype(np.float32),           # K-modes 額外在原始 0/1 上算 DB / CH
    }


def quality(prep: dict, labels: np.ndarray, idx: np.ndarray) -> dict:
    """在同一份抽樣上算 silhouette(Jaccard) / DB / CH。labels 是全部 10,000 筆的。"""
    return evaluation.cluster_quality(prep["D"], prep["emb_sub"], labels[idx])


# ------------------------------------------------------------------------------
# 四個方法：每個都提供 fit(k, seed) -> (model, labels)
# ------------------------------------------------------------------------------
def fit_kmeans(prep, k, seed):
    m = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(prep["emb"])
    return m, m.labels_


def fit_minibatch(prep, k, seed):
    m = MiniBatchKMeans(n_clusters=k, n_init=N_INIT, batch_size=1024, random_state=seed).fit(prep["emb"])
    return m, m.labels_


def fit_kmodes(prep, k, seed):
    # init="Cao" 是「確定性」初始化（依密度挑起點，不用亂數）：kmodes 0.12.2 會悄悄把
    # n_init 降成 1，換 random_state 結果也完全一樣。掃 k 與存模型用它（可重現），
    # 但穩定度不能用它（ARI 永遠 = 1，測不到東西），見 fit_kmodes_random。
    m = KModes(n_clusters=k, init="Cao", n_init=KMODES_N_INIT, random_state=seed, verbose=0)
    labels = m.fit_predict(prep["X_int"])
    return m, np.asarray(labels)


def fit_kmodes_random(prep, k, seed):
    """只給 stability_ari 用：init="Huang" 會隨機挑初始質心，換 seed 才真的會變。"""
    m = KModes(n_clusters=k, init="Huang", n_init=KMODES_N_INIT, random_state=seed, verbose=0)
    labels = m.fit_predict(prep["X_int"])
    return m, np.asarray(labels)


def fit_gmm(prep, k, seed):
    m = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed).fit(prep["emb"])
    return m, m.predict(prep["emb"])


METHODS = {
    # 名稱: (fit 函式, k 範圍, 目標函數欄位名, 目標函數怎麼從模型取)
    "kmeans":    (fit_kmeans,    K_RANGE_KMEANS, "inertia", lambda m, prep: float(m.inertia_)),
    "minibatch": (fit_minibatch, K_RANGE_KMEANS, "inertia", lambda m, prep: float(m.inertia_)),
    "kmodes":    (fit_kmodes,    K_RANGE_KMODES, "cost",    lambda m, prep: float(m.cost_)),
    "gmm":       (fit_gmm,       K_RANGE_GMM,    "bic",     lambda m, prep: float(m.bic(prep["emb"]))),
}


def sweep(name: str, prep: dict, idx: np.ndarray) -> list[dict]:
    """對一個方法掃過所有 k，回傳每個 k 的指標列表。"""
    fit, k_range, obj_name, obj_fn = METHODS[name]
    rows = []
    for k in k_range:
        t0 = time.time()
        model, labels = fit(prep, k, SEED)
        q = quality(prep, labels, idx)
        row = {"k": k, obj_name: obj_fn(model, prep), "seconds": round(time.time() - t0, 2), **q}
        if name == "kmodes":
            # K-modes 額外在原始 0/1 上算一組 DB / CH（Hamming 對應的歐氏空間）
            q_raw = evaluation.cluster_quality(prep["D"], prep["X_sub_raw"], labels[idx])
            row["davies_bouldin_raw01"] = q_raw["davies_bouldin"]
            row["calinski_harabasz_raw01"] = q_raw["calinski_harabasz"]
        if name == "gmm":
            row["aic"] = float(model.aic(prep["emb"]))
            row["converged"] = bool(model.converged_)
        rows.append(row)
    return rows


def pick_best_k(name: str, rows: list[dict]) -> int:
    """GMM 用 BIC 最小；其他用 silhouette 最大。"""
    if name == "gmm":
        return min(rows, key=lambda r: r["bic"])["k"]
    return max(rows, key=lambda r: (r["silhouette_jaccard"] if r["silhouette_jaccard"] is not None else -1))["k"]


# 穩定度用的 fit：K-modes 換成隨機初始化版本，其他方法本來就吃 random_state
STABILITY_FIT = {name: METHODS[name][0] for name in METHODS}
STABILITY_FIT["kmodes"] = fit_kmodes_random


def stability(name: str, prep: dict, k: int) -> float:
    fit = STABILITY_FIT[name]
    return evaluation.stability_ari(lambda seed: fit(prep, k, seed)[1], n_runs=STABILITY_RUNS, seed=SEED)


def kmodes_profile(model, X: np.ndarray, labels: np.ndarray, cols, top=6) -> list[dict]:
    """K-modes 每一群的「長相」：眾數質心裡是 1 的標籤、群大小、群內平均總標籤數，
    以及跟其他人差最多的幾個標籤（群內比例 vs 群外比例）。用來檢查群是在切什麼。"""
    row_sum = X.sum(1)
    out = []
    for c in range(model.n_clusters):
        m = labels == c
        if m.sum() == 0:
            continue
        diff = X[m].mean(0) - X[~m].mean(0)
        top_j = np.argsort(-np.abs(diff))[:top]
        out.append({
            "cluster": int(c), "size": int(m.sum()),
            "mean_total_labels": round(float(row_sum[m].mean()), 2),
            "mode_labels": [cols[j] for j in np.flatnonzero(model.cluster_centroids_[c].astype(int))],
            "most_different_labels": {cols[j]: round(float(diff[j]), 2) for j in top_j},
        })
    return out


# ------------------------------------------------------------------------------
# 畫圖（基礎層沒有「實線 vs 虛線」與多子圖，這裡自己畫）
# ------------------------------------------------------------------------------
def plot_elbow(results: dict, path) -> None:
    """三個子圖：K-means/MiniBatch 的 inertia、K-modes 的 cost、GMM 的 BIC。真實實線、null 虛線。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        (axes[0], ["kmeans", "minibatch"], "inertia", "K-means / MiniBatch：inertia（越低越緊）"),
        (axes[1], ["kmodes"], "cost", "K-modes：cost（Hamming 總距離）"),
        (axes[2], ["gmm"], "bic", "GMM：BIC（越低越好，取最低點）"),
    ]
    for ax, names, key, title in panels:
        for name in names:
            for kind in KINDS:
                rows = results[name]["sweep"][kind]
                ax.plot([r["k"] for r in rows], [r[key] for r in rows], marker="o", linestyle=KIND_STYLE[kind],
                        label=f"{name}（{KIND_LABEL[kind]}）")
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_silhouette_vs_k(results: dict, path) -> None:
    """四個方法同一張：真實實線、null 虛線、區塊 null 點線，同色配對。"""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, name in enumerate(results):
        for kind in KINDS:
            rows = results[name]["sweep"][kind]
            ax.plot([r["k"] for r in rows], [_num(r["silhouette_jaccard"]) for r in rows],
                    marker="o", markersize=4, linestyle=KIND_STYLE[kind], color=colors[i % len(colors)],
                    label=f"{name}（{KIND_LABEL[kind]}）")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.axhline(0.25, color="gray", linewidth=0.8, linestyle="-.", label="0.25：勉強算有群的門檻")
    ax.set_title(f"silhouette（Jaccard，抽樣 {SUBSAMPLE} 筆）vs k：真實實線、null 虛線、區塊 null 點線")
    ax.set_xlabel("k")
    ax.set_ylabel("silhouette_jaccard")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------
# 主程式
# ------------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    ensure_dirs()
    out_dir = OUTPUT_DIR / MODULE
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 載入資料 → 68 維 0/1；真實 vs null 用同一份抽樣索引
    df = data.load_seed()
    cols = data.trait_columns(df)
    X = features.binary_matrix(df, cols)
    n = X.shape[0]
    idx = features.subsample(n, SUBSAMPLE, seed=SEED)
    X_by_kind = {
        "real": X,
        "null": evaluation.column_permutation_null(X, seed=SEED),   # 逐欄打散（硬規則要求的 null）
        "block_null": block_permutation_null(X, cols, seed=SEED),  # 區塊置換（保留類別內配額）
    }
    print(f"資料 {X.shape}，抽樣 {len(idx)} 筆算 silhouette；每人平均 {X.sum(1).mean():.2f} 個標籤")
    print("每人總標籤數 std：" + "，".join(f"{KIND_LABEL[kd]} {M.sum(1).std():.2f}" for kd, M in X_by_kind.items())
          + "（區塊 null 與真實相同，逐欄 null 變大 = 配額被打掉）")

    preps = {kind: prepare(M, cols, idx) for kind, M in X_by_kind.items()}
    hopkins = {kind: evaluation.hopkins_binary(p["X"], seed=SEED) for kind, p in preps.items()}
    print("Hopkins：" + "，".join(f"{KIND_LABEL[kd]} {h:.3f}" for kd, h in hopkins.items()) + "（0.5 = 與隨機無異）")
    print(f"SVD-{SVD_DIM} 解釋變異：" + "，".join(
        f"{KIND_LABEL[kd]} {p['svd'].explained_variance_ratio_.sum():.3f}" for kd, p in preps.items()))

    # 2. 每個方法：掃 k（真實 + 兩種 null）→ 選最佳 k → 穩定度 → 存模型 → 散點圖
    results = {}
    for name in METHODS:
        t0 = time.time()
        sw = {kind: sweep(name, preps[kind], idx) for kind in KINDS}
        best_k = {kind: pick_best_k(name, sw[kind]) for kind in KINDS}
        k = best_k["real"]
        best_row = next(r for r in sw["real"] if r["k"] == k)
        # 同一個 k 的 null，才是公平對照
        same_k = {kind: next(r for r in sw[kind] if r["k"] == k) for kind in ("null", "block_null")}
        stab = {kind: stability(name, preps[kind], k) for kind in ("real", "null")}

        # 重新 fit 一次最佳模型存檔（連同前處理，之後才能對新使用者用）
        model, labels = METHODS[name][0](preps["real"], k, SEED)
        joblib.dump({
            "model": model, "svd": preps["real"]["svd"], "weights": preps["real"]["weights"],
            "columns": cols, "k": k, "method": name, "svd_dim": SVD_DIM,
            # 新資料要餵什麼：K-modes 吃原始 0/1（uint8）；其他三個吃 weighted_matrix → svd.transform
            "input_space": "raw01_uint8" if name == "kmodes" else "weighted_svd",
        }, MODEL_DIR / f"{MODULE}_{name}.joblib")
        plotting.scatter_2d(preps["real"]["emb"][:, :2], labels,
                            f"{name}（k={k}）在 SVD 前兩維上的分佈；silhouette={_num(best_row['silhouette_jaccard']):.3f}",
                            FIG_DIR / f"{MODULE}_scatter_{name}.png")

        results[name] = {
            "k_range": [int(a) for a in METHODS[name][1]],
            "best_k": best_k,
            "best_real": best_row,
            "null_at_same_k": same_k["null"],
            "block_null_at_same_k": same_k["block_null"],
            "stability_ari": stab,
            "stability_init": "Huang（隨機初始化；Cao 為確定性，ARI 恆為 1）" if name == "kmodes" else "random_state",
            "sweep": sw,
            "seconds": round(time.time() - t0, 1),
        }
        if name == "kmodes":
            # K-modes 的群到底在切什麼：每群的眾數標籤 + 每群平均總標籤數（驗證不是在切「勾了幾個」）
            results[name]["cluster_profile"] = kmodes_profile(model, X, labels, cols)
        print(f"[{name}] 最佳 k={k}：silhouette 真實 {_num(best_row['silhouette_jaccard']):.3f} / "
              f"null {_num(same_k['null']['silhouette_jaccard']):.3f} / 區塊null {_num(same_k['block_null']['silhouette_jaccard']):.3f}，"
              f"stability_ari 真實 {stab['real']:.3f} / null {stab['null']:.3f}（{results[name]['seconds']}s）")

    # 3. 圖
    plot_elbow(results, FIG_DIR / f"{MODULE}_elbow.png")
    plot_silhouette_vs_k(results, FIG_DIR / f"{MODULE}_silhouette_vs_k.png")

    # 4. 結論：完全依數字寫，不硬掰
    sil_real = {name: _num(r["best_real"]["silhouette_jaccard"]) for name, r in results.items()}
    gap_null = {name: sil_real[name] - _num(r["null_at_same_k"]["silhouette_jaccard"]) for name, r in results.items()}
    gap_block = {name: sil_real[name] - _num(r["block_null_at_same_k"]["silhouette_jaccard"]) for name, r in results.items()}
    max_sil_real = np.nanmax(list(sil_real.values()))
    max_gap = np.nanmax(list(gap_null.values()))
    max_gap_block = np.nanmax(list(gap_block.values()))
    widest = max(gap_null, key=gap_null.get)   # 真實 vs 逐欄 null 差距最大的方法（實測是 K-modes）
    if max_sil_real < 0.10 and max_gap < 0.05 and abs(hopkins["real"] - 0.5) < 0.05:
        conclusion = (f"四個分割式方法在真實資料上的最佳 silhouette 最高只有 {max_sil_real:.3f}，"
                      f"與 null（逐欄打散）的差距最多 {max_gap:.3f}，與區塊 null（保留類別內配額）的差距最多 {max_gap_block:.3f}，"
                      f"Hopkins={hopkins['real']:.3f}≈0.5。"
                      "三道保險一致指向：這份標籤資料沒有群結構，分群切出來的只是把均勻的雲硬切成 k 塊。"
                      "這是預期的正確結果（標籤是獨立隨機生成的），不應拿這些群當使用者族群解讀或做推薦。"
                      "stability_ari 偏高的方法只代表最佳化收斂穩定，不代表群存在。"
                      f"{widest} 真實略高於逐欄 null 的那 {gap_null[widest]:.3f}，在區塊 null 上縮到 {gap_block[widest]:.3f}："
                      "多出來的只是產生器「每類別固定配額」造成的類別內互斥（value 5 選 2–3、diet 5 選 1–3），"
                      "不是使用者族群，也不是總標籤數的差異（K-modes 各群的平均總標籤數幾乎相同，見 cluster_profile）。")
    else:
        conclusion = (f"真實資料最佳 silhouette {max_sil_real:.3f}、與逐欄 null 差距 {max_gap:.3f}、"
                      f"與區塊 null 差距 {max_gap_block:.3f}、Hopkins={hopkins['real']:.3f}。"
                      "真實與 null 有差距，值得進一步檢視各群的眾數標籤，但仍需與 Hopkins、穩定度一起判讀。")

    # 5. 摘要表 + metrics.json
    table_rows = []
    for name, r in results.items():
        b, nl, bl = r["best_real"], r["null_at_same_k"], r["block_null_at_same_k"]
        table_rows.append({
            "方法": name, "最佳k": r["best_k"]["real"],
            "silhouette(真實)": b["silhouette_jaccard"], "silhouette(null)": nl["silhouette_jaccard"],
            "silhouette(區塊null)": bl["silhouette_jaccard"],
            "DB": b["davies_bouldin"], "CH": b["calinski_harabasz"],
            "stability_ari": r["stability_ari"]["real"], "stability_ari(null)": r["stability_ari"]["null"],
        })
    runtime = round(time.time() - t_start, 1)
    metrics = {
        "module": MODULE,
        "dataset": {"n": int(n), "n_traits": int(X.shape[1]), "subsample_for_silhouette": int(len(idx)),
                    "seed": SEED, "svd_dim": SVD_DIM,
                    "total_labels_std": {kd: float(M.sum(1).std()) for kd, M in X_by_kind.items()},
                    "svd_explained_variance": {kd: float(p["svd"].explained_variance_ratio_.sum()) for kd, p in preps.items()}},
        "null_models": {
            "null": "evaluation.column_permutation_null：逐欄獨立打散，保留流行度、破壞所有共現（含類別內配額）",
            "block_null": "block_permutation_null（本檔）：每個類別整塊整列重排，保留類別內配額與互斥、只破壞跨類別關聯",
        },
        "hopkins": hopkins,
        "methods": results,
        "summary_table": table_rows,
        "conclusion": conclusion,
        "runtime_seconds": runtime,
    }
    evaluation.save_json(metrics, out_dir / "metrics.json")

    print("\n| 方法 | 最佳k | silhouette(真實) | silhouette(null) | silhouette(區塊null) | DB | CH | stability_ari |")
    print("|---|---|---|---|---|---|---|---|")
    for row in table_rows:
        print(f"| {row['方法']} | {row['最佳k']} | {_num(row['silhouette(真實)']):.3f} | {_num(row['silhouette(null)']):.3f} | "
              f"{_num(row['silhouette(區塊null)']):.3f} | {_num(row['DB']):.2f} | {_num(row['CH']):.0f} | {row['stability_ari']:.2f} |")
    print("\nHopkins：" + " / ".join(f"{KIND_LABEL[kd]} {h:.3f}" for kd, h in hopkins.items()))
    print("註：silhouette 用同一份 3,000 筆抽樣的 Jaccard 矩陣；DB / CH 在 SVD-16 空間；"
          "silhouette(null / 區塊null) 是同一個 k 的兩種 null；K-modes 穩定度用 init=Huang（Cao 為確定性）。")
    for p in results["kmodes"].get("cluster_profile", []):
        print(f"K-modes 群 {p['cluster']}：{p['size']} 人，平均 {p['mean_total_labels']} 個標籤，"
              f"眾數標籤 {p['mode_labels']}")
    print(f"結論：{conclusion}")
    print(f"總耗時 {runtime}s；輸出 {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
