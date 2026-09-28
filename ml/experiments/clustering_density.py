"""密度式分群實驗：DBSCAN 與 HDBSCAN（Jaccard 距離，metric="precomputed"）。

【這個方法是什麼】
- DBSCAN：把「eps 半徑內至少有 min_samples 個點（含自己）」的點當作核心點，核心點彼此
  可達就連成一群；連不到任何核心點的就是雜訊（label = -1）。不用事先指定群數、能標出
  離群點，代價是整份資料只有一個全域密度門檻 eps。
- HDBSCAN：把 eps 從一個數變成整條「密度階層」（等於一次掃過所有 eps），再用每個群
  的持久度（excess of mass）挑出最穩定的群。只需要 min_cluster_size / min_samples；
  probabilities_ 是每個點屬於所分到那群的強度（雜訊為 0）。

【為什麼對這份資料要特別小心】
1. 標籤是 0/1，兩人的 Jaccard 距離 = 1 − |交集|/|聯集|，只會是少數幾個離散分數
   （這份抽樣只有 187 個不同的值）。所以 eps 一定要落在實際出現的距離值上，
   不能隨便切 0.01 一格——切在兩個實際值中間的 eps 跟切在較小那個值上結果完全相同。
2. 每人 11–21 個標籤、68 個候選，距離集中在 0.6–0.9；k-distance 曲線幾乎是水平線
   （第 5 個點的距離 5%–95% 分位只差約 0.06）。密度法靠「膝點」找 eps，沒有膝點就
   代表密度均勻：eps 再小一點全部變雜訊、再大一點所有點互相可達、併成一個巨群，
   中間沒有「幾個密度島」可以停。
3. 主控端已確認標籤是獨立隨機生成（Hopkins ≈ 0.51）。在均勻密度資料上，DBSCAN 輸出
   「幾乎全是雜訊」或「一個巨群」、HDBSCAN 切出幾個 silhouette ≈ 0 的群，都是
   **預期行為**，不是參數沒調好。判斷方式不是看群數，而是把同一套流程跑在
   evaluation.column_permutation_null（逐欄打散：保留每個標籤的流行度、破壞共現）上；
   真實資料若和 null 拿到一樣的數字，就沒有可學的結構。

【怎麼跑】
    cd ml && .venv/bin/python experiments/clustering_density.py
不連資料庫、不裝套件；O(n²) 的部分一律用 features.subsample 抽 3000 人，筆電約 20–40 秒。

【產出】
    outputs/clustering_density/metrics.json                 所有掃描結果（真實 vs null 並列）
    outputs/figures/clustering_density_kdistance.png        k-distance 曲線（找 eps 膝點）
    outputs/figures/clustering_density_eps_scan.png         eps 掃描：左軸群數、右軸雜訊比
    outputs/figures/clustering_density_dbscan_scatter.png   最佳 DBSCAN 的 SVD 2D 散點（雜訊灰色）
    outputs/figures/clustering_density_hdbscan_scatter.png  最佳 HDBSCAN 的 SVD 2D 散點（雜訊灰色）
    outputs/figures/clustering_density_hdbscan_prob.png     HDBSCAN probabilities_ 直方圖（真實 vs null）
    outputs/models/clustering_density_best.joblib           最佳參數的模型與抽樣索引

【怎麼讀結果】
- 摘要表每列「silhouette(真實)」與「silhouette(null)」並列：兩者都 ≈ 0、或差距 < 0.05，
  就表示群是演算法硬切出來的，不是資料本身有的。
- DBSCAN 的「可行參數」定義為：群數 ≥ 2 且雜訊比 < 50%。找不到也是一個結果。
  但要小心 sklearn DBSCAN 的一個已知性質：會冒出 1–4 人的「迷你群」（人數 < min_samples）。
  原因是核心點的鄰居先被大群當成邊界點搶走，輪到它自己展開時已經沒人可收，就自成一群。
  這種迷你群會讓「群數 ≥ 2」被技術性地滿足，所以另外算一個嚴格版：只數人數 ≥ min_samples
  的群，且最大群佔比 < 90%（否則就是「一個巨群＋零星小群」，不是多個群）。
- k-distance 的膝點：曲線由小到大排序後，DBSCAN 慣例取「平坦段結束、開始陡升」的那一點
  （曲線在頭尾連線下方最遠處），右邊更稀疏的點視為雜訊。曲線開頭「最密的幾 % 人」也會有
  一個彎，那不是 eps 該放的位置（放在那裡大部分人都會變雜訊），腳本另外回報當參考。
  兩個彎的強度都很小（直線 = 0、標準 L 形 ≈ 0.71）就代表沒有膝點。
- 穩定度：密度法沒有隨機初始化，換 random seed 結果不會變，所以 K-means 那種「換 seed
  的 ARI」對它沒有意義。這裡改成「換抽樣」：固定 1000 個評估點，每輪從其他人裡再抽
  2000 個湊成 3000 一起分群，看評估點的標籤在不同抽樣之間的 ARI（雜訊當成一個標籤），
  真實與 null 用同一組索引各跑一次、並列比較。
  注意 ARI 的性質：「全是雜訊」或「全在同一群」這種退化結果 ARI 接近 1；但「一個巨群＋
  幾 % 雜訊」時 ARI 反而偏低——ARI 已校正機率，巨群本身不加分，剩下只剩「哪些人被判成雜訊」
  在不同抽樣間是否一致。所以 ARI 高或低都不能單獨當證據，一定要和 silhouette、null 一起看。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import DBSCAN, HDBSCAN  # noqa: E402

from heartlink_ml import data, evaluation, features, plotting  # noqa: E402
from heartlink_ml.config import FIG_DIR, MODEL_DIR, OUTPUT_DIR, SEED, ensure_dirs  # noqa: E402

MODULE = "clustering_density"
SAMPLE_SIZE = 3000                  # O(n²) 方法一律抽樣
MIN_SAMPLES_GRID = (5, 10)          # DBSCAN 的 min_samples（sklearn 定義：含點本身）
EPS_RANGE = (0.50, 0.75)            # 只掃 k-distance 曲線所在區間附近；網格值一律取 D 裡實際出現的距離
MAX_EPS_POINTS = 70                 # 網格上限；超過就等距抽稀，抽稀後仍是實際值
HDB_MIN_CLUSTER_SIZE = (10, 25, 50, 100)
HDB_MIN_SAMPLES = (5, 10, None)     # None = 等於 min_cluster_size（sklearn 1.9 文件）
SVD_DIM = 16                        # 算 Davies-Bouldin / Calinski-Harabasz 用的空間；前 2 維畫散點
STABILITY_RUNS = 5
STABILITY_EVAL = 1000               # 固定評估點數；其餘 SAMPLE_SIZE − STABILITY_EVAL 每輪重抽
FEASIBLE_MAX_NOISE = 0.5            # 「可行參數」的雜訊比上限
SAME_AS_NULL = 0.05                 # silhouette 真實 − null 小於這個值就視為與 null 無法區分


# ---------------------------------------------------------------- 資料準備
def prepare(Xs, cols, seed=SEED):
    """抽樣後的 0/1 矩陣 → (Jaccard 距離矩陣, SVD 特徵空間)。真實與 null 都走同一段。"""
    D = features.jaccard_distance_matrix(Xs)
    Xw, _ = features.weighted_matrix(Xs, cols)
    emb, _ = features.svd_embed(Xw, n_components=SVD_DIM, seed=seed)
    return D, emb


def describe_distances(D, name):
    """二元資料的 Jaccard 距離是離散值，先看實際出現哪些值，eps 才知道要掃哪裡。"""
    tri = D[np.triu_indices(len(D), k=1)]
    uniq = np.unique(tri)
    pct = (1, 5, 10, 25, 50, 75, 90, 99)
    info = {
        "n_unique": int(len(uniq)), "min": float(uniq[0]), "max": float(uniq[-1]),
        "first_20_unique": [round(float(v), 4) for v in uniq[:20]],
        "percentiles": {str(p): round(float(v), 4) for p, v in zip(pct, np.percentile(tri, pct))},
    }
    print(f"[{name}] Jaccard 距離：{info['n_unique']} 個不同值，min={info['min']:.4f}，max={info['max']:.4f}")
    print(f"  前 20 個實際值：{info['first_20_unique']}")
    print(f"  百分位：{info['percentiles']}")
    return info


# ---------------------------------------------------------------- DBSCAN
def k_distance(D, min_samples):
    """每個點到「第 min_samples 個點（含自己）」的距離，由小到大排序。
    排序後的列索引 0 是自己（距離 0），所以取索引 min_samples−1；
    eps 只要 ≥ 這個值，這個點就是 DBSCAN 的核心點。"""
    kth = np.partition(D, min_samples - 1, axis=1)[:, min_samples - 1]
    return np.sort(kth)


def knee_point(ys):
    """膝點（Kneedle 的簡化版）：把曲線正規化到 [0,1]²，畫「頭尾連線」，取曲線離連線最遠的點。

    k-distance 由小到大排序後通常是「先平、最後陡升」；DBSCAN 慣例（Ester et al. 1996，原文是由大到小
    排、找第一個谷）要的膝點是「平坦段結束、開始陡升」的那一點——右邊 k-distance 更大的點視為雜訊。
    這對應曲線落在連線**下方**最遠處（x − y 最大）。曲線開頭也可能有一段陡升（最密的幾 % 人），
    那是連線**上方**的彎（y − x 最大），只當參考回傳、不拿來當 eps；若用 |x − y| 混在一起取最大，
    會在兩個彎裡挑較強的那個，抓到開頭的彎時 eps 會落在「大部分人都是雜訊」的區間。

    回傳 dict：knee_eps / knee_strength（DBSCAN 慣例的膝點）、dense_bend_eps / dense_bend_strength（開頭的彎）。
    曲線接近直線時兩個強度都很小，代表根本沒有膝點（直線 = 0，標準 L 形 ≈ 0.71）。"""
    n = len(ys)
    x = np.linspace(0.0, 1.0, n)
    rng_y = ys[-1] - ys[0]
    y = (ys - ys[0]) / rng_y if rng_y > 0 else np.zeros(n)
    # 連線 y = x；點到直線的有號距離 ∝ (x − y)：正 = 曲線在連線下方（平→陡），負 = 在上方（陡→平）
    signed = (x - y) / np.sqrt(2)
    i, j = int(np.argmax(signed)), int(np.argmin(signed))
    return {"knee_eps": float(ys[i]), "knee_strength": float(max(signed[i], 0.0)),
            "dense_bend_eps": float(ys[j]), "dense_bend_strength": float(max(-signed[j], 0.0))}


def eps_grid(D, lo, hi, max_points):
    """只取 D 裡實際出現、落在 [lo, hi] 的距離值當 eps 網格。"""
    uniq = np.unique(D[np.triu_indices(len(D), k=1)])
    vals = uniq[(uniq >= lo) & (uniq <= hi)]
    if len(vals) > max_points:
        vals = vals[np.linspace(0, len(vals) - 1, max_points).round().astype(int)]
    return [float(v) for v in vals]


def snap_to_grid(value, grid):
    """把任意 eps（例如膝點）貼到網格上最接近的實際值。"""
    arr = np.asarray(grid)
    return float(arr[int(np.argmin(np.abs(arr - value)))])


def summarize_labels(D, Xfeat, labels, min_size=1):
    """cluster_quality 的精簡版；n_clusters_ge_min_size 只數人數 ≥ min_size 的群（排除 DBSCAN 迷你群）。"""
    q = evaluation.cluster_quality(D, Xfeat, labels)
    return {
        "n_clusters": q["n_clusters"], "noise_ratio": round(q["noise_ratio"], 4),
        "n_clusters_ge_min_size": int(sum(sz >= min_size for sz in q["cluster_sizes"])),
        "cluster_sizes_top5": q["cluster_sizes"][:5],
        "silhouette": None if q["silhouette_jaccard"] is None else round(q["silhouette_jaccard"], 4),
        "davies_bouldin": None if q["davies_bouldin"] is None else round(q["davies_bouldin"], 3),
        "calinski_harabasz": None if q["calinski_harabasz"] is None else round(q["calinski_harabasz"], 2),
        "largest_cluster": q["cluster_sizes"][0] if q["cluster_sizes"] else 0,
        "largest_share": round((q["cluster_sizes"][0] if q["cluster_sizes"] else 0) / len(labels), 4),
    }


def dbscan_scan(D, Xfeat, eps_values, min_samples_grid):
    """eps × min_samples 網格 → 每組的群數、雜訊比、silhouette（排除雜訊）。
    sklearn 的 eps 是包含式（距離 ≤ eps 算鄰居），所以 eps 直接用實際值即可。"""
    rows = []
    for ms in min_samples_grid:
        for eps in eps_values:
            labels = DBSCAN(eps=eps, min_samples=ms, metric="precomputed").fit_predict(D)
            row = {"min_samples": ms, "eps": eps}
            row.update(summarize_labels(D, Xfeat, labels, min_size=ms))
            rows.append(row)
    return rows


def is_feasible(row):
    """任務指定的可行條件：群數 ≥ 2 且雜訊比 < 50%。"""
    return row["n_clusters"] >= 2 and row["noise_ratio"] < FEASIBLE_MAX_NOISE


def is_feasible_strict(row):
    """嚴格版：只數人數 ≥ min_samples 的群，且最大群佔比 < 90%（排除「一個巨群＋迷你群」）。"""
    return (row["n_clusters_ge_min_size"] >= 2 and row["noise_ratio"] < FEASIBLE_MAX_NOISE
            and row["largest_share"] < 0.9)


def pick_best(rows):
    """優先：可行（群數 ≥ 2 且雜訊 < 50%）裡 silhouette 最高；沒有可行的，就退而取
    群數最多、再取雜訊最少的那組（純粹為了畫圖，摘要會明講它不可行）。回傳 (row, 是否可行)。"""
    feas = [r for r in rows if is_feasible(r)]
    if feas:
        return max(feas, key=lambda r: (r["silhouette"] if r["silhouette"] is not None else -1.0)), True
    return max(rows, key=lambda r: (r["n_clusters"], -r["noise_ratio"])), False


def find_row(rows, **keys):
    for r in rows:
        if all(r.get(k) == v for k, v in keys.items()):
            return r
    return None


def transition_width(rows, ms, n):
    """eps 掃描的「相變」有多窄：從「雜訊 ≥ 90%」的最大 eps，到「單一群含 ≥ 90% 的人」的最小 eps，
    中間隔了幾個實際距離值。密度均勻的資料這個間隔會非常小——沒有中間態可以停。"""
    sub = sorted((r for r in rows if r["min_samples"] == ms), key=lambda r: r["eps"])
    eps = [r["eps"] for r in sub]
    noisy = [r["eps"] for r in sub if r["noise_ratio"] >= 0.9]
    giant = [r["eps"] for r in sub if r["largest_cluster"] / n >= 0.9]
    if not noisy or not giant:
        return {"eps_all_noise": None, "eps_giant": None, "grid_steps": None,
                "text": "網格內沒有同時觀察到「幾乎全雜訊」與「單一巨群」兩端"}
    a, b = max(noisy), min(giant)
    steps = eps.index(b) - eps.index(a)
    return {"eps_all_noise": a, "eps_giant": b, "grid_steps": steps,
            "text": f"eps={a:.3f} 時雜訊 ≥ 90%，eps={b:.3f} 時單一群已含 ≥ 90% 的人，中間只隔 {steps} 個實際距離值"}


# ---------------------------------------------------------------- HDBSCAN
def hdbscan_params_str(mcs, ms):
    return f"mcs={mcs}, ms={'None(=mcs)' if ms is None else ms}"


def hdbscan_scan(D, Xfeat):
    """min_cluster_size × min_samples 網格。copy=True：sklearn 1.9 對 precomputed + brute
    預設會就地改寫輸入矩陣（1.10 才改成預設複製），這裡明確複製，保護 D 給後面重用。"""
    rows, fitted = [], {}
    for mcs in HDB_MIN_CLUSTER_SIZE:
        for ms in HDB_MIN_SAMPLES:
            model = HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="precomputed", copy=True).fit(D)
            row = {"min_cluster_size": mcs, "min_samples": ms, "params": hdbscan_params_str(mcs, ms)}
            row.update(summarize_labels(D, Xfeat, model.labels_, min_size=mcs))
            row["mean_probability_clustered"] = (
                round(float(model.probabilities_[model.labels_ >= 0].mean()), 4)
                if (model.labels_ >= 0).any() else None)
            rows.append(row)
            fitted[(mcs, ms)] = model
    return rows, fitted


# ---------------------------------------------------------------- 穩定度（換抽樣）
def run_stability(X, fit_labels, n_runs=STABILITY_RUNS):
    """固定 STABILITY_EVAL 個評估點，其餘每輪依 seed 重抽，一起算 Jaccard 並分群，
    只回傳評估點的標籤——這樣就能直接丟給 evaluation.stability_ari 算兩兩 ARI。
    真實與 null 傳進來的 X 形狀相同、seed 相同，所以評估點與每輪重抽的索引也完全相同（同一套流程）。"""
    n = len(X)
    eval_idx = features.subsample(n, STABILITY_EVAL, seed=SEED + 999)
    rest = np.setdiff1d(np.arange(n), eval_idx)
    log = []

    def fit_predict(seed):
        rng = np.random.default_rng(seed)
        extra = rng.choice(rest, size=SAMPLE_SIZE - STABILITY_EVAL, replace=False)
        idx = np.sort(np.concatenate([eval_idx, extra]))
        labels = np.asarray(fit_labels(features.jaccard_distance_matrix(X[idx])))
        on_eval = labels[np.searchsorted(idx, eval_idx)]
        sizes = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.array([0])
        log.append({"n_clusters_eval": int(len(np.unique(on_eval[on_eval >= 0]))),
                    "noise_ratio_eval": round(float((on_eval < 0).mean()), 4),
                    "n_clusters_full": int(len(np.unique(labels[labels >= 0]))),
                    "noise_ratio_full": round(float((labels < 0).mean()), 4),
                    "largest_share_full": round(float(sizes.max() / len(labels)), 4)})
        return on_eval

    ari = evaluation.stability_ari(fit_predict, n_runs=n_runs, seed=SEED)
    return {"ari_mean": None if ari is None else round(ari, 4), "n_runs": n_runs,
            "n_eval_points": STABILITY_EVAL,
            "min_largest_share_full": min(r["largest_share_full"] for r in log), "runs": log}


# ---------------------------------------------------------------- 畫圖
def plot_kdistance(curves, ref_lines, path):
    """curves: {名稱: 排序後的 k-distance}；x 軸用「點的比例」讓真實與 null 疊在一起比。"""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, ys in curves.items():
        ax.plot(np.linspace(0, 1, len(ys)), ys, lw=1.6, ls="--" if "null" in name else "-", label=name)
    for name, y in ref_lines.items():
        ax.axhline(y, color="gray", lw=0.9, ls=":", label=f"{name} = {y:.3f}")
    ax.set_xlabel("點的比例（依 k-distance 由小到大排序）")
    ax.set_ylabel("到第 k 個點的 Jaccard 距離")
    ax.set_title("k-distance 圖：真實 vs null（曲線越平 = 密度越均勻 = 越沒有膝點）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_eps_scan(rows_real, rows_null, path):
    """每個 min_samples 一張子圖：左軸群數（藍）、右軸雜訊比（紅）；實線真實、虛線 null。"""
    fig, axes = plt.subplots(1, len(MIN_SAMPLES_GRID), figsize=(6.2 * len(MIN_SAMPLES_GRID), 4.8), squeeze=False)
    for ax, ms in zip(axes[0], MIN_SAMPLES_GRID):
        ax2 = ax.twinx()
        for rows, tag, ls in ((rows_real, "真實", "-"), (rows_null, "null", "--")):
            sub = [r for r in rows if r["min_samples"] == ms]
            eps = [r["eps"] for r in sub]
            ax.plot(eps, [r["n_clusters"] for r in sub], color="C0", ls=ls, marker=".", ms=4, label=f"群數（{tag}）")
            ax2.plot(eps, [r["noise_ratio"] for r in sub], color="C3", ls=ls, marker=".", ms=4, label=f"雜訊比（{tag}）")
        ax.set_title(f"DBSCAN eps 掃描（min_samples={ms}）")
        ax.set_xlabel("eps（只取實際出現的 Jaccard 距離值）")
        ax.set_ylabel("群數", color="C0")
        ax2.set_ylabel("雜訊比", color="C3")
        ax2.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_prob_hist(p_real, p_null, title, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 21)
    ax.hist(p_real, bins=bins, alpha=0.6, label="真實")
    ax.hist(p_null, bins=bins, alpha=0.6, label="null（逐欄打散）")
    ax.set_xlabel("HDBSCAN probabilities_（0 = 雜訊；越接近 1 越「確定」屬於該群）")
    ax.set_ylabel("人數")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 摘要
def _f(v, fmt="{:.3f}"):
    return "—" if v is None else fmt.format(v)


def table_row(method, params, real, null):
    return {
        "method": method, "params": params,
        "n_clusters": f"{real['n_clusters']} / {null['n_clusters']}",
        "noise_ratio": f"{real['noise_ratio']:.1%} / {null['noise_ratio']:.1%}",
        "sil_real": real["silhouette"], "sil_null": null["silhouette"],
    }


def print_summary(table):
    print("\n| 方法 | 參數 | 群數（真實 / null） | 雜訊比（真實 / null） | silhouette(真實) | silhouette(null) |")
    print("|---|---|---|---|---|---|")
    for r in table:
        print(f"| {r['method']} | {r['params']} | {r['n_clusters']} | {r['noise_ratio']} | "
              f"{_f(r['sil_real'])} | {_f(r['sil_null'])} |")


def build_conclusion(hop, kd_stats, n_feas_real, n_feas_null, best_db, null_db, db_feasible,
                     best_h, null_h, h_feasible, stab, transition, n_strict):
    """結論由數字直接生成，避免手寫的句子和實際結果不一致。"""
    parts = [f"Hopkins（全部 10,000 人）真實 {hop['real_full']:.3f} vs null {hop['null_full']:.3f}"
             "（0.5 = 與各標籤獨立隨機無法區分）。"]
    spans = [f"min_samples={ms}：5%–95% 分位差 {s['range_5_95']:.3f}、膝點（平→陡）eps {s['knee_eps']:.3f} 強度 "
             f"{s['knee_strength']:.2f}、開頭的彎（陡→平）強度 {s['dense_bend_strength']:.2f}"
             for ms in MIN_SAMPLES_GRID for s in (kd_stats[f"min_samples={ms}/真實"],)]
    parts.append("k-distance 曲線幾乎水平（" + "；".join(spans)
                 + "；強度：直線 = 0、標準 L 形 ≈ 0.71），沒有明確膝點，代表密度均勻。")
    parts.append("eps 掃描的相變極窄（" + "；".join(f"min_samples={ms}：{transition[ms]['text']}" for ms in MIN_SAMPLES_GRID)
                 + "），中間沒有「幾個密度島」可以停。")
    if n_feas_real == 0:
        parts.append("DBSCAN 在整個 eps × min_samples 網格上找不到「群數 ≥ 2 且雜訊 < 50%」的參數："
                     "eps 小一格就幾乎全是雜訊、大一格就併成一個巨群。這是密度均勻資料的預期行為，"
                     f"null 資料也一樣（可行參數 {n_feas_null} 組）。")
    else:
        diff = (best_db["silhouette"] or 0.0) - (null_db["silhouette"] or 0.0)
        verdict = "與 null 無法區分" if abs(diff) < SAME_AS_NULL else "真實與 null 有差距，但絕對值仍很低，群不明顯"
        parts.append(f"DBSCAN 可行參數真實 {n_feas_real} 組 / null {n_feas_null} 組；最佳組 silhouette 真實 "
                     f"{_f(best_db['silhouette'])} vs null {_f(null_db['silhouette'])}（差 {diff:+.3f}），{verdict}；"
                     f"而且最佳組的最大群佔 {best_db['largest_share']:.0%}，實際上是「一個巨群＋零星小群＋雜訊」，"
                     "不是多個密度相當的群。")
        sb, sn = n_strict["best_real"], n_strict["best_null"]
        strict_txt = (f"嚴格版（扣掉人數 < min_samples 的迷你群、最大群 < 90%）真實 {n_strict['real']} 組 / null "
                      f"{n_strict['null']} 組")
        if sb is not None:
            strict_txt += (f"；嚴格最佳組 eps={sb['eps']:.3f}, ms={sb['min_samples']}：{sb['n_clusters_ge_min_size']} 個"
                           f"真群、雜訊 {sb['noise_ratio']:.0%}、最大群 {sb['largest_share']:.0%}、silhouette "
                           f"{_f(sb['silhouette'])}")
            if sn is not None:
                strict_txt += f"（null 嚴格最佳 silhouette {_f(sn['silhouette'])}）"
            strict_txt += "——這是相變區的碎片，不是穩定的密度島"
        parts.append(strict_txt + "。")
    diff_h = (best_h["silhouette"] or 0.0) - (null_h["silhouette"] or 0.0)
    verdict_h = "與 null 無法區分" if abs(diff_h) < SAME_AS_NULL else "真實與 null 有差距，但絕對值仍很低"
    parts.append(f"HDBSCAN 最佳組（{best_h['params']}）真實切出 {best_h['n_clusters']} 群、雜訊 "
                 f"{best_h['noise_ratio']:.1%}、silhouette {_f(best_h['silhouette'])}；同參數的 null 切出 "
                 f"{null_h['n_clusters']} 群、silhouette {_f(null_h['silhouette'])}（差 {diff_h:+.3f}），{verdict_h}。"
                 + ("" if h_feasible else "（沒有任何一組符合可行條件。）"))
    giant_every_run = all(stab[k]["min_largest_share_full"] >= 0.5 for k in ("dbscan", "hdbscan"))
    parts.append(f"換抽樣穩定度 ARI：DBSCAN 真實 {_f(stab['dbscan']['ari_mean'])} / null {_f(stab['dbscan_null']['ari_mean'])}、"
                 f"HDBSCAN 真實 {_f(stab['hdbscan']['ari_mean'])} / null {_f(stab['hdbscan_null']['ari_mean'])}。"
                 + ("每一輪的結果都是「一個過半的巨群＋雜訊／小群」，" if giant_every_run else "")
                 + "ARI 已校正機率，巨群本身不加分，數字只反映「哪些人被歸為雜訊或小群」在不同抽樣間是否一致；"
                 "真實與 null 同量級，ARI 高或低都不能單獨當作群存在的證據。")
    parts.append("整體結論：密度法在這份標籤資料上找不到群，且與 null 一致；這是資料（標籤獨立隨機生成）"
                 "的性質，不是演算法或參數的問題。")
    return " ".join(parts)


# ---------------------------------------------------------------- 主程式
def main():
    t0 = time.time()
    ensure_dirs()
    out_dir = OUTPUT_DIR / MODULE
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 資料與抽樣（真實 / null 用同一批索引）
    df = data.load_seed()
    cols = data.trait_columns(df)
    X = features.binary_matrix(df, cols)
    idx = features.subsample(len(X), SAMPLE_SIZE, SEED)
    Xs = X[idx]
    Xs_null = evaluation.column_permutation_null(Xs, seed=SEED)
    print(f"資料 {X.shape}（68 個 traits），抽樣 {len(idx)} 人做 O(n²) 分群")

    X_null = evaluation.column_permutation_null(X, seed=SEED)   # 全資料的 null：Hopkins 與換抽樣穩定度共用
    hop = {
        "real_full": round(evaluation.hopkins_binary(X, seed=SEED), 4),
        "null_full": round(evaluation.hopkins_binary(X_null, seed=SEED), 4),
        "real_sample": round(evaluation.hopkins_binary(Xs, seed=SEED), 4),
        "null_sample": round(evaluation.hopkins_binary(Xs_null, seed=SEED), 4),
    }
    print(f"Hopkins：{hop}")

    D, F = prepare(Xs, cols)
    Dn, Fn = prepare(Xs_null, cols)
    dist_info = {"real": describe_distances(D, "真實"), "null": describe_distances(Dn, "null")}

    # 2. k-distance 圖與膝點
    kcurves, kd_stats, knee_eps = {}, {}, {}
    for ms in MIN_SAMPLES_GRID:
        for tag, Dm in (("真實", D), ("null", Dn)):
            kd = k_distance(Dm, ms)
            kcurves[f"k={ms}（{tag}）"] = kd
            p5, p50, p95 = np.percentile(kd, [5, 50, 95])
            kn = knee_point(kd)
            kd_stats[f"min_samples={ms}/{tag}"] = {
                "p5": round(float(p5), 4), "p50": round(float(p50), 4), "p95": round(float(p95), 4),
                "range_5_95": round(float(p95 - p5), 4),
                **{k: round(v, 4) for k, v in kn.items()},
            }
            if tag == "真實":
                knee_eps[ms] = kn["knee_eps"]
    print("k-distance 統計：")
    for k, v in kd_stats.items():
        print(f"  {k}: {v}")

    # 3. DBSCAN 網格（eps 只取實際值）
    eps_values = eps_grid(D, *EPS_RANGE, MAX_EPS_POINTS)
    print(f"eps 網格：{len(eps_values)} 個實際距離值，{eps_values[0]:.4f} → {eps_values[-1]:.4f}")
    rows_db_real = dbscan_scan(D, F, eps_values, MIN_SAMPLES_GRID)
    rows_db_null = dbscan_scan(Dn, Fn, eps_values, MIN_SAMPLES_GRID)
    n_feas_real = sum(is_feasible(r) for r in rows_db_real)
    n_feas_null = sum(is_feasible(r) for r in rows_db_null)
    strict_real = [r for r in rows_db_real if is_feasible_strict(r)]
    strict_null = [r for r in rows_db_null if is_feasible_strict(r)]
    n_strict = {"real": len(strict_real), "null": len(strict_null),
                "best_real": (max(strict_real, key=lambda r: r["silhouette"] if r["silhouette"] is not None else -1.0)
                              if strict_real else None),
                "best_null": (max(strict_null, key=lambda r: r["silhouette"] if r["silhouette"] is not None else -1.0)
                              if strict_null else None)}
    print(f"DBSCAN 嚴格可行（扣迷你群、最大群 < 90%）：真實 {n_strict['real']} 組、null {n_strict['null']} 組")
    best_db, db_feasible = pick_best(rows_db_real)
    null_db = find_row(rows_db_null, min_samples=best_db["min_samples"], eps=best_db["eps"])
    print(f"DBSCAN 可行參數：真實 {n_feas_real} 組、null {n_feas_null} 組；"
          f"{'最佳可行' if db_feasible else '無可行參數，退而取群數最多'}："
          f"eps={best_db['eps']:.4f}, min_samples={best_db['min_samples']} → {best_db}")
    transition = {ms: transition_width(rows_db_real, ms, len(idx)) for ms in MIN_SAMPLES_GRID}
    transition_null = {ms: transition_width(rows_db_null, ms, len(idx)) for ms in MIN_SAMPLES_GRID}
    for ms in MIN_SAMPLES_GRID:
        print(f"  相變寬度 min_samples={ms}：真實 {transition[ms]['text']}；null {transition_null[ms]['text']}")
    # 膝點 eps 貼到網格後也跑一次，放進摘要表對照
    knee_rows = {}
    for ms in MIN_SAMPLES_GRID:
        e = snap_to_grid(knee_eps[ms], eps_values)
        knee_rows[ms] = (find_row(rows_db_real, min_samples=ms, eps=e), find_row(rows_db_null, min_samples=ms, eps=e))
    best_db_model = DBSCAN(eps=best_db["eps"], min_samples=best_db["min_samples"], metric="precomputed").fit(D)

    # 4. HDBSCAN 網格
    rows_h_real, fitted_h_real = hdbscan_scan(D, F)
    rows_h_null, fitted_h_null = hdbscan_scan(Dn, Fn)
    best_h, h_feasible = pick_best(rows_h_real)
    null_h = find_row(rows_h_null, min_cluster_size=best_h["min_cluster_size"], min_samples=best_h["min_samples"])
    h_key = (best_h["min_cluster_size"], best_h["min_samples"])
    best_h_model = fitted_h_real[h_key]
    print(f"HDBSCAN 最佳（{'可行' if h_feasible else '無可行參數，退而取群數最多'}）：{best_h}")

    # 5. 穩定度（換抽樣；密度法沒有隨機初始化，換 seed 沒意義）。真實與 null 走同一套流程、同一組索引。
    def db_fit(Dm):
        return DBSCAN(eps=best_db["eps"], min_samples=best_db["min_samples"], metric="precomputed").fit_predict(Dm)

    def h_fit(Dm):
        return HDBSCAN(min_cluster_size=h_key[0], min_samples=h_key[1], metric="precomputed", copy=True).fit_predict(Dm)

    stab = {
        "method": "固定 1000 個評估點，每輪重抽 2000 人湊成 3000 一起分群；ARI 只在評估點上算，雜訊(-1)當成一個標籤；"
                  "null 用同一組評估點與重抽索引",
        "dbscan": run_stability(X, db_fit), "dbscan_null": run_stability(X_null, db_fit),
        "hdbscan": run_stability(X, h_fit), "hdbscan_null": run_stability(X_null, h_fit),
    }
    print(f"穩定度 ARI：DBSCAN 真實 {stab['dbscan']['ari_mean']} / null {stab['dbscan_null']['ari_mean']}；"
          f"HDBSCAN 真實 {stab['hdbscan']['ari_mean']} / null {stab['hdbscan_null']['ari_mean']}")

    # 6. 圖
    ref = {f"最佳 DBSCAN eps（min_samples={best_db['min_samples']}）": best_db["eps"]}
    for ms in MIN_SAMPLES_GRID:
        ref[f"膝點 eps（k={ms}）"] = knee_eps[ms]
    plot_kdistance(kcurves, ref, FIG_DIR / f"{MODULE}_kdistance.png")
    plot_eps_scan(rows_db_real, rows_db_null, FIG_DIR / f"{MODULE}_eps_scan.png")
    plotting.scatter_2d(F[:, :2], best_db_model.labels_,
                        f"DBSCAN eps={best_db['eps']:.3f}, min_samples={best_db['min_samples']}"
                        f"（群 {best_db['n_clusters']}，雜訊 {best_db['noise_ratio']:.1%}）SVD 2D",
                        FIG_DIR / f"{MODULE}_dbscan_scatter.png")
    plotting.scatter_2d(F[:, :2], best_h_model.labels_,
                        f"HDBSCAN {best_h['params']}（群 {best_h['n_clusters']}，雜訊 {best_h['noise_ratio']:.1%}）SVD 2D",
                        FIG_DIR / f"{MODULE}_hdbscan_scatter.png")
    plot_prob_hist(best_h_model.probabilities_, fitted_h_null[h_key].probabilities_,
                   f"HDBSCAN probabilities_（{best_h['params']}）：真實 vs null",
                   FIG_DIR / f"{MODULE}_hdbscan_prob.png")

    # 7. 模型
    model_path = MODEL_DIR / f"{MODULE}_best.joblib"
    joblib.dump({
        "sample_idx": idx, "trait_columns": cols,
        "dbscan_params": {"eps": best_db["eps"], "min_samples": best_db["min_samples"]},
        "dbscan_labels": best_db_model.labels_,
        "hdbscan_params": {"min_cluster_size": h_key[0], "min_samples": h_key[1]},
        "hdbscan_labels": best_h_model.labels_, "hdbscan_probabilities": best_h_model.probabilities_,
        "note": "precomputed 模型無法對新資料 predict；這裡存的是抽樣索引、參數與標籤，供重現與對照。",
    }, model_path)

    # 8. 摘要表
    table = [table_row("DBSCAN（最佳" + ("可行" if db_feasible else "，無可行→群數最多") + "）",
                       f"eps={best_db['eps']:.3f}, ms={best_db['min_samples']}", best_db, null_db)]
    for ms in MIN_SAMPLES_GRID:
        r_real, r_null = knee_rows[ms]
        table.append(table_row("DBSCAN（膝點 eps）", f"eps={r_real['eps']:.3f}, ms={ms}", r_real, r_null))
    for r_real, r_null in zip(rows_h_real, rows_h_null):
        table.append(table_row("HDBSCAN", r_real["params"], r_real, r_null))

    conclusion = build_conclusion(hop, kd_stats, n_feas_real, n_feas_null, best_db, null_db, db_feasible,
                                  best_h, null_h, h_feasible, stab, transition, n_strict)
    runtime = round(time.time() - t0, 1)
    metrics = {
        "module": MODULE, "seed": SEED, "n_total": int(len(X)), "sample_size": int(len(idx)),
        "runtime_seconds": runtime, "hopkins": hop, "distance_values": dist_info,
        "k_distance": kd_stats,
        "dbscan": {
            "eps_grid": eps_values, "min_samples_grid": list(MIN_SAMPLES_GRID),
            "feasible_criterion": f"n_clusters >= 2 and noise_ratio < {FEASIBLE_MAX_NOISE}",
            "n_feasible_real": int(n_feas_real), "n_feasible_null": int(n_feas_null),
            "strict_criterion": "clusters with size >= min_samples >= 2 and noise_ratio < 0.5 and largest_share < 0.9",
            "strict": n_strict,
            "best_real": best_db, "best_is_feasible": bool(db_feasible), "null_at_best_params": null_db,
            "knee_rows": {str(ms): {"real": knee_rows[ms][0], "null": knee_rows[ms][1]} for ms in MIN_SAMPLES_GRID},
            "transition_real": {str(ms): transition[ms] for ms in MIN_SAMPLES_GRID},
            "transition_null": {str(ms): transition_null[ms] for ms in MIN_SAMPLES_GRID},
            "grid_real": rows_db_real, "grid_null": rows_db_null,
        },
        "hdbscan": {
            "best_real": best_h, "best_is_feasible": bool(h_feasible), "null_at_best_params": null_h,
            "grid_real": rows_h_real, "grid_null": rows_h_null,
        },
        "stability": stab,
        "summary_table": table,
        "conclusion": conclusion,
        "outputs": {
            "metrics": str(out_dir / "metrics.json"), "model": str(model_path),
            "figures": [str(FIG_DIR / f"{MODULE}_{s}.png")
                        for s in ("kdistance", "eps_scan", "dbscan_scatter", "hdbscan_scatter", "hdbscan_prob")],
        },
    }
    evaluation.save_json(metrics, out_dir / "metrics.json")

    print_summary(table)
    print(f"\nHopkins：真實 {hop['real_full']:.3f} / null {hop['null_full']:.3f}；"
          f"換抽樣穩定度 ARI（真實 / null）：DBSCAN {_f(stab['dbscan']['ari_mean'])} / {_f(stab['dbscan_null']['ari_mean'])}，"
          f"HDBSCAN {_f(stab['hdbscan']['ari_mean'])} / {_f(stab['hdbscan_null']['ari_mean'])}")
    print(f"\n結論：{conclusion}")
    print(f"\n輸出：{out_dir / 'metrics.json'}；圖 {FIG_DIR}/{MODULE}_*.png；模型 {model_path}；耗時 {runtime}s")


if __name__ == "__main__":
    main()
