"""階層式分群（HAC, Hierarchical Agglomerative Clustering）實驗。

這個方法是什麼
--------------
HAC 一開始把每個人當成一群，每一步把「最近」的兩群合併，直到只剩一群，
過程記錄成一棵樹（dendrogram）。「兩群多近」由 linkage 決定：
- single   ：兩群之間「最近」的一對點的距離（容易 chaining：一條鏈把大家串成一大群）
- average  ：兩群所有點對的平均距離
- complete ：兩群之間「最遠」的一對點的距離（傾向切出緊實、大小相近的群）
- ward     ：合併後群內變異數增加最少（只適用歐氏距離）

為什麼用 Jaccard 距離 + 不用 ward
----------------------------------
每個人是 68 個 0/1 標籤，「兩人共同勾了幾個標籤／兩人一共勾了幾個標籤」的 Jaccard
距離比歐氏距離合理（歐氏會把「都沒勾」也算成相似）。scikit-learn 官方文件
（https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html）
指出 metric 可以是 "precomputed"（fit 時直接餵距離矩陣），但 linkage="ward" 只接受
euclidean / l2，所以 Jaccard 距離不能配 ward。官方使用手冊
（https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering）
也說明對非歐氏距離 "for non Euclidean metrics, average linkage is a good alternative"，
並提醒 agglomerative 有 rich-get-richer 的傾向、single linkage 最嚴重。
因此本實驗比較 single / average / complete 三種，把 average 當主要方法。

為什麼「找不到群」是預期結果、以及兩種 null
------------------------------------------
主控端已驗證這份 seed 資料的 68 個標籤是獨立隨機生成的（Hopkins ≈ 0.51、
K-means silhouette ≈ 0.01）。HAC 一樣「一定會」切出群（它不會拒絕分群），
所以每個結果都必須跟 null model 並列：
- null（逐欄打散）：evaluation.column_permutation_null，每一欄獨立打散，保留每個
  標籤的流行度、破壞所有共現。這是必做的主要對照。但它也破壞了「每人 lifestyle
  恰好 1 個、interest 4–8 個」這類每人標籤數的限制，打散後會出現幾個只有 4–8 個
  標籤的人；Jaccard 下這幾個人離所有人都很遠，HAC 會先把他們切成小群，讓 null 的
  silhouette 在 k=2 時反而略高於真實（這是離群人造成的假象，不是 null 有結構）。
- null_block（類別區塊打散）：本檔自己實作。每個類別的欄位一起做同一個列置換，
  保留每人每類別的標籤數與每欄流行度，只打斷跨類別的共現。更貼近這份資料的
  生成方式，是更公平的第二對照。
真實資料的 silhouette 若跟兩種 null 都差不多，就代表分出來的群只是把均勻的雲切塊。

怎麼跑
------
cd ml && .venv/bin/python experiments/clustering_hierarchical.py
（不連資料庫、不安裝套件；3,000 筆抽樣，筆電上約 1 分鐘跑完）

怎麼讀結果
----------
- outputs/clustering_hierarchical/metrics.json：所有數字（真實 / null / null_block 並列）。
- outputs/figures/clustering_hierarchical_dendrogram_<linkage>.png：三張樹狀圖。
  single 的樹會是一條長梯子（chaining）；average / complete 才會看到分叉。
- outputs/figures/clustering_hierarchical_silhouette_vs_k.png：三個子圖各一種 linkage，
  實線＝真實、虛線＝null（逐欄）、點線＝null_block。三條線糾纏在 0 附近就是「沒有結構」。
- outputs/figures/clustering_hierarchical_largest_ratio_vs_k.png：最大群佔比。
  single 幾乎永遠 ≈ 1（一大群 + 零星小群），這是 chaining 的直接證據。
- cophenetic correlation：樹上的合併距離與原始成對距離的相關係數，衡量「這棵樹
  多忠實地保留原始距離」；越接近 1 越好，在均勻資料上通常很低。
- 穩定度：HAC 沒有隨機初始化，同一份距離矩陣跑幾次都一模一樣，所以
  evaluation.stability_ari（同抽樣不同 seed）必然是 1.0、沒有資訊。真正的變異來源
  是「抽到哪 3,000 人」，因此這裡另外實作：換 3 個抽樣種子，各自挑最佳 k，
  在兩兩重疊的樣本（約 900 人）上算 ARI；兩種 null 也用同樣方式算。
  注意（退化的 pair）：k=2 的切法若只是把幾個離群人切出來，兩次抽樣的重疊樣本裡
  很可能一個離群人都沒有，兩邊在重疊樣本上都只剩「一群」；sklearn 的
  adjusted_rand_score 對兩個「全部同一群」的標籤會回 1.0，這種 pair 沒有任何資訊。
  本檔把「重疊樣本上任一邊少於 2 群」的 pair 標為 degenerate、排除在平均／中位數
  之外，並另外回報含退化 pair 的全體平均供對照，pairs 明細裡有每邊在重疊上的群數。

實作備註
--------
- 切群一律用 sklearn.cluster.AgglomerativeClustering(metric="precomputed")。
  scipy 的 fcluster(criterion="maxclust") 在 complete linkage 會因為大量合併距離
  剛好等於 1.0（平手）而切不出指定群數，校準時 maxclust=4 只回傳 1 群，故不採用。
- scipy.cluster.hierarchy.linkage 只用來產生 Z 畫 dendrogram 與算 cophenetic；
  輸入必須是 condensed 距離向量（squareform(D, checks=False)）。
  腳本會順便驗證「sklearn 在最佳 k 的切法」與「scipy cut_tree(Z, n_clusters=k)」
  是否一致（metrics.json 的 sklearn_vs_scipy_cut_tree_ari；三種 linkage 皆為 1.0），
  所以 dendrogram 與實際切法是同一棵樹，只有 fcluster(maxclust) 因平手切不出來。
- 逐欄 null 的 k=2 小群是不是真的由「標籤數異常少的人」組成，不能只用推論，
  腳本會直接量：k=2 最小群的平均標籤數 vs 全體平均（k2_smallest_cluster）。
- Davies-Bouldin / Calinski-Harabasz 是歐氏空間的指標，這裡直接餵 0/1 矩陣當參考，
  主要判讀指標是 silhouette_jaccard。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.cluster.hierarchy import cophenet, cut_tree, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from sklearn.cluster import AgglomerativeClustering  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from heartlink_ml import config, data, evaluation, features, plotting  # noqa: E402

# plotting 匯入時已設定 Agg 後端與中文字型，之後再拿 pyplot 畫自訂的實線／虛線圖。
import matplotlib.pyplot as plt  # noqa: E402

MODULE = "clustering_hierarchical"
SAMPLE_SIZE = 3000                              # O(n²) 方法一律抽樣
LINKAGES = ("single", "average", "complete")    # ward 不能配 Jaccard，見 docstring
K_RANGE = list(range(2, 13))                    # n_clusters 掃 2..12
THRESHOLD_QUANTILES = (0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50)  # distance_threshold 由 D 的分位數決定
STABILITY_SEEDS = (config.SEED, config.SEED + 1, config.SEED + 2)  # 三個不同的抽樣種子
VARIANTS = ("real", "null", "null_block")       # 真實 / 逐欄打散 / 類別區塊打散
VARIANT_LABEL = {"real": "真實", "null": "null（逐欄打散）", "null_block": "null（類別區塊打散）"}
VARIANT_STYLE = {"real": "-o", "null": "--x", "null_block": ":s"}
OUT_DIR = config.OUTPUT_DIR / MODULE


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def condensed(D: np.ndarray) -> np.ndarray:
    """n×n 距離矩陣 → scipy 要的 condensed 向量（上三角攤平，長度 n(n-1)/2）。"""
    return squareform(D.astype(np.float64), checks=False)


def category_block_permutation_null(X: np.ndarray, cols, seed=config.SEED) -> np.ndarray:
    """第二種 null：每個類別的欄位一起做同一個列置換。
    保留每人每類別的標籤數（lifestyle 恰 1、interest 4–8）與每欄流行度，只打斷跨類別的共現。"""
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    for cat in config.TRAIT_CATEGORIES:
        j = [i for i, c in enumerate(cols) if data.column_category(c) == cat]
        Xn[:, j] = X[rng.permutation(len(X))][:, j]
    return Xn


def row_sum_stats(M: np.ndarray, cols) -> dict:
    """每人標籤數的統計：用來說明兩種 null 各保留了什麼。"""
    rs = M.sum(1)
    life = M[:, [i for i, c in enumerate(cols) if c.startswith("lifestyle_")]].sum(1)
    return {"min": int(rs.min()), "max": int(rs.max()), "mean": float(rs.mean()), "std": float(rs.std()),
            "n_rows_le_8_tags": int((rs <= 8).sum()), "lifestyle_exactly_one_ratio": float((life == 1).mean())}


def hac_labels(D: np.ndarray, method: str, n_clusters=None, distance_threshold=None) -> np.ndarray:
    """用 sklearn 在 precomputed 距離上做 HAC。二選一：指定群數 或 指定切樹高度。"""
    model = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage=method,
        distance_threshold=distance_threshold,
        # 官方文件：distance_threshold 不是 None 時，n_clusters 必須是 None、compute_full_tree 必須是 True
        compute_full_tree=True if distance_threshold is not None else "auto",
    )
    return model.fit(D).labels_


def quality(D: np.ndarray, Xs: np.ndarray, labels: np.ndarray) -> dict:
    """cluster_quality + 最大群佔比；cluster_sizes 只留前 10 個，避免 threshold 切出上千群時 JSON 爆掉。"""
    with warnings.catch_warnings():
        # threshold 切法可能出現大量單人群，DB 指標會有 0/0 的 RuntimeWarning，不影響結論
        warnings.simplefilter("ignore", RuntimeWarning)
        q = evaluation.cluster_quality(D, Xs.astype(np.float32), labels)
    sizes = q.pop("cluster_sizes")
    q["largest_cluster_ratio"] = float(sizes[0] / len(labels)) if sizes else None
    q["cluster_sizes_top10"] = sizes[:10]
    return q


def sweep_k(D: np.ndarray, Xs: np.ndarray, method: str) -> list[dict]:
    """n_clusters 掃 K_RANGE，每個 k 記錄 cluster_quality。"""
    rows = []
    for k in K_RANGE:
        labels = hac_labels(D, method, n_clusters=k)
        rows.append({"k": k, **quality(D, Xs, labels)})
    return rows


def sweep_threshold(D: np.ndarray, Xs: np.ndarray, method: str, thresholds: dict) -> list[dict]:
    """distance_threshold 掃幾個值（key 是分位數、value 是實際距離），每個記錄 cluster_quality。"""
    rows = []
    for qtl, t in thresholds.items():
        labels = hac_labels(D, method, distance_threshold=float(t))
        rows.append({"quantile": qtl, "threshold": float(t), **quality(D, Xs, labels)})
    return rows


def best_by_silhouette(rows: list[dict]) -> dict:
    """回傳 silhouette 最高的那一列（silhouette 為 None 的略過；全 None 時回第一列）。"""
    valid = [r for r in rows if r["silhouette_jaccard"] is not None]
    return max(valid, key=lambda r: r["silhouette_jaccard"]) if valid else rows[0]


def tree_and_cophenetic(D: np.ndarray, method: str):
    """scipy linkage 產生 Z（畫 dendrogram 用）並算 cophenetic correlation。"""
    y = condensed(D)
    Z = linkage(y, method=method)
    c, _ = cophenet(Z, y)
    return Z, float(c)


def overlap_ari(idx_a, labels_a, idx_b, labels_b) -> dict:
    """兩次抽樣重疊的人，比較兩次的分群標籤是否一致。
    若任一邊在重疊樣本上只剩 1 群（例如 k=2 只切出幾個離群人、而離群人剛好不在重疊裡），
    adjusted_rand_score 會回 1.0 但毫無意義，標為 degenerate。"""
    common, ia, ib = np.intersect1d(idx_a, idx_b, assume_unique=True, return_indices=True)
    la, lb = labels_a[ia], labels_b[ib]
    ka, kb = int(len(np.unique(la))), int(len(np.unique(lb)))
    degenerate = len(common) < 2 or ka < 2 or kb < 2
    return {"n_overlap": int(len(common)), "n_clusters_a_in_overlap": ka, "n_clusters_b_in_overlap": kb,
            "degenerate": bool(degenerate),
            "ari": float(adjusted_rand_score(la, lb)) if len(common) >= 2 else None}


def smallest_cluster_tag_stats(Xs: np.ndarray, labels: np.ndarray) -> dict:
    """最小群成員的平均標籤數 vs 全體平均：用來檢查「小群是不是只是標籤數異常的離群人」。"""
    sizes = np.bincount(labels)
    small = int(np.argmin(sizes))
    rs = Xs.sum(1)
    return {"cluster_sizes": sorted(sizes.tolist(), reverse=True), "smallest_size": int(sizes[small]),
            "tag_mean_smallest": float(rs[labels == small].mean()), "tag_mean_all": float(rs.mean())}


def resampling_stability(X_full: np.ndarray, seeds, method="average") -> dict:
    """換抽樣種子的穩定度：每個種子各自抽樣、各自挑最佳 k，兩兩在重疊樣本上算 ARI。
    平均／中位數只算非退化的 pair；含退化 pair 的全體平均另外回報，方便對照。"""
    runs = []
    for s in seeds:
        idx = features.subsample(len(X_full), SAMPLE_SIZE, seed=s)
        Xs = X_full[idx]
        D = features.jaccard_distance_matrix(Xs)
        best = best_by_silhouette(sweep_k(D, Xs, method))
        runs.append({"seed": int(s), "idx": idx, "best_k": best["k"],
                     "silhouette": best["silhouette_jaccard"],
                     "largest_cluster_ratio": best["largest_cluster_ratio"],
                     "labels": hac_labels(D, method, n_clusters=best["k"])})
    pairs = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            info = overlap_ari(runs[i]["idx"], runs[i]["labels"], runs[j]["idx"], runs[j]["labels"])
            pairs.append({"seeds": [runs[i]["seed"], runs[j]["seed"]], **info})
    valid = [p["ari"] for p in pairs if p["ari"] is not None and not p["degenerate"]]
    all_aris = [p["ari"] for p in pairs if p["ari"] is not None]
    return {
        "best_k_per_seed": [{k: r[k] for k in ("seed", "best_k", "silhouette", "largest_cluster_ratio")} for r in runs],
        "pairs": pairs,
        "n_pairs": len(pairs),
        "n_degenerate_pairs": int(sum(p["degenerate"] for p in pairs)),
        "mean_overlap_ari": float(np.mean(valid)) if valid else None,       # 只算非退化 pair
        "median_overlap_ari": float(np.median(valid)) if valid else None,
        "mean_overlap_ari_all_pairs": float(np.mean(all_aris)) if all_aris else None,  # 含退化 pair，僅供對照
    }


# ---------------------------------------------------------------------------
# 畫圖（plotting.line_plot 不支援虛線／子圖，這裡自己畫「真實實線 / null 虛線 / null_block 點線」）
# ---------------------------------------------------------------------------
def plot_sweeps(x, per_linkage: dict, title, ylabel, path, hline=None):
    """per_linkage[method][variant] = 沿著 x 的數值序列；三種 linkage 各一個子圖。"""
    fig, axes = plt.subplots(1, len(per_linkage), figsize=(4.2 * len(per_linkage), 4.2), sharey=True)
    for ax, (method, series) in zip(np.atleast_1d(axes), per_linkage.items()):
        for variant, ys in series.items():
            ax.plot(x, ys, VARIANT_STYLE[variant], label=VARIANT_LABEL[variant], ms=4)
        if hline is not None:
            ax.axhline(hline, color="gray", lw=0.8)
        ax.set_title(f"{method} linkage")
        ax.set_xlabel("群數 k")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    t0 = time.perf_counter()
    config.ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 資料：68 個 traits 的 0/1 矩陣；兩種 null 都在全體上打散一次，之後三個版本用同一組抽樣索引。
    df = data.load_seed()
    cols = data.trait_columns(df)
    X_all = {"real": features.binary_matrix(df, cols)}
    X_all["null"] = evaluation.column_permutation_null(X_all["real"], seed=config.SEED)
    X_all["null_block"] = category_block_permutation_null(X_all["real"], cols, seed=config.SEED)
    idx = features.subsample(len(X_all["real"]), SAMPLE_SIZE, seed=config.SEED)
    Xs = {v: X_all[v][idx] for v in VARIANTS}
    Ds = {v: features.jaccard_distance_matrix(Xs[v]) for v in VARIANTS}
    print(f"資料 {X_all['real'].shape}，抽樣 {len(idx)} 筆；Jaccard 距離矩陣 {Ds['real'].shape}")

    # 每人標籤數：說明逐欄 null 破壞了什麼、區塊 null 保留了什麼
    row_stats = {v: row_sum_stats(Xs[v], cols) for v in VARIANTS}
    for v in VARIANTS:
        s = row_stats[v]
        print(f"每人標籤數（{VARIANT_LABEL[v]}，抽樣 {len(idx)} 人內）：min {s['min']} / mean {s['mean']:.2f} / std {s['std']:.2f}；"
              f"≤8 個的有 {s['n_rows_le_8_tags']} 人；lifestyle 恰 1 個佔 {s['lifestyle_exactly_one_ratio']:.0%}")

    # Hopkins：≈0.5 表示與「各標籤獨立隨機」無法區分
    hopkins = {f"sample_{v}": evaluation.hopkins_binary(Xs[v], seed=config.SEED) for v in VARIANTS}
    hopkins["full_real"] = evaluation.hopkins_binary(X_all["real"], seed=config.SEED)
    print("Hopkins：" + "、".join(f"{k} {v:.3f}" for k, v in hopkins.items()))

    # distance_threshold 的候選值：取真實 D 上三角的分位數（null 也用同一組值，才好比較）
    triu = Ds["real"][np.triu_indices(len(idx), 1)]
    thresholds = {str(q): float(np.quantile(triu, q)) for q in THRESHOLD_QUANTILES}
    print("distance_threshold 候選值：" + ", ".join(f"q{q}={t:.3f}" for q, t in thresholds.items()))

    # 2. 三種 linkage：scipy 樹 + cophenetic + dendrogram；sklearn 掃 k 與 threshold（真實 / null / null_block）
    results = {}
    for method in LINKAGES:
        print(f"\n== linkage = {method} ==")
        trees = {v: tree_and_cophenetic(Ds[v], method) for v in VARIANTS}
        plotting.dendrogram_plot(trees["real"][0], f"HAC dendrogram（{method} linkage，Jaccard，n={len(idx)}）",
                                 config.FIG_DIR / f"{MODULE}_dendrogram_{method}.png")
        by_k = {v: sweep_k(Ds[v], Xs[v], method) for v in VARIANTS}
        by_t = {v: sweep_threshold(Ds[v], Xs[v], method, thresholds) for v in VARIANTS}
        best = {v: best_by_silhouette(by_k[v]) for v in VARIANTS}
        k_best = best["real"]["k"]
        # 同一個 k（真實最佳 k）下 null 的 silhouette：跟「各自最佳」並列，才是同流程同 k 的公平比較
        null_at_real_k = {v: by_k[v][K_RANGE.index(k_best)]["silhouette_jaccard"] for v in VARIANTS}
        # k=2 最小群的標籤數：檢查小群是不是標籤數異常的離群人（逐欄 null 的假象來源）
        k2_small = {v: smallest_cluster_tag_stats(Xs[v], hac_labels(Ds[v], method, n_clusters=2)) for v in VARIANTS}
        # 驗證 sklearn 的切法與 scipy 的樹是同一棵：在最佳 k 比較 sklearn labels 與 cut_tree(Z, k)
        labels_best = hac_labels(Ds["real"], method, n_clusters=k_best)
        tree_cut_ari = float(adjusted_rand_score(labels_best, cut_tree(trees["real"][0], n_clusters=k_best).ravel()))
        results[method] = {
            "cophenetic": {v: trees[v][1] for v in VARIANTS},
            "tree_top_merge_distance": {v: float(trees[v][0][-1, 2]) for v in VARIANTS},
            "by_k": by_k, "by_threshold": by_t, "best_k": best,
            "null_silhouette_at_real_best_k": null_at_real_k,
            "k2_smallest_cluster": k2_small,
            "sklearn_vs_scipy_cut_tree_ari": tree_cut_ari,
        }
        print("cophenetic：" + " / ".join(f"{VARIANT_LABEL[v]} {trees[v][1]:.3f}" for v in VARIANTS)
              + f"；真實最後一次合併距離 {trees['real'][0][-1, 2]:.3f}；sklearn 切法 vs scipy cut_tree ARI={tree_cut_ari:.3f}")
        print("k=2 最小群（人數 / 平均標籤數 vs 全體）：" + " / ".join(
            f"{VARIANT_LABEL[v]} {k2_small[v]['smallest_size']}人 {k2_small[v]['tag_mean_smallest']:.1f} vs {k2_small[v]['tag_mean_all']:.1f}"
            for v in VARIANTS))
        for i, k in enumerate(K_RANGE):
            print(f"  k={k:2d}  silhouette " + " / ".join(fmt(by_k[v][i]["silhouette_jaccard"]) for v in VARIANTS)
                  + "   最大群佔比 " + " / ".join(fmt(by_k[v][i]["largest_cluster_ratio"], 2) for v in VARIANTS))
        # 存下最佳 k 的模型、樹與抽樣索引（HAC 沒有 predict，存起來是為了重現與畫圖）
        joblib.dump({"linkage": method, "sample_idx": idx, "Z": trees["real"][0], "best_k": k_best,
                     "model": AgglomerativeClustering(n_clusters=k_best, metric="precomputed",
                                                      linkage=method).fit(Ds["real"])},
                    config.MODEL_DIR / f"{MODULE}_{method}_best.joblib")

    # 3. 圖：silhouette vs k、最大群佔比 vs k（三種 linkage 各一子圖；實線真實 / 虛線 null / 點線 null_block）
    def series_of(key):
        return {m: {v: [r[key] for r in results[m]["by_k"][v]] for v in VARIANTS} for m in LINKAGES}
    plot_sweeps(K_RANGE, series_of("silhouette_jaccard"), "HAC：silhouette（Jaccard）vs k",
                "silhouette", config.FIG_DIR / f"{MODULE}_silhouette_vs_k.png", hline=0.0)
    plot_sweeps(K_RANGE, series_of("largest_cluster_ratio"), "HAC：最大群佔比 vs k（single ≈ 1 就是 chaining）",
                "最大群佔比", config.FIG_DIR / f"{MODULE}_largest_ratio_vs_k.png")

    # 4. 穩定度
    print("\n== 穩定度（average linkage）==")
    # (a) evaluation.stability_ari 的「同抽樣不同 seed」版本：HAC 是決定性的，seed 根本沒用到，必為 1.0
    k_avg = results["average"]["best_k"]["real"]["k"]
    same_sample_ari = evaluation.stability_ari(lambda seed: hac_labels(Ds["real"], "average", n_clusters=k_avg), n_runs=3)
    # (b) 自己實作：換抽樣種子 → 各自最佳 k → 重疊樣本上的 ARI；兩種 null 用同一套流程
    stab = {v: resampling_stability(X_all[v], STABILITY_SEEDS, "average") for v in VARIANTS}
    print(f"同抽樣不同 seed（evaluation.stability_ari）：{same_sample_ari:.3f}（決定性演算法，必為 1，無資訊）")
    for v in VARIANTS:
        s = stab[v]
        print(f"換抽樣種子重疊 ARI（{VARIANT_LABEL[v]}）：非退化 pair 平均 {fmt(s['mean_overlap_ari'])}、中位數 {fmt(s['median_overlap_ari'])}"
              f"（退化 {s['n_degenerate_pairs']}/{s['n_pairs']} 對；含退化的全體平均 {fmt(s['mean_overlap_ari_all_pairs'])}）；"
              + "各對 " + ", ".join(
                  f"{p['seeds'][0]}×{p['seeds'][1]}={fmt(p['ari'], 2)}"
                  + (f"[退化: 重疊上群數 {p['n_clusters_a_in_overlap']}/{p['n_clusters_b_in_overlap']}]" if p["degenerate"] else "")
                  for p in s["pairs"])
              + "；各種子最佳 k=" + ",".join(str(r["best_k"]) for r in s["best_k_per_seed"]))

    # 5. 摘要表與結論
    print("\n## 摘要（抽樣 3000、Jaccard 距離；k 由 silhouette 在 2..12 中挑）\n")
    print("| linkage | 最佳k | silhouette(真實) | silhouette(null 逐欄) 同k / 自身最佳 | silhouette(null 區塊) 同k / 自身最佳 "
          "| cophenetic 真實/逐欄/區塊 | 最大群佔比(真實) |")
    print("|---|---|---|---|---|---|---|")
    summary_rows = []
    for m in LINKAGES:
        r = results[m]
        b = r["best_k"]
        same_k = r["null_silhouette_at_real_best_k"]
        row = {"linkage": m, "best_k": b["real"]["k"], "silhouette_real": b["real"]["silhouette_jaccard"],
               "silhouette_null_at_same_k": same_k["null"], "silhouette_null_block_at_same_k": same_k["null_block"],
               "best_k_null": b["null"]["k"], "silhouette_null": b["null"]["silhouette_jaccard"],
               "best_k_null_block": b["null_block"]["k"], "silhouette_null_block": b["null_block"]["silhouette_jaccard"],
               "cophenetic_real": r["cophenetic"]["real"], "cophenetic_null": r["cophenetic"]["null"],
               "cophenetic_null_block": r["cophenetic"]["null_block"],
               "largest_cluster_ratio_real": b["real"]["largest_cluster_ratio"],
               "sklearn_vs_scipy_cut_tree_ari": r["sklearn_vs_scipy_cut_tree_ari"]}
        summary_rows.append(row)
        print(f"| {m} | {b['real']['k']} | {fmt(b['real']['silhouette_jaccard'])} "
              f"| {fmt(same_k['null'])} / {fmt(b['null']['silhouette_jaccard'])} (k={b['null']['k']}) "
              f"| {fmt(same_k['null_block'])} / {fmt(b['null_block']['silhouette_jaccard'])} (k={b['null_block']['k']}) "
              f"| {fmt(r['cophenetic']['real'])} / {fmt(r['cophenetic']['null'])} / {fmt(r['cophenetic']['null_block'])} "
              f"| {fmt(b['real']['largest_cluster_ratio'], 2)} |")

    print("\n## distance_threshold 切法（群數 / 最大群佔比 / silhouette；真實 → null 逐欄）\n")
    print("| 分位數 | 距離 | " + " | ".join(LINKAGES) + " |")
    print("|---|---|" + "---|" * len(LINKAGES))
    for i, (q, t) in enumerate(thresholds.items()):
        cells = []
        for m in LINKAGES:
            a, b = results[m]["by_threshold"]["real"][i], results[m]["by_threshold"]["null"][i]
            cells.append(f"{a['n_clusters']}群/{fmt(a['largest_cluster_ratio'], 2)}/{fmt(a['silhouette_jaccard'], 2)} → "
                         f"{b['n_clusters']}群/{fmt(b['largest_cluster_ratio'], 2)}/{fmt(b['silhouette_jaccard'], 2)}")
        print(f"| q={q} | {t:.3f} | " + " | ".join(cells) + " |")

    # 結論：直接由數字產生，避免文字與數字不一致
    max_real_sil = max(r["silhouette_real"] for r in summary_rows)
    max_null_sil = max(max(r["silhouette_null"], r["silhouette_null_block"]) for r in summary_rows)
    # average 是唯一「最佳 k」不是 2 的 linkage；看它 k=2..5 的 silhouette 區間有多窄，判斷選 k 有沒有意義
    avg_sils = [r["silhouette_jaccard"] for r in results["average"]["by_k"]["real"] if r["k"] <= 5]
    b_single, b_avg, b_comp = (results[m]["best_k"]["real"] for m in LINKAGES)
    null_avg2 = results["average"]["by_k"]["null"][0]  # 逐欄 null 在 k=2 的切法
    small_null = results["average"]["k2_smallest_cluster"]["null"]
    small_real = results["average"]["k2_smallest_cluster"]["real"]
    conclusions = [
        f"Hopkins={hopkins['sample_real']:.3f}（0.5 = 與獨立隨機無異）。三種 linkage 的真實最佳 silhouette 最高只有 "
        f"{max_real_sil:.3f}，兩種 null 最高 {max_null_sil:.3f}，全部落在 ±0.1 內：HAC 切出的群與打散後的隨機資料"
        "分不出差別，找不到群結構是預期且正確的結果。"
        f"另外，average linkage 在 k=2..5 的真實 silhouette 只在 {min(avg_sils):.3f}–{max(avg_sils):.3f} 之間，"
        f"「最佳 k={b_avg['k']}」只是 argmax 挑出來的，各 k 之間的差距沒有統計意義。",
        f"逐欄 null 的 silhouette 在 k=2 略高於真實（例如 average：{fmt(null_avg2['silhouette_jaccard'])} vs "
        f"{fmt(results['average']['by_k']['real'][0]['silhouette_jaccard'])}）不是 null 有結構：逐欄打散破壞了每人標籤數的限制"
        f"（抽樣內 ≤8 個標籤的人從 {row_stats['real']['n_rows_le_8_tags']} 變 {row_stats['null']['n_rows_le_8_tags']} 人，lifestyle 恰 1 個從 "
        f"{row_stats['real']['lifestyle_exactly_one_ratio']:.0%} 掉到 {row_stats['null']['lifestyle_exactly_one_ratio']:.0%}）。"
        f"直接量 k=2 的最小群：逐欄 null 的最小群 {small_null['smallest_size']} 人平均只有 {small_null['tag_mean_smallest']:.1f} 個標籤"
        f"（全體 {small_null['tag_mean_all']:.1f}），真實資料 k=2 的最小群 {small_real['smallest_size']} 人平均 "
        f"{small_real['tag_mean_smallest']:.1f} 個（全體 {small_real['tag_mean_all']:.1f}）——null 的「群」只是標籤數異常少的離群人，"
        f"其餘人的 silhouette 就被抬高。保留每類別標籤數的區塊 null（≤8 個標籤 {row_stats['null_block']['n_rows_le_8_tags']} 人）"
        "就沒有這個現象。",
        f"single linkage 最佳 k={b_single['k']} 時最大群佔 {b_single['largest_cluster_ratio']:.1%}，k 掃到 12 最大群仍佔 "
        f"{results['single']['by_k']['real'][-1]['largest_cluster_ratio']:.1%}：典型的 chaining（一大群 + 零星離群小群），"
        "這是均勻資料上 single linkage 的固有行為，不是資料真的有一個主流族群。",
        f"average / complete 最佳 k 的最大群佔比為 {b_avg['largest_cluster_ratio']:.1%} / {b_comp['largest_cluster_ratio']:.1%}，"
        f"silhouette 只有 {fmt(b_avg['silhouette_jaccard'])} / {fmt(b_comp['silhouette_jaccard'])}，cophenetic 只有 "
        f"{results['average']['cophenetic']['real']:.3f} / {results['complete']['cophenetic']['real']:.3f}："
        "樹幾乎不保留原始距離，群只是把均勻的雲切塊。",
        f"換抽樣種子的重疊 ARI（average，各自最佳 k，只算非退化 pair）：真實平均 {fmt(stab['real']['mean_overlap_ari'])}"
        f"（退化 {stab['real']['n_degenerate_pairs']}/{stab['real']['n_pairs']} 對）、"
        f"區塊 null {fmt(stab['null_block']['mean_overlap_ari'])}（退化 {stab['null_block']['n_degenerate_pairs']}/{stab['null_block']['n_pairs']} 對）、"
        f"逐欄 null {fmt(stab['null']['mean_overlap_ari'])}（退化 {stab['null']['n_degenerate_pairs']}/{stab['null']['n_pairs']} 對；"
        f"若把退化 pair 算進去平均會變 {fmt(stab['null']['mean_overlap_ari_all_pairs'])}，因為 k=2 只切出幾個離群人、"
        "重疊樣本裡沒有離群人時兩邊都只剩一群，adjusted_rand_score 對兩個常數標籤回 1.0，這是定義上的假象）。"
        "ARI 已校正機率水準，接近 0 代表換一批人就換一套群，分群結果不可重現。",
        "實務建議：這份資料的標籤矩陣不適合用分群做使用者分群；bio 文字才有真訊號（見其他實驗）。",
    ]
    print("\n## 解讀\n")
    for c in conclusions:
        print(f"- {c}")

    runtime = time.perf_counter() - t0
    metrics = {
        "module": MODULE,
        "settings": {"n_total": int(len(X_all["real"])), "sample_size": int(len(idx)), "seed": config.SEED,
                     "n_trait_columns": len(cols), "linkages": list(LINKAGES), "k_range": K_RANGE,
                     "threshold_quantiles": list(THRESHOLD_QUANTILES), "thresholds": thresholds,
                     "distance": "jaccard (precomputed)",
                     "null_models": {"null": "evaluation.column_permutation_null（逐欄打散，同抽樣索引）",
                                     "null_block": "本檔 category_block_permutation_null（類別區塊打散，同抽樣索引）"}},
        "row_sum_stats": row_stats,
        "hopkins": hopkins,
        "summary": summary_rows,
        "by_linkage": results,
        "stability": {
            "same_sample_stability_ari": same_sample_ari,
            "same_sample_note": "HAC 為決定性演算法，同一距離矩陣換 seed 結果完全相同，此值必為 1.0，不代表群存在。",
            "resampling_note": "mean/median 只算非退化 pair（重疊樣本上兩邊都 ≥2 群）；"
                               "mean_overlap_ari_all_pairs 含退化 pair（常數標籤的 ARI 定義為 1.0），僅供對照。",
            "resampling_average": stab,
        },
        "conclusions": conclusions,
        "runtime_seconds": runtime,
        "figures": [str(config.FIG_DIR / f"{MODULE}_dendrogram_{m}.png") for m in LINKAGES]
                   + [str(config.FIG_DIR / f"{MODULE}_silhouette_vs_k.png"),
                      str(config.FIG_DIR / f"{MODULE}_largest_ratio_vs_k.png")],
        "models": [str(config.MODEL_DIR / f"{MODULE}_{m}_best.joblib") for m in LINKAGES],
    }
    evaluation.save_json(metrics, OUT_DIR / "metrics.json")
    print(f"\n已寫入 {OUT_DIR / 'metrics.json'}；總耗時 {runtime:.1f} 秒")


if __name__ == "__main__":
    main()
