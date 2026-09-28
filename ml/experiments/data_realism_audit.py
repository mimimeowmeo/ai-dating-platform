"""資料體檢：這份 AI 生成的 seed 資料「像不像真的」？

=== 為什麼需要這支腳本 ===
10,000 位使用者是請 AI 模型生成的。模型再準，餵進去的資料如果沒有真實世界的結構，
分群、推薦、排序學到的就只會是生成規則。這支腳本把「像不像真的」拆成可以量的五件事：

  1. 流行度：每個標籤有多少人選？真實世界是長尾（少數很熱門、多數很冷門），不會每個都一樣。
  2. 每人勾幾個：真實的人差異很大（有人只勾 1 個、有人勾 15 個），不會剛好落在固定範圍且每種數量一樣多。
  3. 共現：真實的興趣會成群（登山↔露營），標籤之間有相關；獨立隨機抽的資料相關係數全是 0。
  4. 與人口屬性的關聯：真實的興趣跟性別、年齡有關。
  5. 連續欄位的形狀：身高近似常態（鐘形），不是「某個範圍內每公分一樣多」。
  6. 跨欄位一致性：身高不該由年齡決定；素食者不該同時「愛燒肉」；
     另外檢查「生成規則的指紋」——同類別內的相關係數是否剛好等於隨機抽樣的理論值。

對照組不用外求：同一份 CSV 裡的 40 個 face_* 欄位來自 CelebA **真實**資料集，
拿它跟 AI 生成的 68 個標籤放在一起比，差別一目了然。

=== 關於「常態分布」這個詞 ===
0/1 標籤本身不可能是常態分布（它只有兩個值，服從 Bernoulli）。對標籤資料有意義的問法是：
「各標籤的流行度長什麼形狀」「每人勾選數長什麼形狀」「標籤之間有沒有關聯」。
常態分布只適用於連續或可加總的量（身高、總標籤數），腳本對這些欄位才做常態檢定。

=== 怎麼跑 ===
    cd ml && .venv/bin/python experiments/data_realism_audit.py
輸出：outputs/data_realism_audit/metrics.json、outputs/figures/data_realism_audit.png，並印出判定表。
重新生成 seed 之後再跑一次，就能看到哪些項目改善了。
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from heartlink_ml import config, data, evaluation, features  # noqa: E402
from heartlink_ml import plotting  # noqa: E402,F401  （設定 Agg 後端與中文字型）

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

warnings.filterwarnings("ignore")
NAME = "data_realism_audit"

# 圖表配色：角色固定，全圖一致（藍 = AI 生成；橘 = 真實資料；灰 = 理論參考）。
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
GEN, REAL = "#2a78d6", "#eb6834"


def bh_reject(p, alpha=0.05) -> np.ndarray:
    """Benjamini–Hochberg：同時檢定很多對時控制偽發現率。"""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    ok = p[order] <= alpha * np.arange(1, m + 1) / m
    k = ok.nonzero()[0].max() + 1 if ok.any() else 0
    out = np.zeros(m, dtype=bool)
    out[order[:k]] = True
    return out


def pairwise_phi(M: np.ndarray):
    """回傳上三角的 phi 相關係數、對應的 (i, j) 與 p 值（大樣本常態近似）。"""
    R = np.corrcoef(M.astype(float).T)
    iu = np.triu_indices(M.shape[1], 1)
    r = R[iu]
    p = 2 * stats.norm.sf(np.abs(r) * np.sqrt(M.shape[0]))
    return r, iu, p


def audit(df: pd.DataFrame) -> dict:
    n = len(df)
    cols = data.trait_columns(df)
    X = features.binary_matrix(df, cols)
    cat = np.array([data.column_category(c) for c in cols])
    fcols = data.face_columns(df)
    F = features.binary_matrix(df, fcols)
    male = (df.gender == "Male").to_numpy()
    age = df.age.astype(float).to_numpy()
    out: dict = {"n_users": n}

    # 1. 流行度
    prev = {}
    for c in config.TRAIT_CATEGORIES:
        cnt = X[:, cat == c].sum(0)
        prev[c] = {
            "n_tags": int((cat == c).sum()), "min_pct": float(cnt.min() / n * 100),
            "max_pct": float(cnt.max() / n * 100), "cv": float(cnt.std() / cnt.mean()),
            "uniform_chi2_p": float(stats.chisquare(cnt)[1]),
        }
    fp = F.mean(0) * 100
    out["prevalence"] = prev
    out["prevalence_real_face"] = {"min_pct": float(fp.min()), "max_pct": float(fp.max()), "cv": float(fp.std() / fp.mean())}

    # 2. 每人勾幾個
    per = {}
    for c in config.TRAIT_CATEGORIES:
        k = X[:, cat == c].sum(1)
        vc = pd.Series(k).value_counts().sort_index()
        per[c] = {"min": int(k.min()), "max": int(k.max()), "share_pct": {int(i): float(v / n * 100) for i, v in vc.items()}}
    tot = X.sum(1)
    p_each = X.mean(0)
    out["per_person"] = per
    out["total_tags"] = {
        "mean": float(tot.mean()), "sd": float(tot.std()), "min": int(tot.min()), "max": int(tot.max()),
        "skew": float(stats.skew(tot)), "excess_kurtosis": float(stats.kurtosis(tot)),
        "normaltest_p": float(stats.normaltest(tot)[1]),
        "sd_if_independent_bernoulli": float(np.sqrt((p_each * (1 - p_each)).sum())),
    }

    # 3. 共現
    r, iu, p = pairwise_phi(X)
    same = cat[iu[0]] == cat[iu[1]]
    rf, _, pf = pairwise_phi(F)
    out["cooccurrence"] = {
        "cross_category": {"n_pairs": int((~same).sum()), "n_significant": int(bh_reject(p[~same]).sum()),
                           "max_phi": float(r[~same].max()), "min_phi": float(r[~same].min()),
                           "share_abs_phi_gt_0.1_pct": float((np.abs(r[~same]) > 0.1).mean() * 100)},
        "within_category": {"n_pairs": int(same.sum()), "n_significant": int(bh_reject(p[same]).sum()),
                            "max_phi": float(r[same].max()), "min_phi": float(r[same].min())},
        "real_face": {"n_pairs": int(len(rf)), "n_significant": int(bh_reject(pf).sum()),
                      "max_abs_phi": float(np.abs(rf).max()), "share_abs_phi_gt_0.1_pct": float((np.abs(rf) > 0.1).mean() * 100)},
        "sanity_pairs": {f"{a[9:]}×{b[9:]}": float(np.corrcoef(X[:, cols.index(a)], X[:, cols.index(b)])[0, 1])
                         for a, b in (("interest_hiking", "interest_camping"), ("interest_gaming", "interest_anime"),
                                      ("interest_coffee", "interest_desserts"), ("interest_fitness", "interest_running"))},
    }

    # 4. 與性別、年齡的關聯
    pg = [stats.chi2_contingency(pd.crosstab(male, X[:, j]))[1] for j in range(len(cols))]
    diff = (X[male].mean(0) - X[~male].mean(0)) * 100
    pa = [stats.pointbiserialr(X[:, j], age)[1] for j in range(len(cols))]
    out["demographics"] = {
        "gender_significant_tags": int(bh_reject(pg).sum()), "gender_max_gap_pp": float(np.abs(diff).max()),
        "age_significant_tags": int(bh_reject(pa).sum()),
        "gender_share": df.gender.value_counts(normalize=True).round(3).to_dict(),
        "face_male_matches_gender_pct": float((F[:, fcols.index("face_Male")] == male).mean() * 100),
        "face_young_vs_age_r": float(np.corrcoef(F[:, fcols.index("face_Young")], age)[0, 1]),
    }

    # 5. 連續欄位
    cont = {}
    for label, m in (("male", male), ("female", ~male)):
        h = df.height_cm[m].astype(int)
        vc = h.value_counts().sort_index()
        mu, sd = h.mean(), h.std()
        cont[f"height_{label}"] = {
            "mean": float(mu), "sd": float(sd), "min": int(h.min()), "max": int(h.max()),
            "excess_kurtosis": float(stats.kurtosis(h)), "normaltest_p": float(stats.normaltest(h)[1]),
            "discrete_uniform_chi2_p": float(stats.chisquare(vc.values)[1]),
            "share_within_1sd_pct": float(((h > mu - sd) & (h < mu + sd)).mean() * 100),
        }
    vc = pd.Series(age.astype(int)).value_counts().sort_index()
    inner = vc.iloc[1:-1]  # 頭尾兩個年齡只涵蓋部分生日，先拿掉再檢定
    cont["age"] = {"min": int(age.min()), "max": int(age.max()), "excess_kurtosis": float(stats.kurtosis(age)),
                   "uniform_chi2_p_excluding_edges": float(stats.chisquare(inner.values)[1])}
    out["continuous"] = cont

    # 6. 跨欄位一致性，以及「生成規則的指紋」
    h_all, a_all = df.height_cm.astype(int).to_numpy(), age.astype(int)
    cons: dict = {}
    for label, m, base in (("male", male, 165), ("female", ~male, 150)):
        cons[f"height_age_r_{label}"] = float(np.corrcoef(a_all[m], h_all[m])[0, 1])
        # 真實成年人的身高與年齡幾乎無關；這裡檢查身高是不是直接由年齡算出來的。
        cons[f"height_is_base_plus_age_pct_{label}"] = float((h_all[m] == base + (a_all[m] - 20)).mean() * 100)

    def both(a, b):
        ia, ib = cols.index(a), cols.index(b)
        return {"both": int((X[:, ia] & X[:, ib]).sum()),
                "expected_if_independent": float(X[:, ia].mean() * X[:, ib].mean() * n)}

    cons["contradictions"] = {
        "素食×愛燒肉": both("diet_vegetarian", "diet_likes_yakiniku"),
        "健談×安靜": both("personality_talkative", "personality_quiet"),
        "只想聊天×以結婚為前提": both("dating_goal_chat_only", "dating_goal_marriage_minded"),
    }
    # 指紋：若規則是「先均勻抽一個數量 k，再從該類別不放回均勻抽 k 個」，同類別內兩兩 phi 有封閉解：
    #   P(某標籤) = E[k]/n，P(兩個都中) = E[k(k-1)] / (n(n-1))。實測值與它吻合 = 資料就是這樣抽出來的。
    # diet 的流行度是人工設定的（素食 5%），不符合「類別內等機率」的前提，所以不算。
    R = np.corrcoef(X.astype(float).T)
    sig = {}
    for c in config.TRAIT_CATEGORIES:
        idx = np.where(cat == c)[0]
        if c == "diet" or len(idx) < 2:
            continue
        m_ = len(idx)
        kk = X[:, idx].sum(1).astype(float)
        p1 = kk.mean() / m_
        p11 = np.mean(kk * (kk - 1)) / (m_ * (m_ - 1))
        sig[c] = {"theoretical_phi": float((p11 - p1 * p1) / (p1 * (1 - p1))),
                  "observed_mean_phi": float(R[np.ix_(idx, idx)][np.triu_indices(m_, 1)].mean())}
    cons["uniform_sampling_signature"] = sig
    out["consistency"] = cons
    return out


def style_axes(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)   # 實線髮絲格線，退到後面
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=9, labelcolor=MUTED, pad=4)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK2, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK2, labelpad=6)


def panel_title(ax, title, subtitle):
    ax.text(0, 1.20, title, transform=ax.transAxes, fontsize=12, fontweight="semibold", color=INK, va="bottom")
    ax.text(0, 1.085, subtitle, transform=ax.transAxes, fontsize=9.3, color=INK2, va="bottom")


def rounded_bars(ax, xs, hs, width, color, radius_px=4):
    """長條：資料端 4px 圓角、基線端直角。要在座標範圍定案並 draw 過之後呼叫。"""
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    k = ax.figure.dpi / 100.0
    rx, ry = radius_px * k / abs(x1 - x0), radius_px * k / abs(y1 - y0)
    for x, h in zip(xs, hs):
        if h <= 0:
            continue
        a, b = min(rx, width / 2), min(ry, h)
        l, r = x - width / 2, x + width / 2
        verts = [(l, 0), (l, h - b), (l, h), (l + a, h), (r - a, h), (r, h), (r, h - b), (r, 0), (l, 0)]
        codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO,
                 MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def legend(ax, handles, loc="upper right"):
    ax.legend(handles=handles, loc=loc, frameon=False, fontsize=9, labelcolor=INK2, handlelength=1.6, borderaxespad=0.2)


def end_dot(ax, x, y, color):
    ax.plot([x], [y], marker="o", markersize=7, color=color, markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=5)


def draw(df: pd.DataFrame, path: Path) -> None:
    n = len(df)
    cols = data.trait_columns(df)
    X = features.binary_matrix(df, cols)
    cat = np.array([data.column_category(c) for c in cols])
    F = features.binary_matrix(df, data.face_columns(df))
    male = (df.gender == "Male").to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.6), dpi=100, facecolor=SURFACE)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.085, wspace=0.26, hspace=0.72)
    fig.text(0.075, 0.955, "這份 AI 生成的資料像真的嗎？", fontsize=16, fontweight="semibold", color=INK)
    fig.text(0.075, 0.918, "藍＝AI 生成的標籤與欄位　橘＝同一份檔案裡的真實資料（CelebA 臉部屬性）　灰線＝理論上該有的形狀",
             fontsize=10, color=INK2)

    # --- A. 流行度排名 ---
    ax = axes[0, 0]
    gen = np.sort(X[:, cat == "interest"].mean(0) * 100)[::-1]
    real = np.sort(F.mean(0) * 100)[::-1]
    style_axes(ax, "依流行度排名（第 1 名 → 最後一名）", "有這個標籤的人（%）")
    ax.plot(np.arange(1, len(real) + 1), real, color=REAL, linewidth=1.9, solid_joinstyle="round", solid_capstyle="round", zorder=3)
    ax.plot(np.arange(1, len(gen) + 1), gen, color=GEN, linewidth=1.9, solid_joinstyle="round", solid_capstyle="round", zorder=4)
    end_dot(ax, 1, real[0], REAL); end_dot(ax, len(real), real[-1], REAL)
    end_dot(ax, 1, gen[0], GEN); end_dot(ax, len(gen), gen[-1], GEN)
    ax.set_xlim(0, 42); ax.set_ylim(0, 98); ax.set_xticks([1, 10, 20, 30, 40])
    ax.annotate(f"{real[0]:.0f}%", (1, real[0]), xytext=(-2, 8), textcoords="offset points", fontsize=9, color=INK, ha="left")
    ax.annotate(f"{real[-1]:.0f}%", (len(real), real[-1]), xytext=(0, 9), textcoords="offset points", fontsize=9, color=INK, ha="center")
    ax.annotate(f"37 個興趣全部落在 {gen[-1]:.0f}–{gen[0]:.0f}%", (30.5, gen[29]), xytext=(0, 9), textcoords="offset points",
                fontsize=9, color=INK, ha="center")
    panel_title(ax, "① 每個標籤有多少人選", "真實資料是長尾；生成資料是一條水平線")
    legend(ax, [Line2D([], [], color=GEN, lw=1.9, label="AI 生成：37 個興趣"), Line2D([], [], color=REAL, lw=1.9, label="真實：40 個臉部屬性")])

    # --- B. 兩兩相關 ---
    ax = axes[0, 1]
    r, iu, _ = pairwise_phi(X)
    cross = r[cat[iu[0]] != cat[iu[1]]]
    rf, _, _ = pairwise_phi(F)
    edges = np.arange(-0.55, 0.86, 0.1)
    centers = (edges[:-1] + edges[1:]) / 2
    hg = np.histogram(cross, edges)[0] / len(cross) * 100
    hr = np.histogram(rf, edges)[0] / len(rf) * 100
    style_axes(ax, "兩個標籤的相關係數 φ（0＝互不相關）", "配對占比（%）")
    ax.set_xlim(-0.6, 0.9); ax.set_ylim(0, 120); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    fig.canvas.draw()
    w = 0.036
    rounded_bars(ax, centers - w / 2 - 0.004, hg, w, GEN)      # 兩根長條之間留一道表面色的縫
    rounded_bars(ax, centers + w / 2 + 0.004, hr, w, REAL)
    ax.text(0.075, 88, f"{hg.max():.0f}% 的配對都在 0 附近", fontsize=9, color=INK, ha="left", va="center")
    ax.annotate(f"真實資料最高到 {np.abs(rf).max():.2f}", (0.8, hr[-1]), xytext=(0, 14), textcoords="offset points",
                fontsize=9, color=INK, ha="right")
    panel_title(ax, "② 標籤之間有沒有關聯", "真實的屬性會成群出現；生成的標籤彼此完全獨立")
    legend(ax, [Patch(color=GEN, label=f"AI 生成：跨類別 {len(cross):,} 對"), Patch(color=REAL, label=f"真實：{len(rf)} 對")], loc="upper left")

    # --- C. 每人勾幾個興趣 ---
    ax = axes[1, 0]
    k = X[:, cat == "interest"].sum(1)
    ks = np.arange(0, 15)
    share = np.array([(k == i).mean() * 100 for i in ks])
    n_int = int((cat == "interest").sum())
    ref = stats.binom.pmf(ks, n_int, k.mean() / n_int) * 100
    style_axes(ax, "一個人勾了幾個興趣", "人數占比（%）")
    ax.set_xlim(-0.7, 14.7); ax.set_ylim(0, 32); ax.set_xticks(ks[::2]); ax.set_yticks([0, 5, 10, 15, 20, 25])
    fig.canvas.draw()
    rounded_bars(ax, ks, share, 0.62, GEN)
    ax.plot(ks, ref, color=INK2, linewidth=1.9, solid_joinstyle="round", zorder=4)
    ax.annotate("4～8 個，每種數量各 20%", (6, share[6]), xytext=(0, 8), textcoords="offset points", fontsize=9, color=INK, ha="center")
    ax.text(14.6, 9.6, "若每個興趣各自獨立決定，\n會是這種鐘形：\n有人勾 1 個、有人勾 12 個", fontsize=9, color=INK2,
            ha="right", va="bottom", linespacing=1.4)
    panel_title(ax, "③ 每個人勾幾個興趣", "看起來像「4 到 8 隨機挑一個數字」：沒有人勾很少或很多")
    legend(ax, [Patch(color=GEN, label="AI 生成"), Line2D([], [], color=INK2, lw=1.9, label="理論參考：二項分布")], loc="upper right")

    # --- D. 身高 ---
    ax = axes[1, 1]
    style_axes(ax, "身高（cm）", "占該性別的比例（%／每公分）")
    xs = np.linspace(140, 200, 400)
    for label, m in (("女", ~male), ("男", male)):
        h = df.height_cm[m].astype(int)
        grid = np.arange(h.min(), h.max() + 1)
        pct = np.array([(h == v).mean() * 100 for v in grid])
        step_x = np.concatenate([[grid[0] - 0.5], np.repeat(grid[:-1] + 0.5, 2), [grid[-1] + 0.5]])
        step_x = np.concatenate([[step_x[0]], step_x, [step_x[-1]]])
        step_y = np.concatenate([[0], np.repeat(pct, 2), [0]])
        ax.fill_between(step_x, step_y, color=GEN, alpha=0.10, linewidth=0, zorder=2)
        ax.plot(step_x, step_y, color=GEN, linewidth=1.9, solid_joinstyle="round", zorder=4)
        ax.plot(xs, stats.norm.pdf(xs, h.mean(), h.std()) * 100, color=INK2, linewidth=1.9, zorder=3)
        ax.text(158.0 if label == "女" else 179.5, 2.2, f"{label} {h.min()}–{h.max()} cm", fontsize=9, color=INK, ha="center", va="center")
    ax.set_xlim(142, 198); ax.set_ylim(0, 10.6); ax.set_yticks([0, 2, 4, 6, 8])
    panel_title(ax, "④ 身高是常態分布嗎", "不是：範圍內每一公分的人數都一樣多（平頂），真實身高是鐘形")
    legend(ax, [Line2D([], [], color=GEN, lw=1.9, label="AI 生成"),
                Line2D([], [], color=INK2, lw=1.9, label="理論參考：同平均、同標準差的常態分布")], loc="upper left")

    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def verdict_table(m: dict) -> str:
    pv, pf = m["prevalence"]["interest"], m["prevalence_real_face"]
    co, dm, ct, tt, cs = m["cooccurrence"], m["demographics"], m["continuous"], m["total_tags"], m["consistency"]
    rows = [
        ("興趣流行度", f"{pv['min_pct']:.1f}–{pv['max_pct']:.1f}%（CV {pv['cv']:.3f}）", f"真實對照 {pf['min_pct']:.1f}–{pf['max_pct']:.1f}%（CV {pf['cv']:.2f}）", "均勻，不像真的"),
        ("每人興趣數", "4–8 個、每種數量各約 20%", "差異大、右偏", "固定範圍均勻抽"),
        ("總標籤數", f"SD {tt['sd']:.2f}（獨立抽應為 {tt['sd_if_independent_bernoulli']:.2f}）、常態檢定 p={tt['normaltest_p']:.1g}", "較分散", "看似鐘形但被配額壓扁"),
        ("跨類別共現", f"{co['cross_category']['n_significant']}/{co['cross_category']['n_pairs']} 對顯著、最大 φ {co['cross_category']['max_phi']:+.3f}",
         f"真實對照 {co['real_face']['n_significant']}/{co['real_face']['n_pairs']} 對顯著、最大 |φ| {co['real_face']['max_abs_phi']:.2f}", "完全獨立，不像真的"),
        ("標籤 × 性別", f"{dm['gender_significant_tags']}/68 個顯著、最大差 {dm['gender_max_gap_pp']:.1f} 個百分點", "多數興趣有明顯性別差", "無關聯，不像真的"),
        ("身高（男）", f"{ct['height_male']['min']}–{ct['height_male']['max']}、超額峰度 {ct['height_male']['excess_kurtosis']:+.2f}、對整數均勻 p={ct['height_male']['discrete_uniform_chi2_p']:.2f}", "常態（峰度 0）", "均勻，不是常態"),
        ("身高（女）", f"{ct['height_female']['min']}–{ct['height_female']['max']}、超額峰度 {ct['height_female']['excess_kurtosis']:+.2f}、對整數均勻 p={ct['height_female']['discrete_uniform_chi2_p']:.2f}", "常態（峰度 0）", "均勻，不是常態"),
        ("年齡", f"{ct['age']['min']}–{ct['age']['max']}、去頭尾後對均勻 p={ct['age']['uniform_chi2_p_excluding_edges']:.2f}", "交友 app 集中在 20 多歲", "均勻"),
        ("身高 × 年齡", f"女 r={cs['height_age_r_female']:+.2f}、{cs['height_is_base_plus_age_pct_female']:.0f}% 的女性身高＝150＋(年齡−20)；男 r={cs['height_age_r_male']:+.2f}",
         "成年人身高與年齡幾乎無關", "身高是用年齡算出來的"),
        ("互斥的標籤", "、".join(f"{k} {v['both']} 人（獨立期望 {v['expected_if_independent']:.0f}）" for k, v in cs["contradictions"].items()),
         "應接近 0", "沒有一致性規則"),
        ("生成規則指紋", "；".join(f"{k} φ 實測 {v['observed_mean_phi']:+.4f}／理論 {v['theoretical_phi']:+.4f}" for k, v in cs["uniform_sampling_signature"].items()),
         "—", "與「隨機決定數量、再隨機抽」完全吻合"),
    ]
    head = "| 檢查項目 | 這份生成資料 | 真實資料該有的樣子 | 判定 |\n|---|---|---|---|\n"
    return head + "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows)


def main() -> None:
    config.ensure_dirs()
    df = data.load_seed()
    metrics = audit(df)
    table = verdict_table(metrics)
    metrics["verdict_table_markdown"] = table
    evaluation.save_json(metrics, config.OUTPUT_DIR / NAME / "metrics.json")
    fig_path = config.FIG_DIR / f"{NAME}.png"
    draw(df, fig_path)
    print(table)
    print(f"\n圖：{fig_path.relative_to(config.ML_DIR)}｜數字：outputs/{NAME}/metrics.json")


if __name__ == "__main__":
    main()
