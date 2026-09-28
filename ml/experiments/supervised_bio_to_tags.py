"""監督式實驗：從 bio 文字預測 68 個 traits 標籤（multi-label 文字分類）。

【這個方法是什麼】
把每位使用者的中文 bio 轉成「字元 1–2 gram」的 TF-IDF 向量當特徵 X（中文不用斷詞器）；
68 個 traits 的 0/1 當標籤矩陣 Y。一個人可以同時有很多標籤，所以這是 multi-label 問題：
用 One-vs-Rest 的方式替每個標籤各訓練一個二元分類器，最後一起評估。比較的模型：
  (a) DummyClassifier(most_frequent)：每個標籤永遠猜訓練集裡較多的那一類，是「什麼都沒學」的下限。
  (b) OneVsRest(LogisticRegression)：線性模型，係數可以直接讀成「哪個字對哪個標籤有貢獻」。
  (c) OneVsRest(LinearSVC)：另一種線性模型，只有 decision_function、沒有機率。
  (d) TruncatedSVD(200) + 逐標籤 HistGradientBoostingClassifier：先把稀疏 TF-IDF 壓成 200 維，
      再用梯度提升樹（本身就是 ensemble）。
  (e) soft voting：LR 的 predict_proba 與 LinearSVC 的 decision_function 各自做 rank 正規化後平均。
      LinearSVC 沒有 predict_proba，兩個模型的分數尺度不同（一個是 0–1 機率、一個是到超平面的距離），
      不能直接相加，所以先各自轉成「在訓練集分數裡的百分位」再平均。

【為什麼適合／不適合這份資料】
主控 agent 已確認 68 個標籤是彼此獨立隨機生成的，分群找不到結構；但 bio 是「看著標籤生出來的」，
例如 bio 提到「羽球」的人 100% 有 interest_badminton。這是整份資料唯一有真訊號可學的地方，
所以「bio → 標籤」是唯一合理的監督式任務。
不過訊號很稀（腳本會實際算給你看）：有某個標籤的人裡，只有大約 10% 會在 bio 提到對應關鍵字。
bio 生成器的原始碼不在 repo 裡，所以「哪些類別會被寫進 bio」不能用看的，腳本用兩個獨立證據量測：
  - 關鍵字掃描：拿 db/gen_sql.py 的 68 個標籤中文名（加上實測發現的同義詞，如 爬山／電動／唱K）
    掃全部 bio，看每個標籤的名稱出現幾次、出現時命中率多少、有標籤的人有多少比例提到。
  - 逐標籤 AUC：LR 在某標籤的測試 AUC 若高於「null 對照 68 個標籤裡最大的 AUC」才算有訊號；
    這個門檻是資料自己給的（沒訊號時 AUC 最多能飄多高），不是人手訂的。
實測兩者一致：interest / personality / value 幾乎每個標籤都有訊號，diet / lifestyle / dating_goal 全部沒有。
所以預期：recall 天生被卡在 ~10%、F1 不會高；真正能看出「模型有學到東西」的是
  - 每標籤 AUC 平均（只看排序，不受門檻影響；null 對照應該是 0.5），
  - lift@5%（把模型最有信心的前 5% 使用者抓出來，命中率是隨機的幾倍）。

【切分方式：為什麼一般隨機切分就夠】
train_test_split(test_size=0.2, random_state=SEED)：
  - 每一列就是一位使用者，account 不重複，一個人只出現一次，所以沒有「同一個人同時出現在
    訓練與測試」的 group 洩漏，不需要 GroupShuffleSplit。
  - bio 有約 2%（233 列）完全重複（腳本會印出來），但那些都是「我真的不會自介www」這類平均 13 字的
    短模板句，是不同的人剛好抽到同一句，不是同一個人重複出現。用全 68 組標籤關鍵字掃，重複句裡只有
    8.6% 含關鍵字，而且幾乎全是「歡迎直接亂聊」「可以直接密我」這種口語的「直接」（personality_direct
    命中率只有 0–0.5），不是生成器寫進去的標籤訊號，所以不構成洩漏。
  - multi-label 沒有單一的 y 可以 stratify；68 個標籤最少的流行度也有 5%，2000 筆測試集每個標籤
    都有足夠正例，所以不做分層。
  - TF-IDF 與 SVD 都只在訓練集上 fit，測試集只 transform，避免詞表／IDF 偷看測試資料。

【null 對照】
把 Y 的列順序隨機打散（bio 與標籤脫鉤）再訓練同一個 LR，AUC 應該掉回 0.5。
注意：class_weight="balanced" 會讓模型大量猜正例，所以 null 的 F1 不會是 0，而是「亂猜的 F1」；
這正好說明在這份資料上 F1 很難分辨真假訊號，要看 AUC。

【怎麼跑】
cd ml && .venv/bin/python experiments/supervised_bio_to_tags.py    （筆電約 30 秒–1 分鐘，HGB 佔大半）

【怎麼讀結果】
outputs/supervised_bio_to_tags/metrics.json：各模型整體指標、6 類別分組指標、逐標籤指標、null 對照、
  可解釋性（LR 係數最高的 n-gram）、5 個示範標籤的關鍵字覆蓋率、全 68 標籤的關鍵字掃描、
  逐類別「有訊號標籤數」（LR AUC > null 最大 AUC）、結論。
outputs/figures/supervised_bio_to_tags_models.png：各模型 macro-F1 與 mean AUC（含 dummy 與 null）。
outputs/figures/supervised_bio_to_tags_categories.png：依 6 類別分組的 macro-F1 與 mean AUC。
outputs/models/supervised_bio_to_tags_best.joblib / _vectorizer.joblib：最佳模型與 TF-IDF 向量器。
判讀重點：「真的學到東西」的證據是 mean AUC 與 lift 明顯高於 null，而且只出現在 bio 有提到的類別。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from joblib import parallel_config
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC

from heartlink_ml import plotting  # noqa: F401  匯入即設定 Agg 後端與中文字型
from heartlink_ml.config import FIG_DIR, MODEL_DIR, OUTPUT_DIR, SEED, TRAIT_CATEGORIES, ensure_dirs
from heartlink_ml.data import column_category, load_seed, trait_columns
from heartlink_ml.evaluation import save_json
from heartlink_ml.features import bio_tfidf, binary_matrix

import matplotlib.pyplot as plt  # noqa: E402  要在 heartlink_ml.plotting 之後匯入

MODULE = "supervised_bio_to_tags"
OUT_DIR = OUTPUT_DIR / MODULE
TOP_FRACTION = 0.05          # lift@5%：取模型分數最高的 5% 使用者
N_SVD = 200                  # HGB 用的 SVD 維度
EXPLAIN_LABELS = ["interest_badminton", "interest_camping", "personality_humorous",
                  "value_values_companionship", "diet_vegetarian"]
# 用來驗證「bio 關鍵字 ↔ 標籤」對應強度的關鍵字（diet_vegetarian 預期 bio 根本沒提）
KEYWORDS = {"interest_badminton": "羽球", "interest_camping": "露營", "personality_humorous": "幽默",
            "value_values_companionship": "陪伴", "diet_vegetarian": "素"}
# 全 68 個標籤的中文名（抄自 db/gen_sql.py 的 LABELS，去掉「愛／重視／需要」這種前綴），
# 用來掃描「bio 生成器到底寫了哪些類別」。gen_sql.py 在 import 時會讀 sys.argv[1]，不能直接 import，所以複製一份。
# 有些標籤生成器用的是同義詞（先用中文名掃到 0 次、但 LR AUC 仍明顯 > 0.5，再回頭找出來的），一併列入。
TRAIT_KEYWORDS = {
    # dating_goal
    "dating_goal_serious_relationship": ["認真交往"], "dating_goal_friends_first": ["先做朋友"],
    "dating_goal_chat_only": ["只想聊天"], "dating_goal_dining_partner": ["找飯友", "飯友"],
    "dating_goal_marriage_minded": ["以結婚為前提", "結婚"],
    # interest
    "interest_hiking": ["登山", "爬山"], "interest_camping": ["露營"], "interest_surfing": ["衝浪"],
    "interest_fitness": ["健身"], "interest_running": ["跑步"], "interest_skateboarding": ["滑板"],
    "interest_badminton": ["羽球"], "interest_basketball": ["籃球"], "interest_cycling": ["騎車", "單車"],
    "interest_swimming": ["游泳"], "interest_yoga": ["瑜伽"], "interest_skiing": ["滑雪"],
    "interest_tv_series": ["追劇"], "interest_movies": ["電影"], "interest_anime": ["動漫"],
    "interest_gaming": ["電玩", "電動"], "interest_board_games": ["桌遊"], "interest_reading": ["閱讀"],
    "interest_cooking": ["料理", "下廚", "煮飯"], "interest_karaoke": ["KTV", "唱K"], "interest_photography": ["攝影"],
    "interest_exhibitions": ["看展"], "interest_live_music": ["live 音樂", "live", "演唱會"], "interest_singing": ["唱歌"],
    "interest_playing_instruments": ["樂器"], "interest_writing": ["寫作"], "interest_drawing": ["繪畫"],
    "interest_dancing": ["跳舞"], "interest_exploring_shops": ["逛店", "小店"], "interest_food": ["美食"],
    "interest_wine_tasting": ["品酒"], "interest_cat_person": ["貓派"], "interest_dog_person": ["狗派"],
    "interest_travel": ["旅行"], "interest_coffee": ["咖啡"], "interest_desserts": ["甜點"],
    "interest_fashion": ["時尚", "穿搭"],
    # personality
    "personality_humorous": ["幽默"], "personality_slow_to_warm_up": ["慢熱"], "personality_talkative": ["健談"],
    "personality_quiet": ["文靜", "安靜"], "personality_rational": ["理性"], "personality_emotional": ["感性"],
    "personality_optimistic": ["樂觀"], "personality_independent": ["獨立"], "personality_direct": ["直接"],
    "personality_romantic": ["浪漫"], "personality_action_oriented": ["行動派"], "personality_homebody": ["居家", "宅"],
    # diet
    "diet_likes_seafood": ["海鮮"], "diet_likes_japanese_food": ["日式"], "diet_likes_hotpot": ["火鍋"],
    "diet_likes_yakiniku": ["燒肉"], "diet_vegetarian": ["素食", "吃素"],
    # lifestyle
    "lifestyle_nine_to_five": ["朝九晚五"], "lifestyle_shift_work": ["輪班"],
    "lifestyle_two_days_off_weekly": ["週休二日"], "lifestyle_work_from_home": ["遠端工作", "遠端"],
    # value
    "value_likes_sharing_daily_life": ["分享日常", "分享生活"], "value_values_companionship": ["陪伴"],
    "value_needs_personal_space": ["個人空間"], "value_values_communication": ["溝通"], "value_values_trust": ["信任"],
}
# 圖的顏色：真模型用固定順序的類別色，基準與 null 一律灰色（null 加斜線）
COLOR_OF = {"Dummy": "#9a9a94", "LR": "#2a78d6", "LinearSVC": "#eb6834",
            "SVD+HGB": "#1baf7a", "Soft voting": "#eda100", "LR (null)": "#9a9a94"}
HATCH_OF = {"LR (null)": "//"}


# ---------------------------------------------------------------- 評估 helper
def per_label_auc(Y_true, S):
    """逐標籤 ROC-AUC；若測試集某標籤只有一類（這份資料不會發生）就記 nan。"""
    out = np.full(Y_true.shape[1], np.nan)
    for j in range(Y_true.shape[1]):
        y = Y_true[:, j]
        if 0 < y.sum() < len(y):
            out[j] = roc_auc_score(y, S[:, j])
    return out


def per_label_lift(Y_true, S, frac=TOP_FRACTION):
    """lift@frac：模型分數最高的前 frac 使用者裡真的有這個標籤的比例 ÷ 整體流行度。1 = 跟隨機一樣。"""
    k = max(1, int(frac * Y_true.shape[0]))
    out = np.full(Y_true.shape[1], np.nan)
    for j in range(Y_true.shape[1]):
        p = Y_true[:, j].mean()
        if p > 0:
            top = np.argsort(-S[:, j], kind="stable")[:k]
            out[j] = Y_true[top, j].mean() / p
    return out


def evaluate(Y_true, Y_pred, S, cols, cats, train_seconds):
    """回傳整體指標 + 依 6 類別分組 + 逐標籤指標。S 是排序用的分數（機率或距離都可以）。"""
    Y_pred = np.asarray(Y_pred).astype(int)
    f1 = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    auc = per_label_auc(Y_true, S)
    lift = per_label_lift(Y_true, S)
    per_cat = {}
    for cat in TRAIT_CATEGORIES:
        m = cats == cat
        per_cat[cat] = {"n_labels": int(m.sum()), "macro_f1": float(f1[m].mean()),
                        "mean_auc": float(np.nanmean(auc[m])), "mean_lift_at_5pct": float(np.nanmean(lift[m]))}
    return {
        "micro_f1": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1.mean()),
        "mean_auc": float(np.nanmean(auc)),
        "subset_accuracy": float(accuracy_score(Y_true, Y_pred)),   # 68 個標籤全部猜對的比例
        "hamming_loss": float(hamming_loss(Y_true, Y_pred)),
        "mean_lift_at_5pct": float(np.nanmean(lift)),
        "predicted_positive_rate": float(Y_pred.mean()),
        "train_seconds": round(float(train_seconds), 2),
        "per_category": per_cat,
        "per_label": {c: {"f1": round(float(f1[j]), 4), "auc": round(float(auc[j]), 4),
                          "lift_at_5pct": round(float(lift[j]), 3)} for j, c in enumerate(cols)},
    }


# ---------------------------------------------------------------- 模型 helper
def fit_timed(model, X, Y):
    t = time.perf_counter()
    model.fit(X, Y)
    return model, time.perf_counter() - t


def make_lr():
    return OneVsRestClassifier(
        LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"), n_jobs=-1)


def make_svc():
    return OneVsRestClassifier(LinearSVC(C=0.5, random_state=SEED), n_jobs=-1)


def fit_hgb(Z, Y):
    """SVD 空間上逐標籤的梯度提升樹。
    沒有加 class_weight：實測 HGB 一帶 sample_weight 每個標籤就多 ~9 秒固定成本，68 個標籤會超過
    5 分鐘上限；代價是門檻 0.5 對稀有標籤偏保守，F1 會偏低，比較時請看 AUC / lift。
    HGB 內部已用 OpenMP 多執行緒，逐標籤再開多進程會 oversubscribe，所以限制每個 worker 只用 1 條執行緒。"""
    model = MultiOutputClassifier(HistGradientBoostingClassifier(random_state=SEED), n_jobs=-1)
    t = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        model.fit(Z, Y)
    return model, time.perf_counter() - t


def hgb_scores(model, Z):
    """MultiOutputClassifier.predict_proba 回傳「每個標籤一個 (n, 2) 陣列」的 list，取正類那一欄。"""
    return np.column_stack([p[:, 1] for p in model.predict_proba(Z)])


def cdf_rank(train_scores, scores):
    """rank 正規化：把分數換成「在訓練集分數裡的百分位」(0–1)。用訓練集當參考，新使用者也能單獨算。"""
    out = np.empty(scores.shape, dtype=np.float64)
    for j in range(scores.shape[1]):
        ref = np.sort(train_scores[:, j])
        out[:, j] = np.searchsorted(ref, scores[:, j], side="right") / len(ref)
    return out


def rank_soft_vote(S_lr_tr, S_lr_te, S_svc_tr, S_svc_te, prevalence):
    """soft voting：兩個模型的百分位平均。
    百分位沒有機率意義，門檻不能用 0.5；改成「每個標籤標出的正例比例 ≈ 訓練集流行度」，
    也就是分數 ≥ (1 − 流行度) 才判為正。"""
    E_te = (cdf_rank(S_lr_tr, S_lr_te) + cdf_rank(S_svc_tr, S_svc_te)) / 2
    Y_pred = (E_te >= 1.0 - prevalence).astype(int)
    return E_te, Y_pred


def top_ngrams(lr_model, vec, cols, label, n=10):
    names = vec.get_feature_names_out()
    coef = lr_model.estimators_[cols.index(label)].coef_[0]
    idx = np.argsort(-coef)[:n]
    # 字元 n-gram 可能含換行或空白，印出來會破壞表格，先跳脫
    return [(str(names[i]).replace("\n", "\\n").replace(" ", "␣"), round(float(coef[i]), 3)) for i in idx]


def keyword_coverage(bio, Y, cols, keywords):
    """P(有標籤 | bio 提到關鍵字) 與 P(bio 提到關鍵字 | 有標籤)：前者是精確度上限，後者是 recall 上限。"""
    out = {}
    for label, kw in keywords.items():
        mention = bio.str.contains(kw, regex=False).to_numpy()
        y = Y[:, cols.index(label)].astype(bool)
        out[label] = {"keyword": kw, "n_mention": int(mention.sum()),
                      "p_tag_given_mention": float(y[mention].mean()) if mention.any() else None,
                      "p_mention_given_tag": float(mention[y].mean())}
    return out


def keyword_scan_all(bio, Y, cols, cats):
    """用 TRAIT_KEYWORDS 掃全部 68 個標籤：回傳（逐標籤覆蓋率, 依類別彙總）。
    這是「bio 生成器寫了哪些類別」的直接證據。名稱沒出現的標籤若 LR 仍有 AUC，代表生成器用了
    我們沒列到的說法，所以這個掃描是下限，要跟逐標籤 AUC（signal_count_by_category）一起看。"""
    per_label = {}
    for j, c in enumerate(cols):
        mention = np.zeros(len(bio), dtype=bool)
        for kw in TRAIT_KEYWORDS[c]:
            mention |= bio.str.contains(kw, regex=False).to_numpy()
        y = Y[:, j].astype(bool)
        per_label[c] = {"keywords": TRAIT_KEYWORDS[c], "n_mention": int(mention.sum()),
                        "p_tag_given_mention": float(y[mention].mean()) if mention.any() else None,
                        "p_mention_given_tag": float(mention[y].mean())}
    per_cat = {}
    for cat in TRAIT_CATEGORIES:
        items = [per_label[c] for c, k in zip(cols, cats) if k == cat]
        hit = [v for v in items if v["n_mention"] > 0]
        per_cat[cat] = {
            "n_labels": len(items), "n_labels_mentioned": len(hit),
            "mean_p_tag_given_mention": float(np.mean([v["p_tag_given_mention"] for v in hit])) if hit else None,
            "mean_p_mention_given_tag": float(np.mean([v["p_mention_given_tag"] for v in items])),
        }
    return per_label, per_cat


def signal_count_by_category(per_label_real, per_label_null, cols, cats):
    """逐標籤判定「有訊號」：真實 LR 的測試 AUC > null 對照 68 個標籤裡最大的 AUC。
    null 的 68 個 AUC 就是「沒有訊號時 AUC 會飄到多高」的經驗分布，拿它的最大值當門檻比人手訂 0.03 誠實。
    回傳 (null_auc_max, {類別: {"n_labels", "n_signal"}})。"""
    null_max = max(v["auc"] for v in per_label_null.values())
    out = {}
    for cat in TRAIT_CATEGORIES:
        labs = [c for c, k in zip(cols, cats) if k == cat]
        out[cat] = {"n_labels": len(labs),
                    "n_signal": int(sum(per_label_real[c]["auc"] > null_max for c in labs))}
    return float(null_max), out


# ---------------------------------------------------------------- 畫圖 helper
def _style(ax):
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_models(results, path):
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, title, ref in ((axes[0], "macro_f1", "macro-F1（測試集）", None),
                                (axes[1], "mean_auc", "每標籤 AUC 平均（測試集）", 0.5)):
        vals = [results[n][key] for n in names]
        bars = ax.bar(names, vals, width=0.6, color=[COLOR_OF[n] for n in names],
                      hatch=[HATCH_OF.get(n, "") for n in names], edgecolor="white", linewidth=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(vals) * 1.22)   # 留出上方空間給數值標籤與圖例，避免互相重疊
        if ref is not None:
            ax.axhline(ref, color="gray", ls="--", lw=1, label="隨機 = 0.5")
            ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        _style(ax)
    fig.suptitle("bio → 68 個標籤：各模型比較（灰色 = 基準／null）", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_categories(results, prevalence_by_cat, path):
    cats, names = list(TRAIT_CATEGORIES), list(results)
    x, w = np.arange(len(cats)), 0.8 / len(names)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, key, title, ref in ((axes[0], "macro_f1", "各類別 macro-F1", None),
                                (axes[1], "mean_auc", "各類別 mean AUC", 0.5)):
        for i, n in enumerate(names):
            vals = [results[n]["per_category"][c][key] for c in cats]
            ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, width=w * 0.92, label=n,
                   color=COLOR_OF[n], hatch=HATCH_OF.get(n, ""), edgecolor="white", linewidth=0.5)
        if ref is not None:
            ax.axhline(ref, color="gray", ls="--", lw=1)
        ax.set_xticks(x, [f"{c}\n流行度 {prevalence_by_cat[c]:.2f}" for c in cats], fontsize=8)
        ax.set_title(title)
        _style(ax)
    axes[0].legend(fontsize=8, ncol=3, loc="upper left")
    fig.suptitle("依標籤類別分組：F1 高低主要跟著流行度走，AUC 才看得出 bio 有沒有提到該類別", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 主程式
def main():
    t_all = time.perf_counter()
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 資料與切分（先切索引，TF-IDF / SVD 只在訓練集 fit）
    df = load_seed()
    cols = trait_columns(df)
    cats = np.array([column_category(c) for c in cols])
    Y = binary_matrix(df, cols).astype(int)
    bio = df["bio"]
    n = len(df)
    idx_tr, idx_te = train_test_split(np.arange(n), test_size=0.2, random_state=SEED)
    X_tr, vec = bio_tfidf(bio.iloc[idx_tr])
    X_te = vec.transform(bio.iloc[idx_te])
    Y_tr, Y_te = Y[idx_tr], Y[idx_te]
    prevalence_tr = Y_tr.mean(0)
    # 洩漏檢查：account 不能重複；重複的 bio 應該只是不帶任何標籤關鍵字（全 68 組）的短模板句
    dup_mask = bio.duplicated(keep=False)
    all_kws = [kw for kws in TRAIT_KEYWORDS.values() for kw in kws]
    has_kw = bio.apply(lambda s: any(k in s for k in all_kws))
    dup_info = {"n_duplicate_accounts": int(df["account"].duplicated().sum()),
                "n_rows_with_duplicated_bio": int(dup_mask.sum()),
                "keyword_rate_in_duplicated_bios": float(has_kw[dup_mask].mean()) if dup_mask.any() else 0.0,
                "keyword_rate_overall": float(has_kw.mean()),
                "mean_len_duplicated_bios": float(bio[dup_mask].str.len().mean()) if dup_mask.any() else 0.0,
                "mean_len_overall": float(bio.str.len().mean())}
    print(f"使用者 {n}，標籤 {len(cols)}，訓練 {len(idx_tr)} / 測試 {len(idx_te)}，TF-IDF 特徵 {X_tr.shape[1]}")
    print(f"洩漏檢查：account 重複 {dup_info['n_duplicate_accounts']} 筆；bio 重複列 {dup_info['n_rows_with_duplicated_bio']} 筆"
          f"（平均 {dup_info['mean_len_duplicated_bios']:.0f} 字、含任一標籤關鍵字 {dup_info['keyword_rate_in_duplicated_bios']:.1%}；"
          f"全體平均 {dup_info['mean_len_overall']:.0f} 字、含關鍵字 {dup_info['keyword_rate_overall']:.1%}）")

    results = {}

    # 2a. 基準：DummyClassifier 原生支援 2D Y，逐標籤猜最多的那一類；分數用訓練集流行度（常數 → AUC = 0.5）
    dummy, sec = fit_timed(DummyClassifier(strategy="most_frequent"), X_tr, Y_tr)
    S_dummy = np.tile(prevalence_tr, (len(idx_te), 1))
    results["Dummy"] = evaluate(Y_te, dummy.predict(X_te), S_dummy, cols, cats, sec)

    # 2b. LogisticRegression（One-vs-Rest）
    lr, sec = fit_timed(make_lr(), X_tr, Y_tr)
    S_lr_tr, S_lr_te = lr.predict_proba(X_tr), lr.predict_proba(X_te)
    results["LR"] = evaluate(Y_te, lr.predict(X_te), S_lr_te, cols, cats, sec)

    # 2c. LinearSVC（One-vs-Rest）
    svc, sec = fit_timed(make_svc(), X_tr, Y_tr)
    S_svc_tr, S_svc_te = svc.decision_function(X_tr), svc.decision_function(X_te)
    results["LinearSVC"] = evaluate(Y_te, svc.predict(X_te), S_svc_te, cols, cats, sec)

    # 2d. TruncatedSVD(200) + 逐標籤 HistGradientBoosting
    t0 = time.perf_counter()
    svd = TruncatedSVD(n_components=N_SVD, random_state=SEED).fit(X_tr)
    Z_tr, Z_te = svd.transform(X_tr), svd.transform(X_te)
    svd_sec = time.perf_counter() - t0
    hgb, sec = fit_hgb(Z_tr, Y_tr)
    results["SVD+HGB"] = evaluate(Y_te, hgb.predict(Z_te), hgb_scores(hgb, Z_te), cols, cats, sec + svd_sec)
    print(f"SVD-{N_SVD} 解釋變異 {svd.explained_variance_ratio_.sum():.3f}，HGB 訓練 {sec:.1f} 秒")

    # 2e. soft voting（rank 正規化平均；不需要再訓練，訓練秒數 = LR + SVC）
    S_ens_te, Y_ens = rank_soft_vote(S_lr_tr, S_lr_te, S_svc_tr, S_svc_te, prevalence_tr)
    results["Soft voting"] = evaluate(Y_te, Y_ens, S_ens_te, cols, cats,
                                      results["LR"]["train_seconds"] + results["LinearSVC"]["train_seconds"])

    # 3. null 對照：打散 Y 的列順序（bio 與標籤脫鉤），同樣的切分與 LR
    rng = np.random.default_rng(SEED)
    Y_null = Y[rng.permutation(n)]
    lr_null, sec = fit_timed(make_lr(), X_tr, Y_null[idx_tr])
    results["LR (null)"] = evaluate(Y_null[idx_te], lr_null.predict(X_te), lr_null.predict_proba(X_te),
                                    cols, cats, sec)

    # 4. 可解釋性 + 關鍵字覆蓋率（5 個示範標籤）+ 全 68 標籤關鍵字掃描 + 逐標籤「有訊號」計數
    explain = {lab: top_ngrams(lr, vec, cols, lab) for lab in EXPLAIN_LABELS}
    coverage = keyword_coverage(bio, Y, cols, KEYWORDS)
    scan_label, scan_cat = keyword_scan_all(bio, Y, cols, cats)
    null_auc_max, signal_cat = signal_count_by_category(results["LR"]["per_label"], results["LR (null)"]["per_label"],
                                                        cols, cats)

    # 5. 圖
    prevalence_by_cat = {c: float(Y[:, cats == c].mean()) for c in TRAIT_CATEGORIES}
    plot_models(results, FIG_DIR / f"{MODULE}_models.png")
    plot_categories(results, prevalence_by_cat, FIG_DIR / f"{MODULE}_categories.png")

    # 6. 存最佳模型（在單一模型裡以 mean AUC 挑；AUC 不受門檻影響，最能反映排序能力）
    singles = {"LR": lr, "LinearSVC": svc, "SVD+HGB": hgb}
    best_name = max(singles, key=lambda k: results[k]["mean_auc"])
    bundle = {"name": best_name, "model": singles[best_name], "labels": cols,
              "svd": svd if best_name == "SVD+HGB" else None}
    joblib.dump(bundle, MODEL_DIR / f"{MODULE}_best.joblib")
    joblib.dump(vec, MODEL_DIR / f"{MODULE}_vectorizer.joblib")

    # 7. 結論（用實際數字組出來，避免文字與數字不一致）
    lr_cat, null_cat = results["LR"]["per_category"], results["LR (null)"]["per_category"]
    # 「有訊號的類別」= 該類別過半標籤的 LR AUC 高於 null 對照 68 個標籤的最大 AUC
    signal_cats = [c for c in TRAIT_CATEGORIES if signal_cat[c]["n_signal"] > signal_cat[c]["n_labels"] / 2]
    no_signal_cats = [c for c in TRAIT_CATEGORIES if c not in signal_cats]
    signal_txt = "、".join(f"{c} {signal_cat[c]['n_signal']}/{signal_cat[c]['n_labels']}" for c in TRAIT_CATEGORIES)
    scan_txt = "、".join(f"{c} {scan_cat[c]['n_labels_mentioned']}/{scan_cat[c]['n_labels']}" for c in TRAIT_CATEGORIES)
    mentioned = [v for v in scan_label.values() if v["n_mention"] > 0]
    mention_rates = [v["p_mention_given_tag"] for v in mentioned]
    hit_rates = [v["p_tag_given_mention"] for v in mentioned]
    conclusions = [
        f"最佳單一模型（以 mean AUC）：{best_name}，mean AUC {results[best_name]['mean_auc']:.3f}、"
        f"lift@5% {results[best_name]['mean_lift_at_5pct']:.2f}；null 對照 mean AUC "
        f"{results['LR (null)']['mean_auc']:.3f}、lift {results['LR (null)']['mean_lift_at_5pct']:.2f}。",
        f"逐標籤看，LR 測試 AUC 高於 null 對照最大 AUC（{null_auc_max:.3f}）的標籤數：{signal_txt}。"
        f"所以有訊號的類別是 {signal_cats}，沒有訊號的是 {no_signal_cats}。",
        f"關鍵字掃描給出同樣的答案（名稱出現在 bio 的標籤數）：{scan_txt}；"
        f"{no_signal_cats} 的標籤名稱在 10,000 筆 bio 裡一次都沒出現，代表 bio 生成器根本沒寫這幾類。",
        f"訊號很稀：在 {len(mentioned)} 個名稱有出現的標籤裡，有標籤的人只有 {min(mention_rates):.0%}–{max(mention_rates):.0%} "
        f"會在 bio 提到（平均 {np.mean(mention_rates):.0%}），但提到時命中率平均 {np.mean(hit_rates):.2f}"
        f"（最低 {min(hit_rates):.2f}，像「直接」「安靜」「咖啡」在別的語境也會出現）；所以 recall 天生被卡住，F1 不可能高。",
        f"F1 的高低主要跟著流行度走：null 對照在 value 類（流行度 {prevalence_by_cat['value']:.2f}）的 macro-F1 "
        f"也有 {null_cat['value']['macro_f1']:.3f}，所以不能拿 F1 比較類別間學得好不好，要看 AUC / lift。",
        f"Soft voting 的 mean AUC {results['Soft voting']['mean_auc']:.3f}，介於 LR {results['LR']['mean_auc']:.3f} "
        f"與 LinearSVC {results['LinearSVC']['mean_auc']:.3f} 之間；兩個線性模型看的是同一份 TF-IDF，"
        "錯得很像，平均起來沒有互補效果。",
        "監督式模型學到的是「bio 生成器把哪個關鍵字寫給哪個標籤」的對應（例如 羽球→badminton），"
        "不是真實使用者的語言習慣；上線前需要真人 bio 重新驗證。",
    ]

    metrics = {
        "module": MODULE,
        "dataset": {"n_users": n, "n_labels": len(cols), "n_train": len(idx_tr), "n_test": len(idx_te),
                    "n_tfidf_features": int(X_tr.shape[1]), "leakage_check": dup_info,
                    "svd_components": N_SVD, "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
                    "prevalence_by_category": prevalence_by_cat,
                    "label_prevalence": {c: round(float(Y[:, j].mean()), 4) for j, c in enumerate(cols)}},
        "settings": {"split": "train_test_split(test_size=0.2, random_state=SEED)", "seed": SEED,
                     "tfidf": "char 1-2 gram, min_df=3, sublinear_tf, fit on train only",
                     "lr": "OneVsRest(LogisticRegression(C=4.0, max_iter=2000, class_weight='balanced'))",
                     "svc": "OneVsRest(LinearSVC(C=0.5))",
                     "hgb": f"TruncatedSVD({N_SVD}) + MultiOutput(HistGradientBoostingClassifier default, no class_weight)",
                     "soft_voting": "mean of train-CDF ranks of LR proba and SVC decision; threshold = 1 - train prevalence",
                     "null_control": "row-permuted Y, same split, same LR"},
        "models": results,
        "best_model": best_name,
        "explainability_top10_ngrams_lr": explain,
        "keyword_coverage": coverage,
        "keyword_scan_all_labels": {"per_label": scan_label, "per_category": scan_cat},
        "signal_labels": {"rule": "LR test AUC > max AUC over the 68 null-control labels",
                          "null_auc_max": null_auc_max, "per_category": signal_cat,
                          "signal_categories": signal_cats, "no_signal_categories": no_signal_cats},
        "conclusions": conclusions,
        "runtime_seconds": None,
    }

    # 8. 摘要表
    print("\n### 各模型（測試集 2000 人 × 68 標籤）")
    print("| 模型 | micro-F1 | macro-F1 | mean AUC | lift@5% | subset acc | Hamming | 訓練秒數 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['micro_f1']:.3f} | {r['macro_f1']:.3f} | {r['mean_auc']:.3f} | "
              f"{r['mean_lift_at_5pct']:.2f} | {r['subset_accuracy']:.4f} | {r['hamming_loss']:.3f} | {r['train_seconds']:.1f} |")

    print(f"\n### 依類別（LR vs null；「有訊號」= LR 測試 AUC > null 對照 68 標籤最大 AUC {null_auc_max:.3f}）")
    print("| 類別 | 標籤數 | 流行度 | LR macro-F1 | null macro-F1 | LR AUC | null AUC | LR lift@5% | 有訊號標籤數 | 名稱出現在 bio 的標籤數 | 平均 P(提到\\|標籤) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in TRAIT_CATEGORIES:
        a, b, s, k = lr_cat[c], null_cat[c], signal_cat[c], scan_cat[c]
        print(f"| {c} | {a['n_labels']} | {prevalence_by_cat[c]:.2f} | {a['macro_f1']:.3f} | {b['macro_f1']:.3f} | "
              f"{a['mean_auc']:.3f} | {b['mean_auc']:.3f} | {a['mean_lift_at_5pct']:.2f} | "
              f"{s['n_signal']}/{s['n_labels']} | {k['n_labels_mentioned']}/{k['n_labels']} | {k['mean_p_mention_given_tag']:.2f} |")

    print("\n### 關鍵字覆蓋率（精確度上限 / recall 上限）")
    print("| 標籤 | 關鍵字 | bio 提到人數 | P(標籤|提到) | P(提到|標籤) |")
    print("|---|---|---|---|---|")
    for lab, v in coverage.items():
        p1 = "—" if v["p_tag_given_mention"] is None else f"{v['p_tag_given_mention']:.2f}"
        print(f"| {lab} | {v['keyword']} | {v['n_mention']} | {p1} | {v['p_mention_given_tag']:.2f} |")

    print("\n### LR 係數最高的 10 個 n-gram")
    for lab, items in explain.items():
        print(f"- {lab}: " + "、".join(f"{g}({w:.1f})" for g, w in items))

    metrics["runtime_seconds"] = round(time.perf_counter() - t_all, 1)
    save_json(metrics, OUT_DIR / "metrics.json")
    print("\n### 結論")
    for c in conclusions:
        print(f"- {c}")
    print(f"\n總耗時 {metrics['runtime_seconds']} 秒；輸出：{OUT_DIR / 'metrics.json'}、"
          f"{FIG_DIR / (MODULE + '_*.png')}、{MODEL_DIR / (MODULE + '_*.joblib')}")


if __name__ == "__main__":
    main()
