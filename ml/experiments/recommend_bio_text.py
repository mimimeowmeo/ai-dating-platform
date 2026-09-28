"""用 bio 文字加入推薦：不需下載任何模型的五種做法 + 混合 + 冷啟動模擬（零網路、零 API）。

【問題】
使用者的 bio 不在 68 個標籤裡，但我們也想「依照他寫的文字」加入推薦。主控 agent 已實測：
整段 bio 直接做 TF-IDF cosine 幾乎沒用（recall@50 只比隨機高 1.2 倍），因為 93% 的文字是
口頭禪、梗、表情符號這類「填充文字」，它們主導了 cosine，真正跟標籤有關的字只佔 6.9%。
所以重點不是「要不要用 bio」，而是「怎麼把訊號從填充文字裡分離出來」，以及
「填充文字本身（風格）算不算一個獨立維度」。

【做法（全部只用 scikit-learn，不下載任何模型權重）】
  (1) 整段 TF-IDF（char 1–2 gram）cosine —— 天真做法，當作基準。
  (2) 關鍵字抽取 → 標籤空間：用 db/03_seed_traits.sql 的 68 個 label_zh（加上同義詞）掃 bio，
      得到「bio 提到的標籤」0/1 矩陣 M。附兩個字典健檢：scan_missing_keywords()（bio 完全沒掃到的標籤，
      有沒有字典沒收、但命中率 ≥ 0.9 的說法——「烹飪／聽團／探店／慢熟」就是這樣補進來的）與
      每個歧義關鍵字最常見的誤判句（「今天也需要咖啡續命」不代表興趣是咖啡）。
      (2a) 只用 M（IDF 加權 cosine，bio↔bio）。
      (2b) augment：原標籤 X ∨ M 再做 IDF×類別加權 cosine，示範「用 bio 補標籤」。
  (3) bio→標籤機率填補：對每人算 P(tag|bio) 68 維 → cosine；
      (3z) 先把 68 欄各自標準化再 cosine（去掉「每個標籤的平均機率」這個人人都有的共同成分）；
      (3b) soft augment：X ∨ (P > 0.5)；另外掃門檻 0.7 / 0.9，看高門檻的 P 是不是只剩關鍵字比對。
      ⚠ 機率的來源：腳本會先載入上一輪存的 OneVsRest(LogisticRegression) bundle 並印出結構，但那個模型是用
      80% 的使用者訓練的，直接對同一批人推論會「記住」他們的標籤（實測 lift 會被灌水 2–3.4 倍）。
      所以 (3)(3z)(3b)(7) 的主數字一律用本腳本重算的 5-fold out-of-fold（OOF）機率——每個人的機率都來自
      沒看過他的模型，這才是「新使用者寫了 bio」的真實情境；bundle 的 in-sample 結果只留一列 (3z*) 當洩漏示範。
  (4) 主題模型：NMF(10) 與 LDA(10)（char 1–2 gram）→ 主題分佈 cosine；印每個主題前 8 個 n-gram。
  (5) 風格特徵：長度、句數、換行、空格、emoji 比例、顏文字、注音文、www／哈哈、省略號、驚嘆號、問號、
      波浪號、第一人稱「我」、標點密度、英文字母比例、數字 → StandardScaler → cosine。
      這是標籤完全沒有的維度；用 pair_rank_correlation 證明它與標籤相似度幾乎獨立。
  (6) 混合：對 (2b)、(3)、(5) 各自與 S_tag 做 textrec.blend，w ∈ {0, 0.25, 0.5, 0.75, 1}。
  (7) 冷啟動／半冷啟動模擬（本腳本自己加的，因為這才是「bio 補標籤」真正派得上用場的情境）：
      seeker 端只有 bio（或隨機遮掉一半已勾的標籤），候選端用完整的真實標籤，
      看 bio 推出來的標籤向量能把 recall 拉回多少。

【評估 —— 同一把尺】
所有方法都用 features.subsample(10000, 3000, seed=42) 這 3,000 人，以及同一個
S_tag = cosine(IDF×類別加權標籤)。指標來自 heartlink_ml/textrec.py：
  - neighbor_recovery(k=50 / 10)：文字找到的前 k 位有多少也在標籤的前 k 位；隨機基準 = k/(n-1)，
    lift = recall / 隨機基準。另外附上標準誤與 z 值，分辨「統計上顯著」與「實務上有用」。
  - pair_rank_correlation：隨機抽 20 萬對，兩種相似度的 Spearman。接近 1 = 只是把標籤抽回來；
    接近 0 = 獨立維度（可能是新資訊、也可能是雜訊，這份資料分不出來）。
  - exposure_gini(k=10)：每位候選被推進前 10 名的次數的 Gini，越高代表曝光越集中。
  - 本腳本補充：前 10 名的平均標籤相似度（比 top-k 重疊寬鬆的連續版本）。

【怎麼讀結果（誠實版）】
  - 「recall 對 S_tag」量的是「跟標籤排序有多像」。它能回答冷啟動（沒標籤時能不能逼近標籤排序），
    但**不能**回答「加了文字之後推薦有沒有變好」——任何不是標籤的訊號混進去，這個數字都只會下降。
    要回答後者需要真實互動資料（like / 配對 / 聊天長度），這份專案沒有。
  - (2b) 的 recall ≈ 1 只是因為 X ∨ M 幾乎等於 X（bio 是看著標籤寫的，97% 的提及本來就勾了），
    不是「bio 很有用」。
  - 純文字 bio↔bio 的 (2a)(3)(3z) 實測 lift 只有 1.1–1.9 倍（以 metrics.json 為準）：bio 平均只寫到每人
    15.6 個標籤裡的 1 個左右，兩邊都只靠 bio 時幾乎對不上。改成 (7)「seeker 用 bio、候選用真實標籤」會好一些。
  - (2a) 的 Gini@10 很高有一半是同分的人工現象（全 0 向量的 seeker 對所有人同分，argpartition 永遠取同一批人）；
    extra.2a.gini_at_10_random_tiebreak 是加極小隨機數打破同分後的值，兩個都要看。
  - 每個 lift 旁邊都有 z 值：z < 2 = 與隨機無法區分；z 很大但 lift < 1.5 = 統計上顯著、實務上沒用。
  - (5) Spearman ≈ 0、lift ≈ 1：風格是獨立於標籤的維度；「風格像的人是不是比較合」**無法驗證**。

【怎麼跑】
  cd ml && .venv/bin/python experiments/recommend_bio_text.py
不需要網路、不呼叫 API、不連資料庫；只讀 seed CSV、db/03_seed_traits.sql 與上一輪存的 joblib。約 40 秒。
輸出：
  outputs/recommend_bio_text/metrics.json        所有數字 + 結論
  outputs/recommend_bio_text/keyword_map.json    68 個標籤的關鍵字→標籤映射（含每個關鍵字的命中率），供其他腳本用
  outputs/recommend_bio_text/bio_keyword_matrix.npz / bio_tag_proba_oof.npy / style_features.npz  三種 bio 向量（全 10,000 人）
  outputs/models/recommend_bio_text_*.joblib     風格 StandardScaler、主題模型（NMF / LDA / 向量器）
  outputs/figures/recommend_bio_text_recall.png / _blend.png / _coldstart.png / _style_corr.png

【已知限制】
  - (3) 系列用的是本腳本 5-fold OOF 重訓的 LR（超參數與上一輪相同），不是 bundle 裡那一個模型本身；
    bundle 只用來示範洩漏，並用「還原上一輪切分後只看沒被訓練過的 seeker」交叉驗證 OOF 的數字。
  - 關鍵字匹配是純子字串（沒有斷詞器），像「直接」「安靜」「咖啡」「慢熟」會撞到生成器的填充句；
    每個關鍵字的命中率與最常見的誤判句都存在 keyword_map.json 裡，下游可以自己設門檻。
    同義詞與字典缺口檢查是看著這份資料的標籤整理的（整理字典、不是評估），換成真人資料要重掃。
  - textrec 沒有「seeker 與候選用不同向量」與「前 k 名平均標籤相似度」的函式，這兩個在本腳本內實作。
  - bio 是生成器看著標籤寫的，不是真人自介；所有結論都要在真人資料上重驗。
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from heartlink_ml import plotting  # noqa: F401  匯入即設定 Agg 後端與中文字型
from heartlink_ml import textrec
from heartlink_ml.config import FIG_DIR, MODEL_DIR, OUTPUT_DIR, PROJECT_ROOT, SEED, TRAIT_CATEGORIES, ensure_dirs
from heartlink_ml.data import column_category, load_seed, trait_columns
from heartlink_ml.evaluation import save_json
from heartlink_ml.features import binary_matrix, bio_tfidf, idf_weights, subsample, weighted_matrix

import matplotlib.pyplot as plt  # noqa: E402  要在 heartlink_ml.plotting 之後匯入

MODULE = "recommend_bio_text"
OUT_DIR = OUTPUT_DIR / MODULE
SQL_TRAITS = PROJECT_ROOT / "db" / "03_seed_traits.sql"
BUNDLE_PATH = MODEL_DIR / "supervised_bio_to_tags_best.joblib"
VEC_PATH = MODEL_DIR / "supervised_bio_to_tags_vectorizer.joblib"

N_SUB = 3000                 # n×n 相似度矩陣一律用 3,000 人抽樣（36 MB / 個）
K_MAIN, K_SMALL = 50, 10     # recall@50 為主、recall@10 與 Gini@10 為輔
N_TOPICS = 10
TOP_NGRAMS = 8
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
BLEND_METHODS = ("2b", "3", "5")
PROBA_THRESHOLD = 0.5
EXTRA_THRESHOLDS = (0.7, 0.9)   # (3b) 補充：把門檻拉高會怎樣（0.5 是規格指定的主數字）
NOISY_PRECISION = 0.9        # 命中率低於這個值的關鍵字視為「子字串歧義」
MASK_KEEP = 0.5              # (7) 半冷啟動：每個已勾的標籤以 0.5 的機率被保留
DEMO_TOP = 3
DEMO_METHODS = ("1", "2b", "3", "5")
# label_zh 裡的前綴：「愛海鮮」在 bio 裡通常寫成「海鮮」，所以去掉前綴後的字也當關鍵字
LABEL_PREFIXES = ("愛", "重視", "需要")
# 標籤名以外的同義詞。來源有兩個：
#  (a) experiments/supervised_bio_to_tags.py 的 TRAIT_KEYWORDS（上一輪用「中文名掃到 0 次、但 LR AUC 仍 > 0.5」
#      反推出來的說法）。這裡只複製、不修改那支腳本。
#  (b) 本腳本 scan_missing_keywords() 掃出來的 4 個字典缺口：生成器寫的是「烹飪／聽團／探店／慢熟」，
#      但 label_zh 是「料理／live 音樂／逛店／慢熱」，上一輪猜的「下廚、煮飯、演唱會、小店」實際出現 0 次。
#      沒補之前這 4 個標籤（約佔全部提及的 6%）完全掃不到，會低估關鍵字法。
SYNONYMS = {
    "dating_goal_dining_partner": ["飯友"], "dating_goal_marriage_minded": ["結婚"],
    "interest_hiking": ["爬山"], "interest_cycling": ["單車"], "interest_gaming": ["電動"],
    "interest_cooking": ["下廚", "煮飯", "烹飪"], "interest_karaoke": ["唱K"],
    "interest_live_music": ["live", "演唱會", "聽團"], "interest_exploring_shops": ["小店", "探店"],
    "interest_fashion": ["穿搭"], "personality_quiet": ["安靜"], "personality_homebody": ["宅"],
    "personality_slow_to_warm_up": ["慢熟"],
    "diet_vegetarian": ["吃素"], "lifestyle_work_from_home": ["遠端"],
    "value_likes_sharing_daily_life": ["分享生活"],
}
# 風格特徵用的正規表示式（全部是字元層級，不需要斷詞）
RE_EMOJI = re.compile(r"[\U0001F000-\U0001FAFF☀-➿⬀-⯿]")
RE_KAOMOJI = re.compile(r"[ω･｡•｀´╥▽∀︶＾]")
RE_ZHUYIN = re.compile(r"[ㄅ-ㄩ]")            # ㄅㄆㄇ… 注音符號（「ㄅ知道」「聊ㄅ」）
RE_LAUGH = re.compile(r"(?:w{3,}|哈哈+|XD|ㄎㄎ|笑死)", re.IGNORECASE)
RE_ELLIPSIS = re.compile(r"(?:\.{3,}|…+)")
RE_EXCLAIM = re.compile(r"[!！]")
RE_QUESTION = re.compile(r"[?？]")
RE_TILDE = re.compile(r"[~～]")
RE_PUNCT = re.compile(r"[，。、！？；：「」『』（）()…～~,.!?;:\-—【】《》]")
RE_ASCII = re.compile(r"[A-Za-z]")
RE_DIGIT = re.compile(r"\d")
RE_SENT_SPLIT = re.compile(r"[。！？!?\n]+")
# 生成器（與多數真人）用來分隔短句的寫法：「 / 」、連續空格、「...」、句號、換行、波浪號、逗號、頓號
RE_SEGMENT = re.compile(r"\s*/\s*|\s{2,}|\.{3,}|…+|[。\n～~，、]")

N_FOLDS = 5                  # (3) out-of-fold 機率的折數
METHOD_ORDER = ["tag", "random", "1", "2a", "2b", "3", "3z", "3z_in", "3b", "4a", "4b", "5"]
METHOD_NAME = {
    "tag": "標籤本身 S_tag（參考）", "random": "random（基準）",
    "1": "(1) 整段 TF-IDF", "2a": "(2a) 關鍵字→標籤 M", "2b": "(2b) 標籤 ∨ 關鍵字",
    "3": "(3) P(tag|bio)（OOF）", "3z": "(3z) P 欄標準化（OOF）", "3z_in": "(3z*) bundle in-sample（洩漏示範）",
    "3b": "(3b) 標籤 ∨ (P>0.5)",
    "4a": "(4a) NMF-10 主題", "4b": "(4b) LDA-10 主題", "5": "(5) 風格特徵",
}
METHOD_SHORT = {
    "tag": "S_tag", "random": "random", "1": "(1) TF-IDF", "2a": "(2a) 關鍵字", "2b": "(2b) 標籤∨關鍵字",
    "3": "(3) P(tag|bio)", "3z": "(3z) P標準化", "3z_in": "(3z*) 洩漏示範", "3b": "(3b) 標籤∨P>0.5",
    "4a": "(4a) NMF", "4b": "(4b) LDA", "5": "(5) 風格",
}
# 圖的顏色：含原標籤的（augment）綠色、純文字藍色、風格橘色、基準灰色
COLOR_OF = {"tag": "#5a5a55", "random": "#9a9a94", "1": "#2a78d6", "2a": "#2a78d6", "2b": "#1baf7a",
            "3": "#2a78d6", "3z": "#2a78d6", "3z_in": "#b9b9b3", "3b": "#1baf7a", "4a": "#2a78d6", "4b": "#2a78d6", "5": "#eb6834"}
# (7) 的情境：名稱 → 說明
COLD_NAME = {
    "cold_M": "冷啟動：seeker = bio 關鍵字標籤 M",
    "cold_P": "冷啟動：seeker = P(tag|bio) − 平均",
    "half_obs": "半冷啟動：seeker = 只剩一半的標籤 X_obs",
    "half_obs_or_M": "半冷啟動：X_obs ∨ M",
    "half_obs_plus_P": "半冷啟動：X_obs + (P − 平均)",
    "half_obs_or_Pbin": f"半冷啟動：X_obs ∨ (P>{PROBA_THRESHOLD})",
}


# ---------------------------------------------------------------- (2) 關鍵字 → 標籤
def parse_traits_sql(path=SQL_TRAITS) -> list[dict]:
    """從 db/03_seed_traits.sql 讀出 68 列 (category, code, label_zh, csv_column)。"""
    pat = re.compile(r"\('([a-z_]+)',\s*'([a-z_]+)',\s*'([^']+)',\s*'([a-z_]+)'\)")
    rows = [{"category": c, "code": k, "label_zh": z, "col": col}
            for c, k, z, col in pat.findall(path.read_text(encoding="utf-8"))]
    if len(rows) != 68:
        raise RuntimeError(f"{path} 應該有 68 個標籤，實際解析到 {len(rows)} 個")
    return rows


def strip_prefix(label: str) -> str:
    for p in LABEL_PREFIXES:
        if label.startswith(p) and len(label) > len(p):
            return label[len(p):]
    return label


def build_keyword_map(traits: list[dict]) -> dict:
    """每個標籤的關鍵字 = label_zh + 去掉前綴的 label_zh + 同義詞（全部小寫、去重、保序）。"""
    out = {}
    for t in traits:
        kws = [t["label_zh"], strip_prefix(t["label_zh"]), *SYNONYMS.get(t["col"], [])]
        seen, uniq = set(), []
        for k in kws:
            k = k.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(k)
        out[t["col"]] = {"category": t["category"], "label_zh": t["label_zh"], "keywords": uniq}
    return out


def keyword_matrix(bio_lower: pd.Series, cols, kw_map: dict, X: np.ndarray):
    """回傳 (M: bool (n, 68)「bio 提到該標籤」, 每個關鍵字的統計)。
    命中率 P(標籤|提到) 用全體 X 算，存進 keyword_map.json 讓下游能自己設門檻。"""
    n = len(bio_lower)
    M = np.zeros((n, len(cols)), dtype=bool)
    stats = {}
    for j, c in enumerate(cols):
        per_kw = []
        for kw in kw_map[c]["keywords"]:
            hit = bio_lower.str.contains(kw, regex=False).to_numpy(dtype=bool)
            M[:, j] |= hit
            per_kw.append({"keyword": kw, "n_mention": int(hit.sum()),
                           "precision_p_tag_given_mention": float(X[hit, j].mean()) if hit.any() else None})
        y = X[:, j]
        stats[c] = {"category": kw_map[c]["category"], "label_zh": kw_map[c]["label_zh"], "keywords": per_kw,
                    "n_mention_any": int(M[:, j].sum()),
                    "p_tag_given_mention": float(y[M[:, j]].mean()) if M[:, j].any() else None,
                    "p_mention_given_tag": float(M[y, j].mean())}
    return M, stats


def scan_missing_keywords(bio_lower: pd.Series, X: np.ndarray, cols, M: np.ndarray,
                          min_df=60, min_precision=NOISY_PRECISION, top=3) -> dict:
    """字典缺口檢查：對「bio 完全沒掃到關鍵字」的標籤，找 P(有該標籤 | bio 含此 n-gram) ≥ 0.9 的 char 2–4 gram。
    有找到 = bio 用了字典沒收的說法，請補進 SYNONYMS；沒找到 = bio 真的沒寫這個標籤（例如 diet / lifestyle）。
    這一步用到標籤，屬於「整理字典」而不是評估；換成真人資料要重掃。回傳 {標籤欄名: [候選 n-gram…]}。"""
    zero_cols = [j for j in range(len(cols)) if not M[:, j].any()]
    if not zero_cols:
        return {}
    cv = CountVectorizer(analyzer="char", ngram_range=(2, 4), min_df=min_df, binary=True)
    B = cv.fit_transform(bio_lower)
    vocab = cv.get_feature_names_out()
    df_ng = np.asarray(B.sum(0)).ravel()
    out = {}
    for j in zero_cols:
        prec = np.asarray(B[np.where(X[:, j])[0]].sum(0)).ravel() / df_ng
        best = [i for i in np.argsort(-prec, kind="stable")[:top] if prec[i] >= min_precision]
        out[cols[j]] = [{"ngram": str(vocab[i]), "n_bio": int(df_ng[i]), "precision": float(prec[i])} for i in best]
    return out


def false_positive_example(bio_lower: pd.Series, kw: str, fp_mask: np.ndarray):
    """關鍵字誤判（bio 有這個字、但使用者沒勾該標籤）時，最常見的那個短句與次數——讓人一眼看出是哪種歧義。"""
    seen = {}
    for text in bio_lower[fp_mask]:
        for seg in RE_SEGMENT.split(text):
            if kw in seg:
                seen[seg.strip()] = seen.get(seg.strip(), 0) + 1
    if not seen:
        return None, 0
    seg = max(seen, key=seen.get)
    return seg, seen[seg]


def augment_stats(X: np.ndarray, A: np.ndarray, cats: np.ndarray) -> dict:
    """X 是原標籤、A 是從 bio 推出來的標籤：平均每人幾個、其中多少本來就勾了（重疊率）、真正新增幾個。"""
    per_person = A.sum(1)
    both = A & X
    new = A & ~X
    per_cat = {}
    for cat in TRAIT_CATEGORIES:
        m = cats == cat
        per_cat[cat] = {"mean_from_bio": float(A[:, m].sum(1).mean()),
                        "overlap_rate": float(both[:, m].sum() / max(A[:, m].sum(), 1))}
    return {
        "rate_with_any": float((per_person > 0).mean()),
        "mean_tags_from_bio": float(per_person.mean()),
        "mean_already_ticked": float(both.sum(1).mean()),
        "mean_new_tags": float(new.sum(1).mean()),
        "overlap_rate": float(both.sum() / max(A.sum(), 1)),
        "mean_tags_after_union": float((X | A).sum(1).mean()),
        "mean_tags_original": float(X.sum(1).mean()),
        "per_category": per_cat,
    }


# ---------------------------------------------------------------- (3) bio → 標籤機率
def load_bundle_proba(bio: pd.Series, cols):
    """載入上一輪的 bundle，回傳 (P_in: (n, 68) 或 None, 說明 dict)。
    P_in 是「模型對自己訓練過的人」的機率（80% in-sample），只拿來示範洩漏，不當主數字。"""
    info = {"source": None, "loaded": False, "bundle_types": None}
    try:
        bundle = joblib.load(BUNDLE_PATH)
        vec = joblib.load(VEC_PATH)
        info["bundle_types"] = ({k: type(v).__name__ for k, v in bundle.items()} if isinstance(bundle, dict)
                                else type(bundle).__name__)
        print(f"bundle keys/型別：{info['bundle_types']}；vectorizer：{type(vec).__name__}")
        model, labels, svd = bundle["model"], bundle["labels"], bundle.get("svd")
        if list(labels) != list(cols):
            raise ValueError("bundle 的 labels 順序與 trait_columns 不一致")
        if not hasattr(model, "predict_proba"):
            raise ValueError(f"bundle 的模型 {type(model).__name__} 沒有 predict_proba")
        Z = vec.transform(bio)
        if svd is not None:
            Z = svd.transform(Z)
        P_in = np.asarray(model.predict_proba(Z), dtype=np.float32)
        if P_in.shape != (len(bio), len(cols)):
            raise ValueError(f"predict_proba 形狀 {P_in.shape} 不是 (n, 68)")
        info["source"] = f"loaded {BUNDLE_PATH.name}（{bundle.get('name')}）"
        info["loaded"] = True
        return P_in, info
    except Exception as e:  # noqa: BLE001  bundle 壞了不影響主流程（主數字用 OOF），只是少了洩漏示範那一列
        print(f"無法使用 bundle（{e}）：跳過洩漏示範，主數字仍用本腳本的 OOF 機率")
        info["source"] = f"bundle unusable: {e}"
        return None, info


def oof_bio_proba(bio: pd.Series, X: np.ndarray) -> np.ndarray:
    """5-fold out-of-fold 的 P(tag|bio)：TF-IDF 與 LR 都只在訓練折 fit，再對沒看過的那一折推論。
    超參數沿用上一輪 supervised_bio_to_tags.py 的 LR（C=4、class_weight=balanced）。約 6 秒。"""
    P = np.zeros(X.shape, dtype=np.float32)
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.arange(len(bio))):
        Xt, vec = bio_tfidf(bio.iloc[tr])
        model = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"), n_jobs=-1)
        model.fit(Xt, X[tr].astype(int))
        P[te] = model.predict_proba(vec.transform(bio.iloc[te]))
    return P


def mean_label_auc(X: np.ndarray, P: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(X[:, j], P[:, j]) for j in range(X.shape[1])]))


# ---------------------------------------------------------------- (4) 主題模型
def top_ngrams(components: np.ndarray, vocab: np.ndarray, k=TOP_NGRAMS) -> list[list[str]]:
    return [[str(vocab[i]) for i in np.argsort(-row)[:k]] for row in components]


def topic_keyword_share(topics: list[list[str]], all_keywords: list[str]) -> float:
    """主題前 k 個 n-gram 裡的「2 字元 n-gram」，有多少與某個標籤關鍵字互為子字串。
    只看 2-gram 是因為單一字元（生、活、歡…）幾乎一定是某個關鍵字的子字串，會灌水。
    這只是粗略代理指標：低 = 主題抓的是虛字／模板句，不是標籤。"""
    hits = total = 0
    for words in topics:
        for w in words:
            if len(w.strip()) < 2:
                continue
            total += 1
            hits += any((w in kw) or (kw in w) for kw in all_keywords)
    return hits / max(total, 1)


# ---------------------------------------------------------------- (5) 風格特徵
def style_features(bio: pd.Series):
    """回傳 (DataFrame 風格特徵, 被丟掉的常數特徵)。全部是字元層級統計，標籤裡完全沒有這些資訊。"""
    s = bio.astype(str)
    length = s.str.len().to_numpy(dtype=np.float64)
    length_safe = np.maximum(length, 1)

    def count(rx):
        return s.apply(lambda t: len(rx.findall(t))).to_numpy(dtype=np.float64)

    def n_sentences(t):
        return max(1, sum(1 for seg in RE_SENT_SPLIT.split(t) if seg.strip()))

    n_sent = s.apply(n_sentences).to_numpy(dtype=np.float64)
    F = pd.DataFrame({
        "length": length,
        "n_sentences": n_sent,
        "mean_sentence_len": length / n_sent,
        "n_newlines": s.str.count("\n").to_numpy(dtype=np.float64),
        "n_spaces": s.str.count(" ").to_numpy(dtype=np.float64),
        "emoji_ratio": count(RE_EMOJI) / length_safe,
        "kaomoji_count": count(RE_KAOMOJI),
        "zhuyin_count": count(RE_ZHUYIN),
        "laugh_count": count(RE_LAUGH),
        "ellipsis_count": count(RE_ELLIPSIS),
        "exclaim_count": count(RE_EXCLAIM),
        "question_count": count(RE_QUESTION),
        "tilde_count": count(RE_TILDE),
        "first_person_count": s.str.count("我").to_numpy(dtype=np.float64),
        "punct_density": count(RE_PUNCT) / length_safe,
        "ascii_ratio": count(RE_ASCII) / length_safe,
        "digit_count": count(RE_DIGIT),
    })
    dropped = [c for c in F.columns if F[c].std() == 0]
    return F.drop(columns=dropped), dropped


# ---------------------------------------------------------------- 評估
def mean_tag_sim_topk(S: np.ndarray, S_tag: np.ndarray, k: int, seekers=None) -> float:
    """S 的前 k 名（排除自己）在 S_tag 上的平均值：比 top-k 重疊寬鬆的連續版本。textrec 沒有，這裡自己算。"""
    seekers = np.arange(S.shape[0]) if seekers is None else np.asarray(seekers)
    R = S[seekers].astype(np.float32, copy=True)
    R[np.arange(len(seekers)), seekers] = -np.inf
    top = np.argpartition(-R, k, axis=1)[:, :k]
    return float(np.take_along_axis(S_tag[seekers], top, axis=1).mean())


def evaluate_method(S: np.ndarray, S_tag: np.ndarray, seekers=None) -> dict:
    """同一把尺：recall@50 / @10（textrec）、Spearman vs 標籤、Gini@10，外加標準誤、z 值與前 10 名平均標籤相似度。"""
    r50 = textrec.neighbor_recovery(S, S_tag, k=K_MAIN, seekers=seekers)
    r10 = textrec.neighbor_recovery(S, S_tag, k=K_SMALL, seekers=seekers)
    rho = textrec.pair_rank_correlation(S, S_tag, seed=SEED)
    se = r50["recall_std"] / np.sqrt(max(r50["n_seekers"], 1))
    return {
        "recall_at_50": r50["recall_at_k"], "recall_se_50": float(se),
        "random_baseline_50": r50["random_baseline"], "lift_50": r50["lift_over_random"],
        "z_vs_random_50": float((r50["recall_at_k"] - r50["random_baseline"]) / se) if se > 0 else None,
        "recall_at_10": r10["recall_at_k"], "random_baseline_10": r10["random_baseline"], "lift_10": r10["lift_over_random"],
        "spearman_vs_tag": rho["spearman"], "spearman_p_value": rho["p_value"],
        "gini_at_10": textrec.exposure_gini(S, k=K_SMALL, seekers=seekers),
        "mean_tag_sim_top10": mean_tag_sim_topk(S, S_tag, K_SMALL, seekers),
        "n_seekers": r50["n_seekers"],
    }


def top_recs(S: np.ndarray, pos: int, k=DEMO_TOP) -> np.ndarray:
    row = S[pos].astype(np.float32, copy=True)
    row[pos] = -np.inf
    return np.argsort(-row, kind="stable")[:k]


def short(text: str, n=60) -> str:
    text = text.replace("\n", "⏎")
    return text if len(text) <= n else text[:n] + "…"


# ---------------------------------------------------------------- 畫圖
def _style(ax):
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def _bars(ax, names, vals, colors, base, fmt="{:.3f}"):
    hatches = ["//" if "洩漏" in nm else "" for nm in names]      # 洩漏示範那一條加斜線，避免被誤讀成最好的方法
    bars = ax.bar(names, vals, color=colors, width=0.65, edgecolor="white", hatch=hatches)
    top = max(max(vals), base)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + top * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=8)
    ax.axhline(base, color="gray", ls="--", lw=1, label=f"隨機基準 k/(n-1) = {base:.3f}")
    ax.set_ylim(0, top * 1.15)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    _style(ax)


def plot_recall(methods: dict, order: list, path):
    """左：全部方法（(2b) 因為含原標籤所以接近 1）；右：把純文字方法放大，才看得出跟隨機基準的差距。"""
    base = methods["random"]["random_baseline_50"]
    keys_all = [k for k in order if k != "tag"]
    keys_text = [k for k in keys_all if k not in ("2b", "3b")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    _bars(axes[0], [METHOD_SHORT[k] for k in keys_all], [methods[k]["recall_at_50"] for k in keys_all],
          [COLOR_OF[k] for k in keys_all], base)
    axes[0].set_title("全部方法（綠 = 含原標籤的 augment）")
    axes[0].set_ylabel("recall@50（以標籤前 50 位鄰居為正解）")
    _bars(axes[1], [METHOD_SHORT[k] for k in keys_text], [methods[k]["recall_at_50"] for k in keys_text],
          [COLOR_OF[k] for k in keys_text], base)
    axes[1].set_title("只看純文字方法（放大）")
    fig.suptitle("bio 文字做法的 recall@50（灰色 (3z*) = 模型對自己訓練過的人推論，是洩漏、不是效果）", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_blend(blend: dict, base_recall: float, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for key, rows in blend.items():
        ws = [r["w_text"] for r in rows]
        axes[0].plot(ws, [r["recall_at_50"] for r in rows], marker="o", color=COLOR_OF[key], label=METHOD_SHORT[key])
        axes[1].plot(ws, [r["gini_at_10"] for r in rows], marker="o", color=COLOR_OF[key], label=METHOD_SHORT[key])
    axes[0].axhline(base_recall, color="gray", ls="--", lw=1, label="隨機基準")
    axes[0].set_title("recall@50 vs 文字權重 w（w=0 全標籤、w=1 全文字）")
    axes[1].set_title("曝光 Gini@10 vs 文字權重 w")
    for ax, yl in zip(axes, ("recall@50（= 與標籤排序的相似程度）", "Gini@10")):
        ax.set_xlabel("w（文字相似度的權重）")
        ax.set_ylabel(yl)
        ax.set_xticks(BLEND_WEIGHTS)
        ax.legend(fontsize=8, frameon=False)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_coldstart(cold: dict, base: float, path):
    keys = list(COLD_NAME)
    names = ["冷：M→標籤", "冷：P→標籤", "半：X_obs", "半：X_obs∨M", "半：X_obs+P", f"半：X_obs∨P>{PROBA_THRESHOLD}"]
    colors = ["#2a78d6", "#2a78d6", "#9a9a94", "#1baf7a", "#1baf7a", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    _bars(ax, names, [cold[k]["all"]["recall_at_50"] for k in keys], colors, base)
    ax.set_ylabel("recall@50")
    ax.set_title("(7) seeker 端缺標籤、候選端用真實標籤：bio 能把 recall 拉回多少（灰 = 不用 bio 的對照）")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_style_corr(F: pd.DataFrame, path):
    C = np.corrcoef(F.to_numpy().T)
    names = list(F.columns)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{C[i, j]:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(C[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title("風格特徵之間的 Pearson 相關（全 10,000 人）")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return C


# ---------------------------------------------------------------- 主程式
def main():
    t0 = time.perf_counter()
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 0. 資料、抽樣、共同的標籤相似度 S_tag
    df = load_seed()
    cols = trait_columns(df)
    cats = np.array([column_category(c) for c in cols])
    X = binary_matrix(df, cols)
    bio = df["bio"]
    bio_lower = bio.str.lower()
    n = len(df)
    idx = subsample(n, N_SUB, seed=SEED)
    Xw, w_tag = weighted_matrix(X, cols)
    S_tag = textrec.cosine_sim(Xw[idx])
    rng = np.random.default_rng(SEED)
    print(f"使用者 {n}，標籤 {len(cols)}，抽樣 {len(idx)} 人做 n×n 相似度；bio 中位長度 {int(bio.str.len().median())} 字")

    S = {}          # 各方法的 3000×3000 相似度矩陣
    methods = {}    # 各方法的評估數字
    extra = {}      # 各方法的附加資訊（提到率、主題詞…）
    S["tag"] = S_tag
    methods["tag"] = evaluate_method(S_tag, S_tag)
    S["random"] = rng.standard_normal((len(idx), len(idx))).astype(np.float32)
    methods["random"] = evaluate_method(S["random"], S_tag)

    # (1) 整段 TF-IDF：天真做法
    Xt, tfidf_vec = bio_tfidf(bio)
    S["1"] = textrec.cosine_sim(Xt[idx])
    methods["1"] = evaluate_method(S["1"], S_tag)
    print(f"(1) 整段 TF-IDF：{Xt.shape[1]} 個 char 1–2 gram，recall@50 {methods['1']['recall_at_50']:.3f}"
          f"（lift {methods['1']['lift_50']:.2f}）")

    # (2) 關鍵字 → 標籤空間
    traits = parse_traits_sql()
    kw_map = build_keyword_map(traits)
    M, kw_stats = keyword_matrix(bio_lower, cols, kw_map, X)
    all_keywords = sorted({kw for c in cols for kw in kw_map[c]["keywords"]})
    aug_kw = augment_stats(X, M, cats)
    M_idf = M.astype(np.float32) * idf_weights(M)
    S["2a"] = textrec.cosine_sim(M_idf[idx])
    methods["2a"] = evaluate_method(S["2a"], S_tag)
    has_kw = np.where(M[idx].sum(1) > 0)[0]
    no_kw = np.where(M[idx].sum(1) == 0)[0]
    extra["2a"] = {"keyword_stats": aug_kw, "n_keywords": len(all_keywords),
                   "seekers_with_keyword": evaluate_method(S["2a"], S_tag, seekers=has_kw),
                   "seekers_without_keyword": evaluate_method(S["2a"], S_tag, seekers=no_kw)}
    Xw_aug, _ = weighted_matrix(X | M, cols)
    S["2b"] = textrec.cosine_sim(Xw_aug[idx])
    methods["2b"] = evaluate_method(S["2b"], S_tag)
    extra["2b"] = {"augment": aug_kw}
    mentioned_cats = {cat: int(sum(kw_stats[c]["n_mention_any"] > 0 for c in cols if kw_stats[c]["category"] == cat))
                      for cat in TRAIT_CATEGORIES}
    noisy_kws = sorted(((k["keyword"], k["precision_p_tag_given_mention"], k["n_mention"], j)
                        for j, c in enumerate(cols) for k in kw_stats[c]["keywords"]
                        if k["n_mention"] >= 20 and k["precision_p_tag_given_mention"] < NOISY_PRECISION), key=lambda t: t[1])
    # 「bio 補進來的新標籤」(M & ~X) 有多少其實是歧義關鍵字的誤判？（bio 是看著標籤寫的，理論上不該有新標籤）
    new_mask = M & ~X
    noisy_cols = sorted({j for *_, j in noisy_kws})
    new_from_noisy_share = float(new_mask[:, noisy_cols].sum() / max(new_mask.sum(), 1))
    noisy_rows = []
    for kw, prec, cnt, j in noisy_kws:
        fp = bio_lower.str.contains(kw, regex=False).to_numpy(dtype=bool) & ~X[:, j]
        seg, seg_n = false_positive_example(bio_lower, kw, fp)
        noisy_rows.append({"keyword": kw, "precision": prec, "n_mention": cnt, "n_false_positive": int(fp.sum()),
                           "most_common_false_positive_segment": seg, "segment_count": seg_n})
    extra["2a"]["noisy_keywords_precision_below_0.9"] = noisy_rows
    extra["2b"]["new_tags_total"] = int(new_mask.sum())
    extra["2b"]["new_tags_share_from_noisy_keywords"] = new_from_noisy_share
    # 同分的人工現象：全 0 向量的 seeker 對所有人 cosine = 0，argpartition 永遠取同一批人。加極小的隨機數打破同分再算一次 Gini。
    jitter = np.random.default_rng(SEED + 2).random(S["2a"].shape, dtype=np.float32) * np.float32(1e-5)
    extra["2a"]["gini_at_10_random_tiebreak"] = textrec.exposure_gini(S["2a"] + jitter, k=K_SMALL)
    extra["2a"]["gini_at_10_random_tiebreak_seekers_with_keyword"] = textrec.exposure_gini(S["2a"] + jitter, k=K_SMALL, seekers=has_kw)
    del jitter
    # 字典缺口檢查：還有哪些標籤 bio 完全沒掃到？它們有沒有「字典沒收、但命中率很高」的說法？
    missing = scan_missing_keywords(bio_lower, X, cols, M)
    missing_found = {c: v for c, v in missing.items() if v}
    extra["2a"]["zero_mention_traits"] = missing
    print(f"(2) 關鍵字：{len(all_keywords)} 個關鍵字；{aug_kw['rate_with_any']:.1%} 的人 bio 至少提到一個標籤名，"
          f"平均每人從 bio 得到 {aug_kw['mean_tags_from_bio']:.2f} 個標籤，其中 {aug_kw['overlap_rate']:.1%} 本來就勾了"
          f"（平均只新增 {aug_kw['mean_new_tags']:.2f} 個）；名稱有出現的標籤數依類別：{mentioned_cats}")
    print(f"    命中率 < {NOISY_PRECISION} 的關鍵字（子字串歧義）：" + "；".join(
        f"{r['keyword']} {r['precision']:.2f}（誤判 {r['n_false_positive']} 次，最常見：「{r['most_common_false_positive_segment']}」×{r['segment_count']}）"
        for r in noisy_rows))
    print(f"    bio「新增」的 {int(new_mask.sum())} 個標籤裡，{new_from_noisy_share:.0%} 來自上面這些歧義關鍵字（= 誤判，不是真的新標籤）")
    print(f"    字典缺口檢查：bio 完全沒掃到的標籤 {len(missing)} 個；其中有高命中率 n-gram 可補的："
          f"{missing_found if missing_found else '無（這些標籤 bio 真的沒寫）'}")
    print(f"    (2a) Gini@10 {methods['2a']['gini_at_10']:.3f} → 隨機打破同分後 {extra['2a']['gini_at_10_random_tiebreak']:.3f}"
          f"（只看有提到標籤名的 seeker：{extra['2a']['gini_at_10_random_tiebreak_seekers_with_keyword']:.3f}）")
    print(f"    (2a) 有提到者 recall@50 {extra['2a']['seekers_with_keyword']['recall_at_50']:.3f} / "
          f"沒提到者 {extra['2a']['seekers_without_keyword']['recall_at_50']:.3f}（後者向量全 0，全部同分）")

    # (3) bio → 標籤機率：主數字用 OOF（每個人都沒被自己的模型看過），bundle in-sample 只當洩漏示範
    P_in, model_info = load_bundle_proba(bio, cols)
    P = oof_bio_proba(bio, X)
    auc_oof = mean_label_auc(X, P)
    S["3"] = textrec.cosine_sim(P[idx])
    methods["3"] = evaluate_method(S["3"], S_tag)
    P_c = P - P.mean(0)                       # 減掉每個標籤的平均機率（大家都一樣的成分）
    P_z = P_c / (P.std(0) + 1e-9)
    S["3z"] = textrec.cosine_sim(P_z[idx])
    methods["3z"] = evaluate_method(S["3z"], S_tag)
    P_bin = P > PROBA_THRESHOLD
    aug_p = augment_stats(X, P_bin, cats)
    Xw_p, _ = weighted_matrix(X | P_bin, cols)
    S["3b"] = textrec.cosine_sim(Xw_p[idx])
    methods["3b"] = evaluate_method(S["3b"], S_tag)
    common_share = float(np.linalg.norm(P.mean(0)) / np.linalg.norm(P, axis=1).mean())
    extra["3"] = {"proba": f"{N_FOLDS}-fold out-of-fold OvR(LogisticRegression C=4, class_weight=balanced)",
                  "oof_mean_label_auc": auc_oof, "bundle": model_info,
                  "mean_proba_per_label_min": float(P.mean(0).min()), "mean_proba_per_label_max": float(P.mean(0).max()),
                  "common_component_share": common_share}
    extra["3b"] = {"augment": aug_p, "threshold": PROBA_THRESHOLD}
    # 補充：0.5 是規格指定的門檻；把門檻拉高，P 會變成什麼？（看每人幾個、多少本來就勾了、多少其實就是關鍵字掃得到的）
    sweep = [{"threshold": PROBA_THRESHOLD, "mean_tags_predicted": aug_p["mean_tags_from_bio"],
              "overlap_rate": aug_p["overlap_rate"], "mean_new_tags": aug_p["mean_new_tags"],
              "share_also_found_by_keywords": float((P_bin & M).sum() / max(P_bin.sum(), 1)),
              "recall_at_50": methods["3b"]["recall_at_50"]}]
    for thr in EXTRA_THRESHOLDS:
        Pb = P > thr
        Xw_t, _ = weighted_matrix(X | Pb, cols)
        r_t = textrec.neighbor_recovery(textrec.cosine_sim(Xw_t[idx]), S_tag, k=K_MAIN)
        sweep.append({"threshold": thr, "mean_tags_predicted": float(Pb.sum(1).mean()),
                      "overlap_rate": float((Pb & X).sum() / max(Pb.sum(), 1)),
                      "mean_new_tags": float((Pb & ~X).sum(1).mean()),
                      "share_also_found_by_keywords": float((Pb & M).sum() / max(Pb.sum(), 1)),
                      "recall_at_50": r_t["recall_at_k"]})
    extra["3b"]["threshold_sweep"] = sweep
    # 洩漏示範：bundle 直接對全部人推論（80% 是它的訓練集）。還原上一輪切分，另外只看沒被訓練過的 seeker。
    held_pos, P_in_c = None, None
    if P_in is not None:
        _, idx_te = train_test_split(np.arange(n), test_size=0.2, random_state=SEED)
        held_pos = np.where(np.isin(idx, idx_te))[0]
        in_train = np.ones(n, dtype=bool)
        in_train[idx_te] = False
        P_in_c = P_in - P_in.mean(0)
        S["3z_in"] = textrec.cosine_sim((P_in_c / (P_in.std(0) + 1e-9))[idx])
        methods["3z_in"] = evaluate_method(S["3z_in"], S_tag)
        hh = np.ix_(held_pos, held_pos)          # seeker 與候選都只留 bundle 沒訓練過的人（候選變少，隨機基準跟著變）
        extra["3z_in"] = {"auc_on_its_train_users": mean_label_auc(X[in_train], P_in[in_train]),
                          "auc_on_its_test_users": mean_label_auc(X[idx_te], P_in[idx_te]),
                          "held_out_seekers": evaluate_method(S["3z_in"], S_tag, seekers=held_pos),
                          "held_out_seekers_oof": evaluate_method(S["3z"], S_tag, seekers=held_pos),
                          "held_out_seekers_and_candidates": textrec.neighbor_recovery(S["3z_in"][hh], S_tag[hh], k=K_MAIN),
                          "held_out_seekers_and_candidates_oof": textrec.neighbor_recovery(S["3z"][hh], S_tag[hh], k=K_MAIN)}
    order = [k for k in METHOD_ORDER if k != "3z_in" or P_in is not None]
    print(f"(3) P(tag|bio)：{N_FOLDS}-fold OOF mean AUC {auc_oof:.3f}；機率向量的共同成分佔 {common_share:.0%}；"
          f"(3) lift {methods['3']['lift_50']:.2f} → (3z) 標準化後 {methods['3z']['lift_50']:.2f}")
    if P_in is not None:
        print(f"    洩漏示範：{model_info['source']} 對訓練過的人 AUC {extra['3z_in']['auc_on_its_train_users']:.3f}、"
              f"沒訓練過的人 {extra['3z_in']['auc_on_its_test_users']:.3f}；(3z*) in-sample lift {methods['3z_in']['lift_50']:.2f} "
              f"vs OOF {methods['3z']['lift_50']:.2f}")
    print(f"    (3b) P>{PROBA_THRESHOLD} 平均每人 {aug_p['mean_tags_from_bio']:.1f} 個、其中 {aug_p['overlap_rate']:.0%} 本來就勾了、"
          f"平均新增 {aug_p['mean_new_tags']:.1f} 個（原本每人 {aug_p['mean_tags_original']:.1f} 個）")
    print("    (3b) 門檻掃描：" + "；".join(
        f"P>{r['threshold']} 每人 {r['mean_tags_predicted']:.2f} 個、{r['overlap_rate']:.0%} 本來就勾了、"
        f"{r['share_also_found_by_keywords']:.0%} 關鍵字也掃得到 → recall@50 {r['recall_at_50']:.3f}" for r in sweep))

    # (4) 主題模型
    nmf = NMF(n_components=N_TOPICS, init="nndsvda", random_state=SEED, max_iter=300)
    W_nmf = nmf.fit_transform(Xt)
    S["4a"] = textrec.cosine_sim(W_nmf[idx])
    methods["4a"] = evaluate_method(S["4a"], S_tag)
    count_vec = CountVectorizer(analyzer="char", ngram_range=(1, 2), min_df=3, max_features=20000)
    Xc = count_vec.fit_transform(bio)
    lda = LatentDirichletAllocation(n_components=N_TOPICS, learning_method="online", max_iter=10, batch_size=500,
                                    random_state=SEED)
    W_lda = lda.fit_transform(Xc)
    S["4b"] = textrec.cosine_sim(W_lda[idx])
    methods["4b"] = evaluate_method(S["4b"], S_tag)
    topics_nmf = top_ngrams(nmf.components_, tfidf_vec.get_feature_names_out())
    topics_lda = top_ngrams(lda.components_, count_vec.get_feature_names_out())
    share_nmf, share_lda = topic_keyword_share(topics_nmf, all_keywords), topic_keyword_share(topics_lda, all_keywords)
    extra["4a"] = {"topics_top8": topics_nmf, "bigram_keyword_share_in_top8": share_nmf,
                   "reconstruction_err": float(nmf.reconstruction_err_)}
    extra["4b"] = {"topics_top8": topics_lda, "bigram_keyword_share_in_top8": share_lda,
                   "perplexity": float(lda.perplexity(Xc))}
    # 人眼解讀主題詞時對到的兩句生成器模板句；附上實際出現次數讓結論可以驗證（換資料後若為 0 代表這段解讀已過期）
    template_counts = {t: int(bio.str.contains(t, regex=False).sum())
                       for t in ("不喜歡把聊天搞得太有壓力", "希望相處起來可以自然一點")}
    extra["4a"]["template_sentence_counts"] = template_counts
    print(f"(4) 主題：前 8 n-gram 的 2-gram 裡與標籤關鍵字重疊的比例 NMF {share_nmf:.0%}、LDA {share_lda:.0%}")
    for name, tp in (("NMF", topics_nmf), ("LDA", topics_lda)):
        for t, words in enumerate(tp):
            print(f"    {name} 主題 {t}: {' | '.join(words)}")

    # (5) 風格特徵
    F, dropped = style_features(bio)
    scaler = StandardScaler().fit(F)
    Fz = scaler.transform(F).astype(np.float32)
    S["5"] = textrec.cosine_sim(Fz[idx])
    methods["5"] = evaluate_method(S["5"], S_tag)
    C_style = plot_style_corr(F, FIG_DIR / f"{MODULE}_style_corr.png")
    style_summary = {c: {"mean": float(F[c].mean()), "std": float(F[c].std()),
                         "nonzero_rate": float((F[c] != 0).mean())} for c in F.columns}
    extra["5"] = {"features": list(F.columns), "dropped_constant_features": dropped, "feature_summary": style_summary,
                  "max_abs_corr_between_features": float(np.abs(C_style - np.eye(len(F.columns))).max())}
    print(f"(5) 風格：{len(F.columns)} 個特徵（丟掉常數特徵 {dropped}），與標籤相似度 Spearman "
          f"{methods['5']['spearman_vs_tag']:.4f}，recall@50 {methods['5']['recall_at_50']:.3f}（lift {methods['5']['lift_50']:.2f}）")

    # 方法之間的 Spearman：整段 TF-IDF / 主題到底比較像「標籤」還是比較像「風格」
    cross_pairs = [("1", "5"), ("4a", "5"), ("4b", "5"), ("1", "4a"), ("2a", "3z"), ("1", "3z")]
    cross = {f"{a}_vs_{b}": textrec.pair_rank_correlation(S[a], S[b], seed=SEED)["spearman"] for a, b in cross_pairs}
    print("方法之間的 Spearman：" + "、".join(f"{METHOD_SHORT[a]}↔{METHOD_SHORT[b]} {cross[f'{a}_vs_{b}']:.3f}"
                                         for a, b in cross_pairs))

    # (6) 混合
    blend = {}
    for key in BLEND_METHODS:
        rows = []
        for w in BLEND_WEIGHTS:
            Sb = textrec.blend(S[key], S_tag, w)
            r = textrec.neighbor_recovery(Sb, S_tag, k=K_MAIN)
            rows.append({"w_text": w, "recall_at_50": r["recall_at_k"], "lift_50": r["lift_over_random"],
                         "gini_at_10": textrec.exposure_gini(Sb, k=K_SMALL),
                         "mean_tag_sim_top10": mean_tag_sim_topk(Sb, S_tag, K_SMALL)})
        blend[key] = rows

    # (7) 冷啟動 / 半冷啟動模擬：seeker 端用 bio 推出的標籤向量，候選端用完整的真實標籤（同一組 IDF×類別權重）
    cand = Xw[idx]
    keep = np.random.default_rng(SEED + 1).random(X.shape) < MASK_KEEP
    X_obs = X & keep
    seeker_vectors = {
        "cold_M": M.astype(np.float32),
        "cold_P": P_c,
        "half_obs": X_obs.astype(np.float32),
        "half_obs_or_M": (X_obs | M).astype(np.float32),
        "half_obs_plus_P": X_obs.astype(np.float32) + P_c,
        "half_obs_or_Pbin": (X_obs | P_bin).astype(np.float32),
    }
    cold = {}
    for key, A in seeker_vectors.items():
        Sx = textrec.cosine_sim(A[idx] * w_tag, cand)      # 列 = seeker（不完整）、欄 = 候選（完整標籤）
        cold[key] = {"all": evaluate_method(Sx, S_tag)}
        if key == "cold_M":
            cold[key]["seekers_with_keyword"] = evaluate_method(Sx, S_tag, seekers=has_kw)
    if P_in_c is not None:   # 洩漏示範 + 交叉驗證：bundle 的 held-out seeker 應該跟 OOF 的數字差不多
        Sx = textrec.cosine_sim(P_in_c[idx] * w_tag, cand)
        cold["cold_P"]["bundle_in_sample_all_seekers"] = evaluate_method(Sx, S_tag)
        cold["cold_P"]["bundle_held_out_seekers"] = evaluate_method(Sx, S_tag, seekers=held_pos)
    cold_info = {"mask_keep": MASK_KEEP, "mean_tags_observed": float(X_obs.sum(1).mean()),
                 "mean_hidden_tags_restored_by_M": float((M & X & ~X_obs).sum(1).mean()),
                 "mean_wrong_tags_added_by_M": float((M & ~X).sum(1).mean()),
                 "mean_hidden_tags_restored_by_Pbin": float((P_bin & X & ~X_obs).sum(1).mean()),
                 "mean_wrong_tags_added_by_Pbin": float((P_bin & ~X).sum(1).mean())}
    print(f"(7) 半冷啟動：每人只看得到 {cold_info['mean_tags_observed']:.1f} 個標籤；M 平均補回 "
          f"{cold_info['mean_hidden_tags_restored_by_M']:.2f} 個被遮掉的真標籤（錯加 {cold_info['mean_wrong_tags_added_by_M']:.2f}），"
          f"P>{PROBA_THRESHOLD} 補回 {cold_info['mean_hidden_tags_restored_by_Pbin']:.2f} 個（錯加 {cold_info['mean_wrong_tags_added_by_Pbin']:.2f}）")

    base50 = methods["random"]["random_baseline_50"]
    plot_blend(blend, base50, FIG_DIR / f"{MODULE}_blend.png")
    plot_recall(methods, order, FIG_DIR / f"{MODULE}_recall.png")
    plot_coldstart(cold, base50, FIG_DIR / f"{MODULE}_coldstart.png")

    # Demo：3 位使用者，各方法 top-3 的 bio 讓人眼判斷
    n_kw_sub = M[idx].sum(1)
    style_sub = (F["zhuyin_count"].to_numpy()[idx] > 0) | (F["emoji_ratio"].to_numpy()[idx] > 0)
    cand_a = np.where(n_kw_sub >= 2)[0]                                       # bio 提到 ≥2 個標籤名
    cand_b = np.where((n_kw_sub == 0) & style_sub)[0]                         # 沒提標籤、但有注音文或 emoji
    cand_c = np.where((bio.str.len().to_numpy()[idx] >= 80) & (n_kw_sub == 1))[0]  # 長 bio、只提 1 個標籤
    demo_pos = [int(rng.choice(c)) for c in (cand_a, cand_b, cand_c) if len(c)]
    demo = []
    for pos in demo_pos:
        user = int(idx[pos])
        entry = {"row": user, "bio": bio.iloc[user], "n_keyword_tags": int(n_kw_sub[pos]),
                 "tags": [c for c, v in zip(cols, X[user]) if v], "recs": {}}
        for key in DEMO_METHODS:
            recs = []
            for p in top_recs(S[key], pos):
                u = int(idx[p])
                recs.append({"row": u, "sim": float(S[key][pos, p]), "shared_tags": int((X[user] & X[u]).sum()),
                             "shared_tag_names": [c for c, a, b in zip(cols, X[user], X[u]) if a and b],
                             "bio": bio.iloc[u]})
            entry["recs"][key] = recs
        demo.append(entry)

    # 存向量與模型，供其他腳本重用
    sparse.save_npz(OUT_DIR / "bio_keyword_matrix.npz", sparse.csr_matrix(M.astype(np.uint8)))
    np.save(OUT_DIR / "bio_tag_proba_oof.npy", P)
    np.savez_compressed(OUT_DIR / "style_features.npz", features=F.to_numpy(dtype=np.float32),
                        names=np.array(list(F.columns)))
    joblib.dump(scaler, MODEL_DIR / f"{MODULE}_style_scaler.joblib")
    joblib.dump({"nmf": nmf, "tfidf_vectorizer": tfidf_vec, "lda": lda, "count_vectorizer": count_vec},
                MODEL_DIR / f"{MODULE}_topics.joblib")
    save_json({
        "source": "db/03_seed_traits.sql 的 label_zh + 去前綴 + 同義詞（同義詞參考 experiments/supervised_bio_to_tags.py 的 TRAIT_KEYWORDS）",
        "matching": "bio.lower() 純子字串比對，無斷詞器；precision = P(使用者真的有該標籤 | bio 含此關鍵字)，用全 10,000 人算",
        "column_order": cols,
        "traits": kw_stats,
        "keyword_to_trait": {kw: c for c in cols for kw in kw_map[c]["keywords"]},
        "traits_never_mentioned_in_bio": list(missing),      # 關鍵字掃不到、也找不到高命中率 n-gram 可補的標籤（bio 沒寫）
        "noisy_keywords": noisy_rows,                          # 命中率 < NOISY_PRECISION 的關鍵字與最常見的誤判句
    }, OUT_DIR / "keyword_map.json")

    # ------------------------------------------------------------ 一句話解讀與結論（全部用數字組出來）
    m = methods
    a2 = extra["2a"]
    text_only = ("1", "2a", "3", "3z", "4a", "4b", "5")          # 不含洩漏示範 (3z*)
    best_text = max(text_only, key=lambda k: m[k]["lift_50"])
    leak = P_in is not None

    hi_thr = extra["3b"]["threshold_sweep"][-1]                  # 門檻掃描裡最高的那一個
    hi_verdict = ("高門檻的 P 退化成（比較貴的）關鍵字比對，同樣不是新資訊" if hi_thr["share_also_found_by_keywords"] >= 0.8
                  else "高門檻的 P 找到一些關鍵字掃不到的標籤，但在這份資料上它們本來就勾了，同樣不是新資訊"
                  if hi_thr["overlap_rate"] >= 0.9 else "高門檻的 P 仍有不少錯加的標籤")

    def strength(r) -> str:
        """依 z 值與 lift 給一句不誇大的判定。"""
        z = r["z_vs_random_50"] or 0.0
        if z < 2:
            return "與隨機無法區分"
        if r["lift_50"] < 1.5:
            return "統計上顯著但實務上沒用"
        return "訊號真實但弱" if r["lift_50"] < 5 else "訊號明顯"

    def topic_txt(key, share):
        kind = "主題是模板句／分隔符風格的群組，不是標籤" if share < 0.3 else "部分主題對到標籤關鍵字"
        return (f"lift {m[key]['lift_50']:.2f}（≈ 隨機）；前 8 n-gram 的 2-gram 只有 {share:.0%} 與標籤關鍵字重疊，{kind}；"
                f"與風格相似度 Spearman {cross[f'{key}_vs_5']:.2f}")

    interp = {
        "tag": f"正解本身（recall = 1）；Gini {m['tag']['gini_at_10']:.2f} 與前 10 名平均標籤相似度 {m['tag']['mean_tag_sim_top10']:.2f} 是上限參考",
        "random": (f"隨機基準：lift {m['random']['lift_50']:.2f}、Gini {m['random']['gini_at_10']:.2f}、"
                   f"前 10 名平均標籤相似度 {m['random']['mean_tag_sim_top10']:.2f} 是下限參考"),
        "1": (f"天真做法：lift {m['1']['lift_50']:.2f}（z={m['1']['z_vs_random_50']:.0f}，{strength(m['1'])}）；"
              f"它與風格相似度的 Spearman {cross['1_vs_5']:.2f} 遠高於與標籤的 {m['1']['spearman_vs_tag']:.3f}——量到的主要是填充文字"),
        "2a": (f"{strength(m['2a'])}（lift {m['2a']['lift_50']:.2f}）：只有 {a2['keyword_stats']['rate_with_any']:.0%} 的人 bio 提到標籤名（有提到者 recall "
               f"{a2['seekers_with_keyword']['recall_at_50']:.3f}、沒提到者 {a2['seekers_without_keyword']['recall_at_50']:.3f}）；"
               f"全 0 向量同分 → 固定名單（Gini {m['2a']['gini_at_10']:.2f}；隨機打破同分後 {a2['gini_at_10_random_tiebreak']:.2f}，"
               f"剩下的集中是因為 bio 沒提標籤名的候選幾乎排不進別人的前 10 名），上線一定要有 fallback"),
        "2b": (f"recall {m['2b']['recall_at_50']:.2f} 只是因為 X∨M ≈ X：bio 提到的標籤 {aug_kw['overlap_rate']:.0%} 本來就勾了，"
               f"平均只新增 {aug_kw['mean_new_tags']:.2f} 個（其中 {new_from_noisy_share:.0%} 是歧義關鍵字的誤判）"
               "——在這份資料上對已勾標籤的人不是新資訊"),
        "3": (f"{strength(m['3'])}（lift {m['3']['lift_50']:.2f}、z={m['3']['z_vs_random_50']:.1f}、Spearman {m['3']['spearman_vs_tag']:.2f}）；機率向量 "
              f"{extra['3']['common_component_share']:.0%} 是人人共有的成分，造成 hub（Gini {m['3']['gini_at_10']:.2f}）"),
        "3z": (f"去掉共同成分後 lift {m['3']['lift_50']:.2f}→{m['3z']['lift_50']:.2f}、Gini {m['3']['gini_at_10']:.2f}→{m['3z']['gini_at_10']:.2f}："
               f"要用 P 就先標準化；bio↔bio 的判定：{strength(m['3z'])}（z={m['3z']['z_vs_random_50']:.1f}）"),
        "3b": (f"P>{PROBA_THRESHOLD}（class_weight=balanced）平均錯加 {aug_p['mean_new_tags']:.1f} 個沒勾的標籤"
               f"（只有 {aug_p['overlap_rate']:.0%} 本來就勾了），recall 從 (2b) 的 {m['2b']['recall_at_50']:.2f} 掉到 {m['3b']['recall_at_50']:.2f}：雜訊蓋過補值；"
               f"門檻拉到 {hi_thr['threshold']} 時每人只剩 {hi_thr['mean_tags_predicted']:.2f} 個、{hi_thr['share_also_found_by_keywords']:.0%} 關鍵字也掃得到"
               f"（recall {hi_thr['recall_at_50']:.2f}）——{hi_verdict}"),
        "4a": topic_txt("4a", share_nmf),
        "4b": topic_txt("4b", share_lda),
        "5": (f"與標籤幾乎獨立（Spearman {m['5']['spearman_vs_tag']:.3f}、lift {m['5']['lift_50']:.2f}）：它是獨立維度，"
              "但「風格像＝比較合」在沒有互動資料下無法驗證"),
    }
    if leak:
        e = extra["3z_in"]
        interp["3z_in"] = (f"不是效果、是洩漏：同樣的 LR 直接對自己訓練過的人推論（AUC {e['auc_on_its_train_users']:.2f} vs 沒看過的人 "
                           f"{e['auc_on_its_test_users']:.2f}），lift 被灌到 {m['3z_in']['lift_50']:.2f}（OOF 才 {m['3z']['lift_50']:.2f}）；"
                           f"只看沒被訓練過的 seeker 是 {e['held_out_seekers']['lift_50']:.2f}（同一批 seeker 用 OOF 是 "
                           f"{e['held_out_seekers_oof']['lift_50']:.2f}，差距來自候選端仍有 80% 是被記住的人）；seeker 與候選都限定沒被訓練過的人時 "
                           f"{e['held_out_seekers_and_candidates']['lift_over_random']:.2f} vs OOF "
                           f"{e['held_out_seekers_and_candidates_oof']['lift_over_random']:.2f}")

    def blend_txt(key):
        rows = {r["w_text"]: r for r in blend[key]}
        return (f"{METHOD_SHORT[key]} recall@50 {rows[0.0]['recall_at_50']:.2f}→{rows[0.25]['recall_at_50']:.2f}→{rows[0.5]['recall_at_50']:.2f}"
                f"→{rows[1.0]['recall_at_50']:.2f}、Gini@10 {rows[0.0]['gini_at_10']:.2f}→{rows[0.25]['gini_at_10']:.2f}"
                f"→{rows[0.5]['gini_at_10']:.2f}→{rows[1.0]['gini_at_10']:.2f}")

    c_all = {k: v["all"] for k, v in cold.items()}
    se_half = c_all["half_obs"]["recall_se_50"]
    best_cold = max(("cold_M", "cold_P"), key=lambda k: c_all[k]["lift_50"])
    best_half = max(("half_obs_or_M", "half_obs_plus_P", "half_obs_or_Pbin"), key=lambda k: c_all[k]["recall_at_50"])
    half_gain = c_all[best_half]["recall_at_50"] - c_all["half_obs"]["recall_at_50"]
    half_advice = (f"標籤不完整時用「{COLD_NAME[best_half]}」補（recall@50 {half_gain:+.3f}）" if half_gain > 2 * se_half
                   else "標籤不完整時 bio 補不出可量測的提升，直接用已勾的標籤")

    def delta_txt(key):
        d = c_all[key]["recall_at_50"] - c_all["half_obs"]["recall_at_50"]
        size = "在抽樣誤差內" if abs(d) < 2 * se_half else ("有提升" if d > 0 else "變差")
        return f"{d:+.3f}（{size}）"

    noisy_txt = "".join(f"「{r['keyword']}」" for r in noisy_rows) or "（無）"
    leak_cold = ""
    if leak:
        lc = cold["cold_P"]
        leak_cold = (f"（洩漏對照：bundle in-sample 會顯示 lift {lc['bundle_in_sample_all_seekers']['lift_50']:.2f}，"
                     f"但只看它沒訓練過的 seeker 是 {lc['bundle_held_out_seekers']['lift_50']:.2f}，與 OOF 一致）")

    conclusions = [
        f"整段 bio 直接 TF-IDF cosine 沒用：recall@50 {m['1']['recall_at_50']:.3f}（隨機 {base50:.3f}，lift {m['1']['lift_50']:.2f}），"
        f"與標籤相似度 Spearman {m['1']['spearman_vs_tag']:.3f}，但與風格相似度 Spearman {cross['1_vs_5']:.2f}：它量到的是填充文字與寫法，不是興趣。",
        f"把標籤訊號從 bio 分離出來（bio↔bio）：關鍵字 (2a) lift {m['2a']['lift_50']:.2f}、P(tag|bio) (3) {m['3']['lift_50']:.2f}、"
        f"標準化後 (3z) {m['3z']['lift_50']:.2f}。純文字方法裡最好的是 {METHOD_SHORT[best_text]}（recall@50 {m[best_text]['recall_at_50']:.3f}）。"
        f"判定：{strength(m[best_text])}。原因是 bio 平均只寫到每人 {aug_kw['mean_tags_original']:.1f} 個標籤裡的 {aug_kw['mean_tags_from_bio']:.2f} 個，兩邊都只靠 bio 幾乎對不上；"
        f"而且生成器只寫 interest / personality / value（名稱有出現的標籤數：{mentioned_cats}）。",
        (f"洩漏警告：上一輪存的模型直接對全部人推論時，(3z*) lift {m['3z_in']['lift_50']:.2f}，是 OOF {m['3z']['lift_50']:.2f} 的 "
         f"{m['3z_in']['lift_50'] / m['3z']['lift_50']:.1f} 倍——那是模型記住了訓練過的 80% 使用者的標籤，不是 bio 的資訊量。"
         "本腳本所有 P 系列的主數字都用 out-of-fold 機率。" if leak else
         "bundle 無法載入，所以沒有洩漏示範；P 系列的數字本來就用 out-of-fold 機率，不受影響。"),
        f"用 bio 補標籤 (2b) 的 recall {m['2b']['recall_at_50']:.2f} 不代表 bio 有用，只代表 X∨M ≈ X（{aug_kw['overlap_rate']:.0%} 的提及本來就勾了，平均只新增 "
        f"{aug_kw['mean_new_tags']:.2f} 個，而且其中 {new_from_noisy_share:.0%} 是{noisy_txt}這類歧義關鍵字的誤判——"
        f"(2b) 與標籤排序的那一點差距來自誤判，不是 bio 帶來的新標籤）。(3b) 用 P>{PROBA_THRESHOLD} 會錯加 {aug_p['mean_new_tags']:.1f} 個標籤，"
        f"recall 掉到 {m['3b']['recall_at_50']:.2f}——balanced 權重訓練的機率不能直接用 0.5 當門檻；門檻拉到 {hi_thr['threshold']} 時每人只剩 "
        f"{hi_thr['mean_tags_predicted']:.2f} 個、{hi_thr['overlap_rate']:.0%} 本來就勾了、{hi_thr['share_also_found_by_keywords']:.0%} 關鍵字也掃得到"
        f"（recall {hi_thr['recall_at_50']:.2f}）：{hi_verdict}。",
        f"(7) 冷啟動（seeker 只有 bio、候選用真實標籤）：M→標籤 lift {c_all['cold_M']['lift_50']:.2f}"
        f"（bio 有提到標籤名的 seeker {cold['cold_M']['seekers_with_keyword']['lift_50']:.2f}）、P→標籤 lift {c_all['cold_P']['lift_50']:.2f}{leak_cold}；"
        f"較好的是「{COLD_NAME[best_cold]}」，前 10 名平均標籤相似度 {c_all[best_cold]['mean_tag_sim_top10']:.3f}"
        f"（隨機 {m['random']['mean_tag_sim_top10']:.3f}、標籤本身 {m['tag']['mean_tag_sim_top10']:.3f}）。比隨機好、但離標籤還很遠；"
        f"而且比 bio↔bio 的 {METHOD_SHORT[best_text]}（lift {m[best_text]['lift_50']:.2f}）好，因為候選端用的是完整的真實標籤。",
        f"(7) 半冷啟動（每人隨機遮掉一半標籤，只剩 {cold_info['mean_tags_observed']:.1f} 個）：只用剩下的標籤 recall@50 {c_all['half_obs']['recall_at_50']:.3f}；"
        f"∨M {delta_txt('half_obs_or_M')}、+(P−平均) {delta_txt('half_obs_plus_P')}、∨(P>{PROBA_THRESHOLD}) {delta_txt('half_obs_or_Pbin')}。"
        f"M 平均只能補回 {cold_info['mean_hidden_tags_restored_by_M']:.2f} 個被遮掉的真標籤（錯加 {cold_info['mean_wrong_tags_added_by_M']:.2f} 個）；"
        f"這是 bio 在這份資料上唯一量得到的實際幫助，而且幅度遠小於「多勾幾個標籤」。",
        f"主題模型抓不到標籤：NMF / LDA lift {m['4a']['lift_50']:.2f} / {m['4b']['lift_50']:.2f}。主題可以解讀，但解讀出來的是模板句"
        f"（" + "、".join(f"「{t}」出現在 {c} 則 bio" for t, c in template_counts.items()) + "）與分隔符寫法（「...」「 / 」「空格」），"
        f"跟風格相似度的 Spearman {cross['4a_vs_5']:.2f} / {cross['4b_vs_5']:.2f} 也比跟標籤的高。",
        f"風格特徵是標籤沒有的獨立維度：與標籤相似度 Spearman {m['5']['spearman_vs_tag']:.3f}、lift {m['5']['lift_50']:.2f}。"
        "獨立不等於有用——「風格像的人比較合」需要真實互動資料才能驗證，這裡只能說它是一個可以 A/B 測的候選訊號。",
        "混合（w = 文字權重 0→0.25→0.5→1）：" + "；".join(blend_txt(k) for k in BLEND_METHODS) + "。"
        "注意：對 S_tag 的 recall 量的是「跟標籤排序有多像」，任何非標籤訊號混進去都只會讓它下降，所以這條曲線只能告訴你"
        "「加多少文字權重會偏離標籤排序多少」，不能告訴你推薦變好或變差。",
        "具體建議（依證據強度）：(a) 已勾滿標籤的使用者：主排序維持標籤，bio 的標籤訊號沒有新資訊；"
        f"(b) 沒勾或只勾一點標籤的使用者：用「{COLD_NAME[best_cold]}」去比對別人的真實標籤當過渡（lift {c_all[best_cold]['lift_50']:.2f}），{half_advice}；"
        "同時把 bio 抽到的關鍵字做成「一鍵加入標籤」的 UI 建議（每個關鍵字的命中率在 keyword_map.json，低於 0.9 的不要自動加）；"
        "(c) 風格相似度只放低權重（由混合曲線挑一個你能接受的偏離量）或當同分時的 tie-breaker，並且一定要用 A/B 測驗證；"
        "(d) 不要直接用整段 TF-IDF 或主題模型；(e) 想抓「標籤以外的語意」要靠句子嵌入或 LLM 抽取（另外兩支腳本）。",
        "限制：P 系列是本腳本 5-fold OOF 重訓的 LR，不是 bundle 那一個模型本身；關鍵字是純子字串比對，同義詞與字典缺口檢查是看著這份資料的標籤整理的"
        "（整理字典、不是評估，但換資料要重掃）；關鍵字的「誤判」是以這份合成資料的標籤為準——生成器的填充句（例如"
        + "、".join(f"「{r['most_common_false_positive_segment']}」" for r in noisy_rows[:2]) +
        "）與標籤無關，但真人寫出這種句子很可能真的代表該特質，所以命中率要在真人資料上重算；半冷啟動的遮蔽是均勻隨機，"
        "真人漏勾的標籤不一定隨機；bio 是生成器看著標籤寫的，真人 bio 很可能提到沒勾的興趣，那時 (2b) 的價值會比這裡量到的大，需要重驗。",
    ]

    table = []
    for k in order:
        r = m[k]
        table.append({"方法": METHOD_NAME[k], "recall@50": r["recall_at_50"], "lift": r["lift_50"],
                      "recall@10": r["recall_at_10"], "Spearman vs 標籤": r["spearman_vs_tag"],
                      "Gini@10": r["gini_at_10"], "前10名平均標籤相似度": r["mean_tag_sim_top10"], "解讀": interp[k]})

    runtime = round(time.perf_counter() - t0, 1)
    metrics = {
        "module": MODULE,
        "dataset": {"n_users": n, "n_labels": len(cols), "subsample": int(len(idx)), "seed": SEED,
                    "bio_median_len": int(bio.str.len().median()), "n_tfidf_features": int(Xt.shape[1]),
                    "n_held_out_seekers_in_subsample": int(len(held_pos)) if held_pos is not None else None},
        "settings": {"S_tag": "cosine(features.weighted_matrix(X, cols)[idx])，IDF × 類別權重",
                     "subsample": f"features.subsample({n}, {N_SUB}, seed={SEED})",
                     "recall": f"textrec.neighbor_recovery k={K_MAIN} 與 {K_SMALL}；random baseline = k/(n-1)；se = std/sqrt(n_seekers)",
                     "spearman": "textrec.pair_rank_correlation n_pairs=200000",
                     "gini": f"textrec.exposure_gini k={K_SMALL}",
                     "mean_tag_sim_top10": "本腳本：方法的前 10 名在 S_tag 上的平均值",
                     "tfidf": "features.bio_tfidf char 1-2 gram, min_df=3, sublinear_tf, fit on all 10,000",
                     "keyword": "label_zh + 去前綴 + 同義詞，小寫子字串比對",
                     "proba_model": f"{N_FOLDS}-fold out-of-fold OvR(LogisticRegression C=4, class_weight=balanced)，TF-IDF 每折重新 fit",
                     "proba_threshold": PROBA_THRESHOLD, "bundle": model_info["source"],
                     "leak_demo": "bundle 直接對全部人推論 = (3z*)；train_test_split(test_size=0.2, random_state=SEED) 還原上一輪切分取 held-out seeker",
                     "nmf": f"NMF(n_components={N_TOPICS}, init=nndsvda, max_iter=300) on TF-IDF",
                     "lda": f"LatentDirichletAllocation({N_TOPICS}, online, max_iter=10, batch_size=500) on CountVectorizer char 1-2 gram",
                     "style": f"{len(F.columns)} 個字元層級特徵 → StandardScaler → cosine",
                     "blend": f"textrec.blend(S_text, S_tag, w) w∈{list(BLEND_WEIGHTS)}",
                     "cold_start": f"seeker 向量 × 標籤權重 vs 候選 Xw（完整標籤）；半冷啟動每個已勾標籤以 {MASK_KEEP} 機率保留（seed={SEED + 1}）"},
        "methods": {k: m[k] for k in order},
        "extra": extra,
        "cross_method_spearman": cross,
        "blend": blend,
        "cold_start_simulation": {"info": cold_info, "names": COLD_NAME, "results": cold},
        "demo": demo,
        "summary_table": table,
        "conclusions": conclusions,
        "runtime_seconds": runtime,
    }
    save_json(metrics, OUT_DIR / "metrics.json")

    # ------------------------------------------------------------ 摘要表
    print(f"\n### 各方法（同一份 {len(idx)} 人抽樣、同一個 S_tag；隨機基準 recall@50 = {base50:.3f}）")
    print("| 方法 | recall@50 | lift | recall@10 | Spearman vs 標籤 | Gini@10 | 前10名平均標籤相似度 | 一句話解讀 |")
    print("|---|---|---|---|---|---|---|---|")
    for row in table:
        print(f"| {row['方法']} | {row['recall@50']:.3f} | {row['lift']:.2f} | {row['recall@10']:.3f} | "
              f"{row['Spearman vs 標籤']:.3f} | {row['Gini@10']:.3f} | {row['前10名平均標籤相似度']:.3f} | {row['解讀']} |")

    print("\n### (6) 混合（w = 文字相似度的權重；w=0 全標籤、w=1 全文字）")
    print("| 方法 | w | recall@50 | lift | Gini@10 | 前10名平均標籤相似度 |")
    print("|---|---|---|---|---|---|")
    for key, rows in blend.items():
        for r in rows:
            print(f"| {METHOD_SHORT[key]} | {r['w_text']:.2f} | {r['recall_at_50']:.3f} | {r['lift_50']:.2f} | "
                  f"{r['gini_at_10']:.3f} | {r['mean_tag_sim_top10']:.3f} |")

    print("\n### (7) 冷啟動／半冷啟動模擬（seeker 端缺標籤，候選端用完整真實標籤）")
    print("| 情境 | recall@50 | ±SE | lift | recall@10 | 前10名平均標籤相似度 |")
    print("|---|---|---|---|---|---|")
    for key, name in COLD_NAME.items():
        r = cold[key]["all"]
        print(f"| {name} | {r['recall_at_50']:.3f} | {r['recall_se_50']:.3f} | {r['lift_50']:.2f} | {r['recall_at_10']:.3f} | "
              f"{r['mean_tag_sim_top10']:.3f} |")
    if leak:
        lc = cold["cold_P"]
        print(f"洩漏對照（冷啟動 P→標籤）：bundle in-sample lift {lc['bundle_in_sample_all_seekers']['lift_50']:.2f} ／ "
              f"bundle 只看沒訓練過的 seeker {lc['bundle_held_out_seekers']['lift_50']:.2f} ／ OOF {lc['all']['lift_50']:.2f}")

    print("\n### Demo：3 位使用者與各方法 top-3（「共同標籤」= 與 seeker 共有的原標籤數）")
    for e in demo:
        print(f"\n**seeker row {e['row']}**（bio 提到 {e['n_keyword_tags']} 個標籤名；標籤 {len(e['tags'])} 個）：{short(e['bio'], 100)}")
        for key in DEMO_METHODS:
            print(f"- {METHOD_NAME[key]}")
            for r in e["recs"][key]:
                print(f"    - row {r['row']}（sim {r['sim']:.2f}，共同標籤 {r['shared_tags']}：{'、'.join(r['shared_tag_names'][:4])}）："
                      f"{short(r['bio'])}")

    print("\n### 結論")
    for c in conclusions:
        print(f"- {c}")
    figs = "、".join(f"{MODULE}_{s}.png" for s in ("recall", "blend", "coldstart", "style_corr"))
    print(f"\n總耗時 {runtime}s；輸出 {OUT_DIR}/metrics.json、keyword_map.json；圖 {FIG_DIR}/{{{figs}}}")


if __name__ == "__main__":
    main()
