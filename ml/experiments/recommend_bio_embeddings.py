"""句子嵌入（sentence-transformers）做 bio 相似度推薦，並匯出成 pgvector 可直接載入的格式。

【這個方法是什麼】
把每位使用者的中文 bio 丟進一個「句子嵌入模型」，得到一個固定維度（例如 384 維）的向量；
兩個人的 bio 向量餘弦相似度越高，就越可能「說的是同一類事情」。推薦時拿 seeker 的向量去找
向量最接近的前 k 位候選（k-NN），再跟標籤相似度混合（blend）。
跟 recommend_bio_text.py 的字元 n-gram TF-IDF 不同：TF-IDF 只認「字面上有沒有同樣的字」，
「羽球」跟「打球」、「爬山」跟「登山」在 TF-IDF 眼裡幾乎無關；句子嵌入模型在大量語料上學過，
理論上能把同義／相近的說法放在一起，也比較不受「www」「嗯嗯」「...」這類填充文字影響。

【為什麼在這份資料上要特別小心】
主控 agent 已實測：bio 有 93% 是標籤以外的填充文字（口頭禪、梗、emoji、注音文），整段 TF-IDF cosine
找到的鄰居跟標籤鄰居幾乎無關（主控抽樣 1500 人：recall@50 = 0.040，隨機基準 0.033，lift 1.2；Spearman 0.014。
本腳本抽樣 3000 人，隨機基準 = 50/2999 = 0.017，所以 recall 的絕對值不能跟那組數字直接比，要比 lift）。
句子嵌入是否真的能把「訊號」從填充裡分離出來，是一個實證問題，**本輪無法回答**，因為規則禁止下載模型。
所以這支腳本的定位是：把「嵌入 → 評估 → 匯入 pgvector」整條管線做好、用假嵌入跑通、留好插槽；
等可以下載模型時，只要加 --model 參數，同一張表就會多一列可以直接比較。

【dry-run（預設）是什麼、不是什麼】
沒有給 --model 時，用 features.bio_tfidf（字元 1–2 gram）→ TruncatedSVD(384) → L2 normalize 當「假嵌入」。
它**不是語意嵌入**：只是把 TF-IDF 稀疏向量線性壓成 384 維（等同 LSA），看到的仍然是字面重疊；
它唯一的用途是驗證 pipeline（維度、快取、評估、匯出 SQL）在真模型接上前就能跑，並提供一個「LSA 基準」。
預期它的 recall@50 會跟整段 TF-IDF 一樣接近隨機——如果真的如此，正好證實「線性壓縮救不了字面比對」。

【怎麼跑】
cd ml && .venv/bin/python experiments/recommend_bio_embeddings.py                   # dry-run，零網路，約 10 秒
cd ml && .venv/bin/python experiments/recommend_bio_embeddings.py --model intfloat/multilingual-e5-small --device mps
    第一次會從 Hugging Face 下載權重（見下方候選模型的大小），之後向量快取在
    outputs/recommend_bio_embeddings/<model_slug>.npy（旁邊有同名 .json 記錄維度與編碼秒數），下次直接讀。
    加 --local-files-only 可以保證絕不下載（模型不在本機快取就直接報錯）；也可以設環境變數 HF_HUB_OFFLINE=1。
參數：--model（選填，不給 = dry-run）、--device auto|mps|cpu、--k 50、--batch-size 64、
      --prefix auto|<字串>（e5 系列模型要求輸入加前綴，auto 會替名稱含 "e5" 的模型自動加 "query: "，其他模型不加）。

【候選模型】（規格皆來自各模型卡與 Hugging Face API 的檔案清單，2026-09 查）
| 模型 | 維度 | 最長輸入 | 語言 | 授權 | 首次執行下載量 |
|---|---|---|---|---|---|
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384 | 128 token | 50 種 | Apache-2.0 | model.safetensors 470.6 MB（117.65M 參數）+ tokenizer 約 15 MB，合計約 490 MB |
| intfloat/multilingual-e5-small | 384（12 層） | 512 token | 100 種（含 zh；MTEB 有 zh-TW 評測） | MIT | model.safetensors 470.6 MB + tokenizer 約 22 MB，合計約 495 MB；輸入要加 "query: " 前綴 |
| BAAI/bge-m3 | 1024 | 8192 token | 100+ 種 | MIT | pytorch_model.bin 2,271 MB（約 2.3 GB）+ tokenizer 17 MB |
bio 中位數 39 字、最長 100 字：512 / 8192 token 的兩個模型一定夠；MiniLM 的 128 token 對中位數綽綽有餘，
但最長的 bio（含 emoji／注音文，一個字可能拆成多個 token）有可能被截斷——本輪沒下載 tokenizer，未驗證。
優先建議 multilingual-e5-small（小、支援中文、
初始化自 microsoft/Multilingual-MiniLM-L12-H384）；bge-m3 品質通常更好但 2.3 GB、1024 維，
10,000 則短 bio 在 MPS 上編碼仍可接受，pgvector 每列儲存從 4×384+8 變成 4×1024+8 bytes。
e5 模型卡原文：每段輸入都要以 "query: " 或 "passage: " 開頭（非英文也一樣）；非檢索任務一律用 "query: "。
sentence-transformers 6.1.0 的簽章（已用 inspect.signature 對本機安裝版本確認）：
  SentenceTransformer(model_name_or_path, device=None, cache_folder=None, local_files_only=False, ...)
  encode(inputs, prompt_name=None, prompt=None, batch_size=32, show_progress_bar=None, output_value='sentence_embedding',
         precision='float32', convert_to_numpy=True, convert_to_tensor=False, device=None, normalize_embeddings=False,
         truncate_dim=None, pool=None, chunk_size=None, **kwargs) -> np.ndarray（convert_to_numpy=True 時）

【評估：跟 recommend_bio_text.py 用同一把尺】
同一份 features.subsample(10000, 3000, seed=42) 抽樣、同一個 S_tag = cosine(IDF×類別權重的標籤矩陣)：
  - neighbor_recovery(k=50, 10)：文字前 k 位有多少也在標籤前 k 位（冷啟動時能不能靠文字找到標籤鄰居）。
  - pair_rank_correlation：隨機 200,000 對的 Spearman；接近 0 = 文字跟標籤幾乎是獨立的維度。
  - exposure_gini(k=10)：推薦曝光集中度，越高越集中在少數人。
  - blend 曲線：w_text ∈ {0, .25, .5, .75, 1}，w_text·z(S_text) + (1-w_text)·z(S_tag)，看混入文字後
    標籤鄰居掉多少、曝光集中度怎麼變。
  - 對照：把嵌入的列順序隨機打散（使用者與向量脫鉤）的 null；以及 S_tag 對自己的 recall（應為 1.0）。
  - 另附 recall 的標準誤與 z 值（se = std/√n_seekers，與 recommend_bio_text.py 同公式；seeker 共用候選、並非獨立，
    z 只是粗估）以及「前 10 名平均標籤相似度」（textrec 沒有，本腳本自己算，公式同 recommend_bio_text.py）。
判讀口訣：recall 量的是「跟標籤鄰居的重疊」，所以 recall 高於隨機的那一部分**一定是從 bio 抽回來的標籤訊號，
不是新資訊**（對已經勾標籤的人沒有增量，只對冷啟動有用）；Spearman 也高 = 連整體排序都跟標籤重疊；
recall 低 + Spearman ≈ 0 = 文字是獨立維度（風格／用語／填充），但它是否有助配對，在沒有互動資料下**無法驗證**。

【pgvector 匯出】
outputs/recommend_bio_embeddings/bio_embeddings.csv：account, '[v1,v2,...]'（pgvector 的 vector 文字格式）。
outputs/recommend_bio_embeddings/pgvector_import.sql：CREATE EXTENSION、ALTER TABLE profiles ADD COLUMN
  bio_embedding vector(<dim>)、用 staging 表 COPY CSV 再 UPDATE、HNSW 索引（vector_cosine_ops）、
  以及 ORDER BY <=> $1 LIMIT 50 的查詢。維度由實際向量決定。
  查詢用 PREPARE 包起來：$1 是給應用程式用的參數佔位符，在 psql 裡裸寫會報 "there is no parameter $1"，
  包成 PREPARE 後整個檔案才能用 psql -f 從頭跑到尾。SQL 依規定沒有連資料庫實跑過，語法以 README 為準。
pgvector README：預設是精確（暴力）最近鄰搜尋、recall 100%；加索引才是近似搜尋，用一點 recall 換速度；
「先載入資料再建索引」比較快。10,000 列 × 384 維的暴力掃描只有約 15 MB 浮點數，單次查詢毫秒級，
**先不用建 HNSW**。什麼規模才需要索引，pgvector 沒有給官方數字；實務上是「精確掃描的延遲超出你的
預算時」（通常是數十萬到百萬列以上）——這是經驗法則，不是官方門檻。

【怎麼讀結果】
outputs/recommend_bio_embeddings/metrics.json：mode（dry-run / model）、embedding（維度、encode 秒數）、
  methods.{tag, null_permuted, emb_*}（鍵名與 recommend_bio_text.py 相同：recall_at_50 / recall_se_50 / lift_50 /
  z_vs_random_50 / recall_at_10 / spearman_vs_tag / gini_at_10 / mean_tag_sim_top10，可直接併表）、blend、summary_table、reference_recommend_bio_text（若那支跑過，
  把它的摘要列讀進來並排）、示範鄰居、pgvector 匯出資訊、候選模型、結論。
outputs/figures/recommend_bio_embeddings_recall.png：recall@50 長條（文字 vs null vs 隨機基準；標籤自檢 = 1.0 寫在標題）。
outputs/figures/recommend_bio_embeddings_blend.png：blend 曲線（左：recall@50 與 Spearman；右：Gini@10）。
outputs/figures/recommend_bio_embeddings_simhist.png：文字 vs 標籤相似度的分布。

【來源】
- sentence-transformers 文件 https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
  （device / cache_folder / local_files_only 的說明），以及本機 6.1.0 的 inspect.signature 結果。
- 模型卡：https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2、
  https://huggingface.co/intfloat/multilingual-e5-small、https://huggingface.co/BAAI/bge-m3；
  檔案大小來自 https://huggingface.co/api/models/<id>?blobs=true 與 /tree/main。
- pgvector README https://github.com/pgvector/pgvector（vector(n)、'[1,2,3]' 文字格式、COPY、<=>、
  hnsw / vector_cosine_ops、m=16、ef_construction=64、hnsw.ef_search=40、精確搜尋為預設、每列 4×d+8 bytes）。
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from heartlink_ml import plotting  # noqa: F401  匯入即設定 Agg 後端與中文字型
from heartlink_ml.config import FIG_DIR, OUTPUT_DIR, SEED, ensure_dirs
from heartlink_ml.data import load_seed, trait_columns
from heartlink_ml.evaluation import save_json
from heartlink_ml.features import binary_matrix, bio_tfidf, subsample, weighted_matrix
from heartlink_ml.textrec import blend, cosine_sim, exposure_gini, neighbor_recovery, pair_rank_correlation

import matplotlib.pyplot as plt  # noqa: E402  要在 heartlink_ml.plotting 之後匯入

MODULE = "recommend_bio_embeddings"
OUT_DIR = OUTPUT_DIR / MODULE
EVAL_SIZE = 3000                       # n×n 相似度矩陣一律用 3000 抽樣（與其他文字推薦腳本一致）
DRYRUN_DIM = 384                       # 假嵌入維度：故意跟 MiniLM / e5-small 一樣，讓 SQL 的 vector(384) 直接可用
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_EXAMPLES = 3                         # 印幾組「seeker → 文字最近鄰 / 標籤最近鄰」示範
CSV_DECIMALS = 5                       # CSV 每個分量保留幾位小數（384 維 × 10,000 列約 30 MB）

# 只列實際查過模型卡與 HF API 檔案清單的模型（大小為權重檔本身；tokenizer 另加 15–22 MB）
CANDIDATE_MODELS = [
    {"name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "dim": 384, "max_seq_length": 128,
     "languages": "50", "license": "apache-2.0", "weights_mb": 470.6, "params": "117.65M", "prefix": ""},
    {"name": "intfloat/multilingual-e5-small", "dim": 384, "max_seq_length": 512,
     "languages": "100（含 zh）", "license": "mit", "weights_mb": 470.6, "params": "12 層 MiniLM", "prefix": "query: "},
    {"name": "BAAI/bge-m3", "dim": 1024, "max_seq_length": 8192,
     "languages": "100+", "license": "mit", "weights_mb": 2271.1, "params": "XLM-R large 系", "prefix": ""},
]


# ---------------------------------------------------------------- CLI
def parse_args():
    ap = argparse.ArgumentParser(description="句子嵌入做 bio 相似度推薦（預設 dry-run，不下載任何模型）")
    ap.add_argument("--model", default=None, help="Hugging Face 模型名稱；不給 = dry-run（TF-IDF→SVD 假嵌入）")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"], help="編碼裝置（只在 --model 時用到）")
    ap.add_argument("--k", type=int, default=50, help="neighbor_recovery 的 k")
    ap.add_argument("--batch-size", type=int, default=64, help="encode 的 batch_size")
    ap.add_argument("--prefix", default="auto", help="每則 bio 前面加的字串；auto = e5 系列加 'query: '，其他不加")
    ap.add_argument("--local-files-only", action="store_true", help="傳給 SentenceTransformer，保證不從網路下載")
    return ap.parse_args()


def resolve_device(name: str) -> str:
    """auto → 有 MPS 就用 MPS，否則 CPU。只在 --model 模式呼叫，dry-run 不 import torch。"""
    if name != "auto":
        return name
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def model_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", name)


# ---------------------------------------------------------------- 嵌入
def dryrun_embeddings(texts) -> tuple[np.ndarray, dict]:
    """假嵌入：字元 1–2 gram TF-IDF → TruncatedSVD(384) → L2 normalize。不是語意嵌入，只驗證 pipeline。"""
    t0 = time.time()
    X_tfidf, vec = bio_tfidf(texts)
    n_comp = min(DRYRUN_DIM, X_tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
    E = normalize(svd.fit_transform(X_tfidf)).astype(np.float32)
    sec = time.time() - t0
    info = {
        "mode": "dry-run", "label": f"dry-run 假嵌入（TF-IDF→SVD-{n_comp}→L2）", "model": None,
        "device": "n/a", "dim": int(E.shape[1]), "encode_seconds": round(sec, 2), "cached": False,
        "tfidf_features": int(X_tfidf.shape[1]),
        "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
        "note": "這不是語意嵌入：只是把 TF-IDF 線性壓縮成 384 維（LSA），用來驗證 pipeline 與當 LSA 基準。",
    }
    return E, info


def model_embeddings(texts, args) -> tuple[np.ndarray, dict]:
    """真嵌入：SentenceTransformer(model, device=...).encode(..., normalize_embeddings=True)，向量快取成 .npy。"""
    slug = model_slug(args.model)
    cache_npy, cache_json = OUT_DIR / f"{slug}.npy", OUT_DIR / f"{slug}.json"
    # e5 系列（名稱裡有獨立的 "e5" 片段，例如 multilingual-e5-small）才自動加前綴；避免 "base5" 這類名稱誤判
    is_e5 = re.search(r"(^|[-_/])e5([-_]|$)", args.model.lower()) is not None
    prefix = ("query: " if is_e5 else "") if args.prefix == "auto" else args.prefix
    if cache_npy.exists():
        E = np.load(cache_npy).astype(np.float32)
        meta = json.loads(cache_json.read_text(encoding="utf-8")) if cache_json.exists() else {}
        if "prefix" in meta and meta["prefix"] != prefix:
            print(f"警告：快取是用前綴 {meta['prefix']!r} 編碼的，這次指定的是 {prefix!r}；"
                  f"快取不會重算，要換前綴請先刪掉 {cache_npy}")
        if E.shape[0] != len(texts):
            raise SystemExit(f"快取 {cache_npy} 有 {E.shape[0]} 列，但資料有 {len(texts)} 列；請刪掉快取重跑")
        print(f"讀取快取 {cache_npy}（{E.shape}）")
        info = {"mode": "model", "label": args.model, "model": args.model, "device": meta.get("device", "cached"),
                "dim": int(E.shape[1]), "encode_seconds": meta.get("encode_seconds"), "cached": True,
                "prefix": meta.get("prefix", prefix), "max_seq_length": meta.get("max_seq_length")}
        return E, info

    from sentence_transformers import SentenceTransformer  # 只在需要時 import；dry-run 完全不碰

    device = resolve_device(args.device)
    print(f"載入模型 {args.model}（device={device}，local_files_only={args.local_files_only}）")
    model = SentenceTransformer(args.model, device=device, local_files_only=args.local_files_only)
    inputs = [prefix + t for t in texts]
    t0 = time.time()
    E = model.encode(inputs, batch_size=args.batch_size, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    sec = time.time() - t0
    info = {"mode": "model", "label": args.model, "model": args.model, "device": device, "dim": int(E.shape[1]),
            "encode_seconds": round(sec, 2), "cached": False, "prefix": prefix,
            "max_seq_length": getattr(model, "max_seq_length", None), "batch_size": args.batch_size}
    np.save(cache_npy, E)
    cache_json.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"編碼 {len(texts)} 則 bio 花 {sec:.1f} 秒，向量已快取到 {cache_npy}")
    return E, info


# ---------------------------------------------------------------- 評估
def mean_tag_sim_topk(S: np.ndarray, S_tag: np.ndarray, k: int) -> float:
    """S 的前 k 名（排除自己）在 S_tag 上的平均值：比 top-k 重疊寬鬆的連續版本。
    textrec 沒有這個函式；公式與 recommend_bio_text.py 的同名函式相同，所以數字可以直接比。"""
    R = S.astype(np.float32, copy=True)
    np.fill_diagonal(R, -np.inf)
    top = np.argpartition(-R, k, axis=1)[:, :k]
    return float(np.take_along_axis(S_tag, top, axis=1).mean())


def evaluate_against_tags(S_txt: np.ndarray, S_tag: np.ndarray, k: int) -> dict:
    """同一把尺：recall@k / @10、Spearman、Gini@10，外加標準誤、z 值與前 10 名平均標籤相似度。
    鍵名刻意跟 recommend_bio_text.py 的 methods.<key> 完全一樣，兩份 metrics.json 可以直接併表。
    se = std/√n_seekers；seeker 之間共用候選、並非獨立，所以 z 只是粗估，用來分辨「≈隨機」與「略高於隨機」。"""
    a = neighbor_recovery(S_txt, S_tag, k=k)
    b = neighbor_recovery(S_txt, S_tag, k=10)
    c = pair_rank_correlation(S_txt, S_tag, seed=SEED)
    se = a["recall_std"] / np.sqrt(max(a["n_seekers"], 1))
    return {
        f"recall_at_{k}": a["recall_at_k"], f"recall_std_{k}": a["recall_std"], f"recall_se_{k}": float(se),
        f"random_baseline_{k}": a["random_baseline"], f"lift_{k}": a["lift_over_random"],
        f"z_vs_random_{k}": float((a["recall_at_k"] - a["random_baseline"]) / se) if se > 0 else None,
        "recall_at_10": b["recall_at_k"], "random_baseline_10": b["random_baseline"], "lift_10": b["lift_over_random"],
        "spearman_vs_tag": c["spearman"], "spearman_p_value": c["p_value"],
        "gini_at_10": exposure_gini(S_txt, k=10), "mean_tag_sim_top10": mean_tag_sim_topk(S_txt, S_tag, 10),
        "n_seekers": a["n_seekers"],
    }


def blend_curve(S_txt: np.ndarray, S_tag: np.ndarray, k: int) -> list[dict]:
    """w_text 掃 5 個值：混合分數對標籤鄰居的保留率、曝光集中度、與標籤的 Spearman。
    w_text / recall_at_k / lift_k / gini_at_10 / mean_tag_sim_top10 五個鍵與 recommend_bio_text.py 的 blend 列相同。
    Spearman 用跟主表相同的 200,000 對與種子，所以 w_text=1.0 那一列會等於主表的 Spearman（排名不受 z-score 影響）。"""
    rows = []
    for w in BLEND_WEIGHTS:
        S_b = blend(S_txt, S_tag, w)
        r = neighbor_recovery(S_b, S_tag, k=k)
        rows.append({
            "w_text": w, f"recall_at_{k}": r["recall_at_k"], f"lift_{k}": r["lift_over_random"],
            "gini_at_10": exposure_gini(S_b, k=10), "mean_tag_sim_top10": mean_tag_sim_topk(S_b, S_tag, 10),
            f"recall_at_{k}_vs_text": neighbor_recovery(S_b, S_txt, k=k)["recall_at_k"],
            "spearman_vs_tag": pair_rank_correlation(S_b, S_tag, seed=SEED)["spearman"],
        })
    return rows


def neighbor_examples(S_txt, S_tag, X_sub, bios_sub, n=N_EXAMPLES) -> list[dict]:
    """幾組肉眼可讀的示範：seeker 的 bio、文字最近鄰的 bio、標籤最近鄰的 bio，以及共同標籤數。"""
    rng = np.random.default_rng(SEED)
    seekers = rng.choice(S_txt.shape[0], size=n, replace=False)
    out = []
    for i in seekers:
        shared_all = (X_sub & X_sub[i]).sum(1)
        row = {"seeker_bio": bios_sub[i][:40], "n_tags": int(X_sub[i].sum()),
               "mean_shared_tags_with_anyone": float(np.delete(shared_all, i).mean())}
        for name, S in (("text", S_txt), ("tag", S_tag)):
            s = S[i].copy()
            s[i] = -np.inf
            j = int(np.argmax(s))
            row[f"{name}_nn_bio"] = bios_sub[j][:40]
            row[f"{name}_nn_similarity"] = float(S[i, j])
            row[f"{name}_nn_shared_tags"] = int(shared_all[j])
        out.append(row)
    return out


def interpret(info: dict, res: dict, k: int) -> str:
    """一句話解讀；門檻只用來選措辭（lift<1.5 視為實務上≈隨機、z>3 視為統計上高於隨機、|Spearman|≥0.3 視為
    整體排序與標籤高度重疊），不是驗證過的標準。
    注意：recall 量的是「跟標籤鄰居的重疊」，所以 recall 高於隨機的部分一定是標籤訊號，不可以說成新資訊
    （例如任何「bio→標籤」分類器的輸出，lift 可以 >1.5 而整體 Spearman 仍 <0.3，但它依定義就是純標籤訊號）。"""
    lift, rho, z = res[f"lift_{k}"], res["spearman_vs_tag"], res.get(f"z_vs_random_{k}")
    head = "不是語意嵌入（LSA 基準）；" if info["mode"] == "dry-run" else ""
    indep = "與標籤幾乎獨立" if abs(rho) < 0.1 else "與標籤只有弱相關"
    if lift < 1.5:
        sig = "統計上略高於隨機、實務上沒用" if (z is not None and z > 3) else "與隨機無法區分"
        return head + f"幾乎找不到標籤鄰居（{sig}），{indep}"
    if abs(rho) >= 0.3:
        return head + "找得到標籤鄰居、整體排序也與標籤高度相關＝把標籤訊號抽回來，不是新資訊"
    return head + ("找回部分標籤鄰居＝從 bio 抽回來的標籤訊號（不是新資訊，只對冷啟動有用）；"
                   "其餘排序與標籤無關，是獨立維度，是否有助配對無法驗證")


def load_text_reference() -> list[dict]:
    """若 recommend_bio_text.py 已經跑過，把它的 summary_table 讀進來並排比較（唯讀；沒有或格式不同就回空）。"""
    p = OUTPUT_DIR / "recommend_bio_text" / "metrics.json"
    try:
        rows = json.loads(p.read_text(encoding="utf-8")).get("summary_table", [])
        return [r for r in rows if isinstance(r, dict) and "方法" in r]
    except (OSError, ValueError, AttributeError):
        return []


# ---------------------------------------------------------------- pgvector 匯出
def export_pgvector(E: np.ndarray, accounts, label: str) -> dict:
    """寫 bio_embeddings.csv（account, '[...]'）與 pgvector_import.sql；維度依實際向量決定。"""
    dim = int(E.shape[1])
    csv_path, sql_path = OUT_DIR / "bio_embeddings.csv", OUT_DIR / "pgvector_import.sql"
    # 零向量的 cosine distance 是 NaN，而且 pgvector README 明載「zero vectors for cosine distance」不會進索引
    n_zero = int((np.linalg.norm(E, axis=1) < 1e-6).sum())
    if n_zero:
        print(f"警告：有 {n_zero} 列是零向量（cosine distance 會是 NaN，也不會被 HNSW 索引），匯入前請先處理")
    label_sql = label.replace("'", "''")            # SQL 字串常值裡的單引號要寫成兩個
    fmt = f"{{:.{CSV_DECIMALS}f}}".format
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)                       # 向量欄含逗號，csv 模組會自動加雙引號
        w.writerow(["account", "embedding"])
        for acc, vec in zip(accounts, E):
            w.writerow([acc, "[" + ",".join(map(fmt, vec)) + "]"])

    sql = f"""-- pgvector 匯入範例（由 experiments/recommend_bio_embeddings.py 產生）
-- 模型：{label}；維度：{dim}；列數：{E.shape[0]}
-- 對應 db/01_schema.sql 的 users(id, account) / profiles(user_id, bio)。App 端的 uuid schema 只要把 join 改掉即可。
-- 執行：cd ml/outputs/recommend_bio_embeddings && psql -d dating -f pgvector_import.sql
-- （\\copy 的檔名相對於 psql 的工作目錄；語法出處：https://github.com/pgvector/pgvector README）
-- 注意：dry-run 產生的是 TF-IDF→SVD 假嵌入，只能用來測試匯入流程，不要拿去線上推薦。
-- 注意：這份 SQL 由腳本產生，依專案規定沒有連資料庫實際執行過；第一次請在測試庫跑。
\\set ON_ERROR_STOP on

-- 0. 啟用擴充（每個資料庫一次）
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. 在 profiles 加向量欄位；維度必須跟模型輸出一致（這裡是 {dim}；每列儲存 4*{dim}+8 bytes）
--    換成不同維度的模型（例如 384 → 1024）時，IF NOT EXISTS 會保留舊欄位、後面的 UPDATE 會報維度不符；
--    要先 ALTER TABLE profiles DROP COLUMN bio_embedding; 再重跑。
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS bio_embedding vector({dim});
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS bio_embedding_model text;

-- 2. 先把 CSV 載進 staging 表，再用 account 對回 users.id。
--    vector 型別接受 '[0.1,0.2,...]' 文字格式（README 的 INSERT 範例就是這樣寫），所以一般的 CSV COPY 就能載；
--    README 的大量載入範例是 COPY ... WITH (FORMAT BINARY)，更快但要用程式產生二進位串流，10,000 列用文字格式就夠。
DROP TABLE IF EXISTS bio_embeddings_staging;
CREATE TEMP TABLE bio_embeddings_staging (account text PRIMARY KEY, embedding vector({dim}));
\\copy bio_embeddings_staging (account, embedding) FROM 'bio_embeddings.csv' WITH (FORMAT csv, HEADER true)

UPDATE profiles p
SET bio_embedding = s.embedding, bio_embedding_model = '{label_sql}'
FROM bio_embeddings_staging s
JOIN users u ON u.account = s.account
WHERE p.user_id = u.id;

-- 單筆寫入（線上新使用者填完 bio、後端算完向量後）也直接用文字格式：
-- UPDATE profiles SET bio_embedding = '[0.01,-0.02,...]', bio_embedding_model = '{label_sql}' WHERE user_id = 123;

-- 3. 查詢：找 bio 最像的前 50 人。<=> 是 cosine distance（越小越像），相似度 = 1 - distance。
--    $1、$2 是參數化查詢的佔位符：應用程式（node-postgres、psycopg…）把 AS 後面那段 SELECT 當查詢字串、另外傳參數即可。
--    在 psql 裡裸寫 $1 會報 "there is no parameter $1"，所以這裡包成 PREPARE，整個檔案才能用 psql -f 跑完。
--    $1 = seeker 的向量（'[...]' 字串，必須剛好 {dim} 維），$2 = seeker 自己的 user_id（排除自己）
PREPARE bio_knn (vector, bigint) AS
SELECT p.user_id, u.account, 1 - (p.bio_embedding <=> $1) AS cosine_similarity
FROM profiles p
JOIN users u ON u.id = p.user_id
WHERE p.bio_embedding IS NOT NULL AND p.user_id <> $2
ORDER BY p.bio_embedding <=> $1
LIMIT 50;
-- EXECUTE bio_knn('[0.01,-0.02,...]', 123);

--    或直接拿資料庫裡的向量當查詢（前端不用把 {dim} 維向量傳來傳去）；$1 = seeker 的 user_id。
--    寫法照 README 的「Get the nearest neighbors to a row」：ORDER BY 右邊放純量子查詢，日後建了 HNSW 也用得到索引
--    （README：要用到索引，查詢必須有 ORDER BY + LIMIT，且 ORDER BY 直接是距離運算子、遞增排序）。
PREPARE bio_knn_by_user (bigint) AS
SELECT p.user_id, u.account,
       1 - (p.bio_embedding <=> (SELECT bio_embedding FROM profiles WHERE user_id = $1)) AS cosine_similarity
FROM profiles p
JOIN users u ON u.id = p.user_id
WHERE p.bio_embedding IS NOT NULL AND p.user_id <> $1
ORDER BY p.bio_embedding <=> (SELECT bio_embedding FROM profiles WHERE user_id = $1)
LIMIT 50;
EXECUTE bio_knn_by_user(1);      -- 1 換成實際存在的 user_id；該使用者沒有向量時會回傳 0 列

-- 4. 索引。10,000 人先不要建：pgvector 預設精確搜尋（recall 100%），這個規模暴力掃描是毫秒級；
--    索引是近似搜尋，用一點 recall 換速度，規模到數十萬列以上再考慮（經驗法則，pgvector 沒有官方門檻）。
--    要建時：先載完資料再建（README：載入後建索引比較快）；m = 16、ef_construction = 64 是預設值。
-- CREATE INDEX profiles_bio_embedding_hnsw ON profiles USING hnsw (bio_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
--    查詢時可調 hnsw.ef_search（預設 40）換 recall：
-- SET hnsw.ef_search = 100;
"""
    sql_path.write_text(sql, encoding="utf-8")
    return {"csv": str(csv_path), "sql": str(sql_path), "dim": dim, "n_rows": int(E.shape[0]),
            "csv_mb": round(csv_path.stat().st_size / 1e6, 1), "n_zero_vectors": n_zero,
            "sql_executed_against_db": False,
            "index_recommendation": "10,000 列不建 HNSW（精確掃描即可）；SQL 內附建索引語法供日後使用"}


# ---------------------------------------------------------------- 圖
# 配色沿用專案既有的固定順序（藍 / 橘 / 青綠），基準與 null 一律灰色；文字一律用墨色，不跟著序列顏色走。
C_BLUE, C_ORANGE, C_AQUA, C_GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a94"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(short, res_text, res_null, res_tag, blend_rows, S_txt, S_tag, k):
    # (1) recall@k：只畫三根小的。標籤→標籤自檢 = 1.0 寫在標題，畫出來會把其他柱子壓扁到看不見。
    names = [short, "隨機打散 null", "隨機基準 k/(n-1)"]
    vals = [res_text[f"recall_at_{k}"], res_null[f"recall_at_{k}"], res_text[f"random_baseline_{k}"]]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bars = ax.bar(names, vals, width=0.55, color=[C_BLUE, C_GREY, C_GREY])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_ylabel(f"recall@{k}")
    ax.set_title(f"文字鄰居找回標籤鄰居的比例 recall@{k}（n={S_txt.shape[0]}）\n"
                 f"標籤→標籤自檢 = {res_tag[f'recall_at_{k}']:.3f}（未畫出，否則其他柱子會被壓扁）", fontsize=11)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{MODULE}_recall.png", dpi=130)
    plt.close(fig)

    # (2) blend：左邊三條都是「跟誰的排序比較像」（0–1），右邊單獨放 Gini，避免不同意義的量擠在同一個面板。
    ws = [b["w_text"] for b in blend_rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    for name, key, color in ((f"recall@{k} vs 標籤鄰居", f"recall_at_{k}", C_BLUE),
                             (f"recall@{k} vs 文字鄰居", f"recall_at_{k}_vs_text", C_ORANGE),
                             ("Spearman vs 標籤相似度", "spearman_vs_tag", C_AQUA)):
        axes[0].plot(ws, [b[key] for b in blend_rows], marker="o", markersize=6, linewidth=2, color=color, label=name)
    axes[0].set_title("混合後的排序跟誰比較像", fontsize=11)
    axes[0].set_ylabel("比例 / 相關係數")
    # 圖例放在座標軸下方，避免蓋到任何一條線
    axes[0].legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ginis = [b["gini_at_10"] for b in blend_rows]
    axes[1].plot(ws, ginis, marker="o", markersize=6, linewidth=2, color=MUTED)
    for x, y in ((ws[0], ginis[0]), (ws[-1], ginis[-1])):          # 只標頭尾，不在每個點上放數字
        axes[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=INK)
    axes[1].set_ylim(0, max(ginis) * 1.3)
    axes[1].set_title("曝光集中度 Gini@10", fontsize=11)
    for ax in axes:
        ax.set_xticks(ws)
        ax.set_xlabel("w_text（文字相似度的權重）")
        _clean(ax)
    fig.suptitle(f"混合權重掃描 w_text·z(S_text) + (1-w_text)·z(S_tag)（{short}）", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{MODULE}_blend.png", dpi=130)
    plt.close(fig)

    # (3) 兩種相似度的分布（隨機 200,000 對）
    rng = np.random.default_rng(SEED)
    n = S_txt.shape[0]
    i, j = rng.integers(0, n, 200_000), rng.integers(0, n, 200_000)
    m = i != j
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(S_txt[i[m], j[m]], bins=60, color=C_BLUE)
    axes[0].set_title(f"文字相似度分布（{short}）", fontsize=11)
    axes[1].hist(S_tag[i[m], j[m]], bins=60, color=C_GREY)
    axes[1].set_title("標籤相似度分布（IDF×類別權重）", fontsize=11)
    for ax in axes:
        ax.set_xlabel("cosine")
        ax.set_ylabel("配對數")
        _clean(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{MODULE}_simhist.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 結論
def build_conclusions(info, res_text, res_null, blend_rows, k) -> list[str]:
    rec, base, lift = res_text[f"recall_at_{k}"], res_text[f"random_baseline_{k}"], res_text[f"lift_{k}"]
    rn, rho = res_null[f"recall_at_{k}"], res_text["spearman_vs_tag"]
    c = []
    if info["mode"] == "dry-run":
        c.append(f"這一列是 dry-run 假嵌入（TF-IDF→SVD-{info['dim']}→L2），不是語意嵌入；它的數字只代表"
                 "「線性壓縮後的字面相似度」，不能拿來支持或否定句子嵌入的效果。")
    se, z = res_text[f"recall_se_{k}"], res_text[f"z_vs_random_{k}"]
    z_txt = "n/a" if z is None else f"{z:.1f}"
    if lift < 1.5:
        sig = ("統計上確實略高於隨機（合理的解釋是 bio 偶爾會提到標籤關鍵字；本腳本沒有另外驗證），但實務上沒用"
               if (z is not None and z > 3)
               else "與隨機無法區分")
        c.append(f"{info['label']}：recall@{k} = {rec:.4f} ± {se:.4f}（隨機基準 {base:.4f}，lift {lift:.2f}，z ≈ {z_txt}；"
                 f"隨機打散的 null = {rn:.4f}）。{sig}：每 {k} 位推薦平均只比隨機多找回 {(rec - base) * k:.2f} 位標籤鄰居——"
                 "光靠這種向量找不到標籤鄰居，冷啟動時不能取代標籤。")
    elif abs(rho) >= 0.3:
        c.append(f"{info['label']}：recall@{k} = {rec:.3f}（lift {lift:.2f}，z ≈ {z_txt}），而且 Spearman = {rho:.3f} 也高——"
                 "它主要是把標籤訊號從 bio 抽回來，不是新資訊；對已經勾標籤的人沒有增量，只對冷啟動有用。")
    else:
        c.append(f"{info['label']}：recall@{k} = {rec:.3f}（lift {lift:.2f}，z ≈ {z_txt}）高於隨機——這一部分依定義就是"
                 "從 bio 抽回來的標籤訊號，不是新資訊（對已經勾標籤的人沒有增量，只對冷啟動有用）。"
                 f"Spearman = {rho:.3f} 低，代表其餘的排序與標籤無關，是獨立維度（語意／風格／用語）；"
                 "它是否有助配對，沒有互動資料無法驗證，所以這裡不能下「值得混合」的結論。")
    c.append(f"Spearman = {rho:.3f}：文字相似度與標籤相似度{'幾乎獨立' if abs(rho) < 0.1 else '有部分相關'}。"
             "「風格／用語相似的人配對是否更成功」在沒有互動資料下無法驗證，這裡只能說它是"
             f"{'一個獨立維度' if abs(rho) < 0.1 else '與標籤部分重疊的維度'}。")
    b25 = next(b for b in blend_rows if b["w_text"] == 0.25)
    b50 = next(b for b in blend_rows if b["w_text"] == 0.5)
    c.append(f"混合曲線：w_text=0.25 時標籤鄰居保留 {b25[f'recall_at_{k}']:.0%}、w_text=0.5 時 {b50[f'recall_at_{k}']:.0%}；"
             f"曝光 Gini@10 由 {blend_rows[0]['gini_at_10']:.3f}（純標籤）變成 {blend_rows[-1]['gini_at_10']:.3f}（純文字）。"
             "混合權重要等有互動資料才能校準。")
    if info["mode"] == "dry-run":
        c.append("要真的回答「句子嵌入能不能把訊號從 93% 的填充文字裡分離出來」，請用 --model <名稱> 重跑"
                 "（首次下載約 490 MB；bge-m3 約 2.3 GB），同一張表會多一列可直接比較。")
    c.append("pgvector 路徑已產出（CSV + SQL）；10,000 人用精確搜尋即可，不用建 HNSW。")
    return c


# ---------------------------------------------------------------- 主程式
def _f(v, nd=3) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def main():
    t_start = time.time()
    args = parse_args()
    k = args.k
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_seed()
    cols = trait_columns(df)
    X = binary_matrix(df, cols)
    Xw, _ = weighted_matrix(X, cols)
    bios = df["bio"].tolist()
    accounts = df["account"].tolist()
    n = len(df)
    print(f"使用者 {n}，標籤 {len(cols)}，bio 平均 {np.mean([len(b) for b in bios]):.1f} 字")

    # 1. 嵌入（dry-run 或真模型）
    if args.model is None:
        print("模式：dry-run（沒有 --model）。用 TF-IDF→SVD 假嵌入驗證 pipeline，不下載任何東西。")
        E, info = dryrun_embeddings(bios)
        key, short = "emb_dryrun", "dry-run 假嵌入"
    else:
        E, info = model_embeddings(bios, args)
        key, short = f"emb_{model_slug(args.model)}", args.model.split("/")[-1]
    print(f"嵌入矩陣 {E.shape}，encode {info['encode_seconds']} 秒")

    # 2. 同一把尺：同一份抽樣、同一個 S_tag
    idx = subsample(n, EVAL_SIZE, seed=SEED)
    S_tag = cosine_sim(Xw[idx])
    S_txt = cosine_sim(E[idx])
    rng = np.random.default_rng(SEED)
    S_null = cosine_sim(E[idx][rng.permutation(len(idx))])     # 使用者與向量脫鉤的 null 對照

    res_text = evaluate_against_tags(S_txt, S_tag, k)
    res_null = evaluate_against_tags(S_null, S_tag, k)
    res_tag = evaluate_against_tags(S_tag, S_tag, k)           # 自檢：recall 應為 1.0、Spearman 應為 1.0
    blend_rows = blend_curve(S_txt, S_tag, k)
    examples = neighbor_examples(S_txt, S_tag, X[idx], [bios[i] for i in idx])

    # 3. pgvector 匯出 + 圖
    pg = export_pgvector(E, accounts, info["label"])
    make_figures(short, res_text, res_null, res_tag, blend_rows, S_txt, S_tag, k)

    def table_row(name, dim, res, sec, note):
        return {"方法": name, "維度": dim, f"recall@{k}": res[f"recall_at_{k}"], "lift": res[f"lift_{k}"],
                "recall@10": res["recall_at_10"], "Spearman vs 標籤": res["spearman_vs_tag"],
                "Gini@10": res["gini_at_10"], "前10名平均標籤相似度": res["mean_tag_sim_top10"],
                "encode 秒數": sec, "解讀": note}

    summary_table = [
        table_row(info["label"], info["dim"], res_text, info["encode_seconds"], interpret(info, res_text, k)),
        table_row("隨機打散 null（同向量、使用者脫鉤）", info["dim"], res_null, None, "對照：recall / Spearman 應 ≈ 隨機；Gini 與上一列相同是必然的（只是把使用者重新編號，曝光分布不變）"),
        table_row("標籤本身 S_tag（自檢）", len(cols), res_tag, None, "正解本身：recall = 1、Spearman = 1"),
    ]
    reference = load_text_reference() if k == 50 else []
    conclusions = build_conclusions(info, res_text, res_null, blend_rows, k)
    metrics = {
        "module": MODULE, "mode": info["mode"], "embedding": info,
        "dataset": {"n_users": n, "n_labels": len(cols), "subsample": int(len(idx)), "seed": SEED},
        "settings": {"S_tag": "cosine(features.weighted_matrix(X, cols)[idx])，IDF × 類別權重",
                     "subsample": f"features.subsample({n}, {EVAL_SIZE}, seed={SEED})",
                     "recall": f"textrec.neighbor_recovery k={k} 與 10；random baseline = k/(n-1)；se = std/sqrt(n_seekers)",
                     "mean_tag_sim_top10": "本腳本：方法的前 10 名在 S_tag 上的平均值（公式同 recommend_bio_text.py）",
                     "spearman": "textrec.pair_rank_correlation n_pairs=200000（主表與 blend 曲線相同）",
                     "gini": "textrec.exposure_gini k=10",
                     "blend": f"textrec.blend(S_text, S_tag, w) w∈{list(BLEND_WEIGHTS)}",
                     "null": "同一批嵌入向量，列順序用 default_rng(SEED).permutation 打散"},
        "k": k,
        "methods": {"tag": res_tag, "null_permuted": res_null, key: res_text},     # 鍵名同 recommend_bio_text.py
        "blend": {key: blend_rows},
        "summary_table": summary_table,
        "reference_recommend_bio_text": reference,
        "examples": examples, "pgvector": pg, "candidate_models": CANDIDATE_MODELS,
        "conclusions": conclusions, "runtime_seconds": round(time.time() - t_start, 1),
    }
    save_json(metrics, OUT_DIR / "metrics.json")

    # 4. 精簡 markdown 摘要
    print(f"\n### 文字相似度 vs 標籤相似度（mode = {info['mode']}；抽樣 {len(idx)} 人；"
          f"隨機基準 recall@{k} = {res_text[f'random_baseline_{k}']:.3f}）")
    print(f"| 方法 | 維度 | recall@{k} | lift | recall@10 | Spearman vs 標籤 | Gini@10 | 前10名平均標籤相似度 | encode 秒數 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in summary_table:
        print(f"| {r['方法']} | {r['維度']} | {_f(r[f'recall@{k}'])} | {_f(r['lift'], 2)} | {_f(r['recall@10'])} | "
              f"{_f(r['Spearman vs 標籤'])} | {_f(r['Gini@10'])} | {_f(r['前10名平均標籤相似度'])} | {_f(r['encode 秒數'], 1)} |")
    if reference:
        print("| *以下為 recommend_bio_text.py 上次執行的結果（同一份抽樣與 S_tag，可直接比）* | | | | | | | | |")
        for r in reference:
            print(f"| {r.get('方法')} | - | {_f(r.get('recall@50'))} | {_f(r.get('lift'), 2)} | {_f(r.get('recall@10'))} | "
                  f"{_f(r.get('Spearman vs 標籤'))} | {_f(r.get('Gini@10'))} | {_f(r.get('前10名平均標籤相似度'))} | - |")
    print(f"（recall@{k} 的標準誤 {res_text[f'recall_se_{k}']:.4f}、z vs 隨機 ≈ "
          f"{_f(res_text[f'z_vs_random_{k}'], 1)}；null 的 z ≈ {_f(res_null[f'z_vs_random_{k}'], 1)}。seeker 並非獨立，z 只是粗估）")
    print(f"\n### 混合曲線 w_text·z(S_text) + (1-w_text)·z(S_tag)")
    print(f"| w_text | recall@{k} vs 標籤鄰居 | lift | recall@{k} vs 文字鄰居 | Gini@10 | 前10名平均標籤相似度 | Spearman vs 標籤 |")
    print("|---|---|---|---|---|---|---|")
    for b in blend_rows:
        print(f"| {b['w_text']:.2f} | {b[f'recall_at_{k}']:.3f} | {b[f'lift_{k}']:.2f} | {b[f'recall_at_{k}_vs_text']:.3f} | "
              f"{b['gini_at_10']:.3f} | {b['mean_tag_sim_top10']:.3f} | {b['spearman_vs_tag']:.3f} |")
    print("\n### 示範鄰居（seeker → 文字最近鄰 / 標籤最近鄰；「共同」= 共同標籤數）")
    for e in examples:
        print(f"- seeker「{e['seeker_bio']}」（{e['n_tags']} 標籤；與任意一人平均共同 {e['mean_shared_tags_with_anyone']:.1f}）")
        print(f"    文字 NN「{e['text_nn_bio']}」（共同 {e['text_nn_shared_tags']}，cos {e['text_nn_similarity']:.2f}）")
        print(f"    標籤 NN「{e['tag_nn_bio']}」（共同 {e['tag_nn_shared_tags']}，cos {e['tag_nn_similarity']:.2f}）")
    print(f"\n### pgvector\n- {pg['csv']}（{pg['csv_mb']} MB，{pg['n_rows']} 列，vector({pg['dim']})）\n- {pg['sql']}")
    print("\n### 結論")
    for c in conclusions:
        print(f"- {c}")
    print(f"\n總耗時 {metrics['runtime_seconds']} 秒；輸出：{OUT_DIR / 'metrics.json'}、"
          f"{FIG_DIR / (MODULE + '_recall.png')}、{FIG_DIR / (MODULE + '_blend.png')}、{FIG_DIR / (MODULE + '_simhist.png')}")


if __name__ == "__main__":
    main()
