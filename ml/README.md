# HeartLink ML 實驗（學生專案）

用 HeartLink 的 10,000 位 seed 使用者，練習並比較「非監督分群 → 共識分群（ensemble）→ 監督式訓練 → 互惠推薦」
一整條 pipeline。所有腳本都是獨立的，可以單支執行；所有結果都經過**同一套流程跑在 null 對照資料**上驗證。

> **一句話結論**：這份 seed 的 68 個標籤是隨機獨立生成的，所以任何分群（K-means / HAC / DBSCAN /
> HDBSCAN / 共識分群）都**找不到群**（silhouette ≈ 0.01，與 null 對照相同）——這是預期且正確的結果。
> 整份資料唯一有真訊號的是 **bio 文字 → 標籤**；「互惠推薦 + ensemble 排序」則用透明的模擬器示範方法論。

---

## 快速開始

```bash
cd ml
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python run_all.py                     # 全部跑（筆電約 6 分鐘）
.venv/bin/python experiments/clustering_ensemble.py   # 或單支跑
```

輸出都在 `outputs/`（已被專案根目錄的 `.gitignore` 忽略）：

| 位置 | 內容 |
|---|---|
| `outputs/<實驗名>/metrics.json` | 該實驗所有數字（真實 vs null 並列） |
| `outputs/<實驗名>/run.log` | `run_all.py` 執行時的 stdout / stderr |
| `outputs/figures/<實驗名>_*.png` | 圖表（中文字型已處理） |
| `outputs/models/<實驗名>_*.joblib` | 訓練好的模型 |
| `outputs/summary.json` | `run_all.py` 的彙整 |

Python 3.12（用 uv 管理）；不需要資料庫，只讀 `db/seed/heartlink_with_email.csv`
（與 `dating` 資料庫的 `user_traits` 是同一批資料，差 3 筆測試帳號）。`password` / `email` 欄位在載入時就被丟掉。

---

## 目錄結構

```
ml/
├── README.md
├── requirements.txt
├── run_all.py                  # 一鍵執行 + 彙整
├── heartlink_ml/               # 共用基礎層（每支實驗都用）
│   ├── config.py               #   路徑、SEED=42、類別權重
│   ├── data.py                 #   load_seed()、trait_columns()
│   ├── features.py             #   0/1 矩陣、IDF 加權、Jaccard 距離、SVD、bio TF-IDF
│   ├── evaluation.py           #   silhouette/DB/CH、Hopkins、null model、ARI 穩定度、P@k/NDCG、Gini
│   └── plotting.py             #   散點、折線、樹狀圖、長條（Agg 後端、中文字型）
└── experiments/                # 六支獨立實驗（建議依序閱讀）
    ├── clustering_partitional.py       # K-means / MiniBatch / K-modes / GMM
    ├── clustering_hierarchical.py      # HAC：single / average / complete
    ├── clustering_density.py           # DBSCAN / HDBSCAN
    ├── clustering_ensemble.py          # 共識分群（證據累積 EAC）
    ├── supervised_bio_to_tags.py       # bio 文字 → 68 標籤（multi-label）
    └── supervised_reciprocal_ranker.py # 模擬互動 → ensemble 排序 → 互惠融合
```

每支實驗的檔頭 docstring 都用繁體中文解釋：這個方法是什麼、為什麼適合／不適合這份資料、怎麼跑、怎麼讀結果。

---

## 資料與特徵

| 項目 | 數值 |
|---|---|
| 使用者 | 10,000（Female 6,000 / Male 4,000） |
| 標籤 | 68 個 0/1（dating_goal 5 / interest 37 / personality 12 / diet 5 / lifestyle 4 / value 5） |
| 每人平均標籤數 | 15.57（std 1.86；lifestyle 每人剛好 1 個） |
| bio | 中文、100% 非空、平均 40 字 |
| Hopkins 統計量 | **0.512**（0.5 = 與「各標籤獨立隨機」無法區分） |

特徵有三種空間，不同方法用不同空間（這是刻意的，docstring 有解釋）：

| 空間 | 給誰用 | 為什麼 |
|---|---|---|
| 原始 0/1（68 維） | K-modes、Jaccard 距離 | 二元資料的正解：Hamming / Jaccard |
| IDF × 類別權重 → SVD-16 | K-means、GMM、DB/CH 指標 | 給需要歐氏空間的方法一個合理的連續空間 |
| bio 字元 1–2 gram TF-IDF（4,390 維） | 監督式文字分類 | 中文不需斷詞器 |

---

## 怎麼判斷「分群有沒有找到東西」——三道保險

每支分群腳本都做同樣三件事，缺一不可：

| 保險 | 做法 | 在這份資料上的讀數 |
|---|---|---|
| **Null 對照** | 同一套流程跑在 `column_permutation_null(X)`（逐欄打散：保留流行度、破壞共現）以及「類別區塊打散」（保留類別內配額） | 真實與 null 的 silhouette 幾乎相同 |
| **Hopkins** | 資料點的最近鄰距離 vs 隨機點的最近鄰距離 | 0.512（隨機 = 0.5） |
| **穩定度 ARI** | 換 seed／換抽樣，兩兩 ARI | 高也沒用——見下面的陷阱 |

> ⚠️ **ARI 陷阱**：K-means 在這份資料上換 seed 的 ARI 可達 0.36–0.83，看起來「很穩定」，但 silhouette 是 0.01。
> 穩定只代表最佳化收斂良好，**不代表群存在**。不做 null 對照一定會被騙。

---

## 六支實驗的結果

### 1. 分割式分群（`clustering_partitional.py`，129 秒）

| 方法 | 最佳 k | silhouette 真實 | null 逐欄 | null 區塊 | 穩定度 ARI |
|---|---|---|---|---|---|
| K-means（SVD-16） | 8 | 0.012 | 0.012 | 0.013 | 0.36 |
| MiniBatchKMeans | 2 | 0.010 | 0.008 | 0.010 | 0.04 |
| K-modes（原始 0/1） | 2 | **0.020** | 0.004 | **0.018** | 0.14 |
| GaussianMixture（BIC 選 6） | 6 | −0.002 | −0.001 | 0.008 | 0.19 |

教訓：K-modes 唯一「真實 > 逐欄 null」（0.020 vs 0.004），但一對照**區塊 null** 差距只剩 0.001。
多出來的全是資料產生器「每類別固定配額」造成的類別內互斥（value 5 選 2–3），不是族群。
腳本會印出 K-modes 每群的眾數標籤讓你自己檢查（k=2 切的是 value 類別的兩個互補子集）。
另外 kmodes 的 `init="Cao"` 是確定性初始化，換 seed 的 ARI 恆為 1.0，穩定度要改用 `init="Huang"` 算。

### 2. 階層式分群（`clustering_hierarchical.py`，54 秒）

| linkage | 最佳 k | silhouette 真實 | null 逐欄（自身最佳） | null 區塊 | cophenetic | 最大群佔比 |
|---|---|---|---|---|---|---|
| single | 2 | 0.020 | 0.074 | 0.019 | 0.073 | **100%** |
| average | 4 | 0.012 | 0.059 | 0.017 | 0.194 | 30% |
| complete | 2 | 0.001 | 0.014 | 0.004 | 0.139 | 98% |

教訓：single linkage 在均勻資料上會 **chaining**（k=12 時最大群仍佔 99.6%），這是課本現象的活範例。
逐欄 null 的 silhouette 反而比真實高（0.074），是因為打散後出現「只有 5 個標籤」的離群人被切成迷你群——
所以 null 的設計本身也要檢查。cophenetic 相關係數三種 linkage 都 < 0.2，樹狀圖不忠實反映原距離。

### 3. 密度式分群（`clustering_density.py`，20 秒）

| 方法 | 參數 | 群數 真實 / null | 雜訊比 真實 / null | silhouette 真實 / null |
|---|---|---|---|---|
| DBSCAN（膝點 eps） | eps=0.619, min_samples=5 | 1 / 2 | 0.1% / 6.8% | — / 0.032 |
| DBSCAN（最佳可行） | eps=0.591, min_samples=5 | 6 / 4 | 6.8% / 28.5% | −0.072 / −0.034（最大群 93%） |
| HDBSCAN | min_cluster_size=100 | 2 / 2 | 1.3% / 5.8% | 0.024 / 0.043 |

教訓：二元資料的 Jaccard 距離是**離散值**（3000 人只出現 68 種距離值 ∈ [0.50, 0.75]），
eps 只差幾個刻度就從「90% 雜訊」跳到「90% 一大群」，沒有中間地帶——這是密度均勻資料的預期行為，不是參數沒調好。
審查時還抓到一個實作陷阱：k-distance 圖的膝點要抓「平坦段結束、開始陡升」處（Ester et al. 1996 的慣例），
第一版抓到開頭的彎，導致 eps 落在「大部分人是雜訊」的區間。

### 4. 共識分群 / cluster ensemble（`clustering_ensemble.py`，77 秒）

30 個基礎分群器（K-means × 10、K-modes × 10、HAC-average × 10，k 隨機）→ co-association 矩陣 C → 對 1−C 做 HAC。

| 方法 | k | silhouette (Jaccard) | 對基礎器平均 ARI | 兩次共識 ARI | null 區塊 silhouette |
|---|---|---|---|---|---|
| **共識分群 EAC** | 2 | 0.012 | 0.095 | **1.000** | 0.011 |
| 單一 K-means（n_init=1） | 2 | 0.011 | 0.015 | 0.100 | 0.011 |
| 單一 HAC-average | 2 | 0.011 | 0.081 | 1.000（確定性） | 0.011 |

| 診斷 | 真實 | null 逐欄 | null 區塊 |
|---|---|---|---|
| C 值平均 ± 標準差 | 0.227 ± 0.188 | 0.516 ± 0.153 | 0.228 ± 0.182 |
| C 分佈與真實的 JS 距離 | — | 0.656 | **0.075** |
| 成員兩兩 ARI：同家族 / 跨家族 | 0.553 / **0.006** | 0.504 / −0.001 | 0.502 / 0.007 |

教訓（這是回答「能不能用 ensemble」的核心）：

1. **Ensemble 降低變異，不能無中生有。** 共識分群兩次跑的 ARI 從單次 K-means 的 0.10 提升到 1.00，
   但 silhouette 仍是 0.012，與區塊 null 的 0.011 無異；C 矩陣的分佈與 null 幾乎重疊（JS 0.075）。
2. **跨家族 ARI 只有 0.006** —— 三種演算法對「誰跟誰同群」完全沒有共識，這本身就是「沒有群」的證據。
3. **家族主導陷阱**：HAC 的 10 個成員是同一棵樹的不同切法，彼此 ARI 高，會像「加權 10 倍的一票」主導共識
   （共識群 1 的 663 人有 78% 是 `lifestyle_work_from_home`，這是 HAC 家族投出來的，不是全體共識）。
   所以一定要看「共識 vs 各家族的 ARI」。

### 5. 監督式：bio 文字 → 標籤（`supervised_bio_to_tags.py`，31 秒）

整份資料**唯一有真訊號**的任務。訊號稀疏：有某標籤的人只有約 10% 會在 bio 提到對應關鍵字，
但提到的人 98% 都有該標籤。所以看 **AUC 與 lift@5%**（排序能力），不要只看 F1。

| 模型 | macro-F1 | mean AUC | lift@5% |
|---|---|---|---|
| Dummy（most_frequent） | 0.029 | 0.500 | 0.99 |
| **OvR LogisticRegression** | **0.315** | **0.556** | 2.30 |
| OvR LinearSVC | 0.203 | 0.554 | **2.34** |
| SVD(200) + HistGradientBoosting | 0.140 | 0.544 | 1.96 |
| Soft voting（LR + SVC rank 平均） | 0.291 | 0.555 | 2.33 |
| LR — null 對照（標籤列打散） | 0.279 | **0.499** | 1.00 |

| 類別 | LR AUC | null AUC | 有訊號的標籤數 |
|---|---|---|---|
| interest | 0.570 | 0.498 | 37 / 37 |
| personality | 0.557 | 0.500 | 12 / 12 |
| value | 0.580 | 0.498 | 5 / 5 |
| diet | 0.515 | 0.497 | 1 / 5 |
| lifestyle | 0.506 | 0.495 | 0 / 4 |
| dating_goal | 0.507 | 0.513 | 0 / 5 |

模型確實學到對的東西：`interest_badminton` 的 LR 係數前三名是 羽(9.4)、羽球(9.4)、球(6.5)；
`personality_humorous` 是 默、幽、幽默。腳本量測（不是用講的）發現 bio 生成器只寫 interest / personality / value 三類，
diet / lifestyle / dating_goal 的名稱在 bio 出現 0 次，所以那三類 AUC = 0.5 是正確的。
null 對照（把標籤列打散再訓練）AUC 掉回 0.499，證明分數不是假的。

### 6. 監督式排序 + ensemble + 互惠融合（`supervised_reciprocal_ranker.py`，37 秒）

**沒有真實互動資料**，所以用一個透明的偏好模擬器（所有係數印在輸出裡）產生 like 標籤：
`logit = b0 + 5.0·標籤覆蓋率 + 0.8·同 lifestyle + 0.8·dating_goal 交集 + 1.0·(−年齡差/10) + 1.5·性別配對 + 1.0·受歡迎度 + 雜訊`。
600 位 seeker × 60 候選 = 36,000 對；以 seeker 分組切分（GroupShuffleSplit），測試 seeker 不曾出現在訓練集。

| 模型 | AUC | PR-AUC | P@10 | NDCG@10 |
|---|---|---|---|---|
| 零訓練基準（IDF 加權 overlap） | 0.650 | 0.301 | 0.337 | 0.360 |
| **LogisticRegression** | **0.751** | **0.419** | 0.431 | **0.484** |
| RandomForest(300) | 0.737 | 0.400 | 0.417 | 0.471 |
| HistGradientBoosting | 0.746 | 0.416 | 0.434 | 0.484 |
| Voting（soft：LR+RF+HGB） | 0.748 | 0.417 | **0.436** | 0.484 |
| Stacking（LR+RF+HGB → LR） | 0.750 | 0.419 | 0.432 | 0.484 |
| 理論上限：可觀測部分（不含受歡迎度） | 0.753 | 0.429 | 0.443 | 0.495 |
| 理論上限：完整 P（含不可觀測項） | 0.835 | 0.583 | 0.567 | 0.651 |

三個教學重點：

| 重點 | 數字 | 解讀 |
|---|---|---|
| 模型學回了生成規則 | LR 0.751 vs 可觀測上限 0.753 | pipeline 正確、無洩漏 |
| Ensemble 沒贏過 LR | Voting 0.748 / Stacking 0.750 | 模擬器本身是線性 logit，LR 是「形式正確」的模型；ensemble 只能追平。真實資料有非線性交互時才會拉開 |
| **分群特徵沒有貢獻** | 消融 ±0.0004；permutation importance 占比 0.28% | 把 K-means / K-modes 群 id 當特徵餵進去，什麼都沒學到——因為資料沒群結構。注意 RF 的 impurity importance 給了 11%，那是已知偏誤，要看 permutation importance |

互惠融合（調和平均 vs 單向 vs 算術平均）在這個模擬器上三者幾乎相同（mutual P@10 0.228 / 0.229 / 0.228），
因為模擬器的可觀測項是對稱的（corr(s(a→b), s(b→a)) = 0.973）。**調和平均的價值要在真實的不對稱偏好資料上才看得到**。

---

## 跨實驗總表

| 實驗 | 核心指標 | 真實 | null | 結論 |
|---|---|---|---|---|
| K-means / K-modes / GMM | 最佳 silhouette | 0.020 | 0.018（區塊） | 無結構 |
| HAC（average） | 最佳 silhouette | 0.012 | 0.017（區塊） | 無結構 |
| DBSCAN / HDBSCAN | 最佳 silhouette | 0.024 | 0.043 | 無結構 |
| 共識分群 | silhouette / 跨家族 ARI | 0.012 / 0.006 | 0.011 | 無結構；ensemble 只降變異 |
| bio → 標籤 | mean AUC（LR） | **0.556** | 0.499 | **有真訊號** |
| 模擬排序 | AUC（LR） | **0.751** | 上限 0.753 | pipeline 正確 |

---

## 五個帶得走的教訓

1. **沒有 null model 的分群結果不能信。** K-means 永遠會給你 k 個群，穩定度 ARI 也可以很高；只有「同流程跑在打散資料上」才能告訴你那是不是雜訊。
2. **Null 的設計本身要檢查。** 逐欄打散會破壞資料產生器的「每類別配額」，讓 K-modes 看起來贏了；類別區塊打散才公平。
3. **二元資料不要直接丟 K-means。** 用 K-modes（Hamming + 眾數）或 Jaccard + HAC；要用 K-means 就先做 IDF 加權 + SVD。
4. **Ensemble 降低變異、不降低偏差。** 共識分群把穩定度從 0.10 拉到 1.00，但 silhouette 一動不動；監督式的 Voting / Stacking 也追不過「形式正確」的單一 LR。
5. **特徵重要度要看 permutation，不看 impurity。** RF 對 14 個純雜訊的分群特徵給了 11% 的重要度。

---

## 怎麼延伸（接上真實資料之後）

| 想做 | 前提 | 做法 |
|---|---|---|
| 真正的排序模型 | 累積真實 like / pass 與**曝光日誌** | 把 `supervised_reciprocal_ranker.py` 的模擬器換成真實標籤；LightGBM `lambdarank`（macOS 先 `brew install libomp`） |
| 讓互惠融合有意義 | 真實的不對稱偏好，或在模擬器加入「a 對 b 屬性的單向偏好項」 | 調和平均 vs 算術平均的差距才會出現 |
| 更好的 bio 模型 | 真人 bio（現在的是生成器寫的） | jieba 斷詞 + word n-gram；或 sentence embedding |
| 分群真正派上用場 | 對**行為向量**（不是標籤）分群 | 用途限於候選分桶、多樣性配額、營運分眾——不是推薦本身 |
| 視覺化 | — | UMAP / t-SNE（目前用 SVD 2D） |

---

## 誠實聲明

- 所有分群結果都**不應**拿去做使用者分眾或推薦；它們是「方法正確、資料無結構」的示範。
- `supervised_reciprocal_ranker.py` 的所有 AUC / P@k 都是「學回已知模擬器」的能力，不是真實推薦品質。
- `supervised_bio_to_tags.py` 學到的是生成器的關鍵字對應，不能外推到真人自介。
- 每支腳本都由第二位審查者實際執行、對照規格找問題並修正（例如 DBSCAN 膝點方向、常數標籤的 ARI 退化、錯誤的機制解釋），未修的限制寫在各腳本 docstring 與 `metrics.json`。

## 參考文獻

- scikit-learn 官方 clustering 使用指南（方法比較表、precomputed 距離限制、評估指標）：<https://scikit-learn.org/stable/modules/clustering.html>
- Fred & Jain (2005). *Combining Multiple Clusterings Using Evidence Accumulation*. IEEE TPAMI 27(6). <https://dl.acm.org/doi/10.1109/TPAMI.2005.113>
- Strehl & Ghosh (2002). *Cluster Ensembles — A Knowledge Reuse Framework for Combining Multiple Partitions*. JMLR 3. <https://www.jmlr.org/papers/v3/strehl02a.html>
- Huang (1998). *Extensions to the k-Means Algorithm for Clustering Large Data Sets with Categorical Values*. Data Mining and Knowledge Discovery 2. <https://link.springer.com/article/10.1023/A:1009769707641>
- kmodes 套件：<https://github.com/nicodv/kmodes>

---

## 把 bio 文字加入推薦（`recommend_bio_*.py`）

bio 不在 68 個標籤裡，但 93% 的 bio 文字是標籤以外的內容（口頭禪、梗、風格），只有 6.9% 的字元是標籤名。
三支腳本用同一把尺（`heartlink_ml/textrec.py`）比較各種做法：以「標籤相似度的前 50 位鄰居」為正解，
看只用文字能找回多少（recall@50，隨機基準 0.0167），以及文字相似度與標籤相似度的 Spearman 相關。

```bash
.venv/bin/python experiments/recommend_bio_text.py          # 34 秒，不需下載任何東西
.venv/bin/python experiments/recommend_bio_embeddings.py    # 預設 dry-run（假嵌入，驗證流程用）
.venv/bin/python experiments/recommend_bio_embeddings.py --model intfloat/multilingual-e5-small   # 會從 Hugging Face 下載 449 MB
.venv/bin/python experiments/recommend_bio_llm_extract.py   # 預設 dry-run：印 prompt / schema / 費用估計，零 API 呼叫
```

額外依賴：`requirements-embeddings.txt`（sentence-transformers、torch）、`requirements-llm.txt`（anthropic）。

### 結果（`recommend_bio_text.py`，抽樣 3000）

| 做法 | recall@50 | lift | Spearman vs 標籤 | 解讀 |
|---|---|---|---|---|
| 隨機 | 0.017 | 1.00 | −0.001 | 基準 |
| (1) 整段 bio 的 TF-IDF cosine | 0.021 | 1.26 | 0.009 | **天真做法，幾乎沒用**：填充文字主導了相似度 |
| (2a) 關鍵字抽取（bio ↔ bio） | 0.031 | 1.86 | 0.057 | 只有 54% 的人有提到標籤名，每人平均 1.23 個 |
| (2b) 標籤 ∨ 關鍵字 | 0.977 | 58.6 | 0.998 | 等於原本的標籤（抽到的 97% 本來就勾了）——**不是新資訊** |
| (3) bio→標籤機率（out-of-fold） | 0.023 | 1.38 | 0.012 | 弱；用 in-sample 預測會虛高到 2.75（洩漏示範） |
| (3b) 標籤 ∨ (P>0.5) | 0.070 | 4.21 | 0.245 | **有害**：每人被加 23.8 個雜訊標籤，把標籤相似度毀掉；門檻要 ≥0.9 |
| (4) 主題模型 NMF / LDA | 0.016 | 0.99 | ≈0 | 40 字短文切不出可用主題 |
| (5) 風格 16 特徵 | 0.017 | 1.02 | −0.005 | 與標籤**完全獨立**的維度；有沒有助於配對，沒有互動資料無法驗證 |
| 假嵌入（TF-IDF→SVD-384，dry-run） | 0.021 | 1.27 | 0.008 | 與 (1) 相同；不能用來推論真正的句子嵌入 |

真正有用的情境是**冷啟動與補漏勾**：

| 情境 | recall@50 | lift |
|---|---|---|
| 冷啟動：新使用者只有 bio → 用關鍵字對上別人的完整標籤 | 0.070 | 4.22（bio 有提到標籤者 7.07） |
| 冷啟動：同上，改用 bio→標籤機率（out-of-fold） | 0.048 | 2.87 |
| 半冷啟動：只勾了一半標籤（7.8 個） | 0.291 | — |
| 半冷啟動 + 關鍵字補回 | **0.330** | +0.039（SE 0.003） |

教訓：
1. **不要把整段文字直接丟去算相似度。** 要先把訊號從填充文字裡分離出來（關鍵字、分類器、或能下指令的嵌入模型）。
2. **「抽回標籤」不等於「新資訊」。** recall 高但與標籤 Spearman 也高的方法，只是把使用者已經勾過的東西再算一次；它的價值在冷啟動與補漏勾。
3. **軟性補標籤要用高門檻。** `P>0.5` 每人多 24 個雜訊標籤；`P>0.9` 才安全（此時等同關鍵字）。
4. **評估 bio→標籤的下游效果一定要用 out-of-fold 預測**，否則訓練過的人會虛高（lift 2.75 vs 1.38）。
5. 風格／語氣是標籤沒有的獨立維度，但它的權重只能等真實 like / 回覆資料來校準。

### 尚未量測的部分

- **真正的句子嵌入**：需要下載模型（建議 `intfloat/multilingual-e5-small`，384 維、449 MB、MIT 授權，文字前要加 `"query: "` 前綴）。
- **Claude 結構化抽取**：需要 API 憑證且會花錢。10,000 則 bio 的估計（一則一個請求、不含思考 token）：
  Opus 5 同步 $153.80 / Batch $76.90；Sonnet 5 $61.52 / $30.76；Haiku 4.5 $30.76 / $15.38。
  共用前綴（system prompt + schema ≈ 2,486 token）占輸入 97%，所以 prompt cache 或「一個請求塞多則 bio」才是省錢關鍵。
  第一次花錢前先 `--live --limit 5` 小量確認。

---

## 資料體檢：這份 AI 生成的 seed 像不像真的（`data_realism_audit.py`）

```bash
.venv/bin/python experiments/data_realism_audit.py     # 約 3 秒；重新生成 seed 後再跑一次就能看到哪些項目改善
```

**結論：不是常態分布，也不符合實際情況。** 這份資料的統計特徵與「每個類別先隨機決定要勾幾個、再從選項裡隨機抽」
完全吻合（同類別內的相關係數實測值與理論值吻合到小數第 4 位）。這也是前面所有分群實驗「找不到群」的原因。

關於「常態分布」：0/1 標籤只有兩個值（Bernoulli），本身不可能是常態分布。對標籤該問的是
「流行度的形狀」「每人勾幾個」「標籤之間有沒有關聯」「跟性別年齡有沒有關係」；常態只適用於身高這類連續欄位。

| 檢查 | 這份生成資料 | 真實世界（已查證的參考值） | 判定 |
|---|---|---|---|
| 興趣流行度 | 37 個全部 15.2–17.3%（最高是衝浪） | 台灣 113 年運動現況調查 37 個項目：散步/健走 57.4% → 衝浪 0.1%（差 574 倍） | 均勻，不像真的 |
| 每人勾幾個興趣 | 4、5、6、7、8 個各約 20% | 差異大；Bumble 官方：60% 的人選滿上限 5 個 | 固定範圍均勻抽 |
| 標籤共現 | 跨類別 1,510 對中 0 對顯著（登山×露營 φ=−0.001） | 同檔案的真實 CelebA 屬性：780 對中 654 對顯著；91% 的露營者也從事其他戶外活動 | 完全獨立 |
| 標籤 × 性別 | 0/68 顯著，最大差 2.3 個百分點 | 台灣籃球 男 13.5%／女 3.7%；瑜珈 男 3.6%／女 14.7% | 無關聯 |
| 身高形狀 | 男 165–190、女 150–170，每公分人數一樣（超額峰度 −1.2） | 近似常態；國健署 19–44 歲 男 172.0 cm／女 159.5 cm | 均勻；男性平均高了 5.4 cm |
| 身高 × 年齡 | 94% 的女性身高＝150＋(年齡−20) | 成年人身高與年齡幾乎無關 | 身高是用年齡算出來的 |
| 年齡 | 20–39 歲每歲一樣多 | 交友 app 使用率隨年齡遞減（Pew：18–29 歲 53%、30–49 歲 37%） | 範圍合理、形狀不合理 |
| 性別比 | 女 60%／男 40% | 交友 app 男多於女（Ofcom 2024：男 65%／女 35%） | 方向相反 |
| 作息 | 4 種各 25%、互斥 | 勞動部 114 年：固定班 88.5%（白天班 80.9%）、輪班 8.2%；全部可遠距 2.7% | 不合理 |
| 城市 | 6 城各 16.7%、沒有新北、同城市所有人座標相同 | 戶政司：新北 17.4%、台中 12.3%、高雄 11.7%、台北 10.4%、桃園 10.2%、台南 8.0%、新竹市 2.0% | 不合理 |
| 互斥標籤 | 素食×愛燒肉 188 人、健談×安靜 526 人 | 應接近 0 | 沒有一致性規則 |
| 星座 vs 生日 | 99.1% 一致 | — | 合理 |

圖：`outputs/figures/data_realism_audit.png`（藍＝AI 生成、橘＝同檔案的真實 CelebA 屬性、灰＝理論參考）。

### 怎麼讓資料變得像真的

把「逐欄位各自隨機」改成「一條有結構的機率式」，LLM 只負責提供機率表與人設，抽樣交給 numpy：

```
logit P(某人有標籤 j) = b_j            ← 長尾的基礎流行度（旅遊、美食高；衝浪、滑雪低）
                      + 人設 × 主題載荷   ← 先抽「戶外型／藝文型／宅系／美食社交型…」，同主題的興趣一起升高 → 產生共現
                      + γ_j · 性別 + δ_j · 年齡
                      + u_i              ← 每個人的「活躍度」，讓有人勾很少、有人勾很多
再加後置規則：互斥標籤（素食 vs 燒肉）、最少標籤數、身高 ~ 常態且與年齡無關、城市依人口加權並加座標抖動
```

沒有真實對照資料時，算不出標準的擬真度分數（SDMetrics 的 TVComplement、ContingencySimilarity 等都需要「真實 vs 合成」兩份資料）；
能做的是用官方統計當外部基準，並用本腳本檢查結構指紋。

### 參考來源

- 教育部體育署《113 年運動現況調查》：<https://ws.sports.gov.tw/FS01/FilePath/2/relfile/46/13333/ac911192-d0d9-4282-a23d-4fc7195384af.pdf>
- 國健署《國民營養健康狀況變遷調查 106–109 年成果報告》表 3.5.1：<https://crc.sfaa.gov.tw/Uploadfile/StatiscsKnowledge/34_20220629145208_2497642.pdf>
- 勞動部《114 年勞工生活及就業狀況調查》：<https://statdb.mol.gov.tw/html/s2/svy14/1423analyze.pdf>
- 內政部戶政司 Open Data（各縣市人口）：<https://www.ris.gov.tw/rs-opendata/api/v1/datastore/ODRP014/11508>
- Pew Research Center (2023), Key findings about online dating in the U.S.：<https://www.pewresearch.org/short-reads/2023/02/02/key-findings-about-online-dating-in-the-u-s/>
- Bisbee et al. (2024), *Synthetic Replacements for Human Survey Data? The Perils of Large Language Models*, Political Analysis
- Dominguez-Olmedo, Hardt & Mendler-Dünner (NeurIPS 2024), *Questioning the Survey Responses of Large Language Models*：<https://arxiv.org/abs/2306.07951>
- SDMetrics 官方文件：<https://docs.sdv.dev/sdmetrics>
