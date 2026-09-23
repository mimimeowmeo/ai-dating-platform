"""把 Silent-Face 的 MiniFASNet 權重（.pth）轉成 ONNX，並確認 onnxruntime 與 PyTorch 的輸出一致。

只在 Docker build 的 models 階段執行；執行期映像只帶 ONNX，不裝 torch。不一致就讓 build 失敗。

背景說明：
- MiniFASNet 是 Silent-Face-Anti-Spoofing 專案的被動防偽（passive anti-spoofing）模型：
  輸入一張 80×80 的臉部裁切圖，輸出 3 個類別的分數（logits），執行期取 index 1 當作「真人」的機率
  （見 app/models.py 的 REAL_CLASS）。
- 上游只提供 PyTorch 的 .pth 權重。PyTorch 體積很大，執行期映像不想帶它，
  所以在 build 階段先把模型轉成 ONNX（一種跨框架的模型格式），執行期改用輕量的 onnxruntime 推論。
- 轉換之後用隨機輸入比對 PyTorch 與 onnxruntime 的輸出，確認轉出來的 ONNX 行為沒有走樣。

在 Dockerfile 中的執行方式：`cd tools && python export_minifasnet.py ../downloads ../models`
（先 cd 到 tools/，下面的 `from minifasnet import ...` 才找得到同資料夾的 minifasnet.py）。

用法：`python export_minifasnet.py <權重資料夾> <輸出資料夾>`，兩個參數都必填。

可能的失敗（都會讓程式以非 0 狀態結束，因此 Docker build 失敗）：
- 權重檔不存在或內容和模型結構對不上：torch.load／load_state_dict 丟出例外。
- 轉出的 ONNX 和 PyTorch 結果不一致：check_parity() 呼叫 sys.exit(訊息)。
"""

# sys：讀命令列參數（sys.argv），以及用 sys.exit 帶錯誤訊息結束程式。
import sys
# OrderedDict：保留插入順序的字典；PyTorch 的 state_dict 本身就是 OrderedDict，改 key 後沿用同一種型別。
from collections import OrderedDict
# Path：物件導向的檔案路徑工具，可以用 `/` 組路徑。
from pathlib import Path

# numpy：數值陣列運算，用來產生測試輸入、算 softmax 與比較差異。
import numpy as np
# onnxruntime：微軟的 ONNX 推論引擎，也是執行期實際用來跑 MiniFASNet 的函式庫；這裡用它載入剛轉出的 ONNX 做比對。
import onnxruntime as ort
# torch：PyTorch，負責載入 .pth 權重、執行原始模型，以及匯出 ONNX；只在 build 階段安裝。
import torch

# 從同資料夾的 minifasnet.py（上游模型定義的副本）匯入兩個模型建構函式。
from minifasnet import MiniFASNetV1SE, MiniFASNetV2

# (原始權重, 模型建構函式, 輸出檔名)；輸出檔名要和 app/models.py 的 MINIFASNET_FILES 一致。
# 兩個模型分別對應不同的臉部裁切倍率（2.7 與 4.0，寫在權重檔名開頭），執行期會把兩者的機率平均。
MODELS = (
    # MiniFASNetV2：權重檔名開頭 2.7 表示裁切倍率 2.7；轉出檔名 minifasnet_v2.onnx。
    ("2.7_80x80_MiniFASNetV2.pth", MiniFASNetV2, "minifasnet_v2.onnx"),
    # MiniFASNetV1SE（帶 SE 注意力模組的 V1）：權重檔名開頭 4_0_0 表示裁切倍率 4.0；轉出檔名 minifasnet_v1se.onnx。
    ("4_0_0_80x80_MiniFASNetV1SE.pth", MiniFASNetV1SE, "minifasnet_v1se.onnx"),
)
# 模型輸入影像的邊長（像素）：兩個權重都是用 80×80 訓練的（權重檔名中的 80x80）。
INPUT = 80
# utility.get_kernel(80, 80)；建構函式的預設 (7, 7) 對不上權重的形狀。
# 說明：上游 utility.get_kernel 的算法是 ((高 + 15) // 16, (寬 + 15) // 16)，80 代入得到 (5, 5)。
# 模型裡有 4 個 stride 2 的層，80×80 的輸入到最後剩 5×5 的特徵圖，最後的 conv_6_dw 必須用 5×5 的 kernel
# 才能把它壓成 1×1；權重檔裡這一層的形狀也就是 5×5，用預設 (7, 7) 載入權重會因形狀不符而失敗。
KERNEL = ((INPUT + 15) // 16, (INPUT + 15) // 16)
# 一致性檢查要跑幾組隨機輸入；16 組足以抓出轉換錯誤，又不會拖慢 build。
PARITY_SAMPLES = 16
# PyTorch 與 onnxruntime 輸出的機率（softmax 後，0–1）允許的最大絕對差；超過就判定轉換有問題。
# 兩邊的浮點運算順序不完全相同，本來就會有極小的誤差，所以不要求完全相等。
MAX_PROBABILITY_DIFF = 1e-4


# 建立模型並載入 .pth 權重。
def load(path: Path, factory) -> torch.nn.Module:
    """建立一個 MiniFASNet 模型、載入權重檔，並切到推論（eval）模式。

    參數：
        path：.pth 權重檔的路徑。
        factory：模型建構函式（MiniFASNetV2 或 MiniFASNetV1SE），呼叫它會回傳一個尚未載入權重的模型。

    回傳：
        已載入權重、處於 eval 模式的 torch.nn.Module。

    可能丟出的錯誤：
        FileNotFoundError：權重檔不存在。
        pickle／torch 相關例外：檔案內容不是 weights_only 允許的格式。
        RuntimeError：權重的 key 或張量形狀和模型結構對不上（strict=True 時任何多出或缺少的 key 都算錯）。

    設計理由：
        - `weights_only=True` 讓 torch.load 只還原張量與基本容器，不執行檔案裡任意的 pickle 程式碼，
          是載入外部下載檔案時比較安全的做法（這個專案用的 torch 2.5.1 預設不是 True，所以要明寫）。
        - `strict=True` 確保每一層都真的載入到權重；否則有層沒對上時會靜默保留隨機初始值，模型結果會是錯的。
        - 一定要 eval()：BatchNorm 改用訓練時累積的平均與變異數、Dropout 關閉，推論結果才穩定且可以匯出。
    """
    # 呼叫建構函式建立模型架構，並把最後一層 depthwise 卷積的 kernel 設成 (5, 5) 以符合 80×80 輸入與權重形狀。
    model = factory(conv6_kernel=KERNEL)
    # 讀入權重；map_location="cpu" 不論張量存檔時在哪個裝置上，一律載到 CPU（build 環境不使用 GPU），weights_only=True 見 docstring。
    state = torch.load(path, map_location="cpu", weights_only=True)
    # 原始權重是用 DataParallel 存的，每個 key 都有 module. 前綴。
    # 說明：DataParallel 是 PyTorch 的多 GPU 包裝，存檔時 key 會變成 "module.conv1.conv.weight" 這樣；
    # 這裡的模型沒有包 DataParallel，所以把每個 key 前面的 "module." 去掉（removeprefix 沒有這個前綴時會原樣回傳），
    # 值（權重張量）不變，再組回 OrderedDict 保持原本順序。
    state = OrderedDict((key.removeprefix("module."), value) for key, value in state.items())
    # 把權重填進模型；strict=True 要求 key 完全一致，否則丟出 RuntimeError 讓 build 失敗。
    model.load_state_dict(state, strict=True)
    # 切到推論模式；eval() 會回傳模型本身，所以可以直接 return。
    return model.eval()


# 用 numpy 算 softmax，把 logits 轉成每一列加總為 1 的機率。
def softmax(logits: np.ndarray) -> np.ndarray:
    """對形狀 (批次數, 類別數) 的 logits 逐列做 softmax。

    參數：
        logits：模型輸出的原始分數，形狀 (N, C)。

    回傳：
        同形狀的機率陣列，每一列（axis=1）加總為 1。

    可能丟出的錯誤：
        無（輸入有 NaN／inf 時結果會是 NaN，但不會丟例外）。

    設計理由：
        先減掉每一列的最大值再取 exp，數學上結果不變，但可以避免 exp 遇到很大的數而溢位（數值穩定的標準寫法）。
        比較「機率」而不是直接比較 logits，是因為執行期（app/models.py）用的就是 softmax 後的機率，
        誤差門檻 MAX_PROBABILITY_DIFF 也是以機率為單位訂的。
    """
    # 每一列減掉該列最大值（keepdims=True 保留維度，才能和原陣列逐列相減）後取指數。
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    # 每一列除以該列總和，得到加總為 1 的機率。
    return exp / exp.sum(axis=1, keepdims=True)


# 比對 PyTorch 原模型與轉出的 ONNX 在同樣輸入下的輸出。
def check_parity(model: torch.nn.Module, onnx_path: Path) -> float:
    """用固定亂數種子產生的隨機影像，比較 PyTorch 與 onnxruntime 的輸出機率。

    參數：
        model：已載入權重、處於 eval 模式的 PyTorch 模型（轉換的來源）。
        onnx_path：剛匯出的 ONNX 檔路徑。

    回傳：
        所有測試樣本中，兩邊機率的最大絕對差（float），供呼叫端印在 build log。

    可能丟出的錯誤／結束方式：
        SystemExit：任一樣本兩邊的預測類別（argmax）不同，或最大差異超過 MAX_PROBABILITY_DIFF 時，
        呼叫 sys.exit(訊息) 結束程式（訊息印到 stderr、結束代碼 1），讓 Docker build 失敗。
        onnxruntime 相關例外：ONNX 檔無法載入或執行時直接丟出。

    設計理由：
        ONNX 匯出偶爾會因為運算子轉換差異產生不同結果；在 build 階段就驗證，
        可以確保執行期拿到的 ONNX 和上游 PyTorch 模型的判斷一致，而不是上線後才發現防偽結果不對。
        亂數種子固定為 0，每次 build 用的測試資料都一樣，結果可重現。
    """
    # 用 onnxruntime 載入 ONNX 檔；指定只用 CPU 執行（和執行期的設定相同）。
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # 建立種子固定為 0 的亂數產生器，讓每次產生的測試影像都一樣。
    rng = np.random.default_rng(0)
    # 目前看到的最大差異，從 0 開始累計。
    worst = 0.0
    # 跑 PARITY_SAMPLES 組測試；迴圈變數用不到，所以命名為 `_`。
    for _ in range(PARITY_SAMPLES):
        # 執行期的輸入是 0–255 的 BGR 像素（不正規化），這裡用同樣的範圍測。
        # 產生形狀 (1, 3, 80, 80) 的整數陣列：批次 1、3 個色彩通道、高 80、寬 80（PyTorch／ONNX 慣用的 NCHW 排列）；
        # integers(0, 256) 的上限不含 256，所以值落在 0–255，再轉成模型要的 float32。
        sample = rng.integers(0, 256, size=(1, 3, INPUT, INPUT)).astype(np.float32)
        # torch.no_grad()：推論時不需要計算梯度，關掉可以省記憶體與時間。
        with torch.no_grad():
            # 把 numpy 陣列轉成 torch 張量丟進模型，輸出 logits 轉回 numpy 後算 softmax，當作「預期值」。
            expected = softmax(model(torch.from_numpy(sample)).numpy())
        # 用 onnxruntime 跑同一個輸入："input" 是匯出時指定的輸入名稱；第一個參數 None 表示取回所有輸出，
        # 回傳是輸出陣列的 list，[0] 取第一個（也是唯一一個，logits），再算 softmax 當作「實際值」。
        actual = softmax(session.run(None, {"input": sample})[0])
        # 算這個樣本兩邊機率的最大絕對差，和目前的 worst 取較大者。
        worst = max(worst, float(np.max(np.abs(expected - actual))))
        # 兩邊預測的類別（機率最大的 index）不同，代表判斷結果會不一樣，這是最嚴重的錯誤，立刻中止。
        if int(np.argmax(expected)) != int(np.argmax(actual)):
            # sys.exit 帶字串：印到 stderr 並以代碼 1 結束。
            sys.exit(f"{onnx_path.name}: argmax mismatch")
    # 全部樣本跑完後，最大差異超過門檻也視為轉換失敗。
    if worst > MAX_PROBABILITY_DIFF:
        # 訊息用科學記號（.2e、.0e）顯示差異與門檻，方便在 build log 判讀。
        sys.exit(f"{onnx_path.name}: probability diff {worst:.2e} > {MAX_PROBABILITY_DIFF:.0e}")
    # 通過檢查，回傳最大差異給呼叫端記錄。
    return worst


# 腳本主流程：逐一載入權重、匯出 ONNX、做一致性檢查。
def main(weights_dir: Path, out_dir: Path) -> None:
    """把 MODELS 列出的每個 .pth 轉成 ONNX，存到 `out_dir`，並逐一驗證一致性。

    參數：
        weights_dir：放 .pth 權重的資料夾（Dockerfile 中是 fetch_models.py 下載到的 downloads/）。
        out_dir：ONNX 輸出資料夾（Dockerfile 中是 models/，之後整個複製進執行期映像）；不存在時會自動建立。

    回傳：
        None。每個模型完成後印一行 "<輸出檔名>: exported, max probability diff <差異>"。

    可能丟出的錯誤／結束方式：
        load() 與 torch.onnx.export() 的例外直接往上丟；check_parity() 失敗時以 sys.exit 結束。

    設計理由：
        匯出時用固定大小 (1, 3, 80, 80) 的輸入，而且沒有設定 dynamic_axes，所以 ONNX 的輸入形狀固定為
        批次 1、80×80；執行期（app/models.py）每次也剛好只送一張 80×80 的裁切圖，兩者一致。
    """
    # 建立輸出資料夾；parents=True 一併建立上層資料夾，exist_ok=True 表示已存在也不報錯。
    out_dir.mkdir(parents=True, exist_ok=True)
    # 逐一處理 MODELS 的每一組（權重檔名, 建構函式, 輸出檔名）。
    for weights, factory, output in MODELS:
        # 建立模型並載入權重（路徑 = 權重資料夾 / 權重檔名）。
        model = load(weights_dir / weights, factory)
        # 組出 ONNX 的輸出路徑。
        onnx_path = out_dir / output
        # 匯出 ONNX：PyTorch 會用下面的範例輸入實際跑一次模型，把過程中的運算記錄成 ONNX 計算圖。
        torch.onnx.export(
            # 第 1 個參數：要匯出的模型；第 2 個：範例輸入（全 0 的 (1, 3, 80, 80) 張量，只用來決定形狀，數值不重要）；
            # 第 3 個：輸出檔路徑（轉成字串）。
            model, torch.zeros(1, 3, INPUT, INPUT), str(onnx_path),
            # input_names／output_names：替 ONNX 的輸入、輸出取名為 "input" 與 "logits"（check_parity 用 "input" 餵資料）；
            # opset_version=17：使用 ONNX 運算子集第 17 版，決定匯出時可用的運算子定義，執行期的 onnxruntime 必須支援這一版。
            input_names=["input"], output_names=["logits"], opset_version=17,
        )
        # 做一致性檢查（不通過會直接結束程式），通過就印出最大差異（科學記號兩位小數）。
        print(f"{output}: exported, max probability diff {check_parity(model, onnx_path):.2e}")


# 只有「直接執行這個檔案」時才跑下面這段；被 import 時不會自動執行轉換。
if __name__ == "__main__":
    # 第 1 個命令列參數是權重資料夾、第 2 個是輸出資料夾；沒給足參數會因 IndexError 失敗。
    main(Path(sys.argv[1]), Path(sys.argv[2]))
