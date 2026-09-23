"""僅用在未配置模型的測試 AI API，透過真實 HTTP 驗證介面。

這支檔案是「整合測試腳本」，不是 unittest 測試：
- 檔名不是 test*.py，所以 `python -m unittest discover -s tests` 不會自動執行它。
- 必須先把 AI 服務真的跑起來（例如 `uvicorn app.main:app --port 8000`），再手動執行：
  `AI_INTERNAL_TOKEN='<測試 token>' .venv/bin/python -m tests.integration_http`（見 services/ai/README.md）。
- 被測的 AI 服務必須「沒有設定」AI_VERIFICATION_PROVIDER_URL（未配置模型），
  否則最後一步會拿到 provider 的判定而不是 MODEL_NOT_CONFIGURED，斷言就會失敗。

和 test_verification.py 的差別：那邊用 FastAPI 的 TestClient 在同一個程序內呼叫 app，
不經過網路；這裡透過真正的 HTTP 連線，確認實際部署的服務（含 uvicorn、中介層）行為一致。

驗證三件事：
1. GET /health 不需要 token 就能回 {"status": "ok", ...}。
2. 沒帶 X-Internal-Token 呼叫私有驗證 API 會被 main.py 的 InternalBoundary 擋下，回 401。
3. 帶正確 token、送合法影像時回 200，但因為沒有模型，結果只能是 unavailable / MODEL_NOT_CONFIGURED，
   絕不能因為「影像格式正確」就被當成通過。

任何一個 assert 不成立會丟出 AssertionError，腳本以非 0 結束；全部通過才印出 PASS。
注意：Python 用 `-O` 參數執行時會略過所有 assert，所以這支腳本不要用 -O 執行。
"""

# base64：把影像的二進位內容編成 Base64 字串（API 的 imageBase64 欄位要的格式）。
import base64
# io：提供 BytesIO，一個「存在記憶體裡的檔案」，讓 PIL 把影像存進去而不必寫到硬碟。
import io
# os：讀取環境變數（服務網址與內部 token）。
import os

# httpx：Python 的 HTTP 用戶端，用法類似前端的 fetch／axios。
import httpx
# PIL（Pillow）的 Image：用來在記憶體中產生一張測試用的圖片。
from PIL import Image


def main():
    """對執行中的 AI 服務發出真實 HTTP 請求，檢查健康檢查、內部認證與「未配置模型」的結果。

    參數：無（設定從環境變數讀取）。
        AI_TEST_URL：AI 服務的網址，沒設定時用 http://127.0.0.1:8000。
        AI_INTERNAL_TOKEN：必填，要和被測服務設定的 AI_INTERNAL_TOKEN 相同。

    回傳：None；全部檢查通過時印出 PASS 訊息。

    可能丟出：
        KeyError：沒有設定 AI_INTERNAL_TOKEN 環境變數（刻意用 os.environ[...]，沒設定就立刻失敗）。
        httpx.HTTPError（例如 ConnectError、TimeoutException）：服務沒啟動、連不到或超過 10 秒沒回應。
        AssertionError：任何一項回應不符合預期。
    """
    # 讀取被測服務的網址；.get 在環境變數不存在時回傳第二個參數當預設值（本機 8000 埠）。
    url = os.environ.get("AI_TEST_URL", "http://127.0.0.1:8000")
    # 讀取內部 token；用 [] 取值，環境變數不存在會丟 KeyError，避免在沒有 token 的情況下誤判測試結果。
    token = os.environ["AI_INTERNAL_TOKEN"]
    # 建立一個記憶體中的空「檔案」，用來接收 PIL 產生的圖片位元組。
    buffer = io.BytesIO()
    # 產生一張 128×128 的 RGB 純黑圖片（沒給顏色時預設黑色），存成 PNG 寫進 buffer。
    # 128 px 落在 verification.py 允許的 64～4096 px 範圍內，所以是一張「格式合法」的影像。
    Image.new("RGB", (128, 128)).save(buffer, format="PNG")
    # 組出驗證請求的 JSON 內容（對應 schemas.py 的 VerificationRequest）。
    request = {
        # 取出 buffer 裡的 PNG 位元組 → 編成 Base64（bytes）→ .decode() 轉成一般字串，JSON 才放得進去。
        "imageBase64": base64.b64encode(buffer.getvalue()).decode(),
        # 宣告的格式必須和實際檔案（PNG）一致，否則會被判 IMAGE_TYPE_MISMATCH；
        # requestId 只能用英數字、底線、連字號，這個值符合規則。
        "mimeType": "image/png", "requestId": "http-integration-test",
    }
    # 建立 HTTP 用戶端：base_url 讓後面只要寫路徑；timeout=10 表示每個請求最多等 10 秒；
    # trust_env=False 表示不讀取 HTTP_PROXY 等環境變數，確保請求直接打到被測服務、不被本機代理攔截。
    # with 區塊結束時會自動關閉連線（類似 try/finally 的資源清理）。
    with httpx.Client(base_url=url, timeout=10, trust_env=False) as client:
        # 檢查 1：/health 不在 /internal/ 底下，不需要 token；回應 JSON 的 status 必須是 "ok"。
        assert client.get("/health").json()["status"] == "ok"
        # 檢查 2：不帶 X-Internal-Token 呼叫私有 API，InternalBoundary 會在解析 JSON 之前就回 401。
        assert client.post("/internal/ai/face/verify", json=request).status_code == 401
        # 檢查 3 的請求：帶上正確的內部 token 再送一次同樣的影像。
        result = client.post("/internal/ai/face/verify", json=request, headers={"X-Internal-Token": token})
        # 認證通過、影像也合法，所以 HTTP 狀態碼是 200（驗證「結果」放在 JSON 裡，不是用狀態碼表示）。
        assert result.status_code == 200
        # 沒有配置模型時一律回 unavailable，絕不會回 verified。
        assert result.json()["status"] == "unavailable"
        # 原因代碼必須明確指出「模型未設定」，讓呼叫端知道不是影像的問題。
        assert result.json()["reasonCode"] == "MODEL_NOT_CONFIGURED"
    # 走到這裡代表所有 assert 都通過，印出通過訊息。
    print("PASS: 真實 HTTP 健康檢查、私有認證、未配置模型結果")


# 只有「直接執行這個檔案」（python -m tests.integration_http）時才跑 main()；
# 被其他模組 import 時不會自動執行。這是 Python 常見的進入點寫法。
if __name__ == "__main__":
    # 執行整合測試。
    main()
