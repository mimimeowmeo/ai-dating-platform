"""僅用在未配置模型的測試 AI API，透過真實 HTTP 驗證介面。"""

import base64
import io
import os

import httpx
from PIL import Image


def main():
    url = os.environ.get("AI_TEST_URL", "http://127.0.0.1:8000")
    token = os.environ["AI_INTERNAL_TOKEN"]
    buffer = io.BytesIO()
    Image.new("RGB", (128, 128)).save(buffer, format="PNG")
    request = {
        "imageBase64": base64.b64encode(buffer.getvalue()).decode(),
        "mimeType": "image/png", "requestId": "http-integration-test",
    }
    with httpx.Client(base_url=url, timeout=10, trust_env=False) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.post("/internal/ai/face/verify", json=request).status_code == 401
        result = client.post("/internal/ai/face/verify", json=request, headers={"X-Internal-Token": token})
        assert result.status_code == 200
        assert result.json()["status"] == "unavailable"
        assert result.json()["reasonCode"] == "MODEL_NOT_CONFIGURED"
    print("PASS: 真實 HTTP 健康檢查、私有認證、未配置模型結果")


if __name__ == "__main__":
    main()
