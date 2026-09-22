"""Gemini 向量化（embedding）。

規格決定：所有向量都用同一個模型（預設 gemini-embedding-2、768 維），因為不同模型產生的
向量不能互相比較；額度不夠時是「整套換掉並重建索引」，而不是把別的模型當備援。

gemini-embedding-2 沒有 task_type 參數，依官方文件把任務說明寫在文字前面：
- 查詢：`task: search result | query: {內容}`
- 文件：`title: {標題} | text: {內容}`
一次請求放多段文字時，每段都要各自包成 Content，才會得到各自的向量（否則會合併成一個）。
"""

import asyncio
from typing import Literal

from ..config import Settings
from .errors import AIServiceError

Purpose = Literal["query", "document"]
BATCH_SIZE = 32  # 每次請求最多放幾段文字；官方沒寫上限，保守設定並分批送出
DEFAULT_TITLE = "聊天內容"


def format_for_embedding(text: str, purpose: Purpose, title: str | None = None) -> str:
    """依用途加上 gemini-embedding-2 建議的任務前綴。"""
    if purpose == "query":
        return f"task: search result | query: {text}"
    return f"title: {title or DEFAULT_TITLE} | text: {text}"


class Embedder:
    """呼叫 Gemini 把文字轉成向量；沒有 API key 時 configured 為 False。

    client 參數讓測試可以注入假的 client；正式執行時第一次用到才建立 google-genai 的 Client。
    """

    def __init__(self, settings: Settings, client=None):
        """保存設定；client 為 None 時等第一次呼叫 embed 才建立（服務啟動時不需要 API key）。"""
        self.settings = settings
        self._client = client

    @property
    def configured(self) -> bool:
        """有注入 client，或有 Gemini API key，就算已設定。"""
        return self._client is not None or self.settings.gemini_configured

    @property
    def model(self) -> str:
        """目前使用的向量模型名稱（會寫進結果，讓後端記錄版本）。"""
        return self.settings.embedding_model

    @property
    def dimensions(self) -> int:
        """向量維度；pgvector 欄位宣告為 vector(768)，兩邊必須一致。"""
        return self.settings.embedding_dimensions

    def _get_client(self):
        """延遲建立 google-genai 的 Client，避免服務啟動時就需要 API key。"""
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    async def embed(self, texts: list[str], purpose: Purpose = "document", title: str | None = None) -> list[list[float]]:
        """把多段文字轉成向量，回傳順序與輸入相同。

        - 沒有設定 → EMBEDDING_NOT_CONFIGURED。
        - API 錯誤、逾時、回傳數量或維度不對 → EMBEDDING_UNAVAILABLE。
        """
        if not texts:
            return []
        if not self.configured:
            raise AIServiceError("EMBEDDING_NOT_CONFIGURED")
        from google.genai import types

        client = self._get_client()
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), BATCH_SIZE):
                batch = texts[start:start + BATCH_SIZE]
                contents = [
                    types.Content(parts=[types.Part.from_text(text=format_for_embedding(text, purpose, title))])
                    for text in batch
                ]
                async with asyncio.timeout(self.settings.llm_timeout_seconds):
                    result = await client.aio.models.embed_content(
                        model=self.model,
                        contents=contents,
                        config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
                    )
                embeddings = result.embeddings or []
                if len(embeddings) != len(batch):
                    raise ValueError("EMBEDDING_COUNT_MISMATCH")
                for embedding in embeddings:
                    values = list(embedding.values or [])
                    if len(values) != self.dimensions:
                        raise ValueError("EMBEDDING_DIMENSION_MISMATCH")
                    vectors.append(values)
        except AIServiceError:
            raise
        except Exception:
            # google-genai 的例外可能包含請求內容；只回報固定代碼。
            raise AIServiceError("EMBEDDING_UNAVAILABLE") from None
        return vectors
