"""對話切片：把一個聊天室的訊息切成一段一段，供語意檢索（RAG）使用。規格 5.2。

切法（使用者 2026-09-22 決定）：
- 兩則訊息間隔超過 30 分鐘就切開，視為新的一段聊天。
- 每段最多 12 則，或估計 400 tokens。
- 因為長度上限切開時，下一段重疊前一段最後 2 則，避免一句話的前因後果被切散；
  因為時間間隔切開時不重疊，因為兩段是不同場次的聊天，重疊只會混進無關內容。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

from .schemas import IndexMessage
from .textutil import as_utc, estimate_tokens, single_line, taipei_label

CHUNK_VERSION = "chunk-v1"
SESSION_GAP = timedelta(minutes=30)
MAX_MESSAGES = 12
MAX_TOKENS = 400
OVERLAP = 2


@dataclass
class ChunkDraft:
    """切好的一個片段（尚未向量化）。

    messages 是這段包含的訊息（可能含與上一段重疊的 2 則）；content 是要拿去向量化、
    也會存進資料庫的文字；is_open 表示這段可能還會長大（見 build_chunks）。
    """

    messages: list[IndexMessage]
    content: str
    token_estimate: int
    is_open: bool = False


def sort_messages(messages: Sequence[IndexMessage]) -> list[IndexMessage]:
    """依時間排序；同一時間再依 id 排，讓結果每次都一樣（方便測試與增量更新）。"""
    return sorted(messages, key=lambda message: (as_utc(message.createdAt), message.id))


def format_line(message: IndexMessage) -> str:
    """把一則訊息排成片段裡的一行：`[09-18 21:03] 小美：內容`。

    加上台灣時間讓模型知道事情發生的先後，加上暱稱讓模型知道是誰說的。
    內容壓成單行，避免訊息裡的換行把片段格式打亂。
    """
    return f"[{taipei_label(message.createdAt)}] {message.senderName}：{single_line(message.content)}"


def _make_chunk(messages: list[IndexMessage]) -> ChunkDraft:
    """把一組訊息組成片段文字，並估計 token 數。"""
    content = "\n".join(format_line(message) for message in messages)
    return ChunkDraft(messages=list(messages), content=content, token_estimate=estimate_tokens(content))


def build_chunks(messages: Sequence[IndexMessage], now: datetime | None = None) -> list[ChunkDraft]:
    """把一個聊天室的訊息切成片段。

    逐則加入目前的片段，遇到下列情況就先把目前片段收起來：
    1. 跟上一則相隔超過 30 分鐘 → 新片段從這則開始（不重疊）。
    2. 目前片段已有 12 則，或加入這則會超過 400 tokens → 新片段先放入上一段最後 2 則，
       再放這則；若連重疊的 2 則加上這則都放不下（例如超長訊息），就放棄重疊。
    最後一段的最後一則若在 30 分鐘內（相對於 now），標記為 is_open：對話可能還在進行，
    後端之後要刪掉這段、從它的第一則開始重新切。
    """
    ordered = sort_messages(messages)
    chunks: list[ChunkDraft] = []
    current: list[IndexMessage] = []
    current_tokens = 0
    for message in ordered:
        tokens = estimate_tokens(format_line(message))
        if current:
            gap = as_utc(message.createdAt) - as_utc(current[-1].createdAt)
            if gap > SESSION_GAP:
                chunks.append(_make_chunk(current))
                current, current_tokens = [], 0
            elif len(current) >= MAX_MESSAGES or current_tokens + tokens > MAX_TOKENS:
                chunks.append(_make_chunk(current))
                overlap = current[-OVERLAP:]
                overlap_tokens = sum(estimate_tokens(format_line(item)) for item in overlap)
                if overlap_tokens + tokens > MAX_TOKENS:
                    overlap, overlap_tokens = [], 0
                current, current_tokens = list(overlap), overlap_tokens
        current.append(message)
        current_tokens += tokens
    if current:
        chunks.append(_make_chunk(current))
    if chunks:
        reference = as_utc(now) if now else datetime.now(timezone.utc)
        last_at = as_utc(chunks[-1].messages[-1].createdAt)
        chunks[-1].is_open = reference - last_at <= SESSION_GAP
    return chunks
