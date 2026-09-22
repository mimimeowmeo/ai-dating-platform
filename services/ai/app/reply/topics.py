"""AI 話題區段：找出「AI 推薦開啟了新話題」之後、仍在同一個話題的那段對話。規格 5.3。

為什麼需要：A 用 AI 推薦開了一個話題（例如「你喜歡爬山嗎？」），之後的聊天內容雖然是
真人打的（寫法是真的），但「聊爬山」這件事不是他自己選的。萃取風格卡時，這段期間發起者的
話題喜好要降權。對方的回覆不標記（使用者 2026-09-22 決定）。

偵測方式（取代「後 30 則關鍵字比對」，因為平均 7.1 字的短訊息大多不含關鍵字）：
1. 起點：來源是 AI、而且用途屬於「開新話題」（提問、邀約、呼應舊話題、分享）的訊息。
   用途是「回答」或「幽默」的 AI 訊息只是在回應對方，話題不是它開的。
2. 終點（先發生的就算結束）：
   - 兩則訊息間隔超過 30 分鐘（time_gap）。
   - 又出現另一則開新話題的 AI 訊息（topic_shift）。
   - 每 3 則一組算向量，連續 2 組跟起點的相似度都低於 0.5（topic_shift）。
   - 已滿 30 則（cap）。
3. 沒有向量（沒設定 Gemini 或向量化失敗）時，只用時間間隔與上限判斷。
"""

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal, Sequence

from .schemas import IndexMessage, TopicSpanOut
from .textutil import as_utc, single_line

DETECTOR_VERSION = "topic-span-v1"
OPENING_INTENTS = frozenset({"question", "plan", "callback", "share"})
AI_ORIGINS = frozenset({"ai_verbatim", "ai_edited"})
MAX_SPAN = 30
WINDOW = 3
SESSION_GAP = timedelta(minutes=30)
SHIFT_THRESHOLD = 0.5
SHIFT_PATIENCE = 2

EndReason = Literal["time_gap", "topic_shift", "cap", "end_of_data"]


@dataclass(frozen=True)
class SpanPlan:
    """一個候選區段的「計畫」：還沒看話題相似度前，依時間與上限能延伸到哪裡。

    start：開啟話題的 AI 訊息位置；limit：最遠可以延伸到的位置（含）；
    limit_reason：延伸停在 limit 的原因；windows：每 3 則一組的 [開始, 結束) 位置。
    """

    start: int
    limit: int
    limit_reason: EndReason
    windows: tuple[tuple[int, int], ...]


def is_topic_opener(message: IndexMessage) -> bool:
    """判斷一則訊息是不是「AI 推薦開啟的新話題」。

    來源必須是 AI（原封不動或改過再送），而且用途屬於開新話題的那幾種。
    沒有用途資訊的 AI 訊息保守地不算，避免把整段對話都誤判成 AI 話題。
    """
    return message.origin in AI_ORIGINS and message.suggestionIntent in OPENING_INTENTS


def plan_span(messages: Sequence[IndexMessage], start: int) -> SpanPlan | None:
    """依時間間隔、下一個 AI 話題與 30 則上限，算出一個區段最遠能延伸到哪裡。

    messages 必須已依時間排序。起點後面沒有任何可以納入的訊息時回傳 None。
    """
    limit = start
    reason: EndReason = "end_of_data"
    for index in range(start + 1, len(messages)):
        if index - start > MAX_SPAN:
            reason = "cap"
            break
        gap = as_utc(messages[index].createdAt) - as_utc(messages[index - 1].createdAt)
        if gap > SESSION_GAP:
            reason = "time_gap"
            break
        if is_topic_opener(messages[index]):
            reason = "topic_shift"
            break
        limit = index
    if limit == start:
        return None
    windows = tuple((first, min(first + WINDOW, limit + 1)) for first in range(start + 1, limit + 1, WINDOW))
    return SpanPlan(start=start, limit=limit, limit_reason=reason, windows=windows)


def plan_spans(messages: Sequence[IndexMessage]) -> list[SpanPlan]:
    """找出所有開新話題的 AI 訊息，並為每一則建立區段計畫。"""
    plans: list[SpanPlan] = []
    for index, message in enumerate(messages):
        if is_topic_opener(message):
            plan = plan_span(messages, index)
            if plan:
                plans.append(plan)
    return plans


def similarity_texts(messages: Sequence[IndexMessage], plan: SpanPlan) -> list[str]:
    """產生要拿去向量化的文字：第一個是起點的 AI 訊息，後面是每 3 則一組的內容。"""
    texts = [single_line(messages[plan.start].content)]
    for first, end in plan.windows:
        texts.append(" / ".join(single_line(message.content) for message in messages[first:end]))
    return texts


def cosine(first: Sequence[float], second: Sequence[float]) -> float:
    """兩個向量的 cosine 相似度（-1～1）；任一向量長度為 0 時回傳 0。"""
    dot = sum(a * b for a, b in zip(first, second))
    norm = math.sqrt(sum(a * a for a in first)) * math.sqrt(sum(b * b for b in second))
    return 0.0 if norm == 0 else dot / norm


def resolve_span(
    messages: Sequence[IndexMessage], plan: SpanPlan, similarities: Sequence[float] | None
) -> TopicSpanOut | None:
    """依話題相似度決定區段真正的終點，回傳區段；話題一開始就被換掉時回傳 None。

    similarities 是每一組跟起點的相似度（跟 plan.windows 一一對應）；None 代表沒有向量，
    只能用 plan 算好的時間與上限當終點。連續 2 組低於門檻時，話題在「第一組低於門檻」
    的開頭就已經換了，所以終點是那一組的前一則。
    """
    end, reason = plan.limit, plan.limit_reason
    if similarities is not None:
        low_streak = 0
        for position, score in enumerate(similarities):
            low_streak = low_streak + 1 if score < SHIFT_THRESHOLD else 0
            if low_streak >= SHIFT_PATIENCE:
                first_low = plan.windows[position - SHIFT_PATIENCE + 1][0]
                end, reason = first_low - 1, "topic_shift"
                break
    if end <= plan.start:
        return None
    opener = messages[plan.start]
    return TopicSpanOut(
        initiatingMessageId=opener.id,
        initiatorId=opener.senderId,
        startMessageId=messages[plan.start + 1].id,
        endMessageId=messages[end].id,
        messageCount=end - plan.start,
        endReason=reason,
    )
