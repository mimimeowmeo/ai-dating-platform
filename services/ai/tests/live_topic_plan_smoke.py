"""用真的向量模型與真的推薦模型跑一次「話題安排」（2026-09-29），確認四種話題來源都排得出來、模型照著寫。

這支腳本會實際呼叫 Gemini 向量化與推薦模型（會花時間，也會用到額度），所以檔名刻意不以 test 開頭，
`unittest discover` 不會自動執行它，只在手動確認時使用。

執行（在 services/ai 目錄；環境變數與專案根目錄的 .env 相同，例如 GEMINI_API_KEY、OLLAMA_API_KEY）：
    .venv/bin/python -m tests.live_topic_plan_smoke
情境（每個都先印出話題安排，再印出模型寫的推薦）：
1. A、B 都有聊天紀錄：A 也聊過 2、A 的興趣 1、熱門 2。
2. B 沒有聊天紀錄：B 的話題改用 B 的標籤與自我介紹。
3. A 沒有聊天紀錄：「A 也聊過」與「A 的興趣」都用 A 的標籤與自我介紹，一起挑 3 個。
4. 兩邊都沒有聊天紀錄。
5. 追問（A 的問題 B 還沒回）：換個說法重問 1 則，新話題 4 則。
"""

import asyncio
import os
import time
from datetime import timedelta

from pydantic_ai import models

from app.config import Settings
from app.reply.embeddings import Embedder
from app.reply.prompts import describe_plan
from app.reply.schemas import ReplySuggestionRequest, StyleCard, StyleFacet
from app.reply.style import SITE_DEFAULT_STATS, usable_card
from app.reply.suggest import ReplySuggester, last_requester_question, sort_chat
from tests.reply_helpers import at, chat, profile

# reply_helpers 為了單元測試把真實模型呼叫關掉了；這支腳本就是要呼叫真的模型，所以打開。
models.ALLOW_MODEL_REQUESTS = True

NOW = at(60 * 24 * 30)


def topic(name: str, weight: float, days: float) -> StyleFacet:
    """一條話題特徵句，days 天前最後聊到。"""
    return StyleFacet(
        kind="topic", statement=f"聊到「{name}」會比較熱絡", weight=weight, lastSeenAt=NOW - timedelta(days=days)
    )


def chat_card(*facets: StyleFacet) -> StyleCard:
    """聊天紀錄夠多（120 則）的風格卡。"""
    return StyleCard(
        confidence="high", sampleSource="chat", messageCount=120, stats=SITE_DEFAULT_STATS, facets=list(facets)
    )


# B（小美）最近常聊爬山、手搖飲、寵物；攝影、咖啡、火鍋比較久以前聊的。
PARTNER_CARD = chat_card(
    topic("爬山", 1.0, 2), topic("手搖飲", 0.8, 1), topic("寵物生活", 0.7, 3),
    topic("攝影", 0.9, 30), topic("咖啡", 0.6, 20), topic("火鍋", 0.5, 40),
)
# A（阿明）常聊健行、手沖咖啡、羽球。
REQUESTER_CARD = chat_card(topic("健行", 0.9, 5), topic("手沖咖啡", 0.7, 2), topic("羽球", 1.0, 1))
REQUESTER = profile("阿明", bio="平常喜歡打羽球跟煮飯，最近想開始爬山", interests=["羽球", "料理"], hobbies=["攝影"])
PARTNER = profile("小美", bio="喜歡爬山跟拍照，週末常往山上跑喔", interests=["登山", "攝影"], foods=["火鍋"])
ASKED = [chat("B", "嗨嗨，很高興配對到你", 60 * 24 * 30 - 3), chat("A", "妳週末通常都在做什麼？", 60 * 24 * 30 - 2)]


class CountingEmbedder(Embedder):
    """跟正式的向量服務一樣，另外記下呼叫次數與文字段數（看快取有沒有省下呼叫）。"""

    calls = 0
    texts = 0

    async def embed(self, texts, purpose="document", title=None):
        """累計次數後照常呼叫 Gemini。"""
        CountingEmbedder.calls += 1
        CountingEmbedder.texts += len(texts)
        return await super().embed(texts, purpose, title)


async def main() -> None:
    """依序跑五種情境，印出話題安排與推薦結果。"""
    settings = Settings.from_env()
    suggester = ReplySuggester(settings, embedder=CountingEmbedder(settings))
    base = dict(requester=REQUESTER, partner=PARTNER)
    scenarios = [
        ("1. A、B 都有聊天紀錄（開場）", dict(base, requesterStyle=REQUESTER_CARD, partnerStyle=PARTNER_CARD)),
        ("2. B 沒有聊天紀錄（開場）", dict(base, requesterStyle=REQUESTER_CARD)),
        ("3. A 沒有聊天紀錄（開場）", dict(base, partnerStyle=PARTNER_CARD)),
        ("4. 兩邊都沒有聊天紀錄（開場）", dict(base)),
        ("5. 追問：A 的問題 B 還沒回", dict(base, requesterStyle=REQUESTER_CARD, partnerStyle=PARTNER_CARD, recentMessages=ASKED)),
    ]
    for title, values in scenarios:
        print(f"\n===== {title} =====", flush=True)
        request = ReplySuggestionRequest(requestId="live-topic-plan", now=NOW, **values)
        reask = last_requester_question(sort_chat(list(request.recentMessages))) is not None
        started = time.perf_counter()
        plan = await suggester.planner.plan(
            request,
            usable_card(request.requesterStyle, request.requester, SITE_DEFAULT_STATS),
            usable_card(request.partnerStyle, request.partner, SITE_DEFAULT_STATS),
            NOW,
            reask,
        )
        print(f"話題安排（{time.perf_counter() - started:.2f} 秒；向量化累計 {CountingEmbedder.calls} 次、{CountingEmbedder.texts} 段）")
        print(describe_plan(plan))
        started = time.perf_counter()
        result = await suggester.generate(request)
        print(f"推薦：{time.perf_counter() - started:.1f} 秒｜模型 {result.modelName}｜狀態 {result.status}｜模式 {result.mode}")
        for item in result.suggestions:
            print(f"  {item.rank}. [{item.intent}] {item.text}（依據：{item.reason}）")
        for item in result.rejected:
            print(f"  （刪除：{item.reasonCode}）{item.text}")
        if result.notice:
            print("提示：", result.notice)
    print(f"\n向量化總計 {CountingEmbedder.calls} 次呼叫、{CountingEmbedder.texts} 段文字（同一段文字只轉一次）")


if __name__ == "__main__":
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    asyncio.run(main())
