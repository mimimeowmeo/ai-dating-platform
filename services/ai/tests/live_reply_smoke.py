"""用「真的模型」跑一次推薦回覆的主要流程，確認 Ollama／Gemini 真的接上了。

這支腳本會實際呼叫模型（會花時間，也會用到 Ollama Cloud／Gemini 的額度），所以檔名刻意不以 test 開頭，
`unittest discover` 不會自動執行它，只在手動確認時使用。

執行（在 services/ai 目錄；環境變數與專案根目錄的 .env 相同，例如 OLLAMA_BASE_URL、OLLAMA_API_KEY、GEMINI_API_KEY）：
    .venv/bin/python -m tests.live_reply_smoke
有 Ollama 設定時：測風格卡萃取、聊天室摘要，以及只用萃取模型產生推薦。
有 GEMINI_API_KEY 時：再用預設的備援鏈（AI_REPLY_MODELS）產生推薦，並測 Gemini 向量化。
產生推薦時會跑五種情境（見 run_suggestions）：回覆、開場、追問、重啟，以及完全沒有資料根據。
"""

import asyncio
import os
import time
from dataclasses import replace

from pydantic_ai import models

from app.config import Settings
from app.reply.embeddings import Embedder
from app.reply.errors import AIServiceError
from app.reply.extraction import ConversationSummarizer, StyleProfileBuilder
from app.reply.schemas import ReplySuggestionRequest, StyleProfileRequest, SummaryRequest
from app.reply.suggest import ReplySuggester
from tests.reply_helpers import at, chat, indexed, own, profile

# reply_helpers 為了單元測試把真實模型呼叫關掉了；這支腳本就是要呼叫真的模型，所以打開。
models.ALLOW_MODEL_REQUESTS = True

# 模擬 B（小美）在其他聊天室自己發過的訊息：短句、愛用「啦／欸」、常用 emoji、喜歡爬山與手搖飲。
PARTNER_LINES = [
    "欸你週末有空嗎😆", "我超愛爬山啦", "上次去合歡山超冷", "哈哈哈笑死", "你喜歡喝什麼手搖",
    "我都喝烏龍拿鐵欸", "今天好熱喔🥵", "剛下班～", "你也太可愛了吧", "明天要早起爬山啦",
    "雲海真的超美😍", "你有養寵物嗎", "我家貓超黏人", "哈哈好喔", "晚安啦🌙",
    "欸這家咖啡不錯", "週末想去陽明山", "腳好痠😂", "你平常都幾點睡", "我是夜貓子欸",
    "好想吃火鍋", "你喜歡吃辣嗎", "我超怕辣哈哈", "這部電影好看嗎", "我比較喜歡看動畫",
    "欸你有看那部新番嗎", "真假😳", "哈哈哈哈", "好啦好啦", "下次一起去爬山？",
    "我最近在學攝影", "拍山景超療癒", "你呢你都拍什麼", "我覺得你很會聊欸", "哈哈謝謝",
    "今天工作好累", "但想到週末就開心了😆", "欸你會煮飯嗎", "我只會泡麵哈哈", "晚點聊啦",
]


def section(title: str) -> None:
    """印出區段標題，讓輸出比較好讀。"""
    print(f"\n===== {title} =====", flush=True)


async def check_ollama(settings: Settings):
    """用 Ollama 跑風格卡萃取與聊天室摘要，並用 Ollama 產生一次推薦（不需要 Gemini）。"""
    section(f"Ollama 風格卡萃取（{settings.extraction_models[0]}）")
    builder = StyleProfileBuilder(settings, embedder=Embedder(replace(settings, gemini_api_key="")))
    # 前一半放在 10 天前、後一半是最近：萃取會切成兩批，特徵句的 lastSeenAt 分得出新舊。
    messages = [
        own(text, index * 3 - (60 * 24 * 10 if index < len(PARTNER_LINES) // 2 else 0), in_ai_topic=index in (5, 6))
        for index, text in enumerate(PARTNER_LINES)
    ]
    started = time.perf_counter()
    result = await builder.build(StyleProfileRequest(userId="u-b", bio="喜歡爬山跟拍照，週末常往山上跑喔", messages=messages))
    card = result.card
    print(f"耗時 {time.perf_counter() - started:.1f} 秒｜模型 {card.modelName}｜信心 {card.confidence}｜樣本 {card.messageCount} 則")
    print("寫法統計：", card.stats.model_dump(exclude={"messageCount"}))
    print("語氣：", card.voiceNotes)
    for facet in card.facets:
        print(f"  [{facet.kind}] {facet.statement}（權重 {facet.weight}，最後出現 {facet.lastSeenAt}）")
    print("被抽象化檢查刪掉的特徵句數：", result.rejectedStatements)

    section("Ollama 聊天室摘要")
    summarizer = ConversationSummarizer(settings)
    started = time.perf_counter()
    summary = await summarizer.summarize(
        SummaryRequest(
            conversationId="c-demo",
            messages=[
                indexed("u-a", "嗨嗨，看到妳也喜歡爬山", 0),
                indexed("u-b", "對啊我超愛！上個月才去合歡山", 1),
                indexed("u-a", "我比較常打羽球，但一直想去爬山", 2),
                indexed("u-b", "那下次可以一起去陽明山啊😆", 3),
            ],
            maxChars=200,
        )
    )
    print(f"耗時 {time.perf_counter() - started:.1f} 秒｜模型 {summary.modelName}")
    print(summary.summary)

    section(f"只用萃取模型產生推薦（{settings.extraction_models[0]}，不經備援鏈）")
    reply_settings = replace(settings, reply_models=settings.extraction_models, llm_timeout_seconds=300)
    await run_suggestions(ReplySuggester(reply_settings, embedder=Embedder(settings)), card)
    return card


async def run_suggestions(suggester: ReplySuggester, partner_card) -> None:
    """用五種情境各產生一次推薦並印出結果，確認真的模型照規則走（2026-09-27 的新版規則）。

    1. 回覆（B 剛問了「你週末都在幹嘛？」）：整批 A 80%／B 20%，標 blend，回答類排第 1。
    2. 開場（聊天室還沒有任何訊息）：整批照 B 喜歡的樣子寫，標 partner；用問句引導 B 分享，
       話題照程式排好的話題安排（見 topic_plan；四種話題來源另見 tests.live_topic_plan_smoke）。
    3. 追問（A 問了問題、B 還沒回）：只用 A 的語氣；同時有「換個說法重問」（reask）與新話題。
    4. 重啟（最後一則超過 12 小時）：用開場的做法，整批照 B 喜歡的樣子寫。
    5. 完全沒有資料根據（沒有訊息、沒有共同標籤、雙方只有暱稱與城市）：
       不呼叫模型，直接回「沒有可推薦的句子」。
    """
    requester = profile("阿明", bio="平常喜歡打羽球跟煮飯，最近想開始爬山", interests=["羽球", "料理"])
    partner = profile("小美", bio="喜歡爬山跟拍照，週末常往山上跑喔", interests=["登山", "攝影"])
    conversation = [
        chat("A", "嗨嗨，看到妳也喜歡爬山", 0),
        chat("B", "對啊我超愛！", 1),
        chat("B", "你週末都在幹嘛？", 1.5),
    ]
    asked = [chat("B", "嗨嗨，很高興配對到你", 0), chat("A", "妳週末通常都在做什麼？", 1)]
    rich = dict(requester=requester, partner=partner, sharedTags=["戶外活動"], partnerStyle=partner_card)
    scenarios = [
        ("回覆：B 剛問了問題（預期 A 80%／B 20%，回答類第 1）", dict(rich, recentMessages=conversation), at(5)),
        ("開場：聊天室還沒有訊息（預期整批 B 100%、用問句）", dict(rich, recentMessages=[]), at(5)),
        ("追問：A 的問題 B 還沒回（預期只用 A 的語氣，有 reask 也有新話題）", dict(rich, recentMessages=asked), at(5)),
        ("重啟：超過 12 小時沒人說話（預期整批 B 100%、用開場做法）", dict(rich, recentMessages=conversation), at(60 * 13)),
        (
            "完全沒有資料根據（預期不呼叫模型）",
            dict(requester=profile("阿明", city="台北"), partner=profile("小美", city="台北")),
            at(5),
        ),
    ]
    for title, values, now in scenarios:
        print(f"--- {title}", flush=True)
        request = ReplySuggestionRequest(requestId="live-smoke", now=now, **values)
        started = time.perf_counter()
        result = await suggester.generate(request)
        print(f"耗時 {time.perf_counter() - started:.1f} 秒｜模型 {result.modelName}｜狀態 {result.status}｜模式 {result.mode}")
        print(f"寫法規則 {result.target.rule}／{result.target.source}")
        print(f"token：輸入 {result.usage.inputTokens}、輸出 {result.usage.outputTokens}、請求 {result.usage.requests} 次")
        for item in result.suggestions:
            print(f"  {item.rank}. [{item.styleTarget}/{item.intent}] {item.text}（風格距離 {item.styleDistance}）")
        for item in result.rejected:
            print(f"  （刪除：{item.reasonCode}）{item.text}")
        if result.notice:
            print("提示：", result.notice)


async def check_gemini(settings: Settings, partner_card) -> None:
    """有 GEMINI_API_KEY 時：用預設的備援鏈產生推薦（Gemini 是備援），並用 Gemini 做一次向量化。"""
    section(f"備援鏈產生推薦（{' → '.join(settings.reply_models)}）")
    await run_suggestions(ReplySuggester(settings, embedder=Embedder(settings)), partner_card)
    section(f"Gemini 向量化（{settings.embedding_model}）")
    vectors = await Embedder(settings).embed(["你喜歡爬山嗎", "我家的貓很黏人"], "query")
    print(f"取得 {len(vectors)} 個向量，每個 {len(vectors[0])} 維")


async def main() -> None:
    """依目前設定決定要測哪些模型；任何一段失敗都只印出固定錯誤代碼，不中斷其他段落。"""
    settings = Settings.from_env()
    card = None
    if settings.ollama_configured:
        try:
            card = await check_ollama(settings)
        except AIServiceError as error:
            print("Ollama 測試失敗：", error.code)
    else:
        print("略過 Ollama：沒有設定 OLLAMA_BASE_URL（Ollama Cloud 另外需要 OLLAMA_API_KEY）")
    if settings.gemini_configured:
        try:
            await check_gemini(settings, card)
        except AIServiceError as error:
            print("備援鏈／Gemini 向量化測試失敗：", error.code)
    else:
        print("略過 Gemini：沒有設定 GEMINI_API_KEY")


if __name__ == "__main__":
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    asyncio.run(main())
