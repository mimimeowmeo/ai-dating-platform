import unittest
from datetime import timedelta

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import FunctionModel

from app.reply.errors import AIServiceError
from app.reply.schemas import ReplySuggestionRequest, RetrievedChunk, StyleCard, StyleFacet, StyleStats
from app.reply.style import SITE_DEFAULT_STATS, usable_card
from app.reply.suggest import (
    ReplySuggester,
    detect_mode,
    has_topic_basis,
    last_requester_question,
    partner_asked_question,
)
from tests.reply_helpers import SETTINGS, MappedEmbedder, at, chat, direction, json_model, profile


def draft(text, intent="question", priority=2, reason="依據"):
    return {"text": text, "intent": intent, "priority": priority, "reason": reason}


def card(median, particles=None, emoji=0.0, messages=50, facets=None) -> StyleCard:
    return StyleCard(
        confidence="high" if messages >= 30 else "low",
        sampleSource="chat" if messages >= 30 else "mixed",
        messageCount=messages,
        stats=StyleStats(
            messageCount=messages, medianChars=median, meanChars=median, emojiPerMessage=emoji, questionRatio=0.2,
            exclamationRatio=0.0, laughterRatio=0.3 if emoji else 0.0, particles=particles or {},
        ),
        facets=facets or [],
    )


def facet(kind, statement, weight=0.5, minutes=None) -> StyleFacet:
    return StyleFacet(kind=kind, statement=statement, weight=weight, lastSeenAt=at(minutes) if minutes is not None else None)


# B 在所有聊天室的特徵句：登山權重最高但比較舊，咖啡權重低但最近才聊（10 天後）。
PARTNER_FACETS = [
    facet("topic", "聊到「登山」會比較熱絡", 1.0, 0),
    facet("topic", "聊到「咖啡」會比較熱絡", 0.3, 60 * 24 * 10),
    facet("avoid", "對「工作」話題比較冷淡", 0.8, 0),
]


def block(prompt: str, title: str) -> str:
    """取出 prompt 裡某個【區塊】的內容（到下一個空行為止）。"""
    start = prompt.index(f"【{title}】\n")  # 區塊標題自成一行；【模式】的說明裡也會提到區塊名稱
    end = prompt.find("\n\n", start)
    return prompt[start : end if end != -1 else None]


def make_request(**overrides) -> ReplySuggestionRequest:
    values = dict(
        requestId="req-1",
        requester=profile("阿明", bio="喜歡打羽球和煮飯", interests=["羽球"]),
        partner=profile("小美", bio="週末常去爬山，也喜歡貓", interests=["登山"]),
        sharedTags=["戶外活動"],
        recentMessages=[chat("A", "嗨嗨", 0), chat("B", "哈囉～你週末都在幹嘛？", 1)],
        requesterStyle=card(12, {"欸": 0.4}),
        partnerStyle=card(5, {"啦": 0.5}, emoji=1.0),
        now=at(5),
    )
    values.update(overrides)
    return ReplySuggestionRequest(**values)


def bare_request(**overrides) -> ReplySuggestionRequest:
    """完全沒有資料根據的請求：聊天室沒有訊息、沒有共同標籤、雙方檔案只有暱稱與城市。"""
    values = dict(
        recentMessages=[],
        sharedTags=[],
        requesterStyle=None,
        partnerStyle=None,
        requester=profile("阿明", city="台北", age=30),
        partner=profile("小美", city="台北", age=28),
    )
    values.update(overrides)
    return make_request(**values)


DRAFTS = [
    draft("我週末通常去打羽球欸，妳呢", intent="answer", priority=1),
    draft("爬山啦😂", intent="question", priority=2),
    draft("周末一起去爬山的话要带什么", intent="plan", priority=4),
    draft("打給我 0912345678", intent="share", priority=3),
    draft("我週末通常去打羽球欸，妳呢？", intent="answer", priority=2),
]


class ModeTests(unittest.TestCase):
    def test_detect_mode(self):
        self.assertEqual(detect_mode([], at(0)), "opener")
        self.assertEqual(detect_mode([chat("A", "嗨", 0), chat("B", "嗨", 1)], at(2)), "reply")
        self.assertEqual(detect_mode([chat("B", "嗨", 0), chat("A", "嗨", 1)], at(2)), "follow_up")
        # 2026-09-27：最後一則超過 12 小時就是重啟，不論是誰傳的。
        self.assertEqual(detect_mode([chat("B", "你呢？", 0)], at(0) + timedelta(hours=11)), "reply")
        self.assertEqual(detect_mode([chat("B", "你呢？", 0)], at(0) + timedelta(hours=13)), "revive")
        self.assertEqual(detect_mode([chat("A", "嗨", 0)], at(0) + timedelta(hours=13)), "revive")

    def test_partner_asked_question_only_looks_after_last_a_message(self):
        self.assertTrue(partner_asked_question([chat("A", "嗨", 0), chat("B", "你呢？", 1)]))
        self.assertFalse(partner_asked_question([chat("B", "你呢？", 0), chat("A", "我還好", 1)]))

    def test_last_requester_question_is_the_unanswered_one(self):
        messages = [chat("B", "嗨", 0), chat("A", "妳週末都做什麼？", 1), chat("A", "我最近在學做菜", 2)]
        self.assertEqual(last_requester_question(messages), "妳週末都做什麼？")
        self.assertIsNone(last_requester_question([*messages, chat("B", "爬山啊", 3)]))  # B 回了
        self.assertIsNone(last_requester_question([chat("B", "嗨", 0), chat("A", "我最近在學做菜", 1)]))

    def test_topic_basis_counts_conversation_profiles_tags_and_facets(self):
        bare = bare_request()
        partner_card = usable_card(None, bare.partner, SITE_DEFAULT_STATS)
        # 暱稱、年齡、城市這類人人都有的基本資料不算根據。
        self.assertFalse(has_topic_basis(bare, [], partner_card))
        # 下面任何一項有內容就算有根據。
        self.assertTrue(has_topic_basis(bare, [chat("B", "嗨", 0)], partner_card))
        self.assertTrue(has_topic_basis(bare.model_copy(update={"sharedTags": ["登山"]}), [], partner_card))
        self.assertTrue(has_topic_basis(bare.model_copy(update={"conversationSummary": "兩人聊過登山"}), [], partner_card))
        self.assertTrue(
            has_topic_basis(bare.model_copy(update={"requester": profile("阿明", occupation="工程師")}), [], partner_card)
        )
        self.assertTrue(
            has_topic_basis(bare.model_copy(update={"partner": profile("小美", foods=["火鍋"])}), [], partner_card)
        )


class SuggesterTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_pipeline_filters_converts_and_ranks(self):
        prompts: list[str] = []
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}, captured=prompts))
        result = await suggester.generate(make_request())

        self.assertEqual((result.status, result.mode, result.notice), ("ok", "reply", None))
        texts = [item.text for item in result.suggestions]
        self.assertEqual(texts[0], "我週末通常去打羽球欸，妳呢")  # B 剛問了問題 → 回答類排第 1
        self.assertEqual(texts[1], "爬山啦😂")
        self.assertIn("帶什麼", texts[2])  # 簡體被轉成台灣繁體
        self.assertEqual([item.rank for item in result.suggestions], [1, 2, 3])
        self.assertEqual(
            sorted(item.reasonCode for item in result.rejected), ["CONTACT_PHONE", "DUPLICATE"]
        )
        # 回覆模式：整批都是 A 80%／B 20%（0.8 × 12 + 0.2 × 5），沒有哪一則是 B 100%。
        self.assertEqual({item.styleTarget for item in result.suggestions}, {"blend"})
        self.assertEqual(
            (result.target.rule, result.target.source, result.target.stats.medianChars), ("reply", "blend", 10.6)
        )
        self.assertEqual((result.modelName, result.promptVersion, result.usage.requests), ("fake-model", "reply-v4", 1))

        prompt = prompts[0]
        self.assertIn("【模式】reply", prompt)
        self.assertIn("你週末都在幹嘛", prompt)
        self.assertIn("【寫法目標（A 為主、帶一點 B）】", prompt)
        self.assertNotIn("第 1 則", prompt)
        # 回覆模式維持原本的邏輯：雙方完整檔案、寫法目標含問句比例；但不再寫字數。
        self.assertIn("自我介紹：週末常去爬山", prompt)
        self.assertIn("是問句", block(prompt, "寫法目標（A 為主、帶一點 B）"))
        self.assertNotIn("字；", block(prompt, "寫法目標（A 為主、帶一點 B）"))

    async def test_opener_uses_partner_style_and_ranks_questions_first(self):
        prompts: list[str] = []
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}, captured=prompts))
        result = await suggester.generate(make_request(recentMessages=[]))

        self.assertEqual((result.status, result.mode), ("ok", "opener"))
        # 開場：整批都是 B 100%，不是只有第 1 名。
        self.assertEqual([item.styleTarget for item in result.suggestions], ["partner", "partner", "partner"])
        self.assertEqual(
            (result.target.rule, result.target.source, result.target.stats.medianChars), ("opener", "partner", 5)
        )
        # 開場要用問句引導 B 分享：問句排前面（「爬山啦😂」優先度 2，但不是問句，排到最後）。
        self.assertEqual(
            [item.text for item in result.suggestions],
            ["我週末通常去打羽球欸，妳呢", "週末一起去爬山的話要帶什麼", "爬山啦😂"],
        )
        prompt = prompts[0]
        self.assertIn("還沒有任何訊息", prompt)
        self.assertIn("用問句引導 B 分享", prompt)
        self.assertIn("【寫法目標（每一則都完全照 B 喜歡的樣子寫，不要像 A）】", prompt)
        self.assertNotIn("是問句", block(prompt, "寫法目標（每一則都完全照 B 喜歡的樣子寫，不要像 A）"))

    async def test_style_ratio_follows_mode(self):
        # 2026-09-27：回覆 A 80%／B 20%；追問只用 A 自己的語氣；重啟（超過 12 小時）照 B。
        cases = (
            ([chat("B", "哈囉～", 0)], at(5), "reply", "blend", "blend"),
            ([chat("A", "嗨，很高興配對到你", 0)], at(5), "follow_up", "requester", "blend"),
            ([chat("B", "哈囉～", 0)], at(0) + timedelta(hours=13), "revive", "partner", "partner"),
        )
        for messages, now, mode, source, label in cases:
            prompts: list[str] = []
            suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}, captured=prompts))
            result = await suggester.generate(make_request(recentMessages=messages, now=now))
            self.assertEqual((result.mode, result.target.rule, result.target.source), (mode, mode, source))
            self.assertEqual({item.styleTarget for item in result.suggestions}, {label})
            if mode == "follow_up":
                self.assertIn("【寫法目標（用 A 自己的寫法）】", prompts[0])

    async def test_falls_back_to_requester_style_when_partner_unknown(self):
        prompts: list[str] = []
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}, captured=prompts))
        result = await suggester.generate(
            make_request(recentMessages=[], partnerStyle=None, partner=profile("小美", bio="嗨"))
        )
        # B 沒聊過天、bio 也太短：就算是開場，也只能用 A 的寫法，標成 blend。
        self.assertEqual(
            (result.target.rule, result.target.source, result.target.stats.medianChars), ("opener", "requester", 12)
        )
        self.assertEqual({item.styleTarget for item in result.suggestions}, {"blend"})
        self.assertIn("【寫法目標（B 沒有足夠資料，改用 A 的寫法）】", prompts[0])

    async def test_topic_modes_plan_partner_chat_topics_when_history_is_enough(self):
        prompts: list[str] = []
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}, captured=prompts))
        partner_facets = [
            *PARTNER_FACETS[:1],
            facet("topic", "聊到「咖啡」會比較熱絡", 0.3, 0),
            facet("topic", "聊到「貓」會比較熱絡", 0.8, 0),
            facet("topic", "聊到「電影」會比較熱絡", 0.6, 0),
            facet("topic", "聊到「桌遊」會比較熱絡", 0.5, 0),
            PARTNER_FACETS[2],
        ]
        await suggester.generate(make_request(recentMessages=[], partnerStyle=card(5, messages=50, facets=partner_facets)))
        prompt = prompts[0]
        plan = block(prompt, "這次的話題安排")
        # A 跟 B 沒有相近的話題（也沒有向量服務）：5 則都是 B 的熱門話題，依熱度排。
        lines = plan.splitlines()[2:]
        self.assertEqual([line.split("「")[1].split("」")[0] for line in lines], ["登山", "貓", "電影", "桌遊", "咖啡"])
        self.assertTrue(all("B 跟別人聊到" in line and "A 不知道" in line for line in lines))
        self.assertIn("（共 5 則新話題", plan)
        self.assertIn("對「工作」話題比較冷淡", block(prompt, "B 的寫法與喜好"))
        # B 的聊天紀錄夠多、話題也夠：不放 B 的主頁原文與共同標籤，只留基本資料。
        partner_profile = block(prompt, "B 的檔案（聊天對象）")
        self.assertIn("暱稱：小美", partner_profile)
        self.assertNotIn("自我介紹", partner_profile)
        self.assertNotIn("【共同標籤】", prompt)
        self.assertNotIn("【B 最近在聊的話題】", prompt)
        # A 的風格卡沒有話題特徵句：A 那邊用檔案（標籤與自我介紹）。
        self.assertIn("自我介紹：喜歡打羽球和煮飯", prompt)

    async def test_topic_modes_use_profiles_when_history_is_short(self):
        prompts: list[str] = []
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}, captured=prompts))
        partner_style = card(5, messages=10, facets=PARTNER_FACETS)  # 只有 10 則真人訊息
        requester_style = card(12, messages=40, facets=[facet("topic", "聊到「羽球」會比較熱絡", 1.0, 5)])
        await suggester.generate(
            make_request(recentMessages=[], partnerStyle=partner_style, requesterStyle=requester_style)
        )
        prompt = prompts[0]
        # B 的聊天紀錄太少：B 的話題改用檔案上的標籤與自我介紹（2026-09-29 使用者補充）。
        plan = block(prompt, "這次的話題安排")
        self.assertIn("1～5. 從【B 的檔案】挑 5 個", plan)
        self.assertIn("可以挑：「登山」、B 的自我介紹", plan)
        self.assertIn("不要硬湊", plan)
        self.assertIn("自我介紹：週末常去爬山", block(prompt, "B 的檔案（聊天對象）"))
        self.assertNotIn("聊到「登山」", plan)  # B 的聊天話題（紀錄太少）不用
        self.assertNotIn("【共同標籤】", prompt)
        # A 的聊天紀錄夠多，這次的安排也沒用到 A 的檔案：A 那邊只放基本資料。
        self.assertNotIn("自我介紹：喜歡打羽球和煮飯", prompt)

    async def test_plan_matches_similar_topics_with_vectors(self):
        prompts: list[str] = []
        vectors = {"登山": direction(0), "爬山": direction(0, 15), "咖啡": direction(6), "羽球": direction(2)}
        embedder = MappedEmbedder(vectors)
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}, captured=prompts), embedder=embedder)
        requester_style = card(12, messages=40, facets=[facet("topic", "聊到「爬山」會比較熱絡", 1.0, 5)])
        partner_style = card(5, messages=50, facets=PARTNER_FACETS)
        await suggester.generate(make_request(recentMessages=[], partnerStyle=partner_style, requesterStyle=requester_style))
        plan = block(prompts[0], "這次的話題安排")
        # 登山 ↔ 爬山 相似度 0.97 ≥ 0.88：排在第 1 格「A 也聊過」，要先分享 A 的經驗再問。
        self.assertIn("1. B 跟別人聊到「登山」時比較熱絡", plan)
        self.assertIn("A 自己也常聊「爬山」：先用一句話分享", plan)
        self.assertIn("2. B 跟別人聊到「咖啡」", plan)
        # 只送短詞（標題「話題」）；A 的自我介紹不到 10 個字，不算話題，不必轉向量。
        self.assertEqual(embedder.calls, [(["咖啡", "爬山", "登山", "羽球"], "document", "話題")])

    async def test_plan_limits_how_many_suggestions_are_kept(self):
        # B 只有 2 個聊天話題、檔案也沒有可以聊的：只留 2 則（不硬湊），多寫的記成 OVER_LIMIT。
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}))
        partner_style = card(5, messages=50, facets=PARTNER_FACETS)
        result = await suggester.generate(
            make_request(recentMessages=[], partnerStyle=partner_style, partner=profile("小美", bio="嗨"))
        )
        self.assertEqual((result.status, len(result.suggestions)), ("partial", 2))
        self.assertIn("OVER_LIMIT", [item.reasonCode for item in result.rejected])

    async def test_reply_mode_does_not_plan_topics(self):
        prompts: list[str] = []
        embedder = MappedEmbedder({})
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}, captured=prompts), embedder=embedder)
        result = await suggester.generate(make_request(partnerStyle=card(5, messages=50, facets=PARTNER_FACETS)))
        self.assertEqual(result.mode, "reply")
        self.assertEqual(embedder.calls, [])
        self.assertNotIn("【這次的話題安排】", prompts[0])

    async def test_follow_up_rephrases_the_unanswered_question(self):
        prompts: list[str] = []
        drafts = [
            draft("妳週末都做什麼？", intent="reask", priority=1),  # 照抄原句 → 刪掉
            draft("那妳週末通常怎麼過呀？", intent="reask", priority=2),
            draft("妳最近有去哪裡爬山嗎？", intent="question", priority=1),
        ]
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": drafts}, captured=prompts))
        messages = [chat("B", "嗨嗨", 0), chat("A", "妳週末都做什麼？", 1)]
        result = await suggester.generate(make_request(recentMessages=messages))

        self.assertEqual(result.mode, "follow_up")
        self.assertIn("【A 上一則還沒得到回答的問題】\n妳週末都做什麼？", prompts[0])
        self.assertEqual([item.reasonCode for item in result.rejected], ["REPEATS_LAST_QUESTION"])
        self.assertEqual({item.intent for item in result.suggestions}, {"reask", "question"})

    async def test_follow_up_keeps_both_kinds_within_five(self):
        # 5 則新話題的優先度都比重問高：排序後前 5 名全是新話題，要把最好的重問換進第 5 名。
        topics = ["妳最近有去哪裡爬山嗎？", "最近有看什麼好看的電影嗎？", "妳平常喜歡喝什麼咖啡？", "週末有推薦的早午餐店嗎？", "最近有在追什麼劇嗎？"]
        drafts = [draft(text, intent="question", priority=1) for text in topics]
        drafts.append(draft("那妳週末通常怎麼過呀？", intent="reask", priority=5))
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": drafts}))
        messages = [chat("B", "嗨嗨", 0), chat("A", "妳週末都做什麼？", 1)]
        result = await suggester.generate(make_request(recentMessages=messages))

        self.assertEqual([item.intent for item in result.suggestions], ["question"] * 4 + ["reask"])
        self.assertEqual([item.rank for item in result.suggestions], [1, 2, 3, 4, 5])
        self.assertEqual([item.reasonCode for item in result.rejected], ["OVER_LIMIT"])

    async def test_revive_uses_opener_method_instead_of_answering(self):
        drafts = [
            draft("我週末都在打羽球欸", intent="answer", priority=1),
            draft("妳最近有去哪裡爬山嗎？", intent="question", priority=2),
        ]
        chunk = RetrievedChunk(content="[09-01 12:00] 小美：我喜歡爬山", lastAt=at(0))
        messages = [chat("A", "嗨", 0), chat("B", "你週末都在幹嘛？", 1)]
        for now, mode, first in (
            (at(5), "reply", "我週末都在打羽球欸"),  # 回覆：B 剛問問題 → 回答類第 1
            (at(1) + timedelta(hours=13), "revive", "妳最近有去哪裡爬山嗎？"),  # 重啟：改用開場做法，問句優先
        ):
            prompts: list[str] = []
            suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": drafts}, captured=prompts))
            result = await suggester.generate(make_request(recentMessages=messages, now=now, retrievedChunks=[chunk]))
            self.assertEqual((result.mode, result.suggestions[0].text), (mode, first))
            if mode == "revive":
                self.assertIn("超過 12 小時", prompts[0])
                self.assertNotIn("相關的舊對話片段", prompts[0])
            else:
                self.assertIn("相關的舊對話片段", prompts[0])

    async def test_exclude_texts_and_partial_and_empty(self):
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}))
        result = await suggester.generate(make_request(excludeTexts=["爬山啦😂", "我週末通常去打羽球欸，妳呢"]))
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.notice, "只找到 1 則合適的建議")
        self.assertIn("SIMILAR_TO_EXCLUDED", [item.reasonCode for item in result.rejected])

        unsafe = [draft("最近有個投資機會"), draft("先轉帳給我"), draft("加我賴啦")]
        result = await ReplySuggester(SETTINGS, model=json_model({"suggestions": unsafe})).generate(make_request())
        self.assertEqual((result.status, result.suggestions), ("empty", []))
        self.assertEqual(result.notice, "沒有可推薦的句子")

    async def test_model_may_return_fewer_or_none_without_being_forced(self):
        # 以前少於 3 則會逼模型重寫（硬湊）；現在照單全收，一次請求就結束。
        few = await ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS[:2]})).generate(make_request())
        self.assertEqual((few.status, few.notice, few.usage.requests), ("partial", "只找到 2 則合適的建議", 1))
        none = await ReplySuggester(SETTINGS, model=json_model({"suggestions": []})).generate(make_request())
        self.assertEqual((none.status, none.suggestions, none.notice), ("empty", [], "沒有可推薦的句子"))
        self.assertEqual(none.usage.requests, 1)

    async def test_no_topic_basis_returns_notice_without_calling_model(self):
        prompts: list[str] = []
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}, captured=prompts))
        result = await suggester.generate(bare_request())
        self.assertEqual(prompts, [])  # 完全沒有資料根據：不呼叫模型
        self.assertEqual((result.status, result.mode, result.suggestions), ("empty", "opener", []))
        self.assertEqual(result.notice, "沒有可推薦的句子")
        self.assertEqual((result.modelName, result.usage.requests, result.promptVersion), (None, 0, "reply-v4"))

        # 只要有一項根據（這裡是 B 的興趣），就照常請模型產生；只有一個話題，所以只留 1 則（不硬湊）。
        result = await suggester.generate(bare_request(partner=profile("小美", interests=["登山"])))
        self.assertEqual((len(prompts), result.status, len(result.suggestions)), (1, "partial", 1))

    async def test_not_configured_and_unavailable_errors(self):
        with self.assertRaises(AIServiceError) as caught:
            await ReplySuggester(SETTINGS).generate(make_request())
        self.assertEqual(caught.exception.code, "LLM_NOT_CONFIGURED")

        def rate_limited(messages, info):
            raise ModelHTTPError(status_code=429, model_name="gemini", body={"error": "quota"})

        with self.assertRaises(AIServiceError) as caught:
            await ReplySuggester(SETTINGS, model=FunctionModel(rate_limited)).generate(make_request())
        self.assertEqual(caught.exception.code, "LLM_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
