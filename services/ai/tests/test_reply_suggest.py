import unittest
from datetime import timedelta

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import FunctionModel

from app.reply.errors import AIServiceError
from app.reply.schemas import ReplySuggestionRequest, StyleCard, StyleStats
from app.reply.style import SITE_DEFAULT_STATS, usable_card
from app.reply.suggest import ReplySuggester, detect_mode, has_topic_basis, partner_asked_question
from tests.reply_helpers import SETTINGS, at, chat, json_model, profile


def draft(text, intent="question", priority=2, reason="依據"):
    return {"text": text, "intent": intent, "priority": priority, "reason": reason}


def card(median, particles=None, emoji=0.0) -> StyleCard:
    return StyleCard(
        confidence="high",
        sampleSource="chat",
        messageCount=50,
        stats=StyleStats(
            messageCount=50, medianChars=median, meanChars=median, emojiPerMessage=emoji, questionRatio=0.2,
            exclamationRatio=0.0, laughterRatio=0.3 if emoji else 0.0, particles=particles or {},
        ),
    )


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
        self.assertEqual(detect_mode([chat("B", "嗨", 0)], at(0) + timedelta(days=8)), "revive")

    def test_partner_asked_question_only_looks_after_last_a_message(self):
        self.assertTrue(partner_asked_question([chat("A", "嗨", 0), chat("B", "你呢？", 1)]))
        self.assertFalse(partner_asked_question([chat("B", "你呢？", 0), chat("A", "我還好", 1)]))

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
        # 聊天室已經有訊息：整批都是 A 80%／B 20%（0.8 × 12 + 0.2 × 5），沒有哪一則是 B 100%。
        self.assertEqual({item.styleTarget for item in result.suggestions}, {"blend"})
        self.assertEqual(
            (result.target.rule, result.target.source, result.target.stats.medianChars), ("later", "blend", 10.6)
        )
        self.assertEqual((result.modelName, result.promptVersion, result.usage.requests), ("fake-model", "reply-v2", 1))

        prompt = prompts[0]
        self.assertIn("【模式】reply", prompt)
        self.assertIn("你週末都在幹嘛", prompt)
        self.assertIn("【寫法目標（A 為主、帶一點 B）】", prompt)
        self.assertNotIn("第 1 則", prompt)
        self.assertIn("自我介紹：週末常去爬山", prompt)

    async def test_chat_first_message_uses_partner_style_for_every_suggestion(self):
        prompts: list[str] = []
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}, captured=prompts))
        result = await suggester.generate(make_request(recentMessages=[]))

        self.assertEqual((result.status, result.mode), ("ok", "opener"))
        # 要寫整個聊天室的第一則訊息：整批都是 B 100%，不是只有第 1 名。
        self.assertEqual([item.styleTarget for item in result.suggestions], ["partner", "partner", "partner"])
        self.assertEqual(
            (result.target.rule, result.target.source, result.target.stats.medianChars),
            ("first_message", "partner", 5),
        )
        self.assertIn("還沒有任何訊息", prompts[0])
        self.assertIn("【寫法目標（聊天室的第一則訊息：每一則都完全照 B 喜歡的樣子寫，不要像 A）】", prompts[0])

    async def test_partner_rule_stops_once_anyone_sent_the_first_message(self):
        # 「第一則」指整個聊天室的第一則，不是 A 或 B 各自的第一則：
        # A 先傳了開場白再按推薦（追問），或 B 先傳、A 要回，都不再套用 B 100%。
        for messages, mode in (
            ([chat("A", "嗨，很高興配對到你", 0)], "follow_up"),
            ([chat("B", "哈囉～", 0)], "reply"),
        ):
            suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}))
            result = await suggester.generate(make_request(recentMessages=messages))
            self.assertEqual((result.mode, result.target.rule), (mode, "later"))
            self.assertEqual({item.styleTarget for item in result.suggestions}, {"blend"})

    async def test_falls_back_to_requester_style_when_partner_unknown(self):
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}))
        result = await suggester.generate(
            make_request(recentMessages=[], partnerStyle=None, partner=profile("小美", bio="嗨"))
        )
        # B 沒聊過天、bio 也太短：就算是聊天室的第一則訊息，也只能用 A 的寫法，標成 blend。
        self.assertEqual(
            (result.target.rule, result.target.source, result.target.stats.medianChars),
            ("first_message", "requester", 12),
        )
        self.assertEqual({item.styleTarget for item in result.suggestions}, {"blend"})

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
        self.assertEqual((result.modelName, result.usage.requests, result.promptVersion), (None, 0, "reply-v2"))

        # 只要有一項根據（這裡是 B 的興趣），就照常請模型產生。
        result = await suggester.generate(bare_request(partner=profile("小美", interests=["登山"])))
        self.assertEqual((len(prompts), result.status), (1, "ok"))

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
