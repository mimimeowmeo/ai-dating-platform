import unittest
from datetime import timedelta

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import FunctionModel

from app.reply.errors import AIServiceError
from app.reply.schemas import ReplySuggestionRequest, StyleCard, StyleStats
from app.reply.suggest import ReplySuggester, detect_mode, partner_asked_question
from tests.reply_helpers import SETTINGS, at, chat, json_model, profile


def draft(text, intent="question", priority=2, target="blend", reason="依據"):
    return {"text": text, "intent": intent, "priority": priority, "styleTarget": target, "reason": reason}


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


DRAFTS = [
    draft("我週末通常去打羽球欸，妳呢", intent="answer", priority=1),
    draft("爬山啦😂", intent="question", priority=2, target="partner"),
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


class SuggesterTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_pipeline_filters_converts_and_ranks(self):
        prompts: list[str] = []
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}, captured=prompts))
        result = await suggester.generate(make_request())

        self.assertEqual((result.status, result.mode, result.notice), ("ok", "reply", None))
        texts = [item.text for item in result.suggestions]
        self.assertEqual(texts[0], "爬山啦😂")  # 第 1 則：B 風格候選中最接近 B 的
        self.assertEqual(result.suggestions[0].styleTarget, "partner")
        self.assertEqual(result.suggestions[1].intent, "answer")  # B 剛問了問題 → 回答優先
        self.assertIn("帶什麼", texts[2])  # 簡體被轉成台灣繁體
        self.assertEqual([item.rank for item in result.suggestions], [1, 2, 3])
        self.assertEqual(
            sorted(item.reasonCode for item in result.rejected), ["CONTACT_PHONE", "DUPLICATE"]
        )
        self.assertEqual(result.targets.first.medianChars, 5)  # B 100%
        self.assertEqual(result.targets.others.medianChars, 10.6)  # 0.8 × 12 + 0.2 × 5
        self.assertEqual((result.modelName, result.promptVersion, result.usage.requests), ("fake-model", "reply-v1", 1))

        prompt = prompts[0]
        self.assertIn("【模式】reply", prompt)
        self.assertIn("你週末都在幹嘛", prompt)
        self.assertIn("【第 1 則的寫法目標（B 的寫法）】", prompt)
        self.assertIn("自我介紹：週末常去爬山", prompt)

    async def test_first_suggestion_falls_back_to_requester_when_partner_unknown(self):
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}))
        result = await suggester.generate(make_request(partnerStyle=None, partner=profile("小美", bio="嗨")))
        self.assertEqual(result.targets.firstSource, "requester")
        self.assertEqual(result.suggestions[0].styleTarget, "blend")

    async def test_exclude_texts_and_partial_and_empty(self):
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS}))
        result = await suggester.generate(make_request(excludeTexts=["爬山啦😂", "我週末通常去打羽球欸，妳呢"]))
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.notice, "只找到 1 則合適的建議")
        self.assertIn("SIMILAR_TO_EXCLUDED", [item.reasonCode for item in result.rejected])

        unsafe = [draft("最近有個投資機會"), draft("先轉帳給我"), draft("加我賴啦")]
        result = await ReplySuggester(SETTINGS, model=json_model({"suggestions": unsafe})).generate(make_request())
        self.assertEqual((result.status, result.suggestions), ("empty", []))
        self.assertEqual(result.notice, "這次沒有產生合適的建議，請再試一次")

    async def test_opener_mode_without_messages(self):
        prompts: list[str] = []
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS[:3]}, captured=prompts))
        result = await suggester.generate(make_request(recentMessages=[]))
        self.assertEqual(result.mode, "opener")
        self.assertIn("還沒有任何訊息", prompts[0])

    async def test_not_configured_and_unavailable_errors(self):
        with self.assertRaises(AIServiceError) as caught:
            await ReplySuggester(SETTINGS).generate(make_request())
        self.assertEqual(caught.exception.code, "LLM_NOT_CONFIGURED")

        def rate_limited(messages, info):
            raise ModelHTTPError(status_code=429, model_name="gemini", body={"error": "quota"})

        with self.assertRaises(AIServiceError) as caught:
            await ReplySuggester(SETTINGS, model=FunctionModel(rate_limited)).generate(make_request())
        self.assertEqual(caught.exception.code, "LLM_UNAVAILABLE")

    async def test_model_that_keeps_returning_too_few_is_unavailable(self):
        suggester = ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS[:2]}))
        with self.assertRaises(AIServiceError) as caught:
            await suggester.generate(make_request())
        self.assertEqual(caught.exception.code, "LLM_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
