import unittest

from pydantic_ai.models.function import FunctionModel

from app.reply.errors import AIServiceError
from app.reply.extraction import ConversationSummarizer, StyleProfileBuilder
from app.reply.schemas import StyleProfileRequest, SummaryRequest
from app.reply.style import SITE_DEFAULT_STATS
from tests.reply_helpers import SETTINGS, FakeEmbedder, indexed, json_model, own

MAP_OUTPUT = {
    "voiceNotes": ["句子很短、常連發", "愛用自嘲的幽默", "句子很短，常常連發"],
    "topics": [
        {"topic": "登山", "heat": 5},
        {"topic": "登山山景", "heat": 4},
        {"topic": "手搖飲", "heat": 3, "fromAiTopic": True},
    ],
    "avoidTopics": ["工作"],
    "habits": ["常反問對方", "我上個月去了合歡山看雲海超美"],
}


def never_called(messages, info):
    raise AssertionError("model should not be called")


def human_messages(count: int):
    messages = [own(f"今天也很好{i}哈哈", i) for i in range(count - 1)]
    messages.append(own("我上個月去了合歡山看雲海超美", count, in_ai_topic=True))
    return messages


class StyleProfileTests(unittest.IsolatedAsyncioTestCase):
    async def test_high_confidence_profile_merges_filters_and_embeds(self):
        prompts: list[str] = []
        embedder = FakeEmbedder()
        builder = StyleProfileBuilder(SETTINGS, model=json_model(MAP_OUTPUT, captured=prompts), embedder=embedder)
        ai_messages = [own("AI 寫的訊息", 900 + i, origin="ai_verbatim") for i in range(3)]
        request = StyleProfileRequest(
            userId="u-b", bio="喜歡爬山跟拍照，週末常往山上跑喔", messages=human_messages(40) + ai_messages
        )
        result = await builder.build(request)
        card = result.card

        self.assertEqual((card.confidence, card.sampleSource, card.messageCount), ("high", "chat", 40))
        self.assertEqual(len(prompts), 2)  # 一批聊天訊息＋一批自我介紹
        self.assertIn("[AI話題] 我上個月去了合歡山", prompts[0])
        self.assertNotIn("AI 寫的訊息", "\n".join(prompts))
        self.assertIn("[自我介紹]", prompts[1])

        self.assertEqual(len(card.voiceNotes), 2)  # 兩條相似的描述被合併
        topics = {facet.statement: facet.weight for facet in card.facets if facet.kind == "topic"}
        self.assertEqual(topics["聊到「登山」會比較熱絡"], 1.0)
        self.assertNotIn("聊到「登山山景」會比較熱絡", topics)  # 相似話題被合併
        self.assertLess(topics["聊到「手搖飲」會比較熱絡"], 0.3)  # 來自 [AI話題] → 降權
        statements = [facet.statement for facet in card.facets]
        self.assertNotIn("我上個月去了合歡山看雲海超美", statements)  # 抄原文 → 被抽象化檢查刪掉
        self.assertGreaterEqual(result.rejectedStatements, 1)
        self.assertEqual(len(result.facetVectors), len(card.facets))
        self.assertEqual(result.embeddingModel, "fake-embedding")
        self.assertEqual(card.modelName, "fake-model")

    async def test_bio_only_mixed_and_none(self):
        prompts: list[str] = []
        builder = StyleProfileBuilder(SETTINGS, model=json_model(MAP_OUTPUT, captured=prompts), embedder=FakeEmbedder())
        bio_only = await builder.build(StyleProfileRequest(userId="u-1", bio="喜歡爬山跟拍照，週末常往山上跑喔"))
        self.assertEqual((bio_only.card.confidence, bio_only.card.sampleSource), ("low", "bio"))
        self.assertEqual(bio_only.card.stats.meanChars, SITE_DEFAULT_STATS.meanChars)
        self.assertEqual(len(prompts), 1)

        mixed = await builder.build(StyleProfileRequest(userId="u-2", messages=[own("好啊", i) for i in range(5)]))
        self.assertEqual((mixed.card.confidence, mixed.card.sampleSource, mixed.card.stats.messageCount), ("low", "mixed", 5))
        self.assertGreater(mixed.card.stats.medianChars, 2)
        self.assertLess(mixed.card.stats.medianChars, SITE_DEFAULT_STATS.medianChars)

        silent = StyleProfileBuilder(SETTINGS, model=FunctionModel(never_called), embedder=FakeEmbedder())
        none = await silent.build(StyleProfileRequest(userId="u-3", bio="嗨"))
        self.assertEqual((none.card.confidence, none.card.facets), ("none", []))

    async def test_embedding_failure_still_returns_card(self):
        builder = StyleProfileBuilder(SETTINGS, model=json_model(MAP_OUTPUT), embedder=FakeEmbedder(fail=True))
        result = await builder.build(StyleProfileRequest(userId="u-b", messages=human_messages(35)))
        self.assertIsNone(result.facetVectors)
        self.assertTrue(result.card.facets)

    async def test_extraction_not_configured(self):
        builder = StyleProfileBuilder(SETTINGS, embedder=FakeEmbedder())
        with self.assertRaises(AIServiceError) as caught:
            await builder.build(StyleProfileRequest(userId="u-b", messages=human_messages(35)))
        self.assertEqual(caught.exception.code, "EXTRACTION_NOT_CONFIGURED")

    async def test_waits_for_online_requests_before_every_model_call(self):
        # worker 設定了 before_model_call：每一批（這裡是聊天訊息一批＋自我介紹一批）呼叫模型前都先等。
        events: list[str] = []

        async def wait():
            events.append("wait")

        builder = StyleProfileBuilder(SETTINGS, model=json_model(MAP_OUTPUT, captured=events), embedder=FakeEmbedder())
        builder.before_model_call = wait
        await builder.build(
            StyleProfileRequest(userId="u-b", bio="喜歡爬山跟拍照，週末常往山上跑喔", messages=human_messages(40))
        )
        self.assertEqual([event == "wait" for event in events], [True, False, True, False])


class SummaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_is_converted_truncated_and_versioned(self):
        prompts: list[str] = []
        summarizer = ConversationSummarizer(SETTINGS, model=json_model({"summary": "小美喜欢爬山。" * 50}, captured=prompts))
        request = SummaryRequest(
            conversationId="c-1",
            messages=[indexed("u-b", "我喜歡爬山", 5), indexed("u-a", "我也是", 1)],
            maxChars=100,
        )
        result = await summarizer.summarize(request)
        self.assertEqual(len(result.summary), 100)
        self.assertIn("喜歡", result.summary)
        self.assertEqual((result.untilMessageId, result.promptVersion), (request.messages[0].id, "summary-v1"))
        self.assertIn("（沒有，這是第一次整理）", prompts[0])
        self.assertLess(prompts[0].index("我也是"), prompts[0].index("我喜歡爬山"))  # 依時間排序

    async def test_summary_waits_for_online_requests_before_calling_model(self):
        events: list[str] = []

        async def wait():
            events.append("wait")

        summarizer = ConversationSummarizer(SETTINGS, model=json_model({"summary": "兩人聊到爬山"}, captured=events))
        summarizer.before_model_call = wait
        await summarizer.summarize(
            SummaryRequest(conversationId="c-1", messages=[indexed("u-a", "我也喜歡爬山", 1)], maxChars=100)
        )
        self.assertEqual([event == "wait" for event in events], [True, False])


if __name__ == "__main__":
    unittest.main()
