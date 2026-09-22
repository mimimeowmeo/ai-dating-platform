import unittest

from app.reply.chunking import sort_messages
from app.reply.topics import MAX_SPAN, is_topic_opener, plan_span, plan_spans, resolve_span
from tests.reply_helpers import indexed


def ai(content, minutes, intent="question", sender="u-a"):
    return indexed(sender, content, minutes, origin="ai_verbatim", intent=intent)


class TopicSpanTests(unittest.TestCase):
    def test_only_ai_messages_that_open_topics_count(self):
        self.assertTrue(is_topic_opener(ai("你喜歡爬山嗎？", 0)))
        self.assertTrue(is_topic_opener(indexed("u-a", "要不要一起吃飯", 0, origin="ai_edited", intent="plan")))
        self.assertFalse(is_topic_opener(ai("我也覺得", 0, intent="answer")))
        self.assertFalse(is_topic_opener(ai("哈哈", 0, intent="humor")))
        self.assertFalse(is_topic_opener(ai("沒有用途", 0, intent=None)))
        self.assertFalse(is_topic_opener(indexed("u-a", "你喜歡爬山嗎？", 0)))

    def test_span_stops_at_time_gap(self):
        messages = sort_messages([ai("你喜歡爬山嗎？", 0), indexed("u-b", "會啊", 1), indexed("u-b", "隔天的新話題", 100)])
        plan = plan_span(messages, 0)
        self.assertEqual((plan.limit, plan.limit_reason), (1, "time_gap"))

    def test_span_stops_before_next_ai_opener(self):
        messages = sort_messages([
            ai("你喜歡爬山嗎？", 0), indexed("u-b", "會啊", 1), ai("那你養貓嗎？", 2), indexed("u-b", "有", 3),
        ])
        plans = plan_spans(messages)
        self.assertEqual(len(plans), 2)
        self.assertEqual((plans[0].limit, plans[0].limit_reason), (1, "topic_shift"))

    def test_span_is_capped(self):
        messages = sort_messages([ai("你喜歡爬山嗎？", 0)] + [indexed("u-b", f"回{i}", 1 + i) for i in range(40)])
        plan = plan_span(messages, 0)
        self.assertEqual((plan.limit, plan.limit_reason), (MAX_SPAN, "cap"))

    def test_similarity_ends_span_before_first_low_window(self):
        messages = sort_messages([ai("你喜歡爬山嗎？", 0)] + [indexed("u-b", f"回{i}", 1 + i) for i in range(12)])
        plan = plan_span(messages, 0)  # 4 組：1-3、4-6、7-9、10-12
        span = resolve_span(messages, plan, [0.9, 0.8, 0.2, 0.1])
        self.assertEqual((span.endMessageId, span.endReason), (messages[6].id, "topic_shift"))
        self.assertEqual((span.initiatorId, span.messageCount), ("u-a", 6))

    def test_no_vectors_uses_time_limit_and_immediate_shift_returns_none(self):
        messages = sort_messages([ai("你喜歡爬山嗎？", 0)] + [indexed("u-b", f"回{i}", 1 + i) for i in range(6)])
        plan = plan_span(messages, 0)
        self.assertEqual(resolve_span(messages, plan, None).endReason, "end_of_data")
        self.assertIsNone(resolve_span(messages, plan, [0.1, 0.1]))

    def test_no_follow_up_messages_means_no_span(self):
        self.assertIsNone(plan_span(sort_messages([ai("你喜歡爬山嗎？", 0)]), 0))


if __name__ == "__main__":
    unittest.main()
