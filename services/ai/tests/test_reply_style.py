import unittest

from app.reply.schemas import BlendConfig, StyleStats
from app.reply.style import (
    SITE_DEFAULT_STATS,
    blend_stats,
    classify_message_type,
    cold_start_card,
    compute_style_stats,
    partner_reactions,
    resolve_target,
    style_distance,
    usable_card,
)
from tests.reply_helpers import chat, own, profile


def stats(**overrides) -> StyleStats:
    values = dict(
        messageCount=10, medianChars=6, meanChars=6, emojiPerMessage=0.0, questionRatio=0.2,
        exclamationRatio=0.0, laughterRatio=0.0, burstMean=1.0, particles={},
    )
    values.update(overrides)
    return StyleStats(**values)


class StyleStatsTests(unittest.TestCase):
    def test_compute_style_stats(self):
        messages = [
            own("好啊😂", 0),
            own("哈哈你呢？", 0.2),  # 12 秒後、同一聊天室 → 同一次連發
            own("我週末要去爬山啦", 30),
            own("欸好喔", 90, conversation="c-2"),
        ]
        result = compute_style_stats(messages)
        self.assertEqual(result.messageCount, 4)
        self.assertEqual(result.medianChars, 4.0)  # 字數 3、5、8、3
        self.assertEqual(result.emojiPerMessage, 0.25)
        self.assertEqual(result.questionRatio, 0.25)
        self.assertEqual(result.laughterRatio, 0.5)
        self.assertEqual(result.burstMean, round(4 / 3, 3))
        self.assertIn("啦", result.particles)
        self.assertIn("喔", result.particles)

    def test_empty_messages_fall_back_to_site_defaults(self):
        result = compute_style_stats([])
        self.assertEqual(result.messageCount, 0)
        self.assertEqual(result.meanChars, SITE_DEFAULT_STATS.meanChars)

    def test_blend_interpolates(self):
        a = stats(medianChars=8, emojiPerMessage=0.1, particles={"欸": 0.5})
        b = stats(medianChars=15, emojiPerMessage=0.5, particles={"啦": 0.6})
        self.assertEqual(blend_stats(a, b, 0.0).medianChars, 8)
        self.assertEqual(blend_stats(a, b, 1.0).medianChars, 15)
        mixed = blend_stats(a, b, 0.2)
        self.assertEqual(mixed.medianChars, 9.4)
        self.assertEqual(mixed.emojiPerMessage, 0.18)
        self.assertEqual(mixed.particles, {"欸": 0.4, "啦": 0.12})

    def test_style_distance_prefers_closer_length_and_habits(self):
        target = stats(medianChars=5, particles={"啦": 0.5}, emojiPerMessage=1.0)
        close = style_distance("好啊啦😂", target)
        far = style_distance("我覺得這個週末如果天氣不錯的話我們可以考慮一起去郊外走走看看", target)
        self.assertLess(close, far)
        self.assertGreaterEqual(close, 0.0)

    def test_cold_start_card_uses_bio_only_for_wording(self):
        card = cold_start_card(profile("小美", bio="喜歡爬山跟拍照啦，週末常常往山上跑喔"), SITE_DEFAULT_STATS)
        self.assertEqual((card.confidence, card.sampleSource), ("low", "bio"))
        self.assertEqual(card.stats.meanChars, SITE_DEFAULT_STATS.meanChars)  # 數值不看 bio 長度
        self.assertEqual(set(card.stats.particles), {"啦", "喔"})
        empty = cold_start_card(profile("小美", bio="嗨"), SITE_DEFAULT_STATS)
        self.assertEqual((empty.confidence, empty.sampleSource), ("none", "none"))

    def test_resolve_target_uses_partner_only_for_the_chat_first_message(self):
        requester = cold_start_card(profile("阿明", bio="平常喜歡打羽球跟煮飯，也愛看電影"), SITE_DEFAULT_STATS)
        requester = requester.model_copy(update={"stats": stats(medianChars=8)})
        partner = requester.model_copy(update={"stats": stats(medianChars=15)})
        # 聊天室還沒有任何訊息：整批照 B（B 100%）。
        first = resolve_target(requester, partner, BlendConfig(), first_message=True)
        self.assertEqual((first.rule, first.source, first.stats.medianChars), ("first_message", "partner", 15))
        # 有人傳過訊息之後：A 80%／B 20%（0.8 × 8 + 0.2 × 15）。
        later = resolve_target(requester, partner, BlendConfig(), first_message=False)
        self.assertEqual((later.rule, later.source, later.stats.medianChars), ("later", "blend", 9.4))
        # B 沒有足夠資料：不論哪一種都只用 A 的寫法。
        unknown = usable_card(None, profile("小美", bio="嗨"), SITE_DEFAULT_STATS)
        for first_message in (True, False):
            fallback = resolve_target(requester, unknown, BlendConfig(), first_message=first_message)
            self.assertEqual((fallback.source, fallback.stats.medianChars), ("requester", 8))

    def test_resolve_target_labels_source_by_weight(self):
        requester = cold_start_card(profile("阿明", bio="平常喜歡打羽球跟煮飯，也愛看電影"), SITE_DEFAULT_STATS)
        requester = requester.model_copy(update={"stats": stats(medianChars=8)})
        partner = requester.model_copy(update={"stats": stats(medianChars=15)})
        blend = BlendConfig(firstMessagePartnerWeight=0.5, laterPartnerWeight=0.0)
        self.assertEqual(resolve_target(requester, partner, blend, first_message=True).source, "blend")
        self.assertEqual(resolve_target(requester, partner, blend, first_message=False).source, "requester")


class ReactionTests(unittest.TestCase):
    def test_classify_message_type(self):
        self.assertEqual(classify_message_type("要不要一起吃飯？"), "plan")
        self.assertEqual(classify_message_type("你週末都在幹嘛？"), "question")
        self.assertEqual(classify_message_type("你好可愛"), "compliment")
        self.assertEqual(classify_message_type("哈哈哈"), "humor")
        self.assertEqual(classify_message_type("我今天加班"), "share")

    def test_partner_reacts_warmer_to_questions_than_shares(self):
        messages = [
            chat("A", "你平常喜歡做什麼？", 0),
            chat("B", "我超愛爬山！上週才去", 1),
            chat("B", "你呢？", 1.5),
            chat("A", "我今天加班", 10),
            chat("B", "喔", 70),
            chat("A", "你最喜歡哪座山？", 80),
            chat("B", "合歡山！雲海超美", 81),
        ]
        reactions = partner_reactions(messages)
        by_type = {item.type: item for item in reactions}
        self.assertGreater(by_type["question"].heat, by_type["share"].heat)
        self.assertEqual(by_type["question"].samples, 2)
        self.assertEqual(reactions[0].type, "question")

    def test_no_partner_messages_means_no_reactions(self):
        self.assertEqual(partner_reactions([chat("A", "嗨", 0)]), [])


if __name__ == "__main__":
    unittest.main()
