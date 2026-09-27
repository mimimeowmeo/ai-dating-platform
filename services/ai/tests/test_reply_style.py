import unittest

from app.reply.schemas import BlendConfig, StyleStats
from app.reply.style import (
    SITE_DEFAULT_STATS,
    blend_stats,
    classify_message_type,
    cold_start_card,
    compute_style_stats,
    has_chat_history,
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

    def test_fast_back_and_forth_chat_does_not_break_burst_limit(self):
        # 一來一往聊得很快：這個人每則都在上一則的 60 秒內，整段會被算成同一次連發。
        # 以前 burstMean 會超過欄位上限 50，StyleStats 驗證失敗，整個風格卡萃取跟著失敗。
        messages = [own("好喔", index * 0.5) for index in range(120)]
        result = compute_style_stats(messages)
        self.assertEqual((result.messageCount, result.burstMean), (120, 50.0))

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

    def test_style_distance_ignores_length_and_checks_habits(self):
        target = stats(medianChars=5, particles={"啦": 0.5}, emojiPerMessage=1.0)
        short = style_distance("好啦👍", target)
        long = style_distance("那我們這個週末一起去看那個展覽好啦👍", target)
        # 字數不列入（2026-09-27）：寫法一樣時，長句和短句一樣近，短句也不會因為太短被扣分。
        self.assertEqual((short, long), (0.0, 0.0))
        # 沒有 emoji、也沒用常用的語助詞：emoji 差 1/3、語助詞扣一半 → 0.3 × 1/3 + 0.4 × 0.5。
        self.assertEqual(style_distance("我覺得可以", target), 0.3)

    def test_cold_start_card_uses_bio_only_for_wording(self):
        card = cold_start_card(profile("小美", bio="喜歡爬山跟拍照啦，週末常常往山上跑喔"), SITE_DEFAULT_STATS)
        self.assertEqual((card.confidence, card.sampleSource), ("low", "bio"))
        self.assertEqual(card.stats.meanChars, SITE_DEFAULT_STATS.meanChars)  # 數值不看 bio 長度
        self.assertEqual(set(card.stats.particles), {"啦", "喔"})
        empty = cold_start_card(profile("小美", bio="嗨"), SITE_DEFAULT_STATS)
        self.assertEqual((empty.confidence, empty.sampleSource), ("none", "none"))

    def test_resolve_target_follows_mode(self):
        requester = cold_start_card(profile("阿明", bio="平常喜歡打羽球跟煮飯，也愛看電影"), SITE_DEFAULT_STATS)
        requester = requester.model_copy(update={"stats": stats(medianChars=8)})
        partner = requester.model_copy(update={"stats": stats(medianChars=15)})
        # 2026-09-27：開場、重啟 B 100%；回覆 A 80%／B 20%（0.8 × 8 + 0.2 × 15）；追問只用 A 的語氣。
        expected = {
            "opener": ("partner", 15),
            "revive": ("partner", 15),
            "reply": ("blend", 9.4),
            "follow_up": ("requester", 8),
        }
        for mode, (source, median_chars) in expected.items():
            target = resolve_target(requester, partner, BlendConfig(), mode)
            self.assertEqual((target.rule, target.source, target.stats.medianChars), (mode, source, median_chars))
        # B 沒有足夠資料：不論哪一種模式都只用 A 的寫法。
        unknown = usable_card(None, profile("小美", bio="嗨"), SITE_DEFAULT_STATS)
        for mode in expected:
            fallback = resolve_target(requester, unknown, BlendConfig(), mode)
            self.assertEqual((fallback.source, fallback.stats.medianChars), ("requester", 8))

    def test_resolve_target_labels_source_by_weight(self):
        requester = cold_start_card(profile("阿明", bio="平常喜歡打羽球跟煮飯，也愛看電影"), SITE_DEFAULT_STATS)
        requester = requester.model_copy(update={"stats": stats(medianChars=8)})
        partner = requester.model_copy(update={"stats": stats(medianChars=15)})
        blend = BlendConfig(openerPartnerWeight=0.5, replyPartnerWeight=0.0)
        self.assertEqual(resolve_target(requester, partner, blend, "opener").source, "blend")
        self.assertEqual(resolve_target(requester, partner, blend, "reply").source, "requester")

    def test_chat_history_needs_thirty_human_messages(self):
        card = cold_start_card(profile("小美", bio="喜歡爬山跟拍照，週末常往山上跑喔"), SITE_DEFAULT_STATS)
        self.assertFalse(has_chat_history(card))  # 只有 bio
        self.assertFalse(has_chat_history(card.model_copy(update={"messageCount": 29})))
        self.assertTrue(has_chat_history(card.model_copy(update={"messageCount": 30})))


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
