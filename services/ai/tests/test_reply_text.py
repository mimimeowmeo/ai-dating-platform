import unittest
from datetime import datetime, timezone

from app.reply import textutil as t


class TextUtilTests(unittest.TestCase):
    def test_visible_chars_counts_emoji_sequences_once_and_ignores_spaces(self):
        self.assertEqual(t.visible_chars("好 啊"), 2)
        self.assertEqual(t.visible_chars("哈哈😂"), 3)
        self.assertEqual(t.visible_chars("👨‍👩‍👧"), 1)
        self.assertEqual(t.visible_chars("👍🏻"), 1)

    def test_estimate_tokens_weights_cjk_higher_than_ascii(self):
        self.assertEqual(t.estimate_tokens(""), 0)
        self.assertEqual(t.estimate_tokens("你好"), 3)  # 2 × 1.3 = 2.6 → 3
        self.assertEqual(t.estimate_tokens("abcd"), 1)
        self.assertGreater(t.estimate_tokens("今天天氣很好"), t.estimate_tokens("today"))

    def test_question_exclamation_laughter(self):
        self.assertTrue(t.has_question("你週末都在幹嘛？"))
        self.assertTrue(t.has_question("那你呢😂"))
        self.assertTrue(t.has_question("真的嗎～"))
        self.assertFalse(t.has_question("好啊"))
        self.assertTrue(t.has_exclamation("太好了！"))
        self.assertTrue(t.has_laughter("笑死XD"))
        self.assertFalse(t.has_laughter("好喔"))

    def test_particles(self):
        self.assertEqual(t.particles_in("好啦欸"), {"啦", "欸"})
        self.assertEqual(t.particles_in("我知道"), set())

    def test_opencc_converts_to_taiwan_wording(self):
        self.assertEqual(t.to_taiwan_traditional("软件的信息"), "軟體的資訊")
        self.assertEqual(t.to_taiwan_traditional("週末一起去爬山"), "週末一起去爬山")

    def test_similarity_and_single_line(self):
        self.assertEqual(t.text_similarity("週末去爬山嗎", "週末去爬山嗎"), 1.0)
        self.assertLess(t.text_similarity("週末去爬山嗎", "你喜歡貓嗎"), 0.5)
        self.assertEqual(t.single_line(" 第一行\n第二行  "), "第一行 第二行")

    def test_ngram_index_detects_copied_text(self):
        index = t.build_ngram_index(["我上個月去了合歡山看雲海超美"], 8)
        self.assertTrue(t.shares_long_substring("他說上個月去了合歡山看雲海", index, 8))
        self.assertFalse(t.shares_long_substring("喜歡登山與戶外活動", index, 8))

    def test_identifying_details(self):
        for text in ("電話0912", "看 https://x.tw", "@hello_world", "a@b.com"):
            with self.subTest(text=text):
                self.assertTrue(t.has_identifying_detail(text))
        self.assertFalse(t.has_identifying_detail("喜歡聊旅行"))

    def test_time_helpers(self):
        naive = datetime(2026, 9, 1, 12, 0)
        self.assertEqual(t.as_utc(naive).tzinfo, timezone.utc)
        self.assertEqual(t.taipei_label(datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)), "09-01 20:00")


if __name__ == "__main__":
    unittest.main()
