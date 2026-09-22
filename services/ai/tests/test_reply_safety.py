import unittest

from app.reply.safety import check_suggestion


class SafetyTests(unittest.TestCase):
    def test_blocked_examples(self):
        cases = {
            "打給我 0912-345-678": "CONTACT_PHONE",
            "看這個 https://example.com": "CONTACT_LINK",
            "寄到 me@example.org": "CONTACT_EMAIL",
            "加我賴好不好": "CONTACT_ID",
            "我的 IG: sunny_day": "CONTACT_ID",
            "最近有個投資機會": "MONEY",
            "可以先轉帳給我嗎": "MONEY",
            "今晚想約炮嗎": "EXPLICIT",
            "驗證碼是 123456": "LONG_DIGITS",
        }
        for text, code in cases.items():
            with self.subTest(text=text):
                self.assertEqual(check_suggestion(text), code)

    def test_normal_messages_pass(self):
        for text in ("週末要不要一起去爬山？", "哈哈你也太可愛", "我 3 點下班", "你喜歡哪一家咖啡店"):
            with self.subTest(text=text):
                self.assertIsNone(check_suggestion(text))


if __name__ == "__main__":
    unittest.main()
