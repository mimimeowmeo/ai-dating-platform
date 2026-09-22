"""推薦內容的安全規則。規格 4.3。

只檢查「AI 產生的推薦」，不檢查使用者自己打的訊息。規則刻意保守、可以擴充：
命中任何一條，這則推薦就被刪掉，原因代碼會跟著回傳給後端保存，方便之後調整規則。
"""

import re

# (原因代碼, 規則)。順序就是檢查順序，回報第一個命中的原因。
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    # 台灣手機（09xx-xxx-xxx、+886 9xx…）與市話。
    ("CONTACT_PHONE", re.compile(r"(?:\+?886[-\s]?|0)9\d{2}[-\s]?\d{3}[-\s]?\d{3}|\b0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}\b")),
    # Email 要排在網址前面，否則「a@example.org」會先被當成網址。
    ("CONTACT_EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("CONTACT_LINK", re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|tw|net|org|io|me|cc)\b", re.IGNORECASE)),
    # 「LINE ID: xxx」「加我賴」「@帳號」這類把對話帶離平台的聯絡方式。
    (
        "CONTACT_ID",
        re.compile(
            r"(?:line|賴|ig|instagram|telegram|tg|wechat|微信)\s*(?:id)?\s*[:：]\s*\S+|加(?:我)?(?:賴|line)|@[A-Za-z0-9_.]{3,}",
            re.IGNORECASE,
        ),
    ),
    # 詐騙常見的金錢話題。
    ("MONEY", re.compile(r"匯款|轉帳|借錢|借我|投資|虛擬貨幣|加密貨幣|比特幣|usdt|保證獲利|點數卡|遊戲點數|帳戶", re.IGNORECASE)),
    ("EXPLICIT", re.compile(r"約炮|做愛|裸照|性愛|打炮|色色")),
    # 連續 6 位以上數字：可能是帳號、電話或驗證碼。
    ("LONG_DIGITS", re.compile(r"\d{6,}")),
)


def check_suggestion(text: str) -> str | None:
    """檢查一則推薦；安全時回傳 None，否則回傳第一個命中的原因代碼（例如 "MONEY"）。"""
    for code, pattern in RULES:
        if pattern.search(text):
            return code
    return None
