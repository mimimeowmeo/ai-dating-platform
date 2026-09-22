"""AI 推薦回覆功能的錯誤型別。

所有「模型沒設定」「模型暫時不能用」這類可預期的錯誤都用 AIServiceError 表示，
帶一個固定的錯誤代碼（會原樣回給後端），不帶任何使用者內容，避免訊息被寫進日誌或佇列。
"""

# 錯誤代碼 → 給後端看的中文說明。HTTP 狀態一律 503：代表「暫時無法提供」，後端可以稍後重試。
MESSAGES = {
    "LLM_NOT_CONFIGURED": "AI 模型尚未設定",
    "LLM_UNAVAILABLE": "AI 暫時忙碌中，請稍後再試",
    "EMBEDDING_NOT_CONFIGURED": "向量模型尚未設定",
    "EMBEDDING_UNAVAILABLE": "向量服務暫時無法使用",
    "EXTRACTION_NOT_CONFIGURED": "背景萃取模型尚未設定",
    "EXTRACTION_UNAVAILABLE": "背景萃取模型暫時無法使用",
}


class AIServiceError(Exception):
    """可預期的服務錯誤；code 是固定代碼，http_status 是 API 要回的狀態碼。"""

    def __init__(self, code: str, http_status: int = 503):
        """建立錯誤；例外訊息只放固定代碼，不放任何使用者內容。"""
        super().__init__(code)
        self.code = code
        self.http_status = http_status

    @property
    def message(self) -> str:
        """對應的中文說明；未知代碼時回傳通用說明。"""
        return MESSAGES.get(self.code, "AI 服務暫時無法使用")
