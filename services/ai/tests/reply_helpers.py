"""推薦回覆測試共用的工具：建立測試資料、假的向量服務、假的語言模型。

檔名刻意不以 test 開頭，unittest discover 不會把它當成測試檔。
所有測試都不會連網：Pydantic AI 的 ALLOW_MODEL_REQUESTS 設為 False，只能用 FunctionModel 等測試模型。
"""

import json
from datetime import datetime, timedelta, timezone

from pydantic_ai import ModelResponse, TextPart, models
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.config import Settings
from app.reply.schemas import ChatMessage, IndexMessage, OwnMessage, ProfileSnapshot

models.ALLOW_MODEL_REQUESTS = False

BASE_TIME = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
SETTINGS = Settings(internal_token="internal-test-secret")
HEADERS = {"X-Internal-Token": SETTINGS.internal_token}


def at(minutes: float) -> datetime:
    """BASE_TIME 之後幾分鐘的時間，用來排出訊息的先後。"""
    return BASE_TIME + timedelta(minutes=minutes)


def chat(sender: str, content: str, minutes: float, origin: str = "human", message_id: str | None = None) -> ChatMessage:
    """建立一則 A–B 聊天室訊息（產生推薦用）。"""
    return ChatMessage(
        id=message_id or f"m-{sender}-{minutes}",
        sender=sender,
        content=content,
        createdAt=at(minutes),
        origin=origin,
    )


def indexed(
    sender_id: str,
    content: str,
    minutes: float,
    origin: str = "human",
    intent: str | None = None,
    message_id: str | None = None,
) -> IndexMessage:
    """建立一則背景工作用的訊息（切片、話題區段、摘要用）。"""
    return IndexMessage(
        id=message_id or f"i-{sender_id}-{minutes}",
        senderId=sender_id,
        senderName={"u-a": "阿明", "u-b": "小美"}.get(sender_id, sender_id),
        content=content,
        createdAt=at(minutes),
        origin=origin,
        suggestionIntent=intent,
    )


def own(content: str, minutes: float, conversation: str = "c-1", origin: str = "human", in_ai_topic: bool = False) -> OwnMessage:
    """建立一則「自己發出」的訊息（風格卡用）。"""
    return OwnMessage(
        id=f"o-{conversation}-{minutes}",
        conversationId=conversation,
        content=content,
        createdAt=at(minutes),
        origin=origin,
        inAiTopic=in_ai_topic,
    )


def profile(name: str, bio: str = "", **extra) -> ProfileSnapshot:
    """建立一份個人檔案快照。"""
    return ProfileSnapshot(displayName=name, bio=bio, **extra)


class FakeEmbedder:
    """假的向量服務：依文字裡出現的關鍵字產生固定向量，不連網。

    含「山」的文字跟含「貓」的文字向量完全不同，方便測試相似度判斷；calls 記錄每次呼叫。
    """

    configured = True
    model = "fake-embedding"
    dimensions = 4

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple[list[str], str, str | None]] = []

    async def embed(self, texts, purpose="document", title=None):
        """依關鍵字回傳 4 維向量；fail=True 時模擬向量服務暫時無法使用。"""
        from app.reply.errors import AIServiceError

        self.calls.append((list(texts), purpose, title))
        if self.fail:
            raise AIServiceError("EMBEDDING_UNAVAILABLE")
        return [
            [1.0 if "山" in text else 0.0, 1.0 if "貓" in text else 0.0, 1.0 if "吃" in text else 0.0, 0.1]
            for text in texts
        ]


def json_model(payload, name: str = "fake-model", captured: list | None = None) -> FunctionModel:
    """建立一個固定回傳某個 JSON 的假模型；captured 會收到每次呼叫時模型看到的所有文字。"""

    def respond(messages, info: AgentInfo):
        if captured is not None:
            captured.append(prompt_text(messages))
        data = payload() if callable(payload) else payload
        return ModelResponse(parts=[TextPart(json.dumps(data, ensure_ascii=False))])

    return FunctionModel(respond, model_name=name)


def prompt_text(messages) -> str:
    """把 Pydantic AI 傳給模型的訊息攤平成一段文字，方便檢查 prompt 內容。"""
    texts: list[str] = []
    for message in messages:
        for part in getattr(message, "parts", []):
            content = getattr(part, "content", None)
            if isinstance(content, str):
                texts.append(content)
    return "\n".join(texts)
