"""產生推薦的主流程（POST /internal/ai/reply-suggestions）。規格第 3、4 節。

流程：
1. 判斷模式（開場／回覆／追問／重啟）。
2. 準備 A、B 的風格卡（沒有就用 bio 冷啟動），算出這一批 5 則共用的風格目標：
   整個聊天室還沒有任何訊息時 B 100%，只要有人傳過訊息就 A 80%／B 20%。
3. 完全沒有資料根據時（見 has_topic_basis）不呼叫模型，直接回「沒有可推薦的句子」。
4. 計算 B 在這個聊天室的反應熱度。
5. 組 prompt，請模型產生最多 5 則候選（結構化輸出；沒有依據時可以少給甚至不給，不硬湊）。
6. 後處理：單行化、簡轉繁、長度與安全規則、去重。
7. 排序：B 剛問了問題時回答類優先，再依模型給的優先度，最後依離風格目標多近。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic_ai import Agent
from pydantic_ai.models import Model

from ..config import Settings
from .llm import build_chain, model_name_of, run_agent, structured_output, usage_info
from .prompts import REPLY_INSTRUCTIONS, REPLY_PROMPT_VERSION, build_reply_prompt
from .safety import check_suggestion
from .schemas import (
    ChatMessage,
    DraftBatch,
    DraftSuggestion,
    Mode,
    ProfileSnapshot,
    RejectedSuggestion,
    ReplySuggestionRequest,
    ReplySuggestionResponse,
    StyleCard,
    StyleTarget,
    Suggestion,
    UsageInfo,
)
from .style import SITE_DEFAULT_STATS, partner_reactions, resolve_target, style_distance, usable_card
from .textutil import as_utc, has_question, single_line, text_similarity, to_taiwan_traditional, visible_chars

MIN_RETURN = 3
MAX_RETURN = 5
MAX_SUGGESTION_CHARS = 80
DUPLICATE_SIMILARITY = 0.8
REVIVE_AFTER = timedelta(days=7)
# 一則推薦都給不出來時的提示（2026-09-23 使用者指定的文字）；前端只顯示這句話，不會填任何字進輸入框。
NO_SUGGESTION_NOTICE = "沒有可推薦的句子"
# 檔案裡「可以拿來聊」的標籤類欄位；暱稱、年齡、性別、城市、身高不算（見 profile_has_topics）。
PROFILE_TOPIC_LISTS = ("interests", "hobbies", "foods", "traits", "datingGoals")


def sort_chat(messages: list[ChatMessage]) -> list[ChatMessage]:
    """把聊天室訊息依時間由舊到新排序；同一時間再依 id，讓結果固定。"""
    return sorted(messages, key=lambda message: (as_utc(message.createdAt), message.id))


def detect_mode(messages: list[ChatMessage], now: datetime) -> Mode:
    """依聊天室狀態判斷模式（messages 需由舊到新排序）。

    - 沒有訊息 → opener（開場）
    - 最後一則超過 7 天 → revive（重啟）
    - 最後一則是 B 傳的 → reply（回覆）
    - 最後一則是 A 傳的 → follow_up（追問）
    """
    if not messages:
        return "opener"
    last = messages[-1]
    if as_utc(now) - as_utc(last.createdAt) > REVIVE_AFTER:
        return "revive"
    return "reply" if last.sender == "B" else "follow_up"


def partner_asked_question(messages: list[ChatMessage]) -> bool:
    """B 最近一輪（最後一則 A 訊息之後）有沒有問問題；有的話「回答」類推薦要排前面。"""
    for message in reversed(messages):
        if message.sender == "A":
            return False
        if has_question(message.content):
            return True
    return False


def profile_has_topics(profile: ProfileSnapshot) -> bool:
    """這份檔案有沒有「可以拿來聊」的內容：自我介紹、職業、學歷，或任何一類興趣標籤。

    暱稱、年齡、性別、城市、身高不算：這些是註冊時人人都有（或只是數字）的基本資料，
    只靠它們寫出來的只會是「嗨～你好」這類空泛的招呼，也就是使用者說的「硬推薦」。
    """
    if profile.bio.strip() or (profile.occupation or "").strip() or (profile.education or "").strip():
        return True
    return any(getattr(profile, field) for field in PROFILE_TOPIC_LISTS)


def has_topic_basis(request: ReplySuggestionRequest, recent: list[ChatMessage], partner_card: StyleCard) -> bool:
    """這次有沒有任何「資料根據」可以寫推薦（2026-09-23 使用者決定：沒有根據就不要硬推薦）。

    任何一項有內容就算有根據：
    - 對話：最近的訊息、聊天室摘要、檢索到的舊對話片段（有訊息的聊天室一定算有根據）；
    - 共同標籤；
    - A 或 B 的檔案內容（見 profile_has_topics）；
    - B 的喜好特徵句（這次檢索到的，或 B 風格卡上的）。
    全部都沒有時，模型能寫的只剩空泛的招呼，所以呼叫端不呼叫模型，直接回「沒有可推薦的句子」。
    partner_card 要傳 usable_card 處理過的 B 風格卡。
    """
    return bool(
        recent
        or (request.conversationSummary or "").strip()
        or request.retrievedChunks
        or request.sharedTags
        or request.partnerFacets
        or partner_card.facets
        or profile_has_topics(request.partner)
        or profile_has_topics(request.requester)
    )


@dataclass
class Candidate:
    """通過後處理的一則候選，加上它離這一批風格目標的距離（0 最像）。"""

    text: str
    draft: DraftSuggestion
    distance: float


def clean_drafts(
    drafts: list[DraftSuggestion], exclude_texts: list[str]
) -> tuple[list[DraftSuggestion], list[RejectedSuggestion]]:
    """後處理第一階段：整理文字並刪掉不合格的候選（規格 4.4 的 1～4 步）。

    依序：單行化 → 簡轉繁（台灣用語）→ 空白或過長 → 安全規則 → 與「換一批」要避開的句子太像
    → 與前面已保留的候選太像。每則被刪的候選都會記下原因，交給後端保存。
    回傳的 DraftSuggestion 的 text 已經是整理後的版本。
    """
    kept: list[DraftSuggestion] = []
    rejected: list[RejectedSuggestion] = []
    for draft in drafts:
        text = to_taiwan_traditional(single_line(draft.text)).strip("「」\"'")
        reason = None
        if not text:
            reason = "EMPTY"
        elif visible_chars(text) > MAX_SUGGESTION_CHARS:
            reason = "TOO_LONG"
        else:
            reason = check_suggestion(text)
        if reason is None and any(text_similarity(text, old) >= DUPLICATE_SIMILARITY for old in exclude_texts):
            reason = "SIMILAR_TO_EXCLUDED"
        if reason is None and any(text_similarity(text, other.text) >= DUPLICATE_SIMILARITY for other in kept):
            reason = "DUPLICATE"
        if reason:
            rejected.append(RejectedSuggestion(text=single_line(draft.text)[:400], reasonCode=reason))
        else:
            kept.append(draft.model_copy(update={"text": text}))
    return kept, rejected


def rank_candidates(
    drafts: list[DraftSuggestion], target: StyleTarget, answer_first: bool
) -> tuple[list[Suggestion], list[RejectedSuggestion]]:
    """排序並編號（規格 4.1），回傳最多 5 則推薦，以及因為超過 5 則而被捨棄的候選。

    一批 5 則共用同一個風格目標（見 style.resolve_target），所以每一名的排序規則都一樣：
    1. B 剛問了問題時，「回答」類排前面；
    2. 模型給的 priority（1 最推薦）；
    3. 離風格目標越近越前面。
    第 1 名會被前端用打字動畫填進輸入框。
    styleTarget 依這一批的規則標示：目標完全照 B（聊天室的第一則訊息、B 有資料）時是 partner，
    其餘都是 blend。
    """
    if not drafts:
        return [], []
    label: Literal["partner", "blend"] = "partner" if target.source == "partner" else "blend"
    candidates = [
        Candidate(text=draft.text, draft=draft, distance=style_distance(draft.text, target.stats)) for draft in drafts
    ]
    candidates.sort(
        key=lambda item: (
            0 if answer_first and item.draft.intent == "answer" else 1,
            item.draft.priority,
            item.distance,
        )
    )
    suggestions = [
        Suggestion(
            rank=index + 1,
            text=item.text,
            intent=item.draft.intent,
            styleTarget=label,
            styleDistance=item.distance,
            reason=single_line(item.draft.reason)[:120],
        )
        for index, item in enumerate(candidates[:MAX_RETURN])
    ]
    overflow = [RejectedSuggestion(text=item.text, reasonCode="OVER_LIMIT") for item in candidates[MAX_RETURN:]]
    return suggestions, overflow


def notice_for(count: int) -> tuple[Literal["ok", "partial", "empty"], str | None]:
    """依最後保留的則數決定狀態與提示文字（規格 4.1：不硬湊）。

    - 3～5 則：ok，沒有提示。
    - 1～2 則：partial，提示「只找到 N 則合適的建議」，顯示剩下的就好。
    - 0 則：empty，提示「沒有可推薦的句子」；前端只顯示這句話，不會填任何字進輸入框。
    """
    if count >= MIN_RETURN:
        return "ok", None
    if count > 0:
        return "partial", f"只找到 {count} 則合適的建議"
    return "empty", NO_SUGGESTION_NOTICE


class ReplySuggester:
    """產生推薦的服務物件。

    model 參數讓測試可以注入 Pydantic AI 的測試模型；不給時依設定建立 Gemini 備援鏈，
    而且等到第一次使用才建立（服務啟動時不需要 API key）。
    """

    def __init__(self, settings: Settings, model: Model | None = None):
        """保存設定並建立產生推薦的 agent（agent 本身不綁模型，每次執行時才指定）。"""
        self.settings = settings
        self._model = model
        self._model_built = model is not None
        # 結構化輸出方式依模型鏈決定（見 llm.structured_output：Ollama Cloud 用 ToolOutput，其餘用 NativeOutput）；
        # retries={"output": 2}：格式或數量不合時，最多再請模型重寫 2 次。
        self.agent = Agent(
            output_type=structured_output(DraftBatch, settings.reply_models, settings),
            instructions=REPLY_INSTRUCTIONS,
            retries={"output": 2},
            name="reply-suggestions",
        )

    @property
    def model(self) -> Model | None:
        """取得（必要時建立）產生推薦用的模型鏈；沒有任何可用模型時為 None。"""
        if not self._model_built:
            self._model = build_chain(self.settings.reply_models, self.settings, "reply")
            self._model_built = True
        return self._model

    @property
    def configured(self) -> bool:
        """是否至少有一個可用的模型（給 /health 使用）。"""
        return self.model is not None

    async def generate(self, request: ReplySuggestionRequest) -> ReplySuggestionResponse:
        """執行完整的推薦流程並回傳結果；模型不可用時丟出 AIServiceError（LLM_*）。

        完全沒有資料根據時（見 has_topic_basis）不呼叫模型，直接回 status=empty 與
        「沒有可推薦的句子」；這時 modelName 是 None、usage 全是 0。
        """
        now = as_utc(request.now) if request.now else datetime.now(timezone.utc)
        recent = sort_chat(list(request.recentMessages))
        mode = request.mode or detect_mode(recent, now)
        site = request.siteStats or SITE_DEFAULT_STATS
        requester_card = usable_card(request.requesterStyle, request.requester, site)
        partner_card = usable_card(request.partnerStyle, request.partner, site)
        # 「第一則訊息」指的是整個聊天室的第一則：後端送來的是這個聊天室最新的 60 則，
        # 只要有人（A 或 B）傳過任何一則，recent 就不會是空的，B 100% 的規則也就不再適用。
        target = resolve_target(requester_card, partner_card, request.blend, first_message=not recent)
        if not has_topic_basis(request, recent, partner_card):
            return ReplySuggestionResponse(
                requestId=request.requestId,
                status="empty",
                mode=mode,
                suggestions=[],
                rejected=[],
                notice=NO_SUGGESTION_NOTICE,
                modelName=None,
                promptVersion=REPLY_PROMPT_VERSION,
                usage=UsageInfo(),
                target=target,
            )
        reactions = partner_reactions(recent)
        prompt = build_reply_prompt(request, mode, requester_card, partner_card, target, reactions, recent)
        result = await run_agent(
            self.agent, prompt, self.model, self.settings.llm_timeout_seconds, "LLM_NOT_CONFIGURED", "LLM_UNAVAILABLE"
        )
        kept, rejected = clean_drafts(list(result.output.suggestions), list(request.excludeTexts))
        suggestions, overflow = rank_candidates(kept, target, partner_asked_question(recent))
        status, notice = notice_for(len(suggestions))
        return ReplySuggestionResponse(
            requestId=request.requestId,
            status=status,
            mode=mode,
            suggestions=suggestions,
            rejected=[*rejected, *overflow],
            notice=notice,
            modelName=model_name_of(result),
            promptVersion=REPLY_PROMPT_VERSION,
            usage=usage_info(result),
            target=target,
        )
