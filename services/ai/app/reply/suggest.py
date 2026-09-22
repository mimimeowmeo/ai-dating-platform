"""產生推薦的主流程（POST /internal/ai/reply-suggestions）。規格第 3、4 節。

流程：
1. 判斷模式（開場／回覆／追問／重啟）。
2. 準備 A、B 的風格卡（沒有就用 bio 冷啟動），算出第 1 則與其餘的風格目標。
3. 計算 B 在這個聊天室的反應熱度。
4. 組 prompt，請模型產生 5 則候選（結構化輸出，格式不合會自動重試）。
5. 後處理：單行化、簡轉繁、長度與安全規則、去重。
6. 排序：第 1 則挑最接近「第 1 則目標」的 B 風格候選；其餘依內容優先度與風格距離。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

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
    RejectedSuggestion,
    ReplySuggestionRequest,
    ReplySuggestionResponse,
    StyleStats,
    StyleTargets,
    Suggestion,
)
from .style import SITE_DEFAULT_STATS, partner_reactions, resolve_targets, style_distance, usable_card
from .textutil import as_utc, has_question, single_line, text_similarity, to_taiwan_traditional, visible_chars

MIN_RETURN = 3
MAX_RETURN = 5
MAX_SUGGESTION_CHARS = 80
DUPLICATE_SIMILARITY = 0.8
REVIVE_AFTER = timedelta(days=7)


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


@dataclass
class Candidate:
    """通過後處理的一則候選，加上它與兩個風格目標的距離。"""

    text: str
    draft: DraftSuggestion
    distance_first: float
    distance_others: float


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
    drafts: list[DraftSuggestion],
    first_target: StyleStats,
    others_target: StyleStats,
    first_is_partner: bool,
    answer_first: bool,
) -> tuple[list[Suggestion], list[RejectedSuggestion]]:
    """排序並編號（規格 4.1），回傳最多 5 則推薦，以及因為超過 5 則而被捨棄的候選。

    - 第 1 則：優先從 styleTarget=partner 的候選裡挑「最接近第 1 則目標」的；
      模型沒有給 partner 候選時，從全部候選裡挑最接近的。
    - 其餘：B 剛問了問題時「回答」類優先；再依模型給的 priority；最後依與「其餘目標」的距離。
    第 1 則的 styleTarget 會依實際來源標示：B 有資料時是 partner，退回 A 的寫法時是 blend。
    """
    if not drafts:
        return [], []
    candidates = [
        Candidate(
            text=draft.text,
            draft=draft,
            distance_first=style_distance(draft.text, first_target),
            distance_others=style_distance(draft.text, others_target),
        )
        for draft in drafts
    ]
    partner_pool = [item for item in candidates if item.draft.styleTarget == "partner"] or candidates
    first = min(partner_pool, key=lambda item: (item.distance_first, item.draft.priority))
    rest = [item for item in candidates if item is not first]
    rest.sort(
        key=lambda item: (
            0 if answer_first and item.draft.intent == "answer" else 1,
            item.draft.priority,
            item.distance_others,
        )
    )
    ordered = [first, *rest]
    suggestions = [
        Suggestion(
            rank=index + 1,
            text=item.text,
            intent=item.draft.intent,
            styleTarget=("partner" if first_is_partner else "blend") if index == 0 else "blend",
            styleDistance=item.distance_first if index == 0 else item.distance_others,
            reason=single_line(item.draft.reason)[:120],
        )
        for index, item in enumerate(ordered[:MAX_RETURN])
    ]
    overflow = [RejectedSuggestion(text=item.text, reasonCode="OVER_LIMIT") for item in ordered[MAX_RETURN:]]
    return suggestions, overflow


def notice_for(count: int) -> tuple[str, str | None]:
    """依最後保留的則數決定狀態與提示文字（規格 4.1：不足 3 則顯示剩下的並提示，不硬湊）。"""
    if count >= MIN_RETURN:
        return "ok", None
    if count > 0:
        return "partial", f"只找到 {count} 則合適的建議"
    return "empty", "這次沒有產生合適的建議，請再試一次"


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
        """執行完整的推薦流程並回傳結果；模型不可用時丟出 AIServiceError（LLM_*）。"""
        now = as_utc(request.now) if request.now else datetime.now(timezone.utc)
        recent = sort_chat(list(request.recentMessages))
        mode = request.mode or detect_mode(recent, now)
        site = request.siteStats or SITE_DEFAULT_STATS
        requester_card = usable_card(request.requesterStyle, request.requester, site)
        partner_card = usable_card(request.partnerStyle, request.partner, site)
        first_target, others_target, first_source = resolve_targets(requester_card, partner_card, request.blend)
        reactions = partner_reactions(recent)
        prompt = build_reply_prompt(
            request, mode, requester_card, partner_card, first_target, others_target, first_source, reactions, recent
        )
        result = await run_agent(
            self.agent, prompt, self.model, self.settings.llm_timeout_seconds, "LLM_NOT_CONFIGURED", "LLM_UNAVAILABLE"
        )
        kept, rejected = clean_drafts(list(result.output.suggestions), list(request.excludeTexts))
        suggestions, overflow = rank_candidates(
            kept, first_target, others_target, first_source == "partner", partner_asked_question(recent)
        )
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
            targets=StyleTargets(first=first_target, others=others_target, firstSource=first_source),
        )
