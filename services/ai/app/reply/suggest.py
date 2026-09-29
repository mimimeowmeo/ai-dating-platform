"""產生推薦的主流程（POST /internal/ai/reply-suggestions）。規格第 3、4 節。

流程（2026-09-27 使用者決定新版規則）：
1. 判斷模式（開場／回覆／追問／重啟；最後一則超過 12 小時就是重啟，不論誰傳的）。
2. 準備 A、B 的風格卡（沒有就用 bio 冷啟動），依模式算出這一批 5 則共用的風格目標：
   開場與重啟 B 100%，回覆 A 80%／B 20%，追問只用 A 的語氣。
3. 完全沒有資料根據時（見 has_topic_basis）不呼叫模型，直接回「沒有可推薦的句子」。
4. 計算 B 在這個聊天室的反應熱度；追問時找出 A 上一則還沒得到回答的問題。
5. 開場、追問、重啟先排好【這次的話題安排】（A 也聊過 2、A 的興趣 1、熱門 2，見 topic_plan；2026-09-29）。
6. 組 prompt（見 prompts.build_reply_prompt），請模型產生最多 5 則候選
   （結構化輸出；沒有依據時可以少給甚至不給，不硬湊）。
7. 後處理：單行化、簡轉繁、80 字上限與安全規則、去重。
8. 排序：回覆時 B 剛問了問題，回答類優先；開場、追問、重啟時問句優先；再依模型給的優先度，
   最後依離風格目標多近。追問的 5 則裡「換個說法重問」與「新話題」兩種都要有。
   有話題安排時，則數不超過安排的話題數（追問再加 1 則重問）：話題不夠就少給。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic_ai import Agent
from pydantic_ai.models import Model

from ..config import Settings
from .embeddings import Embedder
from .llm import build_chain, model_name_of, run_agent, structured_output, usage_info
from .prompts import REPLY_INSTRUCTIONS, REPLY_PROMPT_VERSION, TOPIC_MODES, build_reply_prompt
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
from .topic_plan import TopicPlan, TopicPlanner

MIN_RETURN = 3
MAX_RETURN = 5
# 推薦長度只保留這個硬上限；沒有下限，也不再要求貼近平常的字數（2026-09-27 使用者決定）。
MAX_SUGGESTION_CHARS = 80
DUPLICATE_SIMILARITY = 0.8
# 最後一則訊息超過 12 小時就算重啟，不論是誰傳的（2026-09-27 使用者決定，原本是 7 天）。
REVIVE_AFTER = timedelta(hours=12)
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
    - 最後一則超過 12 小時 → revive（重啟），不論最後一則是誰傳的
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
    """B 最近一輪（最後一則 A 訊息之後）有沒有問問題；回覆模式下有的話「回答」類推薦要排前面。"""
    for message in reversed(messages):
        if message.sender == "A":
            return False
        if has_question(message.content):
            return True
    return False


def last_requester_question(messages: list[ChatMessage]) -> str | None:
    """追問模式用：A 最後一輪（最後一則 B 訊息之後）裡最後一個問句，B 還沒回，所以還沒得到回答。

    messages 需由舊到新排序。最後一則不是 A 傳的，或 A 那一輪沒有問句時回傳 None；
    這時追問只寫「新話題」那一種推薦。
    """
    for message in reversed(messages):
        if message.sender == "B":
            return None
        if has_question(message.content):
            return single_line(message.content)
    return None


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
    drafts: list[DraftSuggestion], exclude_texts: list[str], last_question: str | None = None
) -> tuple[list[DraftSuggestion], list[RejectedSuggestion]]:
    """後處理第一階段：整理文字並刪掉不合格的候選（規格 4.4 的 1～4 步）。

    依序：單行化 → 簡轉繁（台灣用語）→ 空白或超過 80 字 → 安全規則 → 與「換一批」要避開的句子太像
    → 跟 A 上一則還沒得到回答的問題幾乎一樣（追問要「換個說法」再問）→ 與前面已保留的候選太像。
    每則被刪的候選都會記下原因，交給後端保存。
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
        if reason is None and last_question and text_similarity(text, last_question) >= DUPLICATE_SIMILARITY:
            reason = "REPEATS_LAST_QUESTION"
        if reason is None and any(text_similarity(text, other.text) >= DUPLICATE_SIMILARITY for other in kept):
            reason = "DUPLICATE"
        if reason:
            rejected.append(RejectedSuggestion(text=single_line(draft.text)[:400], reasonCode=reason))
        else:
            kept.append(draft.model_copy(update={"text": text}))
    return kept, rejected


def _include_both_kinds(ranked: list[Candidate], limit: int = MAX_RETURN) -> list[Candidate]:
    """追問：最多 limit 則裡，「換個說法重問」（intent=reask）與「新話題」兩種都要有（2026-09-27 使用者決定）。

    排好序的前 limit 則如果只有其中一種，而後面還有另一種，就用另一種最好的那則換掉最後一名。
    """
    chosen = ranked[:limit]
    for wanted in (True, False):
        if len(chosen) < limit or any((item.draft.intent == "reask") == wanted for item in chosen):
            continue
        extra = next((item for item in ranked[limit:] if (item.draft.intent == "reask") == wanted), None)
        if extra is not None:
            chosen = [*chosen[:-1], extra]
    return chosen


def suggestion_limit(plan: TopicPlan | None, reask: bool) -> int:
    """這一批最多留幾則：有話題安排時是安排的話題數（追問另加 1 則重問），不超過 5；沒有安排時是 5。

    話題不夠就少給（不硬湊，2026-09-29 使用者決定）：模型多寫的會以 OVER_LIMIT 記錄下來。
    """
    if plan is None or plan.size == 0:
        return MAX_RETURN
    return min(MAX_RETURN, plan.size + (1 if reask else 0))


def rank_candidates(
    drafts: list[DraftSuggestion], target: StyleTarget, mode: Mode, answer_first: bool, limit: int = MAX_RETURN
) -> tuple[list[Suggestion], list[RejectedSuggestion]]:
    """排序並編號（規格 4.1），回傳最多 limit 則推薦（預設 5），以及因為超過上限而被捨棄的候選。

    一批 5 則共用同一個風格目標（見 style.resolve_target），所以每一名的排序規則都一樣：
    1. 回覆模式下 B 剛問了問題時（answer_first），「回答」類排前面；
    2. 開場、追問、重啟要用問句引導 B 分享，問句排前面（程式不硬刪非問句，只往後排）；
    3. 模型給的 priority（1 最推薦）；
    4. 離風格目標越近越前面。
    追問時留下的幾則裡「換個說法重問」與「新話題」兩種都要有（見 _include_both_kinds）。
    第 1 名會被前端用打字動畫填進輸入框。
    styleTarget 依這一批的規則標示：目標完全照 B（開場、重啟，而且 B 有資料）時是 partner，其餘都是 blend。
    """
    if not drafts:
        return [], []
    label: Literal["partner", "blend"] = "partner" if target.source == "partner" else "blend"
    questions_first = mode in TOPIC_MODES
    candidates = [
        Candidate(text=draft.text, draft=draft, distance=style_distance(draft.text, target.stats)) for draft in drafts
    ]
    candidates.sort(
        key=lambda item: (
            0 if answer_first and item.draft.intent == "answer" else 1,
            0 if questions_first and has_question(item.text) else 1,
            item.draft.priority,
            item.distance,
        )
    )
    chosen = _include_both_kinds(candidates, limit) if mode == "follow_up" else candidates[:limit]
    suggestions = [
        Suggestion(
            rank=index + 1,
            text=item.text,
            intent=item.draft.intent,
            styleTarget=label,
            styleDistance=item.distance,
            reason=single_line(item.draft.reason)[:120],
        )
        for index, item in enumerate(chosen)
    ]
    overflow = [
        RejectedSuggestion(text=item.text, reasonCode="OVER_LIMIT")
        for item in candidates
        if not any(item is kept for kept in chosen)
    ]
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
    embedder 給話題安排比對「相近的話題」用（見 topic_plan）；不給時只比對文字是否相同。
    """

    def __init__(self, settings: Settings, model: Model | None = None, embedder: Embedder | None = None):
        """保存設定並建立產生推薦的 agent（agent 本身不綁模型，每次執行時才指定）。"""
        self.settings = settings
        self._model = model
        self._model_built = model is not None
        self.planner = TopicPlanner(embedder)
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
        # 寫法比例依模式決定：開場、重啟 B 100%，回覆 A 80%／B 20%，追問只用 A 的語氣。
        target = resolve_target(requester_card, partner_card, request.blend, mode)
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
        last_question = last_requester_question(recent) if mode == "follow_up" else None
        # 開場、追問、重啟先排好這一批要聊的話題（2026-09-29，見 topic_plan）；回覆模式不需要。
        plan = (
            await self.planner.plan(request, requester_card, partner_card, now, reask=last_question is not None)
            if mode in TOPIC_MODES
            else None
        )
        prompt = build_reply_prompt(
            request, mode, requester_card, partner_card, target, reactions, recent, last_question, plan
        )
        result = await run_agent(
            self.agent, prompt, self.model, self.settings.llm_timeout_seconds, "LLM_NOT_CONFIGURED", "LLM_UNAVAILABLE"
        )
        kept, rejected = clean_drafts(list(result.output.suggestions), list(request.excludeTexts), last_question)
        # 「回答」優先只用在回覆模式；重啟改用開場的做法，不再先回答 B 之前的問題。
        answer_first = mode == "reply" and partner_asked_question(recent)
        limit = suggestion_limit(plan, last_question is not None)
        suggestions, overflow = rank_candidates(kept, target, mode, answer_first, limit)
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
