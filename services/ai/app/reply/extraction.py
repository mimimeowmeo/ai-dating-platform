"""背景萃取：風格卡（POST /internal/ai/style-profile）與聊天室摘要（POST /internal/ai/conversation-summary）。

兩者都用「萃取模型鏈」（預設自架 Ollama），不佔用 Gemini 的免費額度；
風格卡的特徵句向量化則用 Gemini（所有向量必須同一個模型）。

風格卡的作法（規格 5.3）：
1. 只留真人訊息（AI 來源的訊息再排除一次當保險）。
2. 依真人訊息數決定信心與寫法來源（≥30 聊天為主／1～29 聊天與 bio 各半／0 只用 bio／都沒有）。
3. 寫法數值用程式統計；語氣、話題、習慣交給模型，分批萃取（map）後在程式裡合併（reduce）。
4. 抽象化檢查：特徵句不得與任何原始訊息共用連續 8 個字，也不得含數字、網址、帳號。
5. 需要時把特徵句向量化，讓後端存進 pgvector 供檢索。
"""

from dataclasses import dataclass, field
from typing import Literal

from pydantic_ai import Agent
from pydantic_ai.models import Model

from ..config import Settings
from .embeddings import Embedder
from .errors import AIServiceError
from .llm import build_chain, model_name_of, run_agent, structured_output
from .prompts import (
    STYLE_MAP_INSTRUCTIONS,
    STYLE_MAP_PROMPT_VERSION,
    SUMMARY_INSTRUCTIONS,
    SUMMARY_PROMPT_VERSION,
    build_style_map_prompt,
    build_summary_prompt,
)
from .schemas import (
    OwnMessage,
    StyleCard,
    StyleFacet,
    StyleMapResult,
    StyleProfileRequest,
    StyleProfileResponse,
    StyleStats,
    SummaryDraft,
    SummaryRequest,
    SummaryResponse,
)
from .chunking import sort_messages
from .style import (
    HIGH_CONFIDENCE_MESSAGES,
    MIN_BIO_CHARS,
    SITE_DEFAULT_STATS,
    blend_stats,
    compute_style_stats,
    stats_from_bio,
)
from .textutil import (
    as_utc,
    build_ngram_index,
    estimate_tokens,
    has_identifying_detail,
    shares_long_substring,
    single_line,
    text_similarity,
    to_taiwan_traditional,
    visible_chars,
)

MAP_BATCH_TOKENS = 6000  # 每批送給萃取模型的訊息量（估計 tokens）；平均 7.1 字的訊息約 500 則
AI_TOPIC_WEIGHT = 0.3  # [AI話題] 裡的話題喜好降權（規格 5.3）
BIO_WEIGHT = {"chat": 0.2, "mixed": 0.5, "bio": 1.0}  # bio 在合併時的權重，依寫法來源而定
NGRAM_SIZE = 8  # 特徵句不得與原始訊息共用連續 8 個字以上
MERGE_SIMILARITY = 0.6  # 不同批次的描述相似度 ≥ 0.6 視為同一條
MAX_VOICE_NOTES = 6
MAX_TOPICS = 20
MAX_AVOID = 5
MAX_HABITS = 5
STATEMENT_CHARS = 40


@dataclass
class _Tally:
    """合併（reduce）時用來累計某一條描述的分數與出現次數。"""

    text: str
    score: float = 0.0
    evidence: int = 0


@dataclass
class _Merged:
    """合併後的結果：語氣描述、話題、會避開的話題、習慣，各自是依分數排序的清單。"""

    voice: list[_Tally] = field(default_factory=list)
    topics: list[_Tally] = field(default_factory=list)
    avoid: list[_Tally] = field(default_factory=list)
    habits: list[_Tally] = field(default_factory=list)


def split_batches(lines: list[str], max_tokens: int = MAP_BATCH_TOKENS) -> list[list[str]]:
    """把訊息行依估計 token 數切成多批，每批不超過 max_tokens（單則超長時自成一批）。"""
    batches: list[list[str]] = []
    current: list[str] = []
    used = 0
    for line in lines:
        cost = estimate_tokens(line) + 2
        if current and used + cost > max_tokens:
            batches.append(current)
            current, used = [], 0
        current.append(line)
        used += cost
    if current:
        batches.append(current)
    return batches


def recency_weights(count: int) -> list[float]:
    """每一批的時間權重：最舊的一批 0.5 左右、最新的一批 1.0，讓近期的樣子影響較大。"""
    return [0.5 + 0.5 * (index + 1) / count for index in range(count)]


def _add_similar(tallies: list[_Tally], text: str, score: float) -> None:
    """把一條描述加進累計清單；與既有描述相似度 ≥ 0.6 就合併成同一條（分數與次數累加）。"""
    for tally in tallies:
        if text_similarity(tally.text, text) >= MERGE_SIMILARITY:
            tally.score += score
            tally.evidence += 1
            return
    tallies.append(_Tally(text=text, score=score, evidence=1))


def merge_map_results(results: list[tuple[StyleMapResult, float]]) -> _Merged:
    """把每一批的萃取結果合併（reduce 步驟），全部在程式裡完成、不再呼叫模型。

    - 語氣與習慣：相似的描述合併，分數 = 各批權重相加。
    - 話題：相似的話題名稱合併（例如「登山」與「登山山景」），分數 = 熱度/5 × 批次權重；
      來自 [AI話題] 的再乘 0.3（降權）。
    - 會避開的話題：相似的合併，出現幾次、權重多少就累加多少。
    """
    merged = _Merged()
    for result, weight in results:
        for note in result.voiceNotes:
            _add_similar(merged.voice, single_line(note), weight)
        for habit in result.habits:
            _add_similar(merged.habits, single_line(habit), weight)
        for item in result.topics:
            score = (item.heat / 5) * weight * (AI_TOPIC_WEIGHT if item.fromAiTopic else 1.0)
            _add_similar(merged.topics, single_line(item.topic), score)
        for topic in result.avoidTopics:
            _add_similar(merged.avoid, single_line(topic), weight)
    for tallies in (merged.voice, merged.topics, merged.avoid, merged.habits):
        tallies.sort(key=lambda tally: (-tally.score, -tally.evidence, tally.text))
    return merged


def is_abstract_enough(statement: str, ngram_index: set[str]) -> bool:
    """抽象化檢查：不得與原始訊息共用連續 8 個字，也不得含數字、網址、帳號或 Email。"""
    return not shares_long_substring(statement, ngram_index, NGRAM_SIZE) and not has_identifying_detail(statement)


def build_facets(merged: _Merged, ngram_index: set[str]) -> tuple[list[str], list[StyleFacet], int]:
    """把合併結果轉成語氣描述與特徵句，並做抽象化檢查。

    回傳（語氣描述清單、特徵句清單、被抽象化檢查刪掉的數量）。
    特徵句的 weight 以同類中最高分為 1 做正規化；文字一律轉台灣繁體並限制 40 字。
    """
    rejected = 0
    voice_notes: list[str] = []
    facets: list[StyleFacet] = []

    def accept(text: str) -> str | None:
        """整理一條描述並檢查；通過回傳整理後的文字，不通過回傳 None 並計數。"""
        nonlocal rejected
        cleaned = to_taiwan_traditional(single_line(text))[:STATEMENT_CHARS]
        if not cleaned or not is_abstract_enough(cleaned, ngram_index):
            rejected += 1
            return None
        return cleaned

    def add_group(
        tallies: list[_Tally], kind: Literal["topic", "tone", "habit", "avoid"], limit: int, template: str
    ) -> None:
        """把某一類的前幾名轉成特徵句；template 用 {} 代表描述本身。"""
        top_score = max((tally.score for tally in tallies), default=0.0) or 1.0
        for tally in tallies[:limit]:
            statement = accept(template.format(tally.text))
            if statement is None:
                continue
            if kind == "tone":
                voice_notes.append(statement)
            facets.append(
                StyleFacet(kind=kind, statement=statement, weight=round(min(1.0, tally.score / top_score), 3), evidence=tally.evidence)
            )

    add_group(merged.voice, "tone", MAX_VOICE_NOTES, "{}")
    add_group(merged.topics, "topic", MAX_TOPICS, "聊到「{}」會比較熱絡")
    add_group(merged.habits, "habit", MAX_HABITS, "{}")
    add_group(merged.avoid, "avoid", MAX_AVOID, "對「{}」話題比較冷淡")
    return voice_notes, facets, rejected


def decide_profile_basis(
    human: list[OwnMessage], bio: str, site: StyleStats
) -> tuple[Literal["high", "low", "none"], Literal["chat", "mixed", "bio", "none"], StyleStats]:
    """依真人訊息數與 bio 決定信心、寫法來源與數值統計（規格 5.3 的冷啟動表）。

    - ≥ 30 則：high／chat／本人統計。
    - 1～29 則：low／mixed／本人統計與全站平均各半。
    - 0 則且 bio ≥ 10 字：low／bio／全站平均（語助詞參考 bio）。
    - 其餘：none／none／全站平均。
    """
    if len(human) >= HIGH_CONFIDENCE_MESSAGES:
        return "high", "chat", compute_style_stats(human)
    if human:
        own = compute_style_stats(human)
        mixed = blend_stats(own, site, 0.5).model_copy(update={"messageCount": own.messageCount})
        return "low", "mixed", mixed
    if visible_chars(bio) >= MIN_BIO_CHARS:
        return "low", "bio", stats_from_bio(bio, site)
    return "none", "none", site.model_copy()


class StyleProfileBuilder:
    """萃取風格卡的服務物件；model 與 embedder 可注入，方便測試。"""

    def __init__(self, settings: Settings, model: Model | None = None, embedder: Embedder | None = None):
        """保存設定並建立萃取用的 agent；model 不給時等第一次使用才依設定建立模型鏈。"""
        self.settings = settings
        self._model = model
        self._model_built = model is not None
        self.embedder = embedder or Embedder(settings)
        # 萃取模型是 Ollama（本機、自架或 Ollama Cloud）；輸出方式依部署位置決定，見 llm.structured_output。
        self.agent = Agent(
            output_type=structured_output(StyleMapResult, settings.extraction_models, settings),
            instructions=STYLE_MAP_INSTRUCTIONS,
            retries={"output": 2},
            name="style-map",
        )

    @property
    def model(self) -> Model | None:
        """取得（必要時建立）萃取模型鏈；沒有任何可用模型時為 None。"""
        if not self._model_built:
            self._model = build_chain(self.settings.extraction_models, self.settings, "extraction")
            self._model_built = True
        return self._model

    async def _map(self, lines: list[str]):
        """對一批訊息呼叫萃取模型（map 步驟），回傳 agent 結果。"""
        return await run_agent(
            self.agent,
            build_style_map_prompt(lines),
            self.model,
            self.settings.extraction_timeout_seconds,
            "EXTRACTION_NOT_CONFIGURED",
            "EXTRACTION_UNAVAILABLE",
        )

    async def build(self, request: StyleProfileRequest) -> StyleProfileResponse:
        """執行完整的風格卡萃取並回傳結果（含特徵句向量，若有要求且向量服務可用）。

        信心為 none 時不呼叫任何模型，直接回傳只有數值的卡片。
        特徵句向量化失敗時不讓整件事失敗：回傳 facetVectors=None，後端之後可以再用 /embed 補。
        """
        human = sorted(
            (m for m in request.messages if m.origin == "human" and m.content.strip()),
            key=lambda message: (as_utc(message.createdAt), message.id),
        )
        bio = request.bio.strip()
        site = request.siteStats or SITE_DEFAULT_STATS
        confidence, source, stats = decide_profile_basis(human, bio, site)
        card = StyleCard(
            userId=request.userId,
            confidence=confidence,
            sampleSource=source,
            stats=stats,
            bioSample=bio[:300] if source in {"bio", "mixed"} else "",
            messageCount=len(human),
            windowFrom=request.windowFrom,
            windowTo=request.windowTo,
            promptVersion=STYLE_MAP_PROMPT_VERSION,
        )
        if confidence == "none":
            return StyleProfileResponse(card=card)

        lines = [f"[AI話題] {single_line(m.content)}" if m.inAiTopic else single_line(m.content) for m in human]
        batches = split_batches(lines) if lines else []
        weighted = list(zip(batches, recency_weights(len(batches)))) if batches else []
        if visible_chars(bio) >= MIN_BIO_CHARS:
            weighted.append(([f"[自我介紹] {single_line(bio)}"], BIO_WEIGHT[source]))
        results: list[tuple[StyleMapResult, float]] = []
        model_name = None
        for batch_lines, weight in weighted:
            result = await self._map(batch_lines)
            results.append((result.output, weight))
            model_name = model_name_of(result) or model_name
        voice_notes, facets, rejected = build_facets(
            merge_map_results(results), build_ngram_index([m.content for m in human], NGRAM_SIZE)
        )
        card = card.model_copy(update={"voiceNotes": voice_notes, "facets": facets, "modelName": model_name})
        vectors = None
        if request.embedFacets and facets and self.embedder.configured:
            try:
                vectors = await self.embedder.embed([facet.statement for facet in facets], "document", "風格特徵")
            except AIServiceError:
                vectors = None
        return StyleProfileResponse(
            card=card,
            facetVectors=vectors,
            embeddingModel=self.embedder.model if vectors is not None else None,
            rejectedStatements=rejected,
        )


class ConversationSummarizer:
    """更新聊天室摘要的服務物件（規格 5.5）；model 可注入，方便測試。"""

    def __init__(self, settings: Settings, model: Model | None = None):
        """保存設定並建立摘要用的 agent；model 不給時等第一次使用才依設定建立模型鏈。"""
        self.settings = settings
        self._model = model
        self._model_built = model is not None
        self.agent = Agent(
            output_type=structured_output(SummaryDraft, settings.extraction_models, settings),
            instructions=SUMMARY_INSTRUCTIONS,
            retries={"output": 2},
            name="conversation-summary",
        )

    @property
    def model(self) -> Model | None:
        """取得（必要時建立）萃取模型鏈；摘要與風格卡共用同一組設定。"""
        if not self._model_built:
            self._model = build_chain(self.settings.extraction_models, self.settings, "extraction")
            self._model_built = True
        return self._model

    async def summarize(self, request: SummaryRequest) -> SummaryResponse:
        """用「舊摘要＋新訊息」產生新摘要；超過字數上限時在上限處截斷。"""
        ordered = sort_messages(request.messages)
        result = await run_agent(
            self.agent,
            build_summary_prompt(request.previousSummary, ordered, request.maxChars),
            self.model,
            self.settings.extraction_timeout_seconds,
            "EXTRACTION_NOT_CONFIGURED",
            "EXTRACTION_UNAVAILABLE",
        )
        summary = to_taiwan_traditional(result.output.summary.strip())[: request.maxChars]
        return SummaryResponse(
            conversationId=request.conversationId,
            summary=summary,
            untilMessageId=ordered[-1].id,
            modelName=model_name_of(result),
            promptVersion=SUMMARY_PROMPT_VERSION,
        )
