"""開場、追問、重啟的「話題安排」：這一批推薦各聊哪個話題（2026-09-29 使用者決定）。

一批最多 5 則，每則只聊一個話題，依下面的名額挑；挑不滿的名額由「熱門」補，再不夠就少給（不硬湊）：
- 「A 也聊過」2 則：B 的話題裡，跟 A 自己常聊的話題相近的（A 可以先分享自己的經驗再問 B）。
- 「A 的興趣」1 則：B 的話題裡，跟 A 的標籤或自我介紹相近的。
- 「熱門」2 則：B 聊得最熱絡的話題，熱度 = 權重 × 0.5^(距今天數 ÷ 14)。
追問而且 A 上一則的問題還沒得到回答時，另外寫 1 則「換個說法重問」，新話題只排 4 則（A 也聊過 1、A 的興趣 1、熱門 2）。

每個人的話題從哪裡來（見 style.uses_recent_topics）：
- 聊天紀錄夠多（≥ 30 則真人訊息，風格卡上也有話題特徵句）：用聊天整理出的話題。
- 不夠時用檔案：標籤（興趣、嗜好、喜歡的食物、其他標籤）與自我介紹（2026-09-29 使用者補充）。
  A 沒有聊天紀錄時，「A 也聊過」改用 A 的檔案，跟「A 的興趣」是同一組資料，兩種名額（共 3 則）一起挑。
  B 有聊天紀錄但聊天話題用完時，剩下的名額從 B 的檔案挑（B 主頁上寫的，A 看得到）。

「相近」怎麼判斷（門檻是 2026-09-29 用 gemini-embedding-2 校準的，見規格第 21 節）：
- 兩個短詞（話題名稱、標籤）：文字相同，或向量相似度 ≥ 0.88。
- 短詞對自我介紹：自我介紹裡直接寫到這個詞，或向量相似度 ≥ 0.76。兩段自我介紹之間不比。
向量服務沒設定或失敗時只看文字，挑不到的名額都給熱門話題，推薦照樣產生。
"""

import asyncio
import math
import re
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Sequence

from .embeddings import Embedder, Purpose
from .errors import AIServiceError
from .schemas import ProfileSnapshot, ReplySuggestionRequest, StyleCard
from .style import MIN_BIO_CHARS, uses_recent_topics
from .textutil import as_utc, single_line, visible_chars

# chat：聊天整理出的話題名稱；tag：檔案上的一個標籤；bio：整段自我介紹。
Source = Literal["chat", "tag", "bio"]

# 話題特徵句的格式（extraction.build_facets 的 topic 範本），用來取出話題名稱，例如「登山」。
_TOPIC_STATEMENT = re.compile(r"^聊到「(.+)」會比較熱絡$")
HALF_LIFE_DAYS = 14.0  # 熱度的半衰期：兩週前最後聊到的話題，熱度剩一半
SHORT_MATCH = 0.88  # 兩個短詞算相近的向量相似度門檻（校準：無關的配對只有 0.07% 會超過）
BIO_MATCH = 0.76  # 短詞對自我介紹的門檻；自我介紹的分數普遍較低，空泛的自我介紹最高也有 0.74
MIN_LITERAL_CHARS = 2  # 「自我介紹直接寫到」的詞至少 2 個字，避免「貓」這種單字到處都對得到
TOPIC_TITLE = "話題"  # 短詞當「文件」轉向量時的標題
CACHE_SIZE = 2048  # 記憶體裡最多留幾個向量（每個約 3 KB，全滿約 6 MB）
PROFILE_TAG_LISTS = ("interests", "hobbies", "foods", "traits")  # 當成話題的標籤欄位（交友目標不算）
BIO_OPTION = "B 的自我介紹"  # 從 B 的檔案挑話題時，自我介紹在清單裡的名稱


@dataclass(frozen=True)
class Quota:
    """各種話題的名額。"""

    shared: int  # A 也聊過
    interest: int  # A 的興趣
    hot: int  # 熱門

    @property
    def total(self) -> int:
        """這一批總共要排幾個新話題。"""
        return self.shared + self.interest + self.hot


FULL_QUOTA = Quota(shared=2, interest=1, hot=2)
REASK_QUOTA = Quota(shared=1, interest=1, hot=2)  # 追問時另外還有 1 則換個說法重問


@dataclass(frozen=True)
class TopicItem:
    """一個可以拿來聊的話題：text 是話題名稱或標籤；自我介紹時是全文。heat 只有聊天話題有。"""

    text: str
    source: Source
    heat: float = 0.0

    @property
    def key(self) -> str:
        """比對用的字串：去掉空白、不分大小寫（「Live 音樂」與「live音樂」算同一個）。"""
        return _normalize(self.text)


@dataclass(frozen=True)
class PlannedTopic:
    """話題安排的一格：topic 是 B 的哪個話題；anchor 是 A 那邊對到的話題（熱門話題沒有，是 None）。"""

    topic: TopicItem
    anchor: TopicItem | None = None


@dataclass(frozen=True)
class TopicPlan:
    """這一批推薦的話題安排。

    - slots：指定好的話題，順序是 A 也聊過 → A 的興趣 → 熱門。
    - profile_picks：另外要從 B 的檔案挑幾個話題（B 沒有聊天紀錄，或聊天話題已經用完）；
      可以挑的放在 profile_options（上面沒用到的標籤，自我介紹寫成「B 的自我介紹」）。
      沒有自我介紹時最多挑標籤的個數；有自我介紹時剩下的名額都開放（自我介紹裡常有好幾個話題）。
    """

    slots: tuple[PlannedTopic, ...] = ()
    profile_picks: int = 0
    profile_options: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        """這次總共要寫幾則新話題。"""
        return len(self.slots) + self.profile_picks

    @property
    def uses_requester_profile(self) -> bool:
        """有沒有哪一格要從 A 的標籤或自我介紹切入；有的話 prompt 要放 A 的完整檔案。"""
        return any(slot.anchor is not None and slot.anchor.source != "chat" for slot in self.slots)

    @property
    def uses_partner_profile(self) -> bool:
        """有沒有用到 B 的檔案（標籤、自我介紹）；有的話 prompt 要放 B 的完整檔案。"""
        return self.profile_picks > 0 or any(slot.topic.source != "chat" for slot in self.slots)


@dataclass(frozen=True)
class _Match:
    """B 的一個話題跟 A 的一個話題相近。tier 越小越可靠：0 文字相同、1 自我介紹直接寫到、2 向量相近。"""

    topic: TopicItem
    anchor: TopicItem
    tier: int
    score: float


def _normalize(text: str) -> str:
    """去掉所有空白並轉小寫，拿來比對文字是否相同。"""
    return "".join(text.split()).casefold()


def _unique(items: list[TopicItem]) -> list[TopicItem]:
    """去掉重複的話題（比對方式見 TopicItem.key），保留第一次出現的。"""
    seen: set[str] = set()
    kept: list[TopicItem] = []
    for item in items:
        if item.key and item.key not in seen:
            seen.add(item.key)
            kept.append(item)
    return kept


def topic_name(statement: str) -> str:
    """從話題特徵句取出話題名稱：「聊到「登山」會比較熱絡」→「登山」；格式不同時回傳整句。"""
    match = _TOPIC_STATEMENT.match(statement)
    return match.group(1) if match else statement


def topic_heat(weight: float, last_seen: datetime | None, now: datetime) -> float:
    """話題熱度 = 權重 × 0.5^(距今天數 ÷ 14)。

    沒有時間（舊版風格卡）時只看權重；時間比 now 還晚時當作剛聊過（不會超過權重本身）。
    """
    if last_seen is None:
        return weight
    days = max(0.0, (as_utc(now) - as_utc(last_seen)).total_seconds() / 86400)
    return weight * 0.5 ** (days / HALF_LIFE_DAYS)


def chat_topics(card: StyleCard, now: datetime) -> list[TopicItem]:
    """風格卡上的話題特徵句 → 話題清單，熱度由高到低（一樣熱時依名稱排，讓結果固定）。"""
    items = [
        TopicItem(topic_name(facet.statement), "chat", topic_heat(facet.weight, facet.lastSeenAt, now))
        for facet in card.facets
        if facet.kind == "topic"
    ]
    items.sort(key=lambda item: (-item.heat, item.text))
    return _unique(items)


def profile_topics(profile: ProfileSnapshot) -> list[TopicItem]:
    """檔案上的話題：標籤依興趣、嗜好、喜歡的食物、其他標籤的順序（去掉重複），最後是自我介紹。

    自我介紹至少要 10 個字（跟冷啟動風格卡的標準相同），太短的通常只是打招呼。
    """
    items = [TopicItem(single_line(tag), "tag") for field in PROFILE_TAG_LISTS for tag in getattr(profile, field)]
    bio = single_line(profile.bio)
    if visible_chars(bio) >= MIN_BIO_CHARS:
        items.append(TopicItem(bio, "bio"))
    return _unique(items)


def _purpose(item: TopicItem) -> Purpose:
    """短詞當「文件」、自我介紹當「查詢」轉向量（校準時就是這樣比的）。"""
    return "query" if item.source == "bio" else "document"


class TopicVectors:
    """話題與自我介紹的向量，留在記憶體裡重複使用。

    Gemini 免費額度每分鐘只能轉 100 段文字（一次送多段，每段各算一次），而標籤是固定的詞彙、
    聊天話題也常重複，留著可以讓大多數請求不必再呼叫向量服務。
    上限 CACHE_SIZE 個，超過時丟掉最久沒用到的（LRU）。整個 AI 服務共用一份，重啟就清空。
    """

    def __init__(self, embedder: Embedder | None, size: int = CACHE_SIZE):
        """embedder 為 None（或沒設定 API key）時完全不轉向量，只能比對文字。"""
        self.embedder = embedder
        self.size = size
        self._cache: OrderedDict[tuple[str, str, str], tuple[array, float]] = OrderedDict()

    def _cache_key(self, text: str, purpose: Purpose) -> tuple[str, str, str]:
        """快取的鍵：模型名稱、用途、文字；換模型時舊向量自然不會被用到。"""
        return (self.embedder.model if self.embedder else "", purpose, text)

    async def lookup(self, items: Sequence[TopicItem]) -> dict[tuple[str, str], tuple[array, float]]:
        """取得這些話題的向量與長度，鍵是（文字, 用途）；缺的一次送去轉換。

        文件（短詞）與查詢（自我介紹）要分兩次呼叫，兩次同時送出。向量服務沒設定或失敗時，
        只回傳快取裡已經有的，呼叫端會把缺向量的配對當作不相近。
        """
        if self.embedder is None or not self.embedder.configured:
            return {}
        found: dict[tuple[str, str], tuple[array, float]] = {}
        missing: set[tuple[str, Purpose]] = set()
        for item in items:
            pair = (item.text, _purpose(item))
            entry = self._cache.get(self._cache_key(*pair))
            if entry is None:
                missing.add(pair)
            else:
                self._cache.move_to_end(self._cache_key(*pair))
                found[pair] = entry
        batches: list[tuple[Purpose, str | None, list[str]]] = [
            ("document", TOPIC_TITLE, sorted(text for text, purpose in missing if purpose == "document")),
            ("query", None, sorted(text for text, purpose in missing if purpose == "query")),
        ]
        batches = [batch for batch in batches if batch[2]]
        results = await asyncio.gather(
            *(self.embedder.embed(texts, purpose, title) for purpose, title, texts in batches), return_exceptions=True
        )
        for (purpose, _, texts), result in zip(batches, results):
            if isinstance(result, AIServiceError):
                continue
            if isinstance(result, BaseException):
                raise result
            for text, vector in zip(texts, result):
                found[(text, purpose)] = self._store(self._cache_key(text, purpose), vector)
        return found

    def _store(self, key: tuple[str, str, str], vector: list[float]) -> tuple[array, float]:
        """存一個向量與它的長度（用 4 bytes 的 float 存，比 Python 的 list 省很多記憶體），超過上限就丟掉最舊的。"""
        values = array("f", vector)
        entry = (values, math.sqrt(math.sumprod(values, values)))
        self._cache[key] = entry
        self._cache.move_to_end(key)
        while len(self._cache) > self.size:
            self._cache.popitem(last=False)
        return entry


def _match(
    topic: TopicItem, anchor: TopicItem, vectors: dict[tuple[str, str], tuple[array, float]]
) -> _Match | None:
    """判斷 B 的話題 topic 跟 A 的話題 anchor 相不相近；不相近（或沒辦法判斷）時回傳 None。"""
    if topic.source == "bio" and anchor.source == "bio":
        return None
    if topic.source != "bio" and anchor.source != "bio":
        if topic.key == anchor.key:
            return _Match(topic, anchor, 0, 1.0)
        threshold = SHORT_MATCH
    else:
        word, bio = (topic, anchor) if anchor.source == "bio" else (anchor, topic)
        if len(word.key) >= MIN_LITERAL_CHARS and word.key in bio.key:
            return _Match(topic, anchor, 1, 1.0)
        threshold = BIO_MATCH
    first = vectors.get((topic.text, _purpose(topic)))
    second = vectors.get((anchor.text, _purpose(anchor)))
    if first is None or second is None or first[1] == 0 or second[1] == 0:
        return None
    score = math.sumprod(first[0], second[0]) / (first[1] * second[1])
    return _Match(topic, anchor, 2, score) if score >= threshold else None


def _pick(matches: list[_Match], quota: int, used_topics: set[str], used_anchors: set[str]) -> list[PlannedTopic]:
    """依可靠程度與相似度由高到低挑配對，最多 quota 個；B 的每個話題、A 的每個話題都只用一次。

    A 的話題也只用一次：同一個「A 也喜歡旅行」拿去接 B 的兩個話題，兩則推薦會變得很像。
    """
    picked: list[PlannedTopic] = []
    for match in sorted(matches, key=lambda item: (item.tier, -item.score, item.topic.text, item.anchor.text)):
        if len(picked) >= quota:
            break
        if match.topic.key in used_topics or match.anchor.key in used_anchors:
            continue
        used_topics.add(match.topic.key)
        used_anchors.add(match.anchor.key)
        picked.append(PlannedTopic(match.topic, match.anchor))
    return picked


class TopicPlanner:
    """排出這一批的話題安排（整個 AI 服務共用一個，向量快取在裡面）。"""

    def __init__(self, embedder: Embedder | None = None):
        """embedder 為 None 時只比對文字（測試或沒有設定向量服務時）。"""
        self.vectors = TopicVectors(embedder)

    async def plan(
        self,
        request: ReplySuggestionRequest,
        requester_card: StyleCard,
        partner_card: StyleCard,
        now: datetime,
        reask: bool = False,
    ) -> TopicPlan:
        """依 A、B 的話題來源與名額排出話題安排（名額見 FULL_QUOTA／REASK_QUOTA）。

        requester_card、partner_card 要傳 usable_card 處理過的風格卡；reask=True 代表追問時
        A 上一則的問題還沒得到回答（這時另外寫 1 則換個說法重問，新話題少排 1 則）。
        """
        quota = REASK_QUOTA if reask else FULL_QUOTA
        partner_chat = uses_recent_topics(partner_card)
        requester_chat = uses_recent_topics(requester_card)
        partner_profile = profile_topics(request.partner)
        topics = chat_topics(partner_card, now) if partner_chat else partner_profile
        interest_anchors = profile_topics(request.requester)
        shared_anchors = chat_topics(requester_card, now) if requester_chat else []
        anchors = [*shared_anchors, *interest_anchors]
        vectors = await self.vectors.lookup([*topics, *anchors]) if topics and anchors else {}

        def matches_with(candidates: list[TopicItem]) -> list[_Match]:
            """B 的每個話題跟 candidates 裡每個 A 的話題兩兩比，留下相近的。"""
            found = (_match(topic, anchor, vectors) for topic in topics for anchor in candidates)
            return [match for match in found if match is not None]

        used_topics: set[str] = set()
        used_anchors: set[str] = set()
        if requester_chat:
            slots = _pick(matches_with(shared_anchors), quota.shared, used_topics, used_anchors)
            slots += _pick(matches_with(interest_anchors), quota.interest, used_topics, used_anchors)
        else:
            # A 沒有聊天紀錄：「A 也聊過」也用 A 的檔案，跟「A 的興趣」同一組資料，名額一起挑。
            slots = _pick(matches_with(interest_anchors), quota.shared + quota.interest, used_topics, used_anchors)
        remaining = quota.total - len(slots)
        if partner_chat:
            hot = [topic for topic in topics if topic.key not in used_topics][:remaining]
            slots += [PlannedTopic(topic) for topic in hot]
            used_topics.update(topic.key for topic in hot)
            remaining -= len(hot)
        # 可以從 B 的檔案挑的話題：還沒用到的標籤，以及自我介紹。自我介紹常常不只一個話題
        # （例如「週末常去爬山，也喜歡貓」），已經配對過也照樣可以再從裡面找別的話題，
        # 所以有自我介紹時剩下的名額都開放；真的找不到就少寫（見 prompts.describe_plan）。
        options = tuple(
            BIO_OPTION if item.source == "bio" else item.text
            for item in partner_profile
            if item.source == "bio" or item.key not in used_topics
        )
        picks = remaining if BIO_OPTION in options else min(remaining, len(options))
        return TopicPlan(slots=tuple(slots), profile_picks=picks, profile_options=options if picks else ())
