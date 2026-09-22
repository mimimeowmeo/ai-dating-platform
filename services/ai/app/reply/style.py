"""風格數值：統計寫法、混合 A／B 的風格目標、計算候選離目標多遠，以及 B 的反應熱度。

這裡全部是確定性的計算（不呼叫模型），對應規格 4.2、5.3、5.4：
- compute_style_stats：從一個人的訊息算出寫法統計（字數、emoji、語助詞…）。
- cold_start_card：沒有風格卡時，用 bio 做冷啟動。
- resolve_target：依「整個聊天室有沒有訊息」算出這一批 5 則共用的目標
  （聊天室的第一則訊息 B 100%；有人傳過訊息之後 A 80%／B 20%）。
- style_distance：一則候選離目標有多遠，用來排序。
- partner_reactions：B 在這個聊天室對 A 各類訊息回得多熱絡。
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta
from statistics import mean, median
from typing import Literal, Sequence

from .schemas import BlendConfig, ChatMessage, OwnMessage, ProfileSnapshot, StyleCard, StyleStats, StyleTarget
from .textutil import (
    as_utc,
    count_emoji,
    has_exclamation,
    has_laughter,
    has_question,
    particles_in,
    visible_chars,
)

# 全站平均的暫定值。平均字數 7.1 來自 2026-09-22 對 dating 資料庫的實測（259 則）；
# 其餘欄位是暫定值，建議由後端定期用 SQL 計算真實數字後，放進請求的 siteStats。
SITE_DEFAULT_STATS = StyleStats(
    messageCount=0,
    medianChars=7.0,
    meanChars=7.1,
    emojiPerMessage=0.15,
    questionRatio=0.25,
    exclamationRatio=0.1,
    laughterRatio=0.15,
    burstMean=1.5,
    particles={},
)

HIGH_CONFIDENCE_MESSAGES = 30  # 真人訊息 ≥ 30 則：風格卡以聊天為主（規格 5.3）
MIN_BIO_CHARS = 10  # bio 少於 10 個字：不足以當寫法樣本
BURST_GAP = timedelta(seconds=60)  # 同一聊天室 60 秒內的連續訊息算同一次「連發」
PARTICLE_MIN_RATIO = 0.05  # 語助詞出現在 5% 以上的訊息才算習慣
MAX_PARTICLES = 6
BIO_PARTICLE_RATIO = 0.3  # 只有 bio 時，bio 裡出現過的語助詞視為三成訊息會用到

MessageType = Literal["plan", "question", "compliment", "humor", "share"]

# 只收「明確邀約」的說法；「週末」「下次」這類詞在一般問句也常出現（例：你週末都在幹嘛？），不列入。
_PLAN_WORDS = ("一起", "要不要", "約你", "約會", "見面", "見個面", "出來走走", "出去走走")
_COMPLIMENT_WORDS = ("好看", "可愛", "漂亮", "帥", "厲害", "好棒", "很棒", "好美", "讚")


def _ratio(count: int, total: int) -> float:
    """安全的比例計算：總數為 0 時回傳 0，並把結果限制在 0～1。"""
    return 0.0 if total <= 0 else max(0.0, min(1.0, count / total))


def _top_particles(counter: Counter, total: int) -> dict[str, float]:
    """把語助詞的出現次數轉成比例，只保留超過門檻的前幾名，當作這個人的語助詞習慣。"""
    ratios = {particle: _ratio(count, total) for particle, count in counter.items()}
    kept = [(particle, ratio) for particle, ratio in ratios.items() if ratio >= PARTICLE_MIN_RATIO]
    kept.sort(key=lambda item: (-item[1], item[0]))
    return {particle: round(ratio, 3) for particle, ratio in kept[:MAX_PARTICLES]}


def _burst_mean(messages: Sequence[OwnMessage]) -> float:
    """計算平均一次連發幾則。

    同一個聊天室裡、跟上一則相隔 60 秒以內的訊息，視為同一次連發。
    我們只有這個人自己發的訊息（看不到對方何時插話），所以用時間間隔來近似。
    """
    if not messages:
        return 1.0
    by_conversation: dict[str, list] = defaultdict(list)
    for message in messages:
        by_conversation[message.conversationId].append(as_utc(message.createdAt))
    runs = 0
    for times in by_conversation.values():
        times.sort()
        runs += 1
        for previous, current in zip(times, times[1:]):
            if current - previous > BURST_GAP:
                runs += 1
    return max(1.0, round(len(messages) / runs, 3))


def compute_style_stats(messages: Sequence[OwnMessage]) -> StyleStats:
    """從一個人「自己發出」的訊息算出寫法統計。

    輸入應該只有真人訊息（AI 來源的訊息要先排除，避免 AI 學 AI）。
    沒有任何有效訊息時回傳全站暫定值，並把 messageCount 設為 0，讓呼叫端知道這是預設值。
    """
    usable = [message for message in messages if message.content.strip()]
    if not usable:
        return SITE_DEFAULT_STATS.model_copy()
    texts = [message.content for message in usable]
    lengths = [visible_chars(text) for text in texts]
    particle_counter: Counter = Counter()
    for text in texts:
        particle_counter.update(particles_in(text))
    total = len(texts)
    return StyleStats(
        messageCount=total,
        medianChars=round(float(median(lengths)), 3),
        meanChars=round(float(mean(lengths)), 3),
        emojiPerMessage=round(min(50.0, mean(count_emoji(text) for text in texts)), 3),
        questionRatio=round(_ratio(sum(has_question(text) for text in texts), total), 3),
        exclamationRatio=round(_ratio(sum(has_exclamation(text) for text in texts), total), 3),
        laughterRatio=round(_ratio(sum(has_laughter(text) for text in texts), total), 3),
        burstMean=_burst_mean(usable),
        particles=_top_particles(particle_counter, total),
    )


def stats_from_bio(bio: str, site: StyleStats) -> StyleStats:
    """只有 bio 時的寫法統計：數值用全站平均，語助詞參考 bio。

    規格 5.3：bio 是精心寫的自我介紹（平均約 40 字），聊天平均只有 7.1 字，
    所以字數、emoji 等數值不能照 bio 算；bio 只拿來參考用詞與語助詞。
    """
    found = sorted(particles_in(bio))[:MAX_PARTICLES]
    return site.model_copy(update={"messageCount": 0, "particles": {particle: BIO_PARTICLE_RATIO for particle in found}})


def blend_stats(base: StyleStats, other: StyleStats, other_weight: float) -> StyleStats:
    """把兩個人的寫法依比例混合：目標 = (1 − w) × base + w × other。

    base 是 A、other 是 B：寫整個聊天室的第一則訊息時 w = 1.0（完全照 B），之後預設 w = 0.2。
    數值欄位做線性內插；語助詞把兩邊的比例加權後重新挑前幾名。
    """
    weight = max(0.0, min(1.0, other_weight))

    def mix(first: float, second: float) -> float:
        """對單一數值做線性內插：w=0 得到 first，w=1 得到 second。"""
        return round((1 - weight) * first + weight * second, 3)

    particle_keys = set(base.particles) | set(other.particles)
    mixed_particles = {
        key: mix(base.particles.get(key, 0.0), other.particles.get(key, 0.0)) for key in particle_keys
    }
    kept = sorted(
        ((key, value) for key, value in mixed_particles.items() if value >= PARTICLE_MIN_RATIO),
        key=lambda item: (-item[1], item[0]),
    )[:MAX_PARTICLES]
    return StyleStats(
        messageCount=0,
        medianChars=mix(base.medianChars, other.medianChars),
        meanChars=mix(base.meanChars, other.meanChars),
        emojiPerMessage=mix(base.emojiPerMessage, other.emojiPerMessage),
        questionRatio=mix(base.questionRatio, other.questionRatio),
        exclamationRatio=mix(base.exclamationRatio, other.exclamationRatio),
        laughterRatio=mix(base.laughterRatio, other.laughterRatio),
        burstMean=max(1.0, mix(base.burstMean, other.burstMean)),
        particles=dict(kept),
    )


@dataclass(frozen=True)
class CandidateFeatures:
    """一則候選訊息的寫法特徵：字數、emoji 數、有沒有笑聲詞、用了哪些語助詞。"""

    chars: int
    emoji: int
    laughter: bool
    particles: frozenset[str]


def candidate_features(text: str) -> CandidateFeatures:
    """擷取一則候選的寫法特徵，給 style_distance 使用。"""
    return CandidateFeatures(
        chars=visible_chars(text),
        emoji=count_emoji(text),
        laughter=has_laughter(text),
        particles=frozenset(particles_in(text)),
    )


def style_distance(text: str, target: StyleStats) -> float:
    """計算一則候選離風格目標有多遠（0 最像，大約到 1）。

    加權組成（權重是可調的經驗值，見規格第 11 節）：
    - 字數 50%：比較字數與目標中位數的「對數比例」，長一倍和短一半算一樣遠；差到 e² 倍以上封頂。
    - emoji 15%：這則的 emoji 數跟目標「每則平均幾個」差多少（最多算 3 個）。
    - 笑聲詞 15%：有沒有笑聲詞，跟目標的笑聲詞比例差多少。
    - 語助詞 20%：沒用到目標常用的語助詞扣一半；用了目標從來不用的語助詞再扣一半。
    問句不列入：單一則訊息「是不是問句」由內容決定，不該被風格目標壓抑。
    """
    features = candidate_features(text)
    length_gap = abs(math.log((features.chars + 1) / (target.medianChars + 1)))
    d_length = min(length_gap, 2.0) / 2.0
    d_emoji = abs(min(features.emoji, 3) - min(target.emojiPerMessage, 3.0)) / 3.0
    d_laughter = abs((1.0 if features.laughter else 0.0) - target.laughterRatio)
    habitual = {particle for particle, ratio in target.particles.items() if ratio >= 0.2}
    known = set(target.particles)
    missing_habit = 0.0 if not habitual or features.particles & habitual else 1.0
    foreign = (
        len(features.particles - known) / len(features.particles) if features.particles and known else 0.0
    )
    d_particles = 0.5 * missing_habit + 0.5 * foreign
    return round(0.5 * d_length + 0.15 * d_emoji + 0.15 * d_laughter + 0.2 * d_particles, 4)


def cold_start_card(profile: ProfileSnapshot, site: StyleStats, user_id: str | None = None) -> StyleCard:
    """沒有風格卡（或信心為 none）時的冷啟動卡片，完全不呼叫模型，適合線上即時使用。

    - bio ≥ 10 字：信心 low、樣本來源 bio；數值用全站平均，語助詞參考 bio；bio 放進 bioSample
      讓 prompt 把它當寫法範例。
    - bio < 10 字：信心 none、樣本來源 none；呼叫端會據此改用 A 的寫法（見 resolve_target）。
    """
    bio = profile.bio.strip()
    if visible_chars(bio) >= MIN_BIO_CHARS:
        return StyleCard(
            userId=user_id,
            confidence="low",
            sampleSource="bio",
            stats=stats_from_bio(bio, site),
            bioSample=bio[:300],
        )
    return StyleCard(userId=user_id, confidence="none", sampleSource="none", stats=site.model_copy(), bioSample=bio[:300])


def usable_card(card: StyleCard | None, profile: ProfileSnapshot, site: StyleStats) -> StyleCard:
    """取得可用的風格卡：有正常的卡就用，沒有或信心為 none 就改用 bio 冷啟動。

    風格卡信心為 none 時再做一次冷啟動，是因為使用者可能在建卡之後才補寫 bio。
    卡片沒有 bioSample 時補上目前的 bio，讓 prompt 有寫法範例可參考。
    """
    if card is None or card.confidence == "none":
        return cold_start_card(profile, site, card.userId if card else None)
    if not card.bioSample and profile.bio.strip():
        return card.model_copy(update={"bioSample": profile.bio.strip()[:300]})
    return card


def resolve_target(requester: StyleCard, partner: StyleCard, blend: BlendConfig, first_message: bool) -> StyleTarget:
    """算出這一批 5 則共用的風格目標（規格 4.1、4.2；2026-09-23 使用者更正）。

    「第一則訊息」指的是整個聊天室的第一則訊息，不是 A 或 B 各自的第一則：
    - first_message=True（聊天室還沒有任何訊息）：依 firstMessagePartnerWeight 混合，
      預設 1.0，5 則全部照 B 喜歡的樣子寫，不像 A。
    - first_message=False（只要有人傳過訊息，不論是誰）：依 laterPartnerWeight 混合，
      預設 0.2，也就是 A 80%／B 20%。
    - B 的風格卡信心為 none（沒聊過天、bio 也太短）：不論哪一種都只用 A 的寫法，
      source 回報 requester（規格 5.3）。

    source 依實際比例標示：權重 1 是 partner、0 是 requester、介於中間是 blend。
    """
    rule: Literal["first_message", "later"] = "first_message" if first_message else "later"
    if partner.confidence == "none":
        return StyleTarget(rule=rule, source="requester", stats=requester.stats)
    weight = blend.firstMessagePartnerWeight if first_message else blend.laterPartnerWeight
    source: Literal["partner", "blend", "requester"] = (
        "partner" if weight >= 1.0 else "requester" if weight <= 0.0 else "blend"
    )
    return StyleTarget(rule=rule, source=source, stats=blend_stats(requester.stats, partner.stats, weight))


def classify_message_type(text: str) -> MessageType:
    """把 A 的一則（或一輪）訊息粗分成五類，用來統計 B 對哪類訊息反應最熱絡。

    判斷順序：邀約 → 提問 → 稱讚 → 幽默 → 分享（其他都算分享）。
    例如「要不要一起吃飯？」同時是提問與邀約，歸在邀約。
    """
    if any(word in text for word in _PLAN_WORDS):
        return "plan"
    if has_question(text):
        return "question"
    if any(word in text for word in _COMPLIMENT_WORDS):
        return "compliment"
    if has_laughter(text):
        return "humor"
    return "share"


@dataclass(frozen=True)
class ReactionSummary:
    """B 對某一類 A 訊息的平均反應熱度（0～1）與樣本數。"""

    type: MessageType
    heat: float
    samples: int


def _turns(messages: list[ChatMessage]) -> list[tuple[str, list[ChatMessage]]]:
    """把依時間排序的訊息切成「輪」：同一個人連續發的訊息算同一輪。"""
    turns: list[tuple[str, list[ChatMessage]]] = []
    for message in messages:
        if turns and turns[-1][0] == message.sender:
            turns[-1][1].append(message)
        else:
            turns.append((message.sender, [message]))
    return turns


def partner_reactions(messages: Sequence[ChatMessage]) -> list[ReactionSummary]:
    """計算 B 在這個聊天室對 A 各類訊息的反應熱度（規格 5.4），由熱到冷排序。

    做法：把對話切成「輪」，看每一輪 A 的訊息之後緊接著的那一輪 B 的回覆：
    - 回覆長度比 35%：B 這輪的總字數 ÷ B 平常一輪的字數中位數（最多算 3 倍）。
    - 回覆速度 25%：A 最後一則到 B 第一則相隔多久，1 小時以上算最慢。
    - 連發則數 20%：B 這輪發了幾則（最多算 3 則）。
    - 是否反問 20%：B 這輪有沒有問句。
    以 B 自己的中位數當基準，是因為有些人本來就習慣回得很短，不能用同一把尺比較。
    """
    ordered = sorted(messages, key=lambda message: as_utc(message.createdAt))
    turns = _turns(ordered)
    b_turn_chars = [sum(visible_chars(m.content) for m in batch) for sender, batch in turns if sender == "B"]
    if not b_turn_chars:
        return []
    typical = max(1.0, float(median(b_turn_chars)))
    heats: dict[str, list[float]] = defaultdict(list)
    for (sender, batch), (next_sender, reply) in zip(turns, turns[1:]):
        if sender != "A" or next_sender != "B":
            continue
        message_type = classify_message_type(" ".join(m.content for m in batch))
        latency = (as_utc(reply[0].createdAt) - as_utc(batch[-1].createdAt)).total_seconds()
        length_score = min(sum(visible_chars(m.content) for m in reply) / typical, 3.0) / 3.0
        speed_score = 1.0 - min(max(latency, 0.0), 3600.0) / 3600.0
        count_score = min(len(reply), 3) / 3.0
        asked_back = 1.0 if any(has_question(m.content) for m in reply) else 0.0
        heats[message_type].append(0.35 * length_score + 0.25 * speed_score + 0.2 * count_score + 0.2 * asked_back)
    summaries = [
        ReactionSummary(type=message_type, heat=round(mean(values), 3), samples=len(values))
        for message_type, values in heats.items()
    ]
    return sorted(summaries, key=lambda item: (-item.heat, -item.samples, item.type))
