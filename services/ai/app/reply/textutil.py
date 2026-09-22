"""文字工具：這個套件裡所有「看字」的小功能都集中在這裡，方便單元測試與重複使用。

設計原則：全部是純函式（同樣輸入永遠得到同樣輸出），不呼叫任何模型或網路。
唯一的例外是簡轉繁：它會延遲載入 OpenCC 的詞庫，但同樣不連網。
"""

import difflib
import math
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache

# 台灣沒有夏令時間，直接用固定的 UTC+8；不依賴系統時區資料庫（Docker slim 映像不一定有）。
TAIPEI = timezone(timedelta(hours=8))

_WHITESPACE = re.compile(r"\s+")
_ZERO_WIDTH = re.compile("[​‌‍︎️]")

# emoji 的比對規則：國旗（兩個區域指示符）、一般 emoji，加上膚色修飾與 ZWJ 串接的組合。
# 「👨‍👩‍👧」這種由多個碼位組成的 emoji 會被算成一個。
_EMOJI_BASE = "[\U0001F300-\U0001FAFF☀-➿⬀-⯿⌀-⏿]"
_EMOJI_MOD = "[\U0001F3FB-\U0001F3FF️]?"
_EMOJI = re.compile(
    f"(?:[\U0001F1E6-\U0001F1FF]{{2}})|(?:{_EMOJI_BASE}{_EMOJI_MOD}(?:‍{_EMOJI_BASE}{_EMOJI_MOD})*)"
)

# 中日韓文字、全形標點、注音符號：粗估每個字約 1.3 個 token。
_CJK = re.compile("[　-〿㄀-ㄯㆠ-ㆿ㐀-䶿一-鿿豈-﫿＀-￯]")

_QUESTION_MARK = re.compile("[?？]")
_EXCLAMATION = re.compile("[!！]")
# 結尾的標點與 emoji，判斷「句尾語氣詞」前要先去掉。
_TRAILING_NOISE = re.compile(f"(?:[\\s~～。.…!！,，、)）]|{_EMOJI.pattern})+$")
_QUESTION_ENDINGS = ("嗎", "呢", "咩", "麼")
_LAUGHTER = re.compile(r"哈哈|呵呵|嘻嘻|笑死|xd|lol|ㄏㄏ|www|😂|🤣|😆", re.IGNORECASE)

# 台灣聊天常見的語助詞；風格統計只看「有沒有出現」，不看位置。
PARTICLES = ("啦", "欸", "喔", "耶", "吧", "呢", "嗎", "啊", "哦", "嘛", "囉", "捏", "齁", "ㄟ")

# 抽象化檢查用：看起來像數字、網址、帳號、Email 的內容都不該出現在特徵句裡。
_IDENTIFYING = re.compile(r"[0-9０-９]|https?://|www\.|\.com\b|\.tw\b|@[A-Za-z0-9_.]+|[\w.+-]+@[\w-]+\.[\w.]+", re.IGNORECASE)


def as_utc(value: datetime) -> datetime:
    """把時間統一成「帶時區的 UTC」。

    後端傳來的 ISO 字串通常帶 `Z`；萬一沒有時區資訊，就當作 UTC。
    統一之後，時間相減、比較先後才不會因為一個有時區、一個沒時區而出錯。
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def taipei_label(value: datetime) -> str:
    """把時間轉成台灣時間的「月-日 時:分」，放在對話片段與 prompt 裡給模型看。"""
    return as_utc(value).astimezone(TAIPEI).strftime("%m-%d %H:%M")


def single_line(text: str) -> str:
    """把換行與連續空白壓成一個空白，並去掉頭尾空白。

    聊天輸入框是單行的 `<input>`，所以推薦內容一律要是一行。
    """
    return _WHITESPACE.sub(" ", text).strip()


def visible_chars(text: str) -> int:
    """計算「看得到的字數」：不算空白與零寬字元，一個 emoji（含組合）算 1 個字。

    用來做字數統計與風格距離，比 `len()` 更接近使用者感覺到的長度。
    """
    emoji_count = len(_EMOJI.findall(text))
    without_emoji = _EMOJI.sub("", text)
    without_invisible = _ZERO_WIDTH.sub("", _WHITESPACE.sub("", without_emoji))
    return len(without_invisible) + emoji_count


def estimate_tokens(text: str) -> int:
    """粗估一段文字的 token 數：中文等每字約 1.3、其他字元約 4 個算 1。

    只用來控制 prompt 與對話片段的大小，不是計費依據；各模型的 tokenizer 不同，
    精確數字要用官方 count_tokens API 量。空字串回傳 0，其餘至少回傳 1。
    """
    if not text:
        return 0
    compact = _WHITESPACE.sub("", text)
    cjk = len(_CJK.findall(compact))
    other = len(compact) - cjk
    return max(1, math.ceil(cjk * 1.3 + other / 4))


def count_emoji(text: str) -> int:
    """計算一段文字裡有幾個 emoji（組合 emoji 算一個）。"""
    return len(_EMOJI.findall(text))


def has_question(text: str) -> bool:
    """判斷是不是問句：有問號，或（去掉句尾標點與 emoji 後）以「嗎、呢、咩、麼」結尾。"""
    if _QUESTION_MARK.search(text):
        return True
    core = _TRAILING_NOISE.sub("", text.strip())
    return core.endswith(_QUESTION_ENDINGS)


def has_exclamation(text: str) -> bool:
    """判斷有沒有驚嘆號（半形或全形）。"""
    return bool(_EXCLAMATION.search(text))


def has_laughter(text: str) -> bool:
    """判斷有沒有笑聲詞，例如「哈哈」「笑死」「XD」「😂」。"""
    return bool(_LAUGHTER.search(text))


def particles_in(text: str) -> set[str]:
    """找出這段文字用到了哪些常見語助詞（見 PARTICLES）。"""
    return {particle for particle in PARTICLES if particle in text}


@lru_cache(maxsize=1)
def _taiwan_converter():
    """延遲建立 OpenCC 的「簡體 → 台灣繁體＋台灣用語」轉換器，整個程序只建立一次。

    載入詞庫需要一點時間，所以用 lru_cache 快取；套件不存在時回傳 None，讓呼叫端照原文處理。
    """
    try:
        import opencc
    except ImportError:
        return None
    return opencc.OpenCC("s2twp.json")


def to_taiwan_traditional(text: str) -> str:
    """把文字轉成台灣繁體與台灣用語（例：「软件的信息」→「軟體的資訊」）。

    模型偶爾會混進簡體字或中國用語；已經是台灣繁體的文字轉換後幾乎不變。
    OpenCC 不可用時原樣回傳，不讓整個功能失敗。
    """
    converter = _taiwan_converter()
    if converter is None or not text:
        return text
    return converter.convert(text)


def text_similarity(first: str, second: str) -> float:
    """兩段文字的相似度（0～1），用 difflib 的最長共同片段演算法。

    用在推薦去重（≥ 0.8 視為重複）與「換一批」時避開舊推薦。
    比較前先壓成單行並轉小寫，避免只差空白或大小寫就被當成不同。
    """
    a = single_line(first).lower()
    b = single_line(second).lower()
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def char_ngrams(text: str, size: int) -> set[str]:
    """取出一段文字所有長度為 size 的連續片段（不含空白）。"""
    compact = _WHITESPACE.sub("", text)
    if len(compact) < size:
        return set()
    return {compact[index:index + size] for index in range(len(compact) - size + 1)}


def build_ngram_index(texts: list[str], size: int) -> set[str]:
    """把一堆原始訊息的所有連續片段收進一個集合，供抽象化檢查快速查詢。

    特徵句不能跟任何一則原始訊息共用連續 size 個字以上（規格 5.3：不保留原文）。
    先建好集合再查，比每條特徵句都跟兩萬則訊息逐一比對快很多。
    """
    index: set[str] = set()
    for text in texts:
        index |= char_ngrams(text, size)
    return index


def shares_long_substring(statement: str, index: set[str], size: int) -> bool:
    """檢查一條特徵句是否跟原始訊息共用了連續 size 個字以上（True 代表太像原文，要丟掉）。"""
    return any(gram in index for gram in char_ngrams(statement, size))


def has_identifying_detail(text: str) -> bool:
    """檢查文字是否含有數字、網址、@帳號或 Email；抽象特徵句不該有這些細節。"""
    return bool(_IDENTIFYING.search(text))
