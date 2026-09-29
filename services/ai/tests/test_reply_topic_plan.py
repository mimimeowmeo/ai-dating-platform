import unittest
from datetime import timedelta

from app.reply.schemas import ProfileSnapshot, ReplySuggestionRequest, StyleCard, StyleFacet
from app.reply.style import SITE_DEFAULT_STATS, usable_card
from app.reply.topic_plan import (
    BIO_OPTION,
    TopicPlan,
    TopicPlanner,
    TopicVectors,
    TopicItem,
    chat_topics,
    profile_topics,
    topic_heat,
    topic_name,
)
from tests.reply_helpers import MappedEmbedder, at, direction, profile

NOW = at(60 * 24 * 60)  # BASE_TIME 之後 60 天


def days_ago(days: float):
    return NOW - timedelta(days=days)


def topic(name: str, weight: float = 0.5, days: float = 0.0) -> StyleFacet:
    """一條話題特徵句（跟 extraction.build_facets 產生的格式相同）。"""
    return StyleFacet(kind="topic", statement=f"聊到「{name}」會比較熱絡", weight=weight, lastSeenAt=days_ago(days))


def chat_card(*facets: StyleFacet, messages: int = 50) -> StyleCard:
    """聊天紀錄夠多（≥ 30 則）的風格卡。"""
    return StyleCard(
        confidence="high", sampleSource="chat", messageCount=messages, stats=SITE_DEFAULT_STATS, facets=list(facets)
    )


async def make_plan(
    planner: TopicPlanner,
    requester_card: StyleCard | None = None,
    partner_card: StyleCard | None = None,
    requester: ProfileSnapshot | None = None,
    partner: ProfileSnapshot | None = None,
    reask: bool = False,
) -> TopicPlan:
    """照正式流程排話題：風格卡先經過 usable_card（沒有卡就用 bio 冷啟動，等於沒有聊天紀錄）。"""
    request = ReplySuggestionRequest(
        requestId="req-plan", requester=requester or profile("阿明"), partner=partner or profile("小美")
    )
    return await planner.plan(
        request,
        usable_card(requester_card, request.requester, SITE_DEFAULT_STATS),
        usable_card(partner_card, request.partner, SITE_DEFAULT_STATS),
        NOW,
        reask,
    )


def summary(plan: TopicPlan) -> list[tuple[str, str | None]]:
    """把安排簡化成（B 的話題, A 對到的話題）清單，方便比對。"""
    return [(slot.topic.text, slot.anchor.text if slot.anchor else None) for slot in plan.slots]


# 兩邊都有聊天紀錄時用的話題與向量：登山↔爬山 0.98、手沖咖啡↔咖啡 0.95、看電影↔電影 0.93，其餘互不相像。
BOTH_VECTORS = {
    "爬山": direction(0),
    "咖啡": direction(1),
    "貓": direction(2),
    "電影": direction(3),
    "桌遊": direction(4),
    "美食": direction(5),
    "登山": direction(0, 12),
    "手沖咖啡": direction(1, 18),
    "看電影": direction(3, 22),
    "最近很常在家煮飯給朋友吃": direction(6),
}
PARTNER_SIX = chat_card(
    topic("爬山", 1.0), topic("咖啡", 0.9), topic("貓", 0.8), topic("電影", 0.7), topic("桌遊", 0.6), topic("美食", 0.5)
)
REQUESTER_THREE = chat_card(topic("登山", 0.9), topic("手沖咖啡", 0.8), topic("看電影", 0.7))
REQUESTER_PROFILE = profile("阿明", bio="最近很常在家煮飯給朋友吃", interests=["桌遊"])


class HelperTests(unittest.TestCase):
    def test_topic_name_and_heat(self):
        self.assertEqual(topic_name("聊到「登山」會比較熱絡"), "登山")
        self.assertEqual(topic_name("常反問對方"), "常反問對方")  # 格式不同：整句當名稱
        self.assertEqual(topic_heat(0.8, None, NOW), 0.8)  # 舊版風格卡沒有時間：只看權重
        self.assertAlmostEqual(topic_heat(1.0, days_ago(14), NOW), 0.5)  # 兩週前：剩一半
        self.assertAlmostEqual(topic_heat(1.0, days_ago(28), NOW), 0.25)
        self.assertEqual(topic_heat(0.8, NOW + timedelta(days=1), NOW), 0.8)  # 時間在未來：當作剛聊過

    def test_chat_topics_are_ordered_by_heat(self):
        card = chat_card(topic("登山", 1.0, 40), topic("咖啡", 0.3, 0), StyleFacet(kind="tone", statement="很愛用哈哈"))
        items = chat_topics(card, NOW)
        # 登山權重高但 40 天前（1.0 × 0.5^(40/14) ≈ 0.14），比不上今天才聊的咖啡（0.3）；語氣特徵不算話題。
        self.assertEqual([(item.text, item.source) for item in items], [("咖啡", "chat"), ("登山", "chat")])

    def test_profile_topics_dedupe_tags_and_skip_short_bios(self):
        person = profile(
            "小美",
            bio="週末常去爬山，也喜歡貓",
            interests=["登山", "Live 音樂"],
            hobbies=["登山"],
            traits=["live音樂", "幽默"],
            datingGoals=["認真交往"],
        )
        self.assertEqual(
            [(item.text, item.source) for item in profile_topics(person)],
            [("登山", "tag"), ("Live 音樂", "tag"), ("幽默", "tag"), ("週末常去爬山，也喜歡貓", "bio")],
        )
        # 自我介紹少於 10 個字不算話題；交友目標也不算。
        self.assertEqual(profile_topics(profile("小美", bio="嗨你好", datingGoals=["認真交往"])), [])


class PlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_have_history_fill_shared_interest_then_hot(self):
        embedder = MappedEmbedder(BOTH_VECTORS)
        plan = await make_plan(TopicPlanner(embedder), REQUESTER_THREE, PARTNER_SIX, requester=REQUESTER_PROFILE)
        # A 也聊過 2（相似度最高的兩組）→ A 的興趣 1（桌遊標籤完全相同）→ 熱門 2（剩下的依熱度）。
        # 電影↔看電影 也相近，但「A 也聊過」只有 2 個名額，電影改排進熱門。
        self.assertEqual(
            summary(plan), [("爬山", "登山"), ("咖啡", "手沖咖啡"), ("桌遊", "桌遊"), ("貓", None), ("電影", None)]
        )
        self.assertEqual([slot.anchor.source for slot in plan.slots[:3]], ["chat", "chat", "tag"])
        self.assertEqual((plan.size, plan.profile_picks), (5, 0))
        self.assertTrue(plan.uses_requester_profile)
        self.assertFalse(plan.uses_partner_profile)
        # 短詞當文件（標題「話題」）、自我介紹當查詢，各送一次。
        self.assertEqual(
            sorted((purpose, title, len(texts)) for texts, purpose, title in embedder.calls),
            [("document", "話題", 9), ("query", None, 1)],
        )

    async def test_follow_up_with_unanswered_question_plans_four(self):
        plan = await make_plan(
            TopicPlanner(MappedEmbedder(BOTH_VECTORS)), REQUESTER_THREE, PARTNER_SIX, requester=REQUESTER_PROFILE, reask=True
        )
        # 追問另外有 1 則換個說法重問，新話題是 A 也聊過 1、A 的興趣 1、熱門 2。
        self.assertEqual(summary(plan), [("爬山", "登山"), ("桌遊", "桌遊"), ("咖啡", None), ("貓", None)])

    async def test_unfilled_slots_go_to_hot_and_each_anchor_is_used_once(self):
        vectors = {**BOTH_VECTORS, "健行": direction(0, 20)}
        partner = chat_card(topic("爬山", 1.0), topic("健行", 0.9), topic("貓", 0.8), topic("電影", 0.7), topic("桌遊", 0.6))
        plan = await make_plan(TopicPlanner(MappedEmbedder(vectors)), chat_card(topic("登山")), partner)
        # 爬山、健行都跟 A 的登山相近，但 A 的每個話題只接一個（健行 0.99 比較像）；A 的興趣沒有，名額都給熱門。
        self.assertEqual(
            summary(plan), [("健行", "登山"), ("爬山", None), ("貓", None), ("電影", None), ("桌遊", None)]
        )

    async def test_partner_topics_run_out_then_profile_then_fewer(self):
        embedder = MappedEmbedder({})
        planner = TopicPlanner(embedder)
        partner_card = chat_card(topic("爬山", 1.0), topic("咖啡", 0.5))
        # B 的聊天話題只有 2 個：剩下的從 B 的檔案挑，跟聊天話題重複的標籤（爬山）不算。
        plan = await make_plan(planner, partner_card=partner_card, partner=profile("小美", interests=["攝影", "爬山"]))
        self.assertEqual(summary(plan), [("爬山", None), ("咖啡", None)])
        self.assertEqual((plan.profile_picks, plan.profile_options, plan.size), (1, ("攝影",), 3))
        self.assertEqual(embedder.calls, [])  # A 沒有任何話題可以比：不用轉向量
        # 檔案也沒有可以聊的：只排 2 則（不硬湊）。
        plan = await make_plan(planner, partner_card=partner_card, partner=profile("小美", interests=["爬山"]))
        self.assertEqual((plan.size, plan.profile_picks, plan.profile_options), (2, 0, ()))
        # 有自我介紹：自我介紹裡可能有好幾個話題，剩下的名額都開放。
        plan = await make_plan(
            planner, partner_card=partner_card, partner=profile("小美", bio="喜歡到處旅行拍照，最想再去日本", interests=["攝影"])
        )
        self.assertEqual((plan.profile_picks, plan.profile_options), (3, ("攝影", BIO_OPTION)))
        self.assertTrue(plan.uses_partner_profile)

    async def test_partner_without_history_uses_tags_and_bio(self):
        bio = "週末常去爬山，也喜歡貓咪"
        vectors = {"登山": direction(0), "咖啡": direction(1), "咖啡廳": direction(1, 15), bio: direction(5)}
        plan = await make_plan(
            TopicPlanner(MappedEmbedder(vectors)),
            requester_card=chat_card(topic("咖啡廳", 0.9)),
            partner_card=chat_card(topic("露營"), messages=10),  # 只有 10 則：聊天話題不用
            requester=profile("阿明", interests=["登山"]),
            partner=profile("小美", bio=bio, interests=["登山", "咖啡"]),
        )
        # B 的話題改用標籤與自我介紹：咖啡↔A 聊過的咖啡廳、登山↔A 的登山標籤；其餘 3 則從 B 的自我介紹找。
        self.assertEqual(summary(plan), [("咖啡", "咖啡廳"), ("登山", "登山")])
        self.assertEqual({slot.topic.source for slot in plan.slots}, {"tag"})
        self.assertEqual((plan.profile_picks, plan.profile_options, plan.size), (3, (BIO_OPTION,), 5))

    async def test_requester_without_history_uses_profile_for_three_slots(self):
        bio = "喜歡看展，也常去咖啡廳看書"
        partner = chat_card(topic("登山", 1.0), topic("咖啡", 0.9), topic("電影", 0.8), topic("貓", 0.7), topic("看展", 0.6))
        plan = await make_plan(TopicPlanner(None), partner_card=partner, requester=profile("阿明", bio=bio, interests=["登山"]))
        # A 沒有聊天紀錄：A 也聊過（2）與 A 的興趣（1）都用 A 的檔案一起挑。沒有向量服務也看得出來：
        # 登山標籤完全相同；自我介紹直接寫到「咖啡」「看展」，但自我介紹只接一個（咖啡）。
        self.assertEqual(
            summary(plan), [("登山", "登山"), ("咖啡", bio), ("電影", None), ("貓", None), ("看展", None)]
        )
        self.assertEqual([slot.anchor.source for slot in plan.slots[:2]], ["tag", "bio"])

    async def test_thresholds(self):
        cases = (
            # （A 的話題, 向量, 預期）：兩個短詞的門檻是 0.88。
            ("健行", direction(0, 27), [("爬山", "健行")]),  # cos 27° = 0.891
            ("露營", direction(0, 29), [("爬山", None)]),  # cos 29° = 0.875
        )
        for name, vector, expected in cases:
            with self.subTest(name=name):
                vectors = {"爬山": direction(0), name: vector}
                plan = await make_plan(TopicPlanner(MappedEmbedder(vectors)), chat_card(topic(name)), chat_card(topic("爬山")))
                self.assertEqual(summary(plan), expected)
        # 自我介紹的門檻是 0.76；只有一個字的「貓」不算「直接寫到」，要靠向量。
        bio = "養了兩隻貓，下班就是當貓奴"
        for degrees, expected in ((35, [("貓", bio)]), (45, [("貓", None)])):  # cos 35° = 0.82、cos 45° = 0.71
            with self.subTest(degrees=degrees):
                vectors = {"貓": direction(2), bio: direction(2, degrees)}
                plan = await make_plan(
                    TopicPlanner(MappedEmbedder(vectors)), partner_card=chat_card(topic("貓")), requester=profile("阿明", bio=bio)
                )
                self.assertEqual(summary(plan), expected)

    async def test_falls_back_to_text_when_vectors_fail(self):
        embedder = MappedEmbedder(BOTH_VECTORS, fail=True)
        plan = await make_plan(
            TopicPlanner(embedder),
            REQUESTER_THREE,
            chat_card(topic("爬山", 1.0), topic("桌遊", 0.6)),
            requester=REQUESTER_PROFILE,
        )
        # 向量服務失敗：登山↔爬山 比不出來，只剩文字完全相同的桌遊；其餘給熱門。推薦照樣排得出來。
        self.assertEqual(summary(plan), [("桌遊", "桌遊"), ("爬山", None)])
        self.assertEqual(len(embedder.calls), 2)

    async def test_vectors_are_cached_between_requests(self):
        embedder = MappedEmbedder(BOTH_VECTORS)
        planner = TopicPlanner(embedder)
        await make_plan(planner, chat_card(topic("登山")), chat_card(topic("爬山"), topic("咖啡")))
        await make_plan(planner, chat_card(topic("登山")), chat_card(topic("爬山"), topic("貓")))
        # 第二次只有「貓」是新的，其他直接用記憶體裡的向量。
        self.assertEqual(
            [(texts, purpose) for texts, purpose, _ in embedder.calls],
            [(["咖啡", "爬山", "登山"], "document"), (["貓"], "document")],
        )

    async def test_vector_cache_drops_the_least_recently_used(self):
        embedder = MappedEmbedder(BOTH_VECTORS)
        vectors = TopicVectors(embedder, size=2)
        climb, coffee, hike = (TopicItem(name, "chat") for name in ("爬山", "咖啡", "登山"))
        await vectors.lookup([climb])
        await vectors.lookup([coffee])
        await vectors.lookup([climb])  # 還在：不必再轉，而且變成「最近用過」
        await vectors.lookup([hike])  # 超過上限 2：擠掉最久沒用的咖啡
        await vectors.lookup([climb, coffee])  # 爬山還在，咖啡要重轉
        self.assertEqual([texts for texts, _, _ in embedder.calls], [["爬山"], ["咖啡"], ["登山"], ["咖啡"]])
        # 一次要的比上限還多時，這次照樣全部拿得到。
        found = await TopicVectors(MappedEmbedder(BOTH_VECTORS), size=2).lookup([climb, coffee, hike])
        self.assertEqual(len(found), 3)


if __name__ == "__main__":
    unittest.main()
