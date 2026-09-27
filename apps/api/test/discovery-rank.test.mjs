// discovery-rank.ts 的單元測試：探索頁 AI 排序怎麼合併外貌與興趣分數。
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
require("ts-node/register/transpile-only");
const {
  percentiles,
  rankCandidates,
  APPEARANCE_WEIGHT,
} = require("../src/discovery-rank.ts");

const ids = ["a", "b", "c", "d"];
const scores = (entries) => new Map(Object.entries(entries));
// 外貌百分位 a=1、b=c=d=0.5；興趣百分位 d=0.25、a=0.5、b=0.75、c=1。
const appearance = scores({ a: 0.9, b: 0.2 });
const interest = scores({ a: 0.1, b: 0.8, c: 0.9, d: 0 });

test("百分位：同分取平均名次；沒有分數的人給中間值 0.5，或照指定值參與排名", () => {
  const p = percentiles(ids, scores({ a: 0.1, b: 0.5, c: 0.5 }));
  assert.equal(p.get("a"), 1 / 3);
  assert.equal(p.get("b"), 2.5 / 3);
  assert.equal(p.get("c"), 2.5 / 3);
  assert.equal(p.get("d"), 0.5);
  const withMissing = percentiles(ids, scores({ a: 1 }), 0);
  assert.equal(withMissing.get("a"), 1);
  assert.equal(withMissing.get("d"), 0.5);
});

test("外貌、興趣都開：外貌 60%、興趣 40%", () => {
  assert.equal(APPEARANCE_WEIGHT, 0.6);
  // a = 0.6×1 + 0.4×0.5 = 0.8；c = 0.6×0.5 + 0.4×1 = 0.7；b = 0.6；d = 0.4
  assert.deepEqual(rankCandidates(ids, appearance, interest), [
    "a",
    "c",
    "b",
    "d",
  ]);
});

test("關掉外貌就只看興趣，關掉興趣就只看外貌", () => {
  assert.deepEqual(rankCandidates(ids, null, interest), ["c", "b", "a", "d"]);
  // 只看外貌時 b、c、d 同分，維持原本順序。
  assert.deepEqual(rankCandidates(ids, appearance, null), ["a", "b", "c", "d"]);
});

test("還沒按過喜歡（外貌分數是空的）就改用興趣 100%", () => {
  assert.deepEqual(rankCandidates(ids, new Map(), interest), [
    "c",
    "b",
    "a",
    "d",
  ]);
});

test("兩個都關或都算不出來時維持原本順序", () => {
  assert.deepEqual(rankCandidates(ids, null, null), ids);
  assert.deepEqual(rankCandidates(ids, new Map(), null), ids);
  assert.deepEqual(rankCandidates(["d", "c", "b", "a"], null, new Map()), [
    "d",
    "c",
    "b",
    "a",
  ]);
});

const {
  PASS_PENALTY_SCALE,
  passPenalty,
  seededOrder,
  EXPLORATION_EVERY,
  buildQueue,
  scoreCandidates,
  explainInterest,
  INTEREST_WEIGHTS,
  projectOut,
} = require("../src/discovery-rank.ts");

test("略過扣分（PASS V2）：附近略過未滿 5 個不扣；附近有喜歡過的人會減輕；原規格最多 0.20，再依性別換算", () => {
  assert.equal(passPenalty(4, 0, "woman"), 0);
  assert.equal(passPenalty(5, 0, "woman"), 0.2 * PASS_PENALTY_SCALE.woman);
  assert.equal(passPenalty(5, 0, "man"), 0.2 * PASS_PENALTY_SCALE.man);
  assert.equal(passPenalty(5, 0, "nonbinary"), 0.2 * PASS_PENALTY_SCALE.other);
  assert.ok(
    Math.abs(
      passPenalty(6, 1, "woman") -
        0.25 * (1 - Math.exp(-2.7)) * Math.exp(-0.8) * PASS_PENALTY_SCALE.woman,
    ) < 1e-12,
  );
  assert.ok(passPenalty(20, 2, "man") < passPenalty(20, 1, "man"));
});

test("固定種子的順序：同種子結果相同；少了某些人，其他人的相對順序不變", () => {
  const ids = Array.from({ length: 50 }, (_, i) => `u${i}`);
  const order = seededOrder(ids, "seed");
  assert.deepEqual(seededOrder(ids, "seed"), order);
  assert.notDeepEqual(seededOrder(ids, "other"), order);
  const fewer = ids.filter((id) => id !== order[0] && id !== order[10]);
  assert.deepEqual(
    seededOrder(fewer, "seed"),
    order.filter((id) => fewer.includes(id)),
  );
});

const ranked = Array.from(
  { length: 40 },
  (_, i) => `u${String(i).padStart(2, "0")}`,
);
const queue = (swipeCount, penalized = new Set()) =>
  buildQueue(ranked, { limit: 30, swipeCount, seed: "s", penalized });
const positions = (list, source) =>
  list.flatMap((card, i) => (card.source === source ? [i + 1] : []));

test("每 5 張推薦卡之後插 1 張未推薦卡，位置依已滑張數決定", () => {
  assert.equal(EXPLORATION_EVERY, 6);
  assert.deepEqual(positions(queue(0), "exploration"), [6, 12, 18, 24, 30]);
  // 已滑 5 張：這次看到的第一張就是第 6 次滑卡，要是未推薦卡。
  assert.deepEqual(positions(queue(5), "exploration"), [1, 7, 13, 19, 25]);
  // 前端每滑一張就重抓、只顯示第一張：連續 18 次滑卡裡第 6、12、18 次是未推薦卡。
  assert.deepEqual(
    Array.from({ length: 18 }, (_, n) => queue(n)[0].source).flatMap(
      (source, n) => (source === "exploration" ? [n + 1] : []),
    ),
    [6, 12, 18],
  );
});

test("推薦卡照排名依序放；未推薦卡從推薦卡之後的排名挑，先挑沒被扣分的人", () => {
  const penalized = new Set(ranked.slice(25, 38));
  const list = queue(0, penalized);
  assert.equal(list.length, 30);
  assert.deepEqual(
    list.filter((c) => c.source === "recommended").map((c) => c.userId),
    ranked.slice(0, 25),
  );
  const exploration = list
    .filter((c) => c.source === "exploration")
    .map((c) => c.userId);
  assert.ok(exploration.every((id) => ranked.indexOf(id) >= 25));
  assert.equal(
    exploration.filter((id) => penalized.has(id)).length,
    3,
    "只有 2 個沒被扣分的人，其餘 3 個才從被扣分的人補",
  );
  assert.deepEqual(queue(0, penalized), list, "同樣的種子，挑到的人不變");
});

test("候選人不滿 30 人時照樣穿插，不會重複也不會漏人", () => {
  const small = ["a", "b", "c", "d", "e", "f", "g"];
  const list = buildQueue(small, {
    limit: 30,
    swipeCount: 5,
    seed: "s",
    penalized: new Set(),
  });
  assert.deepEqual(positions(list, "exploration"), [1, 7]);
  assert.deepEqual(list.map((c) => c.userId).sort(), small);
  assert.deepEqual(
    list.filter((c) => c.source === "recommended").map((c) => c.userId),
    small.slice(0, 5),
  );
});

test("合併分數：回傳排序、各自的百分位與最後分數", () => {
  const scored = scoreCandidates(ids, appearance, interest);
  assert.deepEqual(scored.ranked, ["a", "c", "b", "d"]);
  assert.equal(scored.appearanceWeight, 0.6);
  assert.equal(scored.appearancePct.get("a"), 1);
  assert.equal(scored.interestPct.get("c"), 1);
  assert.ok(Math.abs(scored.score.get("a") - 0.8) < 1e-12);
  const none = scoreCandidates(ids, null, null);
  assert.equal(none.score, null);
  assert.equal(none.appearanceWeight, null);
});

test("標籤比較：各類的共同標籤、聯集與 Jaccard，加權後就是興趣分數", () => {
  const mine = [
    { category: "interest", code: "hiking" },
    { category: "interest", code: "coffee" },
    { category: "diet", code: "likes_hotpot" },
  ];
  const theirs = [
    { category: "interest", code: "coffee" },
    { category: "interest", code: "reading" },
    { category: "diet", code: "likes_hotpot" },
    { category: "personality", code: "humorous" },
  ];
  const parts = explainInterest(mine, theirs);
  const byCategory = Object.fromEntries(parts.map((p) => [p.category, p]));
  assert.deepEqual(byCategory.interest.shared, ["coffee"]);
  assert.equal(byCategory.interest.union, 3);
  assert.equal(byCategory.diet.jaccard, 1);
  assert.equal(byCategory.personality.jaccard, 0);
  const total = parts.reduce((sum, p) => sum + p.weight * p.jaccard, 0);
  assert.ok(
    Math.abs(total - (INTEREST_WEIGHTS.interest / 3 + INTEREST_WEIGHTS.diet)) <
      1e-12,
  );
});

test("扣掉方向：結果和方向垂直、長度是 1", () => {
  const direction = [0.6, 0.8, 0];
  const out = projectOut([0.5, 0.5, 0.7071], direction);
  assert.ok(Math.abs(out.reduce((s, v, i) => s + v * direction[i], 0)) < 1e-12);
  assert.ok(Math.abs(Math.hypot(...out) - 1) < 1e-12);
});
