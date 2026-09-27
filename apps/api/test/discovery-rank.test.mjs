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
  personalizedRatio,
  seededOrder,
  interleave,
  applyQuota,
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

test("個人化名額：依按過的喜歡數 0～5+；興趣開著時至少 60%", () => {
  assert.deepEqual(
    [0, 1, 2, 3, 4, 5, 9].map((n) => personalizedRatio(n, false)),
    [0, 0.6, 0.65, 0.7, 0.75, 0.8, 0.8],
  );
  assert.equal(personalizedRatio(0, true), 0.6);
  assert.equal(personalizedRatio(5, true), 0.8);
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

test("穿插：30 張裡 6 張探索在第 5、10、15、20、25、30 張；第一張是個人化", () => {
  const personalized = Array.from({ length: 24 }, (_, i) => `p${i}`);
  const exploration = Array.from({ length: 6 }, (_, i) => `e${i}`);
  const result = interleave(personalized, exploration);
  assert.deepEqual(
    result.flatMap((id, i) => (id.startsWith("e") ? [i + 1] : [])),
    [5, 10, 15, 20, 25, 30],
  );
  assert.deepEqual(
    result.filter((id) => id.startsWith("p")),
    personalized,
  );
});

test("名額：個人化取排序最前面；探索先挑沒被扣分的人，不夠才用被扣分的人", () => {
  const ranked = Array.from(
    { length: 40 },
    (_, i) => `u${String(i).padStart(2, "0")}`,
  );
  const penalized = new Set(ranked.slice(24, 38));
  const result = applyQuota(ranked, {
    limit: 30,
    ratio: 0.8,
    seed: "s",
    penalized,
  });
  assert.equal(result.length, 30);
  assert.deepEqual(
    result.filter((id) => ranked.indexOf(id) < 24),
    ranked.slice(0, 24),
  );
  const exploration = result.filter((id) => ranked.indexOf(id) >= 24);
  assert.equal(exploration.length, 6);
  assert.equal(
    exploration.filter((id) => penalized.has(id)).length,
    4,
    "只有 2 個沒被扣分的人，其餘 4 個才從被扣分的人補",
  );
});

test("名額：候選人不滿 30 人時照比例縮小；比例 0 時全部是探索", () => {
  const ranked = ["a", "b", "c", "d", "e"];
  assert.equal(
    applyQuota(ranked, {
      limit: 30,
      ratio: 0.6,
      seed: "s",
      penalized: new Set(),
    }).length,
    5,
  );
  assert.deepEqual(
    [
      ...applyQuota(ranked, {
        limit: 30,
        ratio: 0,
        seed: "s",
        penalized: new Set(),
      }),
    ].sort(),
    ranked,
  );
});

test("扣掉方向：結果和方向垂直、長度是 1", () => {
  const direction = [0.6, 0.8, 0];
  const out = projectOut([0.5, 0.5, 0.7071], direction);
  assert.ok(Math.abs(out.reduce((s, v, i) => s + v * direction[i], 0)) < 1e-12);
  assert.ok(Math.abs(Math.hypot(...out) - 1) < 1e-12);
});
