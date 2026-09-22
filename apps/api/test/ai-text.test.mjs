// ai-text.ts 的單元測試：不需要資料庫、Redis 或 AI 服務。
// 相似度是訊息來源標記的依據（規格 5.6），算錯會讓 AI 訊息被當成真人訊息，
// 進而污染風格卡（AI 學 AI），所以這裡把門檻附近的情況都測過。
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
require("ts-node/register/transpile-only");
const { similarity, classifyOrigin, toVector, singleLine } = require("../src/ai-text.ts");

test("相似度：與 Python difflib 的結果一致", () => {
  // 這些數字是拿同一組句子在 services/ai 的 textutil.text_similarity 跑出來的。
  const cases = [
    ["打羽球啦，妳呢", "打羽球啦，妳呢", 1],
    ["打羽球啦，妳呢", "打羽球啦，你呢？", 0.8],
    ["有推薦的新手山嗎", "有沒有推薦的新手山？", 0.777778],
    ["我週末都在打球", "我最近在學攝影，週末常常出門", 0.285714],
    ["下次一起去爬山？", "完全不同的一句話喔", 0.117647],
  ];
  for (const [a, b, expected] of cases)
    assert.equal(Number(similarity(a, b).toFixed(6)), expected, `${a} / ${b}`);
});

test("相似度：只差空白或大小寫算完全相同，空字串是 0", () => {
  assert.equal(similarity("Hello world", "hello   world"), 1);
  assert.equal(similarity("  一起去看展覽 ", "一起去看展覽"), 1);
  assert.equal(similarity("", "abc"), 0);
  assert.equal(similarity("abc", ""), 0);
});

test("來源判定：0.95 與 0.5 是門檻（規格 5.6）", () => {
  assert.equal(classifyOrigin(1), "ai_verbatim");
  assert.equal(classifyOrigin(0.95), "ai_verbatim");
  assert.equal(classifyOrigin(0.949), "ai_edited");
  assert.equal(classifyOrigin(0.5), "ai_edited");
  assert.equal(classifyOrigin(0.499), "human");
  assert.equal(classifyOrigin(0), "human");
});

test("singleLine 把換行壓成單一空白", () => {
  assert.equal(singleLine("第一行\n第二行\t第三行"), "第一行 第二行 第三行");
});

test("向量字面值：維度或數值不對就丟錯，不讓壞資料進資料庫", () => {
  const vector = Array.from({ length: 768 }, (_, i) => i / 1000);
  assert.equal(toVector(vector).startsWith("[0,0.001,0.002"), true);
  assert.throws(() => toVector([1, 2, 3]), /EMBEDDING_DIMENSION_MISMATCH:3/);
  const broken = [...vector];
  broken[10] = Number.NaN;
  assert.throws(() => toVector(broken), /EMBEDDING_NOT_FINITE/);
});
