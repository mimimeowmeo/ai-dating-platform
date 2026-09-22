// AI 推薦回覆會用到的純函式：文字相似度、訊息來源判定、pgvector 字面值。
// 這個檔案刻意不匯入 Prisma、Redis 或設定，方便單獨做單元測試（test/ai-text.test.mjs）。

/** 向量維度；必須與 AI 服務的 AI_EMBEDDING_DIMENSIONS 和資料表的 vector(768) 一致。 */
export const EMBEDDING_DIMENSIONS = 768;

/** 相似度 ≥ 0.95：使用者原封不動送出 AI 的推薦（規格 5.6）。 */
export const VERBATIM_SIMILARITY = 0.95;
/** 相似度 ≥ 0.5：算改寫過的 AI 訊息；低於 0.5 視為使用者自己重寫，算 human。 */
export const EDITED_SIMILARITY = 0.5;

/** 訊息來源：真人打的／AI 推薦原封不動送出／AI 推薦改過再送。 */
export type Origin = "human" | "ai_verbatim" | "ai_edited";

/**
 * 把文字壓成單行、去掉頭尾空白：比較相似度前先正規化，避免只差換行或空格就被當成不同。
 * 與 AI 服務的 textutil.single_line 同樣的處理方式。
 */
export function singleLine(text: string) {
  return text.replace(/\s+/gu, " ").trim();
}

/**
 * 在 a[alo, ahi) 與 b[blo, bhi) 之間找出最長的共同片段。
 *
 * 做法與 Python difflib.SequenceMatcher.find_longest_match 相同：
 * 逐一掃過 a 的每個字，用 b2j（某個字在 b 裡出現的所有位置）更新「以這個位置結尾的共同片段長度」。
 * j2len 記住上一輪的結果，所以 newj2len[j] = j2len[j-1] + 1 就是延長一格。
 * 回傳 [a 的起點, b 的起點, 長度]；沒有共同片段時長度為 0。
 */
function longestMatch(
  a: string,
  b: string,
  alo: number,
  ahi: number,
  blo: number,
  bhi: number,
  b2j: Map<string, number[]>,
): [number, number, number] {
  let bestI = alo;
  let bestJ = blo;
  let bestSize = 0;
  let j2len = new Map<number, number>();
  for (let i = alo; i < ahi; i++) {
    const next = new Map<number, number>();
    for (const j of b2j.get(a[i]) ?? []) {
      if (j < blo) continue;
      if (j >= bhi) break;
      const length = (j2len.get(j - 1) ?? 0) + 1;
      next.set(j, length);
      if (length > bestSize) {
        bestI = i - length + 1;
        bestJ = j - length + 1;
        bestSize = length;
      }
    }
    j2len = next;
  }
  return [bestI, bestJ, bestSize];
}

/**
 * 兩段文字的相似度（0～1）。
 *
 * 與 AI 服務的 textutil.text_similarity 用同一種演算法（Ratcliff／Obershelp）：
 * 先找最長共同片段，再對它左右兩側遞迴，把所有共同片段的長度加總成 M，
 * 相似度 = 2M ÷（兩段文字長度總和）。完全相同是 1.0、完全沒有共同字是 0.0。
 *
 * 用途：判斷使用者送出的訊息跟 AI 推薦有多像（規格 5.6 的 0.95／0.5 門檻）。
 * 兩邊都先壓成單行並轉小寫，所以只差空白或大小寫仍然算一樣。
 * 註：Python 的 difflib 對 200 字以上的第二段文字會自動忽略「太常出現的字」，
 * 這裡沒有實作那個最佳化；推薦長度上限是 80 字，實務上兩邊結果相同。
 */
export function similarity(first: string, second: string) {
  const a = singleLine(first).toLowerCase();
  const b = singleLine(second).toLowerCase();
  if (!a || !b) return 0;
  const b2j = new Map<string, number[]>();
  for (let j = 0; j < b.length; j++) {
    const list = b2j.get(b[j]);
    if (list) list.push(j);
    else b2j.set(b[j], [j]);
  }
  let matched = 0;
  // 用堆疊代替遞迴，避免長訊息把呼叫堆疊塞滿。
  const queue: [number, number, number, number][] = [[0, a.length, 0, b.length]];
  while (queue.length) {
    const [alo, ahi, blo, bhi] = queue.pop()!;
    const [i, j, size] = longestMatch(a, b, alo, ahi, blo, bhi, b2j);
    if (!size) continue;
    matched += size;
    if (alo < i && blo < j) queue.push([alo, i, blo, j]);
    if (i + size < ahi && j + size < bhi)
      queue.push([i + size, ahi, j + size, bhi]);
  }
  return (2 * matched) / (a.length + b.length);
}

/**
 * 依相似度決定訊息來源（規格 5.6）。
 * ≥ 0.95 原封不動、≥ 0.5 改過再送、其餘算使用者自己寫的。
 * ai_verbatim 與 ai_edited 在萃取風格卡時都會被排除，避免 AI 學 AI。
 */
export function classifyOrigin(score: number): Origin {
  if (score >= VERBATIM_SIMILARITY) return "ai_verbatim";
  if (score >= EDITED_SIMILARITY) return "ai_edited";
  return "human";
}

/**
 * 把數字陣列轉成 pgvector 認得的字面值，例如 `[0.1,0.2]`，搭配 `::vector` 使用。
 *
 * 內容與 npm 套件 pgvector 的 toSql()（就是 JSON.stringify）相同，
 * 為了少一個相依套件才自己寫；順便擋掉維度不對或含 NaN／Infinity 的向量，
 * 免得把壞資料寫進資料庫（pgvector 不接受 NaN，錯誤訊息也不好讀）。
 */
export function toVector(values: number[]) {
  if (values.length !== EMBEDDING_DIMENSIONS)
    throw new Error(`EMBEDDING_DIMENSION_MISMATCH:${values.length}`);
  for (const value of values)
    if (!Number.isFinite(value)) throw new Error("EMBEDDING_NOT_FINITE");
  return `[${values.join(",")}]`;
}
