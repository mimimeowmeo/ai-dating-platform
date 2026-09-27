// 探索頁 AI 推薦的純函式：排序、名額、略過扣分、向量方向；不碰資料庫，方便單元測試（test/discovery-rank.test.mjs）。

/** 外貌、興趣都開時外貌的比重（HeartLink Appearance Core V1：外貌 60%、興趣 40%）。 */
export const APPEARANCE_WEIGHT = 0.6;

/**
 * 每個分數在候選池裡的百分位（0～1，同分取平均名次）。
 * 沒有分數的人給 missing；沒給 missing 就不參與排名，直接給中間值 0.5。
 */
export function percentiles(
  ids: string[],
  scores: Map<string, number>,
  missing?: number,
) {
  const value = (id: string) => scores.get(id) ?? missing;
  const sorted = ids
    .filter((id) => value(id) !== undefined)
    .sort((a, b) => value(a)! - value(b)!);
  const result = new Map<string, number>();
  for (let i = 0; i < sorted.length;) {
    let j = i;
    while (j + 1 < sorted.length && value(sorted[j + 1]) === value(sorted[i]))
      j++;
    for (let k = i; k <= j; k++)
      result.set(sorted[k], ((i + j) / 2 + 1) / sorted.length);
    i = j + 1;
  }
  for (const id of ids) if (!result.has(id)) result.set(id, 0.5);
  return result;
}

/**
 * 依外貌、興趣分數排序候選人。兩種分數先換成候選池裡的百分位再相加：
 * 都有時外貌 60%、興趣 40%，只有其中一種就是 100%。
 * 外貌：沒有向量的候選人給中間值 0.5；分數整個是空的（例如還沒按過喜歡）就當作沒有外貌分數。
 * 興趣：沒有計分標籤的候選人算 0 分。
 * 兩種都沒有就維持傳進來的順序；同分也維持原順序（最新註冊在前）。
 */
export function rankCandidates(
  ids: string[],
  appearance: Map<string, number> | null,
  interest: Map<string, number> | null,
) {
  const pa = appearance?.size ? percentiles(ids, appearance) : null;
  const pi = interest ? percentiles(ids, interest, 0) : null;
  if (!pa && !pi) return ids;
  const weight = pa && pi ? APPEARANCE_WEIGHT : pa ? 1 : 0;
  const score = new Map(
    ids.map((id) => [
      id,
      weight * (pa?.get(id) ?? 0) + (1 - weight) * (pi?.get(id) ?? 0),
    ]),
  );
  return [...ids].sort((a, b) => score.get(b)! - score.get(a)!);
}

/**
 * PASS V2 的「附近」門檻：精確去背 + 扣掉戴眼鏡方向之後，同性別隨機兩人相似度的第 95 百分位
 * （ml/experiments/appearance_face_clip.py，1 萬張種子照片，2026-09-27）。原本的 0.45 幾乎把所有人都算成附近。
 */
export const NEAR_THRESHOLDS = { woman: 0.901, man: 0.845, other: 0.889 };

/**
 * PASS V2 扣分的尺度。原規格最多扣 0.20，是以「無關的人 = 0、一模一樣 = 1」的相似度訂的；
 * 這裡無關的人落在同性別隨機兩人相似度的中位數（女 0.828、男 0.746），直接扣 0.20 會從最前面掉到一般人的程度，
 * 變成原規格說不該有的「封鎖」。所以乘上「1 − 中位數」換算到這裡的尺度，和「附近」門檻一樣依候選人性別
 * （1 萬張種子照片的所有同性別配對，2026-09-28；其他性別用男女各半的分布）。
 */
export const PASS_PENALTY_SCALE = { woman: 0.172, man: 0.254, other: 0.211 };

/**
 * PASS V2（HeartLink Appearance Core V1）：候選人附近被我略過的人達 5 個才扣分，
 * 附近也有我喜歡過的人時扣得比較少；原規格最多扣 0.20，再乘上 PASS_PENALTY_SCALE（女最多約 0.034、男約 0.051）。
 */
export function passPenalty(
  nearPass: number,
  nearLike: number,
  gender: string,
) {
  if (nearPass < 5) return 0;
  const passes = 1 - Math.exp(-0.45 * nearPass);
  const protection = 1 - Math.exp(-0.8 * nearLike);
  const scale =
    gender === "woman" || gender === "man"
      ? PASS_PENALTY_SCALE[gender]
      : PASS_PENALTY_SCALE.other;
  return Math.min(0.2, 0.25 * passes * (1 - protection)) * scale;
}

/**
 * 個人化名額比例（Appearance Core V1）：按過的喜歡 0／1／2／3／4／5 個以上 → 0／60／65／70／75／80%，
 * 其餘是探索名額。興趣分數開著時至少 60%：還沒按過喜歡，也能先靠興趣排。
 */
export function personalizedRatio(likeCount: number, interestOn: boolean) {
  const byLikes = [0, 0.6, 0.65, 0.7, 0.75, 0.8][Math.min(likeCount, 5)];
  return interestOn ? Math.max(0.6, byLikes) : byLikes;
}

/** 字串雜湊（FNV-1a，32 位元）。 */
function hash(text: string) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/**
 * 固定種子的隨機順序：每個人各自用「種子＋自己的 id」算雜湊再排序，
 * 候選池少了某些人（剛滑過）也不會打亂其他人的相對順序，探索名單在每次重抓之間很穩定。
 */
export function seededOrder(ids: string[], seed: string) {
  const key = new Map(ids.map((id) => [id, hash(`${seed}:${id}`)]));
  return [...ids].sort((a, b) => key.get(a)! - key.get(b)!);
}

/**
 * 把探索名額平均穿插進個人化名單。前端一次只顯示第一張，探索卡放在最後幾乎不會被看到；
 * 例如 30 張裡 6 張探索，就是第 5、10、15… 張。
 */
export function interleave(personalized: string[], exploration: string[]) {
  const total = personalized.length + exploration.length;
  const result: string[] = [];
  let p = 0;
  let e = 0;
  for (let slot = 0; slot < total; slot++) {
    const explore =
      Math.floor(((slot + 1) * exploration.length) / total) >
      Math.floor((slot * exploration.length) / total);
    if ((explore && e < exploration.length) || p >= personalized.length)
      result.push(exploration[e++]);
    else result.push(personalized[p++]);
  }
  return result;
}

/**
 * 依名額組出最後的清單：個人化名額取排序最前面的人；探索名額從剩下的人裡用固定種子挑，
 * 先挑沒有被略過扣分的人（避開我常略過的長相），不夠再從被扣分的人補，最後平均穿插。
 */
export function applyQuota(
  ranked: string[],
  options: {
    limit: number;
    ratio: number;
    seed: string;
    penalized: Set<string>;
  },
) {
  const total = Math.min(options.limit, ranked.length);
  const personal = Math.round(total * options.ratio);
  const rest = seededOrder(ranked.slice(personal), options.seed);
  const exploration = [
    ...rest.filter((id) => !options.penalized.has(id)),
    ...rest.filter((id) => options.penalized.has(id)),
  ].slice(0, total - personal);
  return interleave(ranked.slice(0, personal), exploration);
}

/** 從向量扣掉某個方向（例如戴眼鏡）再正規化：v − (v·d)d，d 是單位向量。 */
export function projectOut(vector: number[], direction: number[]) {
  const dot = vector.reduce((sum, v, i) => sum + v * direction[i], 0);
  const out = vector.map((v, i) => v - dot * direction[i]);
  const norm = Math.sqrt(out.reduce((sum, v) => sum + v * v, 0));
  return out.map((v) => v / norm);
}
