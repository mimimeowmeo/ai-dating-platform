// 雙向偏好篩選（探索頁的「硬篩選」）：年齡、性別、交往目的、身高、距離，雙方都要符合對方的偏好。
// 同一套規則有兩份：eligible() 在程式裡判斷一對人（按喜歡、測試用搜尋列），eligibleIds() 用 SQL
// 從全部使用者篩出探索頁的候選池。改一邊就要改另一邊；integration.test.mjs 會拿兩者的結果比對。
import { ageAt, distance, type Database } from "./core";

/** 身高偏好拉桿的兩端；停在兩端代表不限。 */
export const preferenceHeightRange = [130, 250] as const;

/** a 的偏好能不能接受 b。 */
export function eligible(a: any, b: any) {
  if (!a?.profile || !b?.profile || !a.preference || !b.preference)
    return false;
  const p = a.preference,
    q = b.profile;
  return (
    ageAt(q.birthDate) >= p.minAge &&
    ageAt(q.birthDate) <= p.maxAge &&
    (p.preferredGender === "any" || p.preferredGender === q.gender) &&
    // 關係期待看 traits 的 dating_goal：偏好「都可以」，或對方的交友目標包含它。
    (p.preferredDatingIntent === "any" ||
      (b.traits || []).some(
        (t: any) =>
          t.trait.category === "dating_goal" &&
          t.trait.code === p.preferredDatingIntent,
      )) &&
    // 沒填身高的人無從判斷，不因身高條件被排除。拉桿停在兩端代表不限：
    // 身高必填後，低於 130 的人沒辦法留空，不能讓預設偏好把他們擋掉。
    (q.heightCm == null ||
      ((p.minHeightCm <= preferenceHeightRange[0] ||
        q.heightCm >= p.minHeightCm) &&
        (p.maxHeightCm >= preferenceHeightRange[1] ||
          q.heightCm <= p.maxHeightCm))) &&
    distance(a.profile, q) <= p.maxDistanceKm
  );
}

/**
 * 探索頁的候選池：全部使用者裡，雙方都符合對方偏好的人的 id（最新註冊在前，同時間再依 id）。
 * 另外排除自己、我按過喜歡或略過的人、任一方封鎖、配對過（不論狀態）的人。
 * 規則和 eligible() 相同：年齡用 UTC 的今天算足歲，距離用同一個球面公式。沒有個人檔案或偏好的人不列入。
 */
export async function eligibleIds(
  db: Database,
  userId: string,
  now = new Date(),
) {
  const today = now.toISOString().slice(0, 10);
  const [minHeight, maxHeight] = preferenceHeightRange;
  const rows = await db.$queryRaw<{ id: string }[]>`
    WITH me AS (
      SELECT u.id, pr.birth_date, pr.gender, pr.latitude, pr.longitude, pr.height_cm,
             pf.min_age, pf.max_age, pf.preferred_gender, pf.preferred_dating_intent,
             pf.min_height_cm, pf.max_height_cm, pf.max_distance_km,
             ARRAY(
               SELECT t.code FROM user_traits ut JOIN traits t ON t.id = ut.trait_id
               WHERE ut.user_id = u.id AND t.category = 'dating_goal'
             ) AS goals
      FROM users u
      JOIN profiles pr ON pr.user_id = u.id
      JOIN preferences pf ON pf.user_id = u.id
      WHERE u.id = ${userId}::uuid
    ),
    candidates AS (
      SELECT u.id, u.created_at, pr.birth_date, pr.gender, pr.height_cm,
             pf.min_age, pf.max_age, pf.preferred_gender, pf.preferred_dating_intent,
             pf.min_height_cm, pf.max_height_cm, pf.max_distance_km,
             6371 * 2 * asin(sqrt(least(1,
               sin(radians(pr.latitude - me.latitude) / 2) ^ 2
               + cos(radians(me.latitude)) * cos(radians(pr.latitude))
                 * sin(radians(pr.longitude - me.longitude) / 2) ^ 2))) AS km
      FROM users u
      JOIN profiles pr ON pr.user_id = u.id
      JOIN preferences pf ON pf.user_id = u.id
      CROSS JOIN me
      WHERE u.id <> me.id
        AND NOT EXISTS (SELECT 1 FROM likes l WHERE l.from_user_id = me.id AND l.to_user_id = u.id)
        AND NOT EXISTS (
          SELECT 1 FROM blocks b
          WHERE (b.user_id = u.id AND b.blocked_user_id = me.id)
             OR (b.user_id = me.id AND b.blocked_user_id = u.id))
        AND NOT EXISTS (
          SELECT 1 FROM matches m
          WHERE (m.user_a_id = u.id AND m.user_b_id = me.id)
             OR (m.user_a_id = me.id AND m.user_b_id = u.id))
    )
    SELECT c.id::text AS id
    FROM candidates c CROSS JOIN me
    WHERE date_part('year', age(${today}::date, c.birth_date)) BETWEEN me.min_age AND me.max_age
      AND (me.preferred_gender = 'any' OR me.preferred_gender = c.gender)
      AND (me.preferred_dating_intent = 'any' OR EXISTS (
            SELECT 1 FROM user_traits ut JOIN traits t ON t.id = ut.trait_id
            WHERE ut.user_id = c.id AND t.category = 'dating_goal'
              AND t.code = me.preferred_dating_intent))
      AND (c.height_cm IS NULL OR (
            (me.min_height_cm <= ${minHeight} OR c.height_cm >= me.min_height_cm)
            AND (me.max_height_cm >= ${maxHeight} OR c.height_cm <= me.max_height_cm)))
      AND c.km <= me.max_distance_km
      AND date_part('year', age(${today}::date, me.birth_date)) BETWEEN c.min_age AND c.max_age
      AND (c.preferred_gender = 'any' OR c.preferred_gender = me.gender)
      AND (c.preferred_dating_intent = 'any' OR c.preferred_dating_intent = ANY(me.goals))
      AND (me.height_cm IS NULL OR (
            (c.min_height_cm <= ${minHeight} OR me.height_cm >= c.min_height_cm)
            AND (c.max_height_cm >= ${maxHeight} OR me.height_cm <= c.max_height_cm)))
      AND c.km <= c.max_distance_km
    ORDER BY c.created_at DESC, c.id ASC
  `;
  return rows.map((row) => row.id);
}
