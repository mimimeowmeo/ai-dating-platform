-- 匯入資料的喜好回填：hl.user_traits（舊 bigint id）→ app 的 user_traits（uuid）。
-- 前置：apps/api 的 20260918030000_traits 已套用（traits / user_traits 存在），且 _map 還在。
-- 這支只新增關聯，不動 users / profiles，重跑安全（ON CONFLICT DO NOTHING）。

-- 1) 先確認兩邊的代碼完全一致；結果必須是 0，否則不要往下跑。
SELECT count(*) AS mismatched
FROM hl.traits h
JOIN traits t ON t.id = h.id
WHERE t.category <> h.category OR t.code <> h.code;

-- 2) 回填（2026-09-18 試算：155655 列，全部對得到 _map）。
INSERT INTO user_traits (user_id, trait_id)
SELECT m.new_id, ut.trait_id
FROM hl.user_traits ut
JOIN _map m ON m.old_id = ut.user_id
JOIN traits t ON t.id = ut.trait_id
ON CONFLICT DO NOTHING;

-- 3) 驗收：交友目標每人最多 2 項（畫面與 API 的上限），結果必須是 0。
SELECT count(*) AS over_limit
FROM (
  SELECT ut.user_id
  FROM user_traits ut
  JOIN traits t ON t.id = ut.trait_id
  WHERE t.category = 'dating_goal'
  GROUP BY 1
  HAVING count(*) > 2
) x;
