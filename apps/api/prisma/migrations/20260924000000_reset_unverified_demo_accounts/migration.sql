-- 匯入的示範帳號（db/07_transform_to_app.sql）當初被直接標成已驗證，但從沒做過真人驗證。
-- 沒有任何一筆 verified 驗證紀錄的帳號一律改回未驗證；真的通過驗證的帳號不受影響。重複執行結果相同。
UPDATE "users" AS u
SET "is_verified" = false, "updated_at" = CURRENT_TIMESTAMP
WHERE u."is_verified" = true
  AND NOT EXISTS (
    SELECT 1 FROM "verification_records" AS v
    WHERE v."user_id" = u."id" AND v."status" = 'verified'
  );
