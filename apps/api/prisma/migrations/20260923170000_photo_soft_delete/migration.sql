-- 照片改成軟刪除：刪除或替換主照片時只寫 deleted_at，照片記錄與 MinIO 物件都保留。
-- 可為空、沒有預設值，既有照片全部視為未刪除，不需要回填。
ALTER TABLE "user_photos" ADD COLUMN "deleted_at" TIMESTAMP(3);
