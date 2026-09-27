-- 外貌推薦用的照片向量（CLIP ViT-B/32，512 維），探索頁的外貌分數用它算「和我按過喜歡的人有多像」。
CREATE TABLE "appearance_embeddings" (
    "photo_id" UUID NOT NULL,
    "user_id" UUID NOT NULL,
    "model_version" TEXT NOT NULL,
    "embedding" vector(512) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "appearance_embeddings_pkey" PRIMARY KEY ("photo_id")
);
CREATE INDEX "appearance_embeddings_user_id_idx" ON "appearance_embeddings"("user_id");
ALTER TABLE "appearance_embeddings" ADD CONSTRAINT "appearance_embeddings_photo_id_fkey" FOREIGN KEY ("photo_id") REFERENCES "user_photos"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "appearance_embeddings" ADD CONSTRAINT "appearance_embeddings_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
