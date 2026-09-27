-- 外貌向量要扣掉的方向（目前只有 glasses），worker 算好的新向量寫回時套用。
CREATE TABLE "appearance_directions" (
    "name" TEXT NOT NULL,
    "model_version" TEXT NOT NULL,
    "direction" vector(512) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "appearance_directions_pkey" PRIMARY KEY ("name")
);

-- 找不到臉的照片也留一列（embedding 是 NULL），表示已經處理過，定期補排時不會一直重送。
ALTER TABLE "appearance_embeddings" ALTER COLUMN "embedding" DROP NOT NULL;
