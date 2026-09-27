-- 探索頁每次滑卡當下的推薦狀態（分數、推薦卡還是未推薦卡、第幾張），用來評估推薦與之後訓練個人化比重。
CREATE TABLE "swipe_logs" (
    "id" UUID NOT NULL,
    "user_id" UUID NOT NULL,
    "target_user_id" UUID NOT NULL,
    "action" TEXT NOT NULL,
    "source" TEXT NOT NULL,
    "position" INTEGER,
    "rank" INTEGER,
    "pool_size" INTEGER,
    "score" DOUBLE PRECISION,
    "appearance_weight" DOUBLE PRECISION,
    "appearance_percentile" DOUBLE PRECISION,
    "appearance_similarity" DOUBLE PRECISION,
    "pass_penalty" DOUBLE PRECISION,
    "interest_percentile" DOUBLE PRECISION,
    "interest_score" DOUBLE PRECISION,
    "like_count" INTEGER,
    "swipe_count" INTEGER,
    "hard_filter" BOOLEAN,
    "appearance_on" BOOLEAN,
    "interest_on" BOOLEAN,
    "ranking_version" TEXT,
    "served_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "swipe_logs_pkey" PRIMARY KEY ("id")
);
CREATE INDEX "swipe_logs_user_id_created_at_idx" ON "swipe_logs"("user_id", "created_at");
CREATE INDEX "swipe_logs_target_user_id_idx" ON "swipe_logs"("target_user_id");
ALTER TABLE "swipe_logs" ADD CONSTRAINT "swipe_logs_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "swipe_logs" ADD CONSTRAINT "swipe_logs_target_user_id_fkey" FOREIGN KEY ("target_user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
