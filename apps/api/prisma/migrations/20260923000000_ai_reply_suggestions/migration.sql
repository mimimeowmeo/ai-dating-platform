-- AI 推薦回覆（docs/ai/REPLY-SUGGESTIONS-SPEC.md 第 9 節）的資料表。
-- 向量欄位需要 pgvector；本機的 dating 資料庫已安裝 0.8.6，CI 用 pgvector/pgvector:pg16。
CREATE EXTENSION IF NOT EXISTS "vector";

-- 萃取風格卡時要撈「某個人自己發出的訊息」（跨聊天室、依時間取視窗），沒有這個索引會全表掃描。
CREATE INDEX "messages_sender_id_created_at_idx" ON "messages"("sender_id", "created_at");

-- 一次「AI 推薦回覆」的請求紀錄；失敗的請求也留下（status=error），才能評估可用率。
CREATE TABLE "ai_suggestion_requests" (
    "id" UUID NOT NULL,
    "conversation_id" UUID NOT NULL,
    "requester_id" UUID NOT NULL,
    "mode" TEXT NOT NULL,
    "last_message_id" UUID,
    "status" TEXT NOT NULL,
    "error_code" TEXT,
    "model_name" TEXT,
    "prompt_version" TEXT,
    "input_tokens" INTEGER NOT NULL DEFAULT 0,
    "output_tokens" INTEGER NOT NULL DEFAULT 0,
    "model_requests" INTEGER NOT NULL DEFAULT 0,
    "latency_ms" INTEGER NOT NULL DEFAULT 0,
    "notice" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ai_suggestion_requests_pkey" PRIMARY KEY ("id")
);
CREATE INDEX "ai_suggestion_requests_conversation_id_created_at_idx" ON "ai_suggestion_requests"("conversation_id", "created_at");
CREATE INDEX "ai_suggestion_requests_requester_id_created_at_idx" ON "ai_suggestion_requests"("requester_id", "created_at");
ALTER TABLE "ai_suggestion_requests" ADD CONSTRAINT "ai_suggestion_requests_conversation_id_fkey" FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "ai_suggestion_requests" ADD CONSTRAINT "ai_suggestion_requests_requester_id_fkey" FOREIGN KEY ("requester_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- 產生過的每一則推薦：rank 1～5 是回給使用者的，rank 0 是被後處理刪掉的候選（決策：全部保存）。
CREATE TABLE "ai_suggestions" (
    "id" UUID NOT NULL,
    "request_id" UUID NOT NULL,
    "rank" INTEGER NOT NULL,
    "text" TEXT NOT NULL,
    "intent" TEXT,
    "style_target" TEXT,
    "style_distance" DOUBLE PRECISION,
    "reason" TEXT,
    "rejected_reason" TEXT,
    "chosen_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ai_suggestions_pkey" PRIMARY KEY ("id")
);
CREATE INDEX "ai_suggestions_request_id_rank_idx" ON "ai_suggestions"("request_id", "rank");
ALTER TABLE "ai_suggestions" ADD CONSTRAINT "ai_suggestions_request_id_fkey" FOREIGN KEY ("request_id") REFERENCES "ai_suggestion_requests"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- 訊息來源；獨立一張表，避免 message:new 把「這則是 AI 產生的」推給聊天對象（規格 5.6）。
CREATE TABLE "message_origins" (
    "message_id" UUID NOT NULL,
    "origin" TEXT NOT NULL,
    "similarity" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "suggestion_id" UUID,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "message_origins_pkey" PRIMARY KEY ("message_id")
);
CREATE INDEX "message_origins_origin_idx" ON "message_origins"("origin");
CREATE INDEX "message_origins_suggestion_id_idx" ON "message_origins"("suggestion_id");
ALTER TABLE "message_origins" ADD CONSTRAINT "message_origins_message_id_fkey" FOREIGN KEY ("message_id") REFERENCES "messages"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "message_origins" ADD CONSTRAINT "message_origins_suggestion_id_fkey" FOREIGN KEY ("suggestion_id") REFERENCES "ai_suggestions"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- 對話切片與它的向量（RAG 的檢索單位）。先做精確搜尋，資料量大到變慢再加 HNSW 索引。
CREATE TABLE "conversation_chunks" (
    "id" UUID NOT NULL,
    "conversation_id" UUID NOT NULL,
    "first_message_id" UUID NOT NULL,
    "last_message_id" UUID NOT NULL,
    "first_at" TIMESTAMP(3) NOT NULL,
    "last_at" TIMESTAMP(3) NOT NULL,
    "message_count" INTEGER NOT NULL,
    "content" TEXT NOT NULL,
    "token_estimate" INTEGER NOT NULL,
    "is_open" BOOLEAN NOT NULL DEFAULT false,
    "chunk_version" TEXT NOT NULL,
    "embedding_model" TEXT,
    "embedding" vector(768),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "conversation_chunks_pkey" PRIMARY KEY ("id")
);
CREATE INDEX "conversation_chunks_conversation_id_last_at_idx" ON "conversation_chunks"("conversation_id", "last_at");
ALTER TABLE "conversation_chunks" ADD CONSTRAINT "conversation_chunks_conversation_id_fkey" FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- 聊天室摘要（每 50 則更新一次），保留歷史版本，只有最新一筆會放進 prompt。
CREATE TABLE "conversation_summaries" (
    "id" UUID NOT NULL,
    "conversation_id" UUID NOT NULL,
    "until_message_id" UUID NOT NULL,
    "summary" TEXT NOT NULL,
    "model_name" TEXT,
    "prompt_version" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "conversation_summaries_pkey" PRIMARY KEY ("id")
);
CREATE INDEX "conversation_summaries_conversation_id_created_at_idx" ON "conversation_summaries"("conversation_id", "created_at");
ALTER TABLE "conversation_summaries" ADD CONSTRAINT "conversation_summaries_conversation_id_fkey" FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- 使用者的風格卡；每次重新萃取新增一個版本，舊版本保留以便比較。
CREATE TABLE "user_style_profiles" (
    "user_id" UUID NOT NULL,
    "version" INTEGER NOT NULL,
    "feature_version" TEXT NOT NULL,
    "confidence" TEXT NOT NULL,
    "sample_source" TEXT NOT NULL,
    "message_count" INTEGER NOT NULL DEFAULT 0,
    "window_from" TIMESTAMP(3),
    "window_to" TIMESTAMP(3),
    "stats" JSONB NOT NULL,
    "voice_notes" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "bio_sample" TEXT NOT NULL DEFAULT '',
    "model_name" TEXT,
    "prompt_version" TEXT,
    "embedding_model" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "user_style_profiles_pkey" PRIMARY KEY ("user_id","version")
);
CREATE INDEX "user_style_profiles_user_id_created_at_idx" ON "user_style_profiles"("user_id", "created_at");
ALTER TABLE "user_style_profiles" ADD CONSTRAINT "user_style_profiles_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- 風格卡上的抽象特徵句與向量；產生推薦時用查詢向量挑出跟目前話題相關的幾條。
CREATE TABLE "user_style_facets" (
    "id" UUID NOT NULL,
    "user_id" UUID NOT NULL,
    "profile_version" INTEGER NOT NULL,
    "kind" TEXT NOT NULL,
    "statement" TEXT NOT NULL,
    "weight" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "evidence" INTEGER NOT NULL DEFAULT 0,
    "embedding" vector(768),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "user_style_facets_pkey" PRIMARY KEY ("id")
);
CREATE INDEX "user_style_facets_user_id_profile_version_idx" ON "user_style_facets"("user_id", "profile_version");
ALTER TABLE "user_style_facets" ADD CONSTRAINT "user_style_facets_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "user_style_facets" ADD CONSTRAINT "user_style_facets_user_id_profile_version_fkey" FOREIGN KEY ("user_id", "profile_version") REFERENCES "user_style_profiles"("user_id", "version") ON DELETE CASCADE ON UPDATE CASCADE;

-- AI 推薦開啟的話題區段；同一則發起訊息只會有一個區段，所以用唯一鍵讓重跑可以覆蓋。
CREATE TABLE "ai_topic_spans" (
    "id" UUID NOT NULL,
    "conversation_id" UUID NOT NULL,
    "initiating_message_id" UUID NOT NULL,
    "initiator_id" UUID NOT NULL,
    "start_message_id" UUID NOT NULL,
    "end_message_id" UUID NOT NULL,
    "message_count" INTEGER NOT NULL,
    "end_reason" TEXT NOT NULL,
    "detector_version" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ai_topic_spans_pkey" PRIMARY KEY ("id")
);
CREATE UNIQUE INDEX "ai_topic_spans_conversation_id_initiating_message_id_key" ON "ai_topic_spans"("conversation_id", "initiating_message_id");
CREATE INDEX "ai_topic_spans_initiator_id_idx" ON "ai_topic_spans"("initiator_id");
ALTER TABLE "ai_topic_spans" ADD CONSTRAINT "ai_topic_spans_conversation_id_fkey" FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id") ON DELETE CASCADE ON UPDATE CASCADE;
