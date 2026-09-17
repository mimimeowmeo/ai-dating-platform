-- AlterTable
ALTER TABLE "sessions" ADD COLUMN     "previous_token_hash" TEXT,
ADD COLUMN     "rotated_at" TIMESTAMP(3);

-- CreateIndex
CREATE UNIQUE INDEX "sessions_previous_token_hash_key" ON "sessions"("previous_token_hash");
