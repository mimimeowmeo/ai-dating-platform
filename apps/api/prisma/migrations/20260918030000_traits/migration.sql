-- 選項清單（traits）與使用者的選擇（user_traits）。
-- 資料與 heartlink 資料集的 traits 表一致，id 沿用同一組編號，方便對應匯入的資料。
CREATE TABLE "traits" (
    "id" SERIAL NOT NULL,
    "category" TEXT NOT NULL,
    "code" TEXT NOT NULL,
    "label_zh" TEXT,
    CONSTRAINT "traits_pkey" PRIMARY KEY ("id")
);
CREATE UNIQUE INDEX "traits_category_code_key" ON "traits"("category", "code");

CREATE TABLE "user_traits" (
    "user_id" UUID NOT NULL,
    "trait_id" INTEGER NOT NULL,
    CONSTRAINT "user_traits_pkey" PRIMARY KEY ("user_id","trait_id")
);
CREATE INDEX "user_traits_trait_id_idx" ON "user_traits"("trait_id");
ALTER TABLE "user_traits" ADD CONSTRAINT "user_traits_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "user_traits" ADD CONSTRAINT "user_traits_trait_id_fkey" FOREIGN KEY ("trait_id") REFERENCES "traits"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

INSERT INTO "traits" ("id", "category", "code", "label_zh") VALUES
  (1, 'dating_goal', 'serious_relationship', '認真交往'),
  (2, 'dating_goal', 'friends_first', '先做朋友'),
  (3, 'dating_goal', 'chat_only', '只想聊天'),
  (4, 'dating_goal', 'dining_partner', '找飯友'),
  (5, 'dating_goal', 'marriage_minded', '以結婚為前提'),
  (6, 'interest', 'hiking', '登山'),
  (7, 'interest', 'camping', '露營'),
  (8, 'interest', 'surfing', '衝浪'),
  (9, 'interest', 'fitness', '健身'),
  (10, 'interest', 'running', '跑步'),
  (11, 'interest', 'skateboarding', '滑板'),
  (12, 'interest', 'badminton', '羽球'),
  (13, 'interest', 'basketball', '籃球'),
  (14, 'interest', 'cycling', '騎車'),
  (15, 'interest', 'swimming', '游泳'),
  (16, 'interest', 'yoga', '瑜伽'),
  (17, 'interest', 'skiing', '滑雪'),
  (18, 'interest', 'tv_series', '追劇'),
  (19, 'interest', 'movies', '電影'),
  (20, 'interest', 'anime', '動漫'),
  (21, 'interest', 'gaming', '電玩'),
  (22, 'interest', 'board_games', '桌遊'),
  (23, 'interest', 'reading', '閱讀'),
  (24, 'interest', 'cooking', '料理'),
  (25, 'interest', 'karaoke', 'KTV'),
  (26, 'interest', 'photography', '攝影'),
  (27, 'interest', 'exhibitions', '看展'),
  (28, 'interest', 'live_music', 'live 音樂'),
  (29, 'interest', 'singing', '唱歌'),
  (30, 'interest', 'playing_instruments', '樂器'),
  (31, 'interest', 'writing', '寫作'),
  (32, 'interest', 'drawing', '繪畫'),
  (33, 'interest', 'dancing', '跳舞'),
  (34, 'interest', 'exploring_shops', '逛店'),
  (35, 'interest', 'food', '美食'),
  (36, 'interest', 'wine_tasting', '品酒'),
  (37, 'interest', 'cat_person', '貓派'),
  (38, 'interest', 'dog_person', '狗派'),
  (39, 'interest', 'travel', '旅行'),
  (40, 'interest', 'coffee', '咖啡'),
  (41, 'interest', 'desserts', '甜點'),
  (42, 'interest', 'fashion', '時尚'),
  (43, 'personality', 'humorous', '幽默'),
  (44, 'personality', 'slow_to_warm_up', '慢熱'),
  (45, 'personality', 'talkative', '健談'),
  (46, 'personality', 'quiet', '文靜'),
  (47, 'personality', 'rational', '理性'),
  (48, 'personality', 'emotional', '感性'),
  (49, 'personality', 'optimistic', '樂觀'),
  (50, 'personality', 'independent', '獨立'),
  (51, 'personality', 'direct', '直接'),
  (52, 'personality', 'romantic', '浪漫'),
  (53, 'personality', 'action_oriented', '行動派'),
  (54, 'personality', 'homebody', '居家'),
  (55, 'diet', 'likes_seafood', '愛海鮮'),
  (56, 'diet', 'likes_japanese_food', '愛日式'),
  (57, 'diet', 'likes_hotpot', '愛火鍋'),
  (58, 'diet', 'likes_yakiniku', '愛燒肉'),
  (59, 'diet', 'vegetarian', '素食'),
  (60, 'lifestyle', 'nine_to_five', '朝九晚五'),
  (61, 'lifestyle', 'shift_work', '輪班'),
  (62, 'lifestyle', 'two_days_off_weekly', '週休二日'),
  (63, 'lifestyle', 'work_from_home', '遠端工作'),
  (64, 'value', 'likes_sharing_daily_life', '愛分享日常'),
  (65, 'value', 'values_companionship', '重視陪伴'),
  (66, 'value', 'needs_personal_space', '需要個人空間'),
  (67, 'value', 'values_communication', '重視溝通'),
  (68, 'value', 'values_trust', '重視信任');
SELECT setval(pg_get_serial_sequence('traits', 'id'), (SELECT max("id") FROM "traits"));

-- dating_intent 改由 dating_goal 取代：保留欄位與資料，新資料用預設值。
ALTER TABLE "profiles" ALTER COLUMN "dating_intent" SET DEFAULT 'serious';

-- 舊的關係期待（serious／casual／friendship）在新的選項裡不存在，改回「都可以」，請使用者重選。
UPDATE "preferences" SET "preferred_dating_intent" = 'any'
WHERE "preferred_dating_intent" <> 'any'
  AND "preferred_dating_intent" NOT IN (SELECT "code" FROM "traits" WHERE "category" = 'dating_goal');
