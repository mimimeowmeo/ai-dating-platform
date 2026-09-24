
TRUNCATE users CASCADE;

DROP TABLE IF EXISTS _map;
CREATE TABLE _map AS
SELECT hu.id AS old_id, gen_random_uuid() AS new_id, hu.account, hu.password_hash
FROM hl.users hu;

INSERT INTO users (id, email, password_hash, is_verified, created_at, updated_at)
-- 匯入的示範帳號沒有做過真人驗證，一律從未驗證開始。
SELECT new_id, lower(account) || '@heartlink.local', password_hash, false, now(), now()
FROM _map;

INSERT INTO profiles (
  user_id, display_name, birth_date, gender, bio, city, latitude, longitude,
  dating_intent, height_cm, occupation, education, interests, hobbies, foods, created_at, updated_at)
SELECT
  m.new_id,
  p.name,
  p.birth_date,
  CASE p.gender WHEN 'Male' THEN 'man' WHEN 'Female' THEN 'woman' ELSE 'nonbinary' END,
  COALESCE(p.bio, ''),
  (ARRAY['台北','台中','高雄','台南','新竹','桃園'])[(m.old_id % 6) + 1],
  (ARRAY[25.033, 24.147, 22.627, 22.999, 24.813, 24.994])[(m.old_id % 6) + 1],
  (ARRAY[121.565, 120.673, 120.301, 120.227, 120.967, 121.301])[(m.old_id % 6) + 1],
  COALESCE((
    SELECT CASE t.code
             WHEN 'serious_relationship' THEN 'serious'
             WHEN 'marriage_minded'      THEN 'serious'
             WHEN 'friends_first'        THEN 'friendship'
             WHEN 'chat_only'            THEN 'casual'
             WHEN 'dining_partner'       THEN 'casual'
           END
    FROM hl.user_traits ut JOIN hl.traits t ON t.id = ut.trait_id
    WHERE ut.user_id = p.user_id AND t.category = 'dating_goal'
    ORDER BY t.id LIMIT 1), 'serious'),
  p.height_cm,
  NULL, NULL,
  COALESCE((SELECT array_agg(DISTINCT mp.label)
            FROM hl.user_traits ut JOIN hl.traits t ON t.id=ut.trait_id
            JOIN (VALUES
              ('travel','旅行'),('coffee','咖啡'),
              ('movies','電影'),('tv_series','電影'),('anime','電影'),
              ('live_music','音樂'),('singing','音樂'),('playing_instruments','音樂'),
              ('drawing','藝術'),('photography','藝術'),('exhibitions','藝術'),('dancing','藝術'),
              ('gaming','科技'),('board_games','科技'),
              ('reading','閱讀'),('writing','閱讀'),
              ('hiking','戶外'),('camping','戶外'),('surfing','戶外'),('running','戶外'),('cycling','戶外'),('skiing','戶外')
            ) AS mp(code,label) ON mp.code=t.code
            WHERE ut.user_id=p.user_id), '{}'),
  COALESCE((SELECT array_agg(DISTINCT mp.label)
            FROM hl.user_traits ut JOIN hl.traits t ON t.id=ut.trait_id
            JOIN (VALUES
              ('photography','攝影'),('hiking','登山'),('fitness','健身'),
              ('cooking','烹飪'),('drawing','繪畫'),('board_games','桌遊'),
              ('yoga','瑜珈'),('running','散步')
            ) AS mp(code,label) ON mp.code=t.code
            WHERE ut.user_id=p.user_id), '{}'),
  COALESCE((SELECT array_agg(DISTINCT mp.label)
            FROM hl.user_traits ut JOIN hl.traits t ON t.id=ut.trait_id
            JOIN (VALUES
              ('likes_japanese_food','日式料理'),('likes_seafood','海鮮'),
              ('likes_hotpot','火鍋'),('likes_yakiniku','台灣小吃'),('vegetarian','蔬食'),
              ('desserts','甜點'),('food','台灣小吃')
            ) AS mp(code,label) ON mp.code=t.code
            WHERE ut.user_id=p.user_id), '{}'),
  now(), now()
FROM hl.profiles p JOIN _map m ON m.old_id = p.user_id;

INSERT INTO user_photos (id, user_id, storage_key, mime_type, is_avatar, display_order, created_at)
SELECT gen_random_uuid(), m.new_id, pp.storage_key, pp.mime_type, pp.is_avatar, pp.display_order, now()
FROM hl.profile_photos pp JOIN _map m ON m.old_id = pp.user_id;

INSERT INTO preferences (user_id, min_age, max_age, preferred_gender, max_distance_km, preferred_dating_intent, updated_at)
SELECT new_id, 18, 99, 'any', 100, 'any', now()
FROM _map;
