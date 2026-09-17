-- 由 gen_sql.py 自動產生：staging_wide → 正規化 5 表

INSERT INTO users (account, password_hash)
SELECT account, crypt(password, gen_salt('bf'))
FROM staging_wide
ON CONFLICT (account) DO NOTHING;

INSERT INTO profiles (user_id, name, gender, birth_date, height_cm, image_file, bio, zodiac)
SELECT u.id, s.name, s.gender,
       to_date(s.birth_date, 'YYYY/MM/DD'),
       s.height_cm::smallint, s.image_file, s.bio, s.zodiac
FROM staging_wide s JOIN users u ON u.account = s.account
ON CONFLICT (user_id) DO NOTHING;

INSERT INTO user_traits (user_id, trait_id)
SELECT u.id, t.id
FROM staging_wide s
JOIN users u ON u.account = s.account
CROSS JOIN LATERAL jsonb_each_text(to_jsonb(s)) AS kv(col, val)
JOIN traits t ON t.csv_column = kv.col
WHERE kv.val = '1'
ON CONFLICT DO NOTHING;

INSERT INTO user_face_features (user_id, attributes)
SELECT u.id, jsonb_object_agg(substring(kv.col from 6), (kv.val = '1'))
FROM staging_wide s
JOIN users u ON u.account = s.account
CROSS JOIN LATERAL jsonb_each_text(to_jsonb(s)) AS kv(col, val)
WHERE kv.col LIKE 'face\_%'
GROUP BY u.id
ON CONFLICT (user_id) DO NOTHING;
