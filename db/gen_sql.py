
import csv, sys

CSV_PATH = sys.argv[1]

with open(CSV_PATH, encoding="utf-8-sig") as f:
    header = next(csv.reader(f))

TRAIT_PREFIXES = ["dating_goal", "interest", "personality", "diet", "lifestyle", "value"]

LABELS = {
    # dating_goal
    "serious_relationship": "認真交往", "friends_first": "先做朋友", "chat_only": "只想聊天",
    "dining_partner": "找飯友", "marriage_minded": "以結婚為前提",
    # interest
    "hiking": "登山", "camping": "露營", "surfing": "衝浪", "fitness": "健身", "running": "跑步",
    "skateboarding": "滑板", "badminton": "羽球", "basketball": "籃球", "cycling": "騎車",
    "swimming": "游泳", "yoga": "瑜伽", "skiing": "滑雪", "tv_series": "追劇", "movies": "電影",
    "anime": "動漫", "gaming": "電玩", "board_games": "桌遊", "reading": "閱讀", "cooking": "料理",
    "karaoke": "KTV", "photography": "攝影", "exhibitions": "看展", "live_music": "live 音樂",
    "singing": "唱歌", "playing_instruments": "樂器", "writing": "寫作", "drawing": "繪畫",
    "dancing": "跳舞", "exploring_shops": "逛店", "food": "美食", "wine_tasting": "品酒",
    "cat_person": "貓派", "dog_person": "狗派", "travel": "旅行", "coffee": "咖啡",
    "desserts": "甜點", "fashion": "時尚",
    # personality
    "humorous": "幽默", "slow_to_warm_up": "慢熱", "talkative": "健談", "quiet": "文靜",
    "rational": "理性", "emotional": "感性", "optimistic": "樂觀", "independent": "獨立",
    "direct": "直接", "romantic": "浪漫", "action_oriented": "行動派", "homebody": "居家",
    # diet
    "likes_seafood": "愛海鮮", "likes_japanese_food": "愛日式", "likes_hotpot": "愛火鍋",
    "likes_yakiniku": "愛燒肉", "vegetarian": "素食",
    # lifestyle
    "nine_to_five": "朝九晚五", "shift_work": "輪班", "two_days_off_weekly": "週休二日",
    "work_from_home": "遠端工作",
    # value
    "likes_sharing_daily_life": "愛分享日常", "values_companionship": "重視陪伴",
    "needs_personal_space": "需要個人空間", "values_communication": "重視溝通",
    "values_trust": "重視信任",
}

def category_of(col):
    for p in sorted(TRAIT_PREFIXES, key=len, reverse=True):
        if col.startswith(p + "_"):
            return p, col[len(p) + 1:]
    return None, None

cols_ddl = ",\n    ".join(f'"{c}" text' for c in header)
with open("db/02_staging.sql", "w", encoding="utf-8") as f:
    f.write("-- 由 gen_sql.py 自動產生：寬表暫存區\n")
    f.write("DROP TABLE IF EXISTS staging_wide;\n")
    f.write(f"CREATE TABLE staging_wide (\n    {cols_ddl}\n);\n")

with open("db/03_seed_traits.sql", "w", encoding="utf-8") as f:
    f.write("-- 由 gen_sql.py 自動產生：把 68 個 one-hot 欄名塞成 traits 資料列\n")
    f.write("INSERT INTO traits (category, code, label_zh, csv_column) VALUES\n")
    rows = []
    for col in header:
        cat, code = category_of(col)
        if cat is None:
            continue
        label = LABELS.get(code, code).replace("'", "''")
        rows.append(f"  ('{cat}', '{code}', '{label}', '{col}')")
    f.write(",\n".join(rows) + ";\n")

face_cols = [c for c in header if c.startswith("face_")]
with open("db/04_load.sql", "w", encoding="utf-8") as f:
    f.write("-- 由 gen_sql.py 自動產生：staging_wide → 正規化 5 表\n\n")

    f.write(
        "INSERT INTO users (account, password_hash)\n"
        "SELECT account, crypt(password, gen_salt('bf'))\n"
        "FROM staging_wide\n"
        "ON CONFLICT (account) DO NOTHING;\n\n"
    )

    f.write(
        "INSERT INTO profiles (user_id, name, gender, birth_date, height_cm, image_file, bio, zodiac)\n"
        "SELECT u.id, s.name, s.gender,\n"
        "       to_date(s.birth_date, 'YYYY/MM/DD'),\n"
        "       s.height_cm::smallint, s.image_file, s.bio, s.zodiac\n"
        "FROM staging_wide s JOIN users u ON u.account = s.account\n"
        "ON CONFLICT (user_id) DO NOTHING;\n\n"
    )

    f.write(
        "INSERT INTO user_traits (user_id, trait_id)\n"
        "SELECT u.id, t.id\n"
        "FROM staging_wide s\n"
        "JOIN users u ON u.account = s.account\n"
        "CROSS JOIN LATERAL jsonb_each_text(to_jsonb(s)) AS kv(col, val)\n"
        "JOIN traits t ON t.csv_column = kv.col\n"
        "WHERE kv.val = '1'\n"
        "ON CONFLICT DO NOTHING;\n\n"
    )

    f.write(
        "INSERT INTO user_face_features (user_id, attributes)\n"
        "SELECT u.id, jsonb_object_agg(substring(kv.col from 6), (kv.val = '1'))\n"
        "FROM staging_wide s\n"
        "JOIN users u ON u.account = s.account\n"
        "CROSS JOIN LATERAL jsonb_each_text(to_jsonb(s)) AS kv(col, val)\n"
        "WHERE kv.col LIKE 'face\\_%'\n"
        "GROUP BY u.id\n"
        "ON CONFLICT (user_id) DO NOTHING;\n"
    )

print("generated: 02_staging.sql, 03_seed_traits.sql, 04_load.sql")
print(f"header cols={len(header)}, trait rows={sum(1 for c in header if category_of(c)[0])}, face cols={len(face_cols)}")
