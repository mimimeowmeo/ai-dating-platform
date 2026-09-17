
CREATE TABLE IF NOT EXISTS profile_photos (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id       bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    storage_key   text NOT NULL UNIQUE,
    bucket        text NOT NULL DEFAULT 'heartlink-media',
    mime_type     text NOT NULL DEFAULT 'image/jpeg',
    is_avatar     boolean NOT NULL DEFAULT false,
    display_order int NOT NULL DEFAULT 0,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_profile_photos_user ON profile_photos(user_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_profile_photos_avatar
    ON profile_photos(user_id) WHERE is_avatar;

INSERT INTO profile_photos (user_id, storage_key, mime_type, is_avatar, display_order)
SELECT user_id, 'profiles/' || image_file, 'image/jpeg', true, 0
FROM profiles
WHERE image_file IS NOT NULL
ON CONFLICT (storage_key) DO NOTHING;
