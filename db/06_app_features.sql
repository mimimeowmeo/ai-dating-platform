

CREATE TABLE IF NOT EXISTS sessions (
    id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id             bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash          text NOT NULL UNIQUE,
    previous_token_hash text UNIQUE,
    rotated_at          timestamptz,
    expires_at          timestamptz NOT NULL,
    revoked_at          timestamptz,
    created_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS preferences (
    user_id                 bigint PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    min_age                 int  NOT NULL DEFAULT 18,
    max_age                 int  NOT NULL DEFAULT 99,
    preferred_gender        text NOT NULL DEFAULT 'any',
    max_distance_km         int  NOT NULL DEFAULT 100,
    preferred_dating_intent text NOT NULL DEFAULT 'any',
    updated_at              timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS profile_extras (
    user_id       bigint PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    city          text,
    latitude      double precision,
    longitude     double precision,
    dating_intent text,
    occupation    text,
    education      text,
    updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS verification_records (
    id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id          bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    storage_key      text,
    mime_type        text NOT NULL,
    status           text NOT NULL DEFAULT 'pending',
    reason_code      text,
    model_name       text,
    model_version    text,
    liveness_score   double precision,
    face_match_score double precision,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_verif_user_created ON verification_records(user_id, created_at);

CREATE TABLE IF NOT EXISTS likes (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    from_user_id bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    to_user_id   bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    action       text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (from_user_id, to_user_id)
);
CREATE INDEX IF NOT EXISTS idx_likes_to ON likes(to_user_id);

CREATE TABLE IF NOT EXISTS matches (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_a_id    bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    user_b_id    bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status       text NOT NULL DEFAULT 'active',
    created_at   timestamptz NOT NULL DEFAULT now(),
    unmatched_at timestamptz,
    UNIQUE (user_a_id, user_b_id)
);

CREATE TABLE IF NOT EXISTS blocks (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id         bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    blocked_user_id bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (user_id, blocked_user_id)
);
CREATE INDEX IF NOT EXISTS idx_blocks_blocked ON blocks(blocked_user_id);

CREATE TABLE IF NOT EXISTS conversations (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    match_id   bigint NOT NULL UNIQUE REFERENCES matches(id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversation_members (
    conversation_id bigint NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id         bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    joined_at       timestamptz NOT NULL DEFAULT now(),
    last_read_at    timestamptz,
    PRIMARY KEY (conversation_id, user_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    conversation_id bigint NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sender_id       bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content         text NOT NULL,
    client_id       uuid NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (sender_id, client_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_conv_created ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS notifications (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id    bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type       text NOT NULL,
    payload    jsonb NOT NULL,
    read_at    timestamptz,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_notifications_user_created ON notifications(user_id, created_at);

INSERT INTO preferences (user_id)
SELECT id FROM users
ON CONFLICT (user_id) DO NOTHING;
