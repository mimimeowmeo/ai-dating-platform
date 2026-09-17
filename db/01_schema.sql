
CREATE EXTENSION IF NOT EXISTS pgcrypto;

DROP TABLE IF EXISTS user_traits, user_face_features, profiles, traits, users, staging_wide CASCADE;

CREATE TABLE users (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    account       text NOT NULL UNIQUE,
    password_hash text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE profiles (
    user_id    bigint PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    name       text,
    gender     text CHECK (gender IN ('Male','Female','Other')),
    birth_date date,
    height_cm  smallint CHECK (height_cm BETWEEN 100 AND 250),
    image_file text,
    bio        text,
    zodiac     text
);

CREATE TABLE traits (
    id         int GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    category   text NOT NULL,
    code       text NOT NULL,
    label_zh   text,
    csv_column text NOT NULL UNIQUE,
    UNIQUE (category, code)
);

CREATE TABLE user_traits (
    user_id  bigint NOT NULL REFERENCES users(id)  ON DELETE CASCADE,
    trait_id int    NOT NULL REFERENCES traits(id) ON DELETE RESTRICT,
    PRIMARY KEY (user_id, trait_id)
);
CREATE INDEX idx_user_traits_trait ON user_traits(trait_id);

CREATE TABLE user_face_features (
    user_id       bigint PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    attributes    jsonb NOT NULL,
    model_version text  NOT NULL DEFAULT 'celeba_v1'
);
CREATE INDEX idx_face_attrs ON user_face_features USING gin (attributes);
