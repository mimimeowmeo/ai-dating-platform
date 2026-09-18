# PostgreSQL ERD — MVP Direction

## Core relationship map

```text
users
  |
  +-- profiles
  +-- preferences
  +-- user_photos
  +-- user_traits -- traits
  +-- user_interests -- interests
  +-- user_hobbies -- hobbies
  +-- user_foods -- foods
  +-- verification_records
  +-- face_embeddings
  +-- user_feature_vectors
  +-- user_clusters
  +-- likes
  +-- matches
  +-- blocks
  +-- conversations
  +-- messages
  +-- conversation_features
  +-- recommendation_events
  +-- notifications
```

## Recommended tables

### users

- id UUID PK
- email unique
- password_hash
- status
- is_verified
- created_at
- updated_at
- deleted_at

### profiles

- user_id UUID PK/FK
- display_name
- birth_date
- gender
- bio
- height_cm nullable
- occupation nullable
- education nullable
- city
- latitude nullable
- longitude nullable
- dating_intent
- created_at
- updated_at

### preferences

- user_id PK/FK
- min_age
- max_age
- preferred_gender
- max_distance_km
- min_height_cm（預設 130）
- max_height_cm（預設 250）
- preferred_dating_intent
- timestamps

### traits

- id SERIAL PK
- category（personality／diet／value／lifestyle／interest／dating_goal）
- code
- label_zh
- unique (category, code)

### user_traits

- user_id FK + trait_id FK 複合 PK
- index (trait_id)
- dating_goal 類別即「想遇見的關係」，其餘五類是「我的小熱愛」
- profiles.dating_intent 與 preferences.preferred_dating_intent 為舊欄位，
  preferred_dating_intent 現在存 `any` 或 dating_goal 的 code

### user_photos

- id UUID PK
- user_id FK
- storage_key
- photo_type
- is_avatar
- display_order
- moderation_status
- timestamps
- deleted_at

### verification_records

- id UUID PK
- user_id FK
- photo_id FK
- status
- liveness_score nullable
- face_match_score nullable
- model_name
- model_version
- reason_code nullable
- created_at

### face_embeddings

- id UUID PK
- user_id FK
- verification_id FK
- embedding VECTOR(N)
- model_name
- model_version
- created_at

### image_embeddings

- id UUID PK
- photo_id FK
- embedding VECTOR(N)
- model_name
- model_version
- created_at

### user_feature_vectors

- user_id PK/FK
- interest_embedding VECTOR(N)
- hobby_embedding VECTOR(N)
- food_embedding VECTOR(N)
- image_embedding VECTOR(N)
- conversation_embedding VECTOR(N)
- behavior_embedding VECTOR(N)
- combined_embedding VECTOR(N)
- feature_version
- model_version
- updated_at

### user_clusters

- id UUID PK
- user_id FK
- cluster_id
- cluster_probability
- cluster_version
- created_at
- expires_at nullable

### likes

- id UUID PK
- from_user_id FK
- to_user_id FK
- action
- created_at

### matches

- id UUID PK
- user_a_id FK
- user_b_id FK
- status
- created_at
- unmatched_at nullable

### conversations

- id UUID PK
- match_id FK
- created_at
- updated_at

### conversation_members

- conversation_id FK
- user_id FK
- joined_at
- last_read_at nullable
- composite PK

### messages

- id UUID PK
- conversation_id FK
- sender_id FK
- content
- message_type
- created_at
- deleted_at nullable

### conversation_features

- id UUID PK
- conversation_id FK
- user_id FK
- feature_type
- feature_key
- feature_value
- confidence
- model_name
- model_version
- created_at

### recommendation_events

- id UUID PK
- user_id FK
- candidate_user_id FK
- position
- score
- algorithm_version
- feature_version
- event_type
- created_at

### blocks

- id UUID PK
- user_id FK
- blocked_user_id FK
- created_at

### notifications

- id UUID PK
- user_id FK
- type
- payload JSONB
- read_at nullable
- created_at
