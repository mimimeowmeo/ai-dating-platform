# API Catalog — MVP v1

Base path: `/api/v1`

## Authentication

- `POST /auth/register`
- `POST /auth/login`
- `POST /auth/logout`
- `POST /auth/refresh`
- `GET /auth/me`

## Profile

- `GET /profile`
- `PUT /profile`
- `GET /profile/:userId`
- `POST /profile/photos`
- `DELETE /profile/photos/:photoId`

## Preferences

- `GET /preferences`
- `PUT /preferences`

## Interests / Hobbies / Foods

- `GET /interests`
- `PUT /me/interests`
- `GET /hobbies`
- `PUT /me/hobbies`
- `GET /foods`
- `PUT /me/foods`

## Verification

- `POST /onboarding/selfie`
- `GET /verification/status`
- `POST /verification/retry`

## Discovery / Interactions

- `GET /discovery`
- `POST /interactions`
- `GET /matches`
- `GET /matches/:matchId`
- `DELETE /matches/:matchId`
- `POST /blocks`
- `DELETE /blocks/:blockedUserId`

## Conversations / Chat

- `GET /conversations`
- `POST /conversations`
- `GET /conversations/:conversationId`
- `GET /conversations/:conversationId/messages`

Socket events should handle realtime delivery, typing, read receipts, and presence.

## Internal AI Endpoints

These are internal service calls and are not browser-facing.

- `POST /internal/ai/face/verify`
- `POST /internal/ai/face/liveness`
- `POST /internal/ai/image/embed`
- `POST /internal/ai/conversation/analyze`
- `POST /internal/ai/users/:userId/recompute-features`
- `POST /internal/ai/clustering/recompute`
- `POST /internal/ai/recommendations/generate`
