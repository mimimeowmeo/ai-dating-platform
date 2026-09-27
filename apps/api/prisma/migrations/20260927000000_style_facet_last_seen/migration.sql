-- 風格卡特徵句最後一次出現的時間（2026-09-27）：開場、追問、重啟找話題時「越近越優先」。
-- 舊的特徵句沒有這個值（NULL），排在有時間的後面；下次重新萃取風格卡時就會補上。
ALTER TABLE "user_style_facets" ADD COLUMN "last_seen_at" TIMESTAMP(3);
