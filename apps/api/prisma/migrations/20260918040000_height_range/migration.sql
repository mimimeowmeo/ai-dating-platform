-- 探索偏好新增身高範圍（公分）；130–250 是畫面拉桿的兩端，等同不限。
ALTER TABLE "preferences" ADD COLUMN "min_height_cm" INTEGER NOT NULL DEFAULT 130;
ALTER TABLE "preferences" ADD COLUMN "max_height_cm" INTEGER NOT NULL DEFAULT 250;
