"""外貌：照片 → 512 維外貌向量（精確去背只留臉 + CLIP ViT-B/32）。

演算法與 ml/experiments/appearance_face_clip.py 的 `--mask seg-face` 相同（seg_cutout 規則第 3 版），
這樣 worker 算的新照片和 ml 匯入的向量才能互相比較。任何一邊的規則有改，兩邊要一起改，並更新 MODEL_VERSION。
「扣掉戴眼鏡方向」不在這裡做：方向存在資料庫（appearance_directions），由 NestJS 寫回向量時套用。
"""
