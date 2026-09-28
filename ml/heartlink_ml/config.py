"""全域設定：路徑、隨機種子、標籤類別權重。"""
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]            # ml/
PROJECT_ROOT = ML_DIR.parent                             # 專案根目錄
SEED_CSV = PROJECT_ROOT / "db" / "seed" / "heartlink_with_email.csv"
OUTPUT_DIR = ML_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIG_DIR = OUTPUT_DIR / "figures"
SEED = 42
# 固定的「今天」，讓年齡計算可重現；不要用 datetime.now()。
REFERENCE_DATE = "2026-09-20"

TRAIT_CATEGORIES = ("dating_goal", "interest", "personality", "diet", "lifestyle", "value")
# 示範用的類別權重（沿用先前討論的設定）。這不是驗證過的門檻，之後用真實互動資料校準。
CATEGORY_WEIGHTS = {
    "value": 3.0, "interest": 2.0, "lifestyle": 1.5,
    "diet": 1.0, "personality": 1.0, "dating_goal": 1.0,
}


def ensure_dirs() -> None:
    for d in (OUTPUT_DIR, MODEL_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
