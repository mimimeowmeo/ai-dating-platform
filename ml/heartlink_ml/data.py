"""載入 seed 資料。

只讀 CSV、不連資料庫，讓實驗可以離線重現。seed CSV 與 dating 庫的 user_traits
是同一批資料（155,655 vs 155,658 筆，差的 3 筆是測試帳號）。
"""
import pandas as pd

from .config import REFERENCE_DATE, SEED_CSV, TRAIT_CATEGORIES

# 帳密不該流進任何模型、圖表或輸出檔。
SENSITIVE_COLUMNS = ("password", "email")
TRUTHY = ("1", "True", "true", "TRUE", "Y")


def load_seed(path=SEED_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.drop(columns=[c for c in SENSITIVE_COLUMNS if c in df.columns])
    birth = pd.to_datetime(df["birth_date"], errors="coerce")
    extra = pd.DataFrame({
        "birth_date": birth,
        "age": ((pd.Timestamp(REFERENCE_DATE) - birth).dt.days // 365.25).astype("Int64"),
        "bio": df["bio"].fillna("").astype(str),
    }, index=df.index)
    # 用 concat 一次合併再 copy()，避免 pandas 對寬表逐欄插入的碎片化警告。
    return pd.concat([df.drop(columns=["birth_date", "bio"]), extra], axis=1).copy()


def column_category(col: str) -> str:
    for cat in TRAIT_CATEGORIES:
        if col.startswith(f"{cat}_"):
            return cat
    return "face" if col.startswith("face_") else "other"


def trait_columns(df: pd.DataFrame, categories=TRAIT_CATEGORIES) -> list[str]:
    return [c for c in df.columns if column_category(c) in categories]


def face_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("face_")]
