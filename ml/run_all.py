"""一鍵執行所有實驗並彙整結果。

用法（在 ml/ 目錄）：
    .venv/bin/python run_all.py                       # 全部跑
    .venv/bin/python run_all.py --only clustering_ensemble,supervised_bio_to_tags
    .venv/bin/python run_all.py --skip supervised_reciprocal_ranker

每支實驗獨立執行（subprocess），stdout/stderr 存到 outputs/<name>/run.log，
所有 outputs/<name>/metrics.json 彙整成 outputs/summary.json。
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
OUT = ML_DIR / "outputs"

# (腳本名, 說明)。順序 = 建議閱讀順序：先非監督、再 ensemble、最後監督式。
EXPERIMENTS = [
    # 先做資料體檢：資料沒有結構，後面所有模型學到的都只會是生成規則。
    ("data_realism_audit", "資料體檢：這份 AI 生成的 seed 像不像真的"),
    ("clustering_partitional", "分割式分群：K-means / MiniBatch / K-modes / GMM"),
    ("clustering_hierarchical", "階層式分群 HAC（single / average / complete）"),
    ("clustering_density", "密度式分群 DBSCAN / HDBSCAN"),
    ("clustering_ensemble", "共識分群 cluster ensemble（證據累積）"),
    ("supervised_bio_to_tags", "監督式：bio 文字 → 68 個標籤（multi-label）"),
    ("supervised_reciprocal_ranker", "監督式排序 + ensemble + 互惠融合（模擬研究）"),
    # 以下三支：把 bio 文字加入推薦。後兩支預設 dry-run（不下載模型、不呼叫 API）。
    ("recommend_bio_text", "bio 文字推薦：關鍵字 / bio→標籤 / 主題 / 風格 / 混合"),
    ("recommend_bio_embeddings", "bio 句子嵌入 + pgvector 匯出（預設 dry-run，不下載模型）"),
    ("recommend_bio_llm_extract", "Claude 結構化抽取 bio 特徵（預設 dry-run，零 API 呼叫）"),
]


def run_one(name: str) -> tuple[int, float]:
    script = ML_DIR / "experiments" / f"{name}.py"
    if not script.exists():
        print(f"  ✗ 找不到 {script.relative_to(ML_DIR)}")
        return 127, 0.0
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(script)], cwd=ML_DIR, capture_output=True, text=True)
    elapsed = time.time() - t0
    log_dir = OUT / name
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "run.log").write_text(
        proc.stdout + "\n\n--- stderr ---\n" + proc.stderr, encoding="utf-8"
    )
    return proc.returncode, elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="逗號分隔，只跑這些實驗")
    ap.add_argument("--skip", default="", help="逗號分隔，跳過這些實驗")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    # 用 --only / --skip 只跑一部分時，保留先前其他實驗的彙整結果。
    summary_path = OUT / "summary.json"
    summary: dict[str, dict] = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    ran: list[str] = []
    print(f"Python: {sys.executable}\n")
    for name, desc in EXPERIMENTS:
        if (only and name not in only) or name in skip:
            continue
        print(f"▶ {name} —— {desc}")
        ran.append(name)
        code, elapsed = run_one(name)
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"  {status}，{elapsed:.1f}s，log: outputs/{name}/run.log")
        metrics_path = OUT / name / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
        summary[name] = {"description": desc, "exit_code": code, "seconds": round(elapsed, 1), "metrics": metrics}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n| 實驗 | 狀態 | 秒數 | metrics.json |")
    print("|---|---|---|---|")
    for name in ran:
        s = summary[name]
        print(f"| {name} | {'✅' if s['exit_code'] == 0 else '❌'} | {s['seconds']} | {'有' if s['metrics'] else '無'} |")
    print("\n彙整檔：outputs/summary.json")
    return 0 if all(summary[n]["exit_code"] == 0 for n in ran) else 1


if __name__ == "__main__":
    sys.exit(main())
