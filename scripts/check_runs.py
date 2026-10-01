# Verify run completeness: for every dataset dir, which arms have metrics.json + best.pt
import json
from pathlib import Path

ROOT = Path("runs")


def check_run_dir(run_dir: Path) -> str:
    has_metrics = (run_dir / "metrics.json").exists()
    has_best = (run_dir / "checkpoints" / "best.pt").exists()
    best_val = None
    if has_metrics:
        try:
            data = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            best_val = round(data.get("ade", float("nan")), 2)
        except Exception:
            best_val = "?"
    return f"{run_dir.name}: metrics={has_metrics}, best.pt={has_best}, ADE={best_val}"


for dataset in sorted(ROOT.iterdir()):
    if not dataset.is_dir():
        continue
    print(f"\n=== {dataset.name} ===")
    run_dirs = sorted(d for d in dataset.iterdir() if d.is_dir() and d.name[0].isdigit())
    if not run_dirs:
        files = [f.name for f in dataset.iterdir()]
        print(f"  (no run dirs; files: {files})")
    for rd in run_dirs:
        print(" ", check_run_dir(rd))
