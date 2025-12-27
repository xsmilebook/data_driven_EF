import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.result_summary.visualize_loadings_similarity import run_for_result_dir


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, required=True)
    ap.add_argument("--atlas", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    base = results_root / "real" / args.atlas
    if not base.exists():
        raise FileNotFoundError(f"real atlas path not found: {base}")

    seed_dirs = sorted(base.glob("**/seed_*/result.json"))
    if not seed_dirs:
        raise RuntimeError(f"No result.json found under: {base}")

    total = 0
    failed = []
    for result_json in seed_dirs:
        result_dir = result_json.parent
        try:
            run_for_result_dir(result_dir)
            total += 1
        except Exception as exc:
            failed.append((result_dir, str(exc)))

    print(f"Completed: {total} runs")
    if failed:
        print("Failures:")
        for path, msg in failed:
            print(f"- {path}: {msg}")


if __name__ == "__main__":
    main()
