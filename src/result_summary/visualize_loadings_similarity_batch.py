import argparse
from pathlib import Path

from src.result_summary.visualize_loadings_similarity import run_for_result_dir
from src.result_summary.cli_utils import resolve_results_root


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, default=None)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    ap.add_argument("--dataset-config", dest="dataset_config", type=str, default=None)
    ap.add_argument("--atlas", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    results_root = resolve_results_root(
        results_root=args.results_root,
        dataset=args.dataset,
        paths_config=args.paths_config,
        dataset_config=args.dataset_config,
    )
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
