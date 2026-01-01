import argparse
from pathlib import Path

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots

def _resolve_defaults(args) -> tuple[Path, Path]:
    if not args.dataset:
        raise ValueError("Missing --dataset (required when defaults are used).")
    repo_root = Path(__file__).resolve().parents[2]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(
        paths_cfg,
        dataset_config_path=args.dataset_config,
        repo_root=repo_root,
    )
    behavioral_cfg = dataset_cfg.get("behavioral", {})
    files_cfg = dataset_cfg.get("files", {})

    app_data_rel = behavioral_cfg.get("app_data_dir")
    metrics_rel = files_cfg.get("behavioral_metrics_file")
    if not app_data_rel:
        raise ValueError("Missing behavioral.app_data_dir in dataset config.")
    if not metrics_rel:
        raise ValueError("Missing files.behavioral_metrics_file in dataset config.")

    data_dir = roots["raw_root"] / app_data_rel
    out_csv = roots["processed_root"] / metrics_rel
    return data_dir, out_csv


def main():
    from src.behavioral_preprocess.metrics.efny.main import run_raw

    parser = argparse.ArgumentParser(description="Compute EFNY behavioral metrics.")
    parser.add_argument("--data-dir", default=None, help="Input app-data directory.")
    parser.add_argument("--out-csv", default=None, help="Output metrics CSV path.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    parser.add_argument("--dataset-config", dest="dataset_config", type=str, default=None)
    parser.add_argument(
        "--metrics-config",
        dest="metrics_config",
        type=str,
        default="configs/behavioral_metrics.yaml",
        help="Behavioral metrics config with compute_tasks/use_metrics.",
    )
    args = parser.parse_args()

    if args.data_dir is None or args.out_csv is None:
        data_dir, out_csv = _resolve_defaults(args)
        args.data_dir = str(data_dir)
        args.out_csv = str(out_csv)

    run_raw(data_dir=args.data_dir, out_csv=args.out_csv, metrics_config_path=args.metrics_config)

if __name__ == '__main__':
    main()
