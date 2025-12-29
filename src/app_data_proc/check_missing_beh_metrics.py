import argparse
from pathlib import Path

import pandas as pd

from src.config_io import load_simple_yaml
from src.path_config import load_paths_config, resolve_dataset_roots
from src.metric_compute.efny.main import DEFAULT_TASK_CONFIG, normalize_task_name


def build_report(metrics_csv, data_dir, output_csv=None):
    metrics_csv = Path(metrics_csv)
    data_dir = Path(data_dir)

    df = pd.read_csv(metrics_csv)
    row_map = {str(r.file_name): r for r in df.itertuples(index=False)}

    rows = []
    missing_files = []
    for fp in sorted(data_dir.glob('*.xlsx')):
        row = row_map.get(fp.name)
        if row is None:
            missing_files.append(fp.name)
            continue

        xl = pd.ExcelFile(fp)
        for sheet in xl.sheet_names:
            task = normalize_task_name(sheet)
            cfg = DEFAULT_TASK_CONFIG.get(task)
            if cfg is None:
                continue

            metrics = cfg.get('metrics', [])
            if not metrics:
                continue

            missing = []
            for metric in metrics:
                col = f'{task}_{metric}'
                if not hasattr(row, col) or pd.isna(getattr(row, col)):
                    missing.append(metric)

            if missing:
                rows.append(
                    {
                        'subject_code': row.subject_code,
                        'file_name': fp.name,
                        'sheet_name': sheet,
                        'task_name': task,
                        'missing_metrics': ';'.join(missing),
                        'missing_count': len(missing),
                        'total_metrics': len(metrics),
                    }
                )

    report = pd.DataFrame(rows)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_csv, index=False, encoding='utf-8')

    return report, missing_files


def _resolve_defaults(args):
    if not args.dataset:
        raise ValueError("Missing --dataset (required when defaults are used).")
    repo_root = Path(__file__).resolve().parents[2]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg_path = (
        Path(args.dataset_config)
        if args.dataset_config is not None
        else (repo_root / "configs" / "datasets" / f"{args.dataset}.yaml")
    )
    dataset_cfg = load_simple_yaml(dataset_cfg_path)
    files_cfg = dataset_cfg.get("files", {})
    behavioral_cfg = dataset_cfg.get("behavioral", {})

    metrics_rel = files_cfg.get("behavioral_metrics_file")
    if not metrics_rel:
        raise ValueError("Missing files.behavioral_metrics_file in dataset config.")
    app_data_rel = behavioral_cfg.get("app_data_dir")
    if not app_data_rel:
        raise ValueError("Missing behavioral.app_data_dir in dataset config.")

    metrics_csv = roots["processed_root"] / metrics_rel
    data_dir = roots["raw_root"] / app_data_rel
    output_csv = (
        roots["outputs_root"]
        / "results"
        / "behavior_data"
        / "behavior_metrics_missing_report.csv"
    )
    return metrics_csv, data_dir, output_csv


def main():
    parser = argparse.ArgumentParser(
        description='Find subjects with sheets present but missing behavioral metrics.'
    )
    parser.add_argument(
        '--metrics_csv',
        default=None,
        help='Metrics CSV produced by the EFNY behavioral pipeline.',
    )
    parser.add_argument(
        '--data_dir',
        default=None,
        help='Directory containing per-subject GameData Excel files.',
    )
    parser.add_argument(
        '--output_csv',
        default=None,
        help='Where to save the missing-metrics report CSV.',
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    parser.add_argument("--dataset-config", dest="dataset_config", type=str, default=None)
    args = parser.parse_args()

    if args.metrics_csv is None or args.data_dir is None or args.output_csv is None:
        metrics_csv, data_dir, output_csv = _resolve_defaults(args)
        args.metrics_csv = str(metrics_csv)
        args.data_dir = str(data_dir)
        args.output_csv = str(output_csv)

    report, missing_files = build_report(
        metrics_csv=args.metrics_csv,
        data_dir=args.data_dir,
        output_csv=args.output_csv,
    )

    if len(missing_files) > 0:
        print(f'WARNING: {len(missing_files)} files missing from metrics CSV.')

    if report.empty:
        print('No missing metrics detected for existing sheets.')
        return

    print(f'Missing metrics rows: {len(report)}')
    print(report.groupby("task_name")["missing_count"].count().sort_values(ascending=False))


if __name__ == '__main__':
    main()
