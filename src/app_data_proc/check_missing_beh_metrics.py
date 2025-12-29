import argparse
from pathlib import Path

import pandas as pd

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


def main():
    parser = argparse.ArgumentParser(
        description='Find subjects with sheets present but missing behavioral metrics.'
    )
    parser.add_argument(
        '--metrics_csv',
        default='data/processed/EFNY/table/metrics/EFNY_beh_metrics.csv',
        help='Metrics CSV produced by the EFNY behavioral pipeline.',
    )
    parser.add_argument(
        '--data_dir',
        default='data/raw/EFNY/behavior_data/cibr_app_data',
        help='Directory containing per-subject GameData Excel files.',
    )
    parser.add_argument(
        '--output_csv',
        default='outputs/EFNY/results/behavior_data/behavior_metrics_missing_report.csv',
        help='Where to save the missing-metrics report CSV.',
    )
    args = parser.parse_args()

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
