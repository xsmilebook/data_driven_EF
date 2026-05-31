from __future__ import annotations

import argparse

import pandas as pd

from scripts.behavior._common import add_common_arguments, load_runtime_config
from src.behavior.app.metrics import calculate_app_metrics
from src.common import write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate THU app behavioral metrics.")
    add_common_arguments(parser)
    args = parser.parse_args()
    _, output_dir, config = load_runtime_config(args)
    trials = pd.read_csv(output_dir / "app_trials_clean.csv")
    task_qc = pd.read_csv(output_dir / "app_task_qc.csv")
    metrics_long, metrics_wide, task_qc = calculate_app_metrics(trials, task_qc, config)
    write_csv(metrics_long, output_dir / "app_metrics_long.csv")
    write_csv(metrics_wide, output_dir / "app_metrics_wide.csv")
    write_csv(task_qc, output_dir / "app_task_qc.csv")
    print(f"Wrote {len(metrics_long)} metric rows to {output_dir / 'app_metrics_long.csv'}")
    print(f"Wrote {len(metrics_wide)} subjects to {output_dir / 'app_metrics_wide.csv'}")
    print(f"Updated {len(task_qc)} task QC rows in {output_dir / 'app_task_qc.csv'}")


if __name__ == "__main__":
    main()
