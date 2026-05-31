from __future__ import annotations

import argparse

from scripts.behavior._common import add_common_arguments, load_runtime_config
from src.behavior.app.clean import clean_app_trials
from src.common import write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean THU app behavioral trials.")
    add_common_arguments(parser, include_limit=True)
    args = parser.parse_args()
    app_data_dir, output_dir, config = load_runtime_config(args)
    trials, task_qc = clean_app_trials(args.dataset, app_data_dir, config, args.limit)
    write_csv(trials, output_dir / "app_trials_clean.csv")
    write_csv(task_qc, output_dir / "app_task_qc.csv")
    print(f"Wrote {len(trials)} cleaned trial rows to {output_dir / 'app_trials_clean.csv'}")
    print(f"Wrote {len(task_qc)} task QC rows to {output_dir / 'app_task_qc.csv'}")


if __name__ == "__main__":
    main()
