from __future__ import annotations

import argparse

from scripts.behavior._common import add_common_arguments, load_runtime_config
from src.behavior.app.check_format import inspect_app_format
from src.common import write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Check THU app behavioral workbook formats.")
    add_common_arguments(parser, include_limit=True)
    args = parser.parse_args()
    app_data_dir, output_dir, config = load_runtime_config(args)
    qc = inspect_app_format(args.dataset, app_data_dir, config, args.limit)
    write_csv(qc, output_dir / "app_format_qc.csv")
    print(f"Wrote {len(qc)} sheet QC rows to {output_dir / 'app_format_qc.csv'}")


if __name__ == "__main__":
    main()
