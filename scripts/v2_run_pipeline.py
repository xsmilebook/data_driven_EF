from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.models.v2_pipeline import (
    V2PipelineConfig,
    build_run_payload,
    profile_behavior_table,
    resolve_behavior_table,
    resolve_output_dir,
    write_v2_stub_outputs,
)
from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run the v2 minimal pipeline skeleton using cleaned behavioral data."
    )
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    ap.add_argument("--dataset-config", dest="dataset_config", type=str, default=None)
    ap.add_argument(
        "--behavior-table",
        type=str,
        default=None,
        help="Override cleaned behavior table path; default uses dataset.files.behavioral_metrics_file.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override output directory; default is outputs/results/v2/<run-id>/",
    )
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier used by default output directory.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Resolve config and print plan without writing files.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(
        paths_cfg,
        dataset_config_path=args.dataset_config,
        repo_root=repo_root,
    )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    behavior_table = resolve_behavior_table(
        behavior_table_arg=args.behavior_table,
        dataset_cfg=dataset_cfg,
        roots=roots,
    )
    out_dir = resolve_output_dir(output_dir_arg=args.out_dir, roots=roots, run_id=run_id)
    cfg = V2PipelineConfig(dataset=args.dataset, behavior_table=behavior_table, output_dir=out_dir)

    if args.dry_run:
        payload = build_run_payload(cfg, dry_run=True, profile=None)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not behavior_table.exists():
        raise FileNotFoundError(f"Cleaned behavior table not found: {behavior_table}")

    profile = profile_behavior_table(behavior_table, subject_candidates=cfg.subject_candidates)
    payload = build_run_payload(cfg, dry_run=False, profile=profile)
    write_v2_stub_outputs(cfg, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

