from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.behavioral_preprocess.app_data.app_sequence_correction import (
    build_effective_visit_sequences,
    correct_workbook,
    extract_observed_task_data,
    infer_subject_visit,
    load_all_visits_sequences,
    parse_subject_id_from_filename,
)
from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


def _resolve_dirs(dataset_cfg: dict, roots: dict[str, Path]) -> tuple[Path, Path, Path]:
    behavioral_cfg = dataset_cfg.get("behavioral", {})
    app_data_rel = behavioral_cfg.get("app_data_dir")
    seq_rel = behavioral_cfg.get("app_sequence_dir")
    out_rel = behavioral_cfg.get("corrected_app_excel_dir")
    if not app_data_rel:
        raise ValueError("Missing dataset.behavioral.app_data_dir in configs/paths.yaml.")
    if not seq_rel:
        raise ValueError("Missing dataset.behavioral.app_sequence_dir in configs/paths.yaml.")
    if not out_rel:
        raise ValueError("Missing dataset.behavioral.corrected_app_excel_dir in configs/paths.yaml.")
    input_dir = roots["raw_root"] / str(app_data_rel)
    seq_dir = roots["raw_root"] / str(seq_rel)
    out_base = roots["processed_root"] / str(out_rel)
    return input_dir, seq_dir, out_base


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Infer and correct app workbook stimulus+answer sequences using visit configs under app_sequence."
    )
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    ap.add_argument("--dataset-config", dest="dataset_config", type=str, default=None)
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Output subfolder name under corrected_app_excel_dir. Default: timestamp.",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional: process only first N workbooks (sorted) for a quick run.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing run directory (otherwise errors).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(paths_cfg, dataset_config_path=args.dataset_config, repo_root=repo_root)

    input_dir, seq_dir, out_base = _resolve_dirs(dataset_cfg, roots)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence dir not found: {seq_dir}")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base / f"run_{run_id}"
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output dir already exists: {out_dir} (use --overwrite or change --run-id)")
    out_dir.mkdir(parents=True, exist_ok=True)

    visits_raw = load_all_visits_sequences(seq_dir)
    visits = build_effective_visit_sequences(visits_raw, baseline_visit="visit1")
    (out_dir / "sequence_library").mkdir(parents=True, exist_ok=True)
    (out_dir / "sequence_library" / "effective_visits_sequences.json").write_text(
        json.dumps(
            {
                visit: {
                    task: {"items": seq.items_raw, "answers": seq.answers_raw, "source": seq.source}
                    for task, seq in sorted(seqs.items())
                }
                for visit, seqs in sorted(visits.items())
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "sequence_library" / "effective_visits_sources.json").write_text(
        json.dumps(
            {
                visit: {task: seq.source for task, seq in sorted(seqs.items())}
                for visit, seqs in sorted(visits.items())
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    excel_files = sorted([p for p in input_dir.glob("*.xlsx") if not p.name.startswith("~$")])
    if args.max_files is not None:
        excel_files = excel_files[: args.max_files]

    manifest_rows: list[dict[str, object]] = []
    per_subject_dir = out_dir / "subjects"
    per_subject_dir.mkdir(parents=True, exist_ok=True)

    for idx, fp in enumerate(excel_files, start=1):
        subject_id = parse_subject_id_from_filename(fp)
        print(f"[{idx}/{len(excel_files)}] {subject_id}")
        observed = extract_observed_task_data(fp)
        inf = infer_subject_visit(fp, subject_id, observed, visits)

        out_excel = per_subject_dir / fp.name
        result = correct_workbook(
            excel_path=fp,
            output_path=out_excel,
            inferred_visit=inf.inferred_visit,
            visits=visits,
        )

        task_rows = []
        for td in result.task_decisions:
            task_rows.append(
                {
                    "task": td.task,
                    "visit": td.visit,
                    "item_match": td.item_match,
                    "answer_match": td.answer_match,
                    "issue_type": td.issue_type,
                }
            )
        (out_dir / "decisions").mkdir(parents=True, exist_ok=True)
        (out_dir / "decisions" / f"{subject_id}.json").write_text(
            json.dumps(
                {
                    "subject_id": subject_id,
                    "inferred_visit": inf.inferred_visit,
                    "expected_visit": inf.expected_visit,
                    "score": inf.score,
                    "scores_by_visit": inf.scores_by_visit,
                    "tasks": task_rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        manifest_rows.append(
            {
                "subject_id": subject_id,
                "file_name": fp.name,
                "inferred_visit": inf.inferred_visit,
                "expected_visit": inf.expected_visit,
                "score": inf.score,
                "out_excel": str(out_excel.relative_to(out_dir)),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(out_dir / "manifest.csv", index=False, encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote corrected workbooks: {per_subject_dir}")
    print(f"Wrote manifest: {out_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
