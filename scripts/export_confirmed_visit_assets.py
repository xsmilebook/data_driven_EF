from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


ITEM_COL = "正式阶段刺激图片/Item名"
ANSWER_COL = "正式阶段正确答案"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Rename run_corrected_v2 sequence directory (conformed -> confirmed), "
            "export per-version task CSVs, and build subject_id-referred_visit table."
        )
    )
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    ap.add_argument("--dataset-config", dest="dataset_config", type=str, default=None)
    ap.add_argument(
        "--run-name",
        type=str,
        default="run_corrected_v2",
        help="Run folder name under corrected_app_excel_dir.",
    )
    ap.add_argument(
        "--source-dir-name",
        type=str,
        default="conformed",
        help="Old directory name to be renamed.",
    )
    ap.add_argument(
        "--target-dir-name",
        type=str,
        default="confirmed",
        help="Target directory name after rename.",
    )
    ap.add_argument(
        "--answer-groups-dir",
        type=str,
        default="behavioral_preprocess/groups_by_answer_regen_20260213",
        help=(
            "Answer grouping directory (absolute or repo-relative or interim-root-relative). "
            "Used to build subject_id-referred_visit mapping."
        ),
    )
    return ap.parse_args()


def _resolve_answer_groups_dir(arg_value: str, *, repo_root: Path, interim_root: Path) -> Path:
    p = Path(arg_value)
    if p.is_absolute():
        return p
    repo_rel = repo_root / p
    if repo_rel.exists():
        return repo_rel
    return interim_root / p


def _pad_list(values: list[object], n: int) -> list[object]:
    padded = list(values)
    if len(padded) < n:
        padded.extend([""] * (n - len(padded)))
    return padded


def _normalize_manifest(manifest_path: Path) -> None:
    if not manifest_path.exists():
        return
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = str(data.get("schema", ""))
    if "conformed" in schema:
        data["schema"] = schema.replace("conformed", "confirmed")
    desc = str(data.get("description", ""))
    if "conformed" in desc:
        data["description"] = desc.replace("conformed", "confirmed")
    files = data.get("files", [])
    if isinstance(files, list):
        for entry in files:
            if not isinstance(entry, dict):
                continue
            if "conformed_file" in entry:
                entry["confirmed_file"] = entry.pop("conformed_file")
            for key in ("source_path",):
                if key in entry and isinstance(entry[key], str):
                    entry[key] = entry[key].replace("/conformed", "/confirmed")
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(paths_cfg, dataset_config_path=args.dataset_config, repo_root=repo_root)

    behavioral_cfg = dataset_cfg.get("behavioral", {})
    corrected_rel = behavioral_cfg.get("corrected_app_excel_dir")
    if not corrected_rel:
        raise ValueError("Missing dataset.behavioral.corrected_app_excel_dir in configs/paths.yaml.")

    run_dir = roots["processed_root"] / str(corrected_rel) / args.run_name
    source_dir = run_dir / args.source_dir_name
    target_dir = run_dir / args.target_dir_name

    if source_dir.exists() and target_dir.exists():
        raise FileExistsError(f"Both source and target dirs exist: {source_dir} and {target_dir}")
    if source_dir.exists():
        source_dir.rename(target_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"Confirmed directory not found: {target_dir}")

    _normalize_manifest(target_dir / "manifest.json")

    version_files = sorted(target_dir.glob("*_confirmed.json"))
    if not version_files:
        raise FileNotFoundError(f"No *_confirmed.json found in {target_dir}")

    # Export per-version 18 task CSVs (2 columns only).
    version_export_rows: list[dict[str, object]] = []
    for vf in version_files:
        version_name = vf.stem
        version_data = json.loads(vf.read_text(encoding="utf-8"))
        if not isinstance(version_data, dict):
            raise ValueError(f"Unexpected JSON structure for {vf}")
        out_dir = target_dir / version_name
        out_dir.mkdir(parents=True, exist_ok=True)
        n_tasks = 0
        for task_name, task_payload in version_data.items():
            if not isinstance(task_payload, dict):
                continue
            items = task_payload.get("items", [])
            answers = task_payload.get("answers", [])
            if not isinstance(items, list):
                items = []
            if not isinstance(answers, list):
                answers = []
            n_rows = max(len(items), len(answers))
            items = _pad_list(items, n_rows)
            answers = _pad_list(answers, n_rows)
            df = pd.DataFrame(
                {
                    ITEM_COL: ["" if x is None else x for x in items],
                    ANSWER_COL: ["" if x is None else x for x in answers],
                }
            )
            df.to_csv(out_dir / f"{task_name}.csv", index=False, encoding="utf-8")
            n_tasks += 1
        version_export_rows.append({"version": version_name, "n_tasks": n_tasks})

    pd.DataFrame(version_export_rows).sort_values("version").to_csv(
        target_dir / "version_task_csv_export_manifest.csv",
        index=False,
        encoding="utf-8",
    )

    # Build subject_id -> referred_visit mapping from current answer groups.
    answer_groups_dir = _resolve_answer_groups_dir(
        args.answer_groups_dir, repo_root=repo_root, interim_root=roots["interim_root"]
    )
    if not answer_groups_dir.exists():
        raise FileNotFoundError(f"Answer groups dir not found: {answer_groups_dir}")

    group_to_visit = {
        "group_001": "visit1_confirmed",
        "group_002": "visit3_confirmed",
        "group_003": "visit3_v1_confirmed",
        "group_004": "visit2_confirmed",
        "group_005": "visit3_v2_confirmed",
        "group_006": "visit1_v1_confirmed",
    }

    found_groups = sorted(p.stem.replace("_sublist", "") for p in answer_groups_dir.glob("group_*_sublist.txt"))
    if sorted(group_to_visit) != found_groups:
        raise ValueError(
            f"Unexpected answer group set. expected={sorted(group_to_visit)} found={found_groups}"
        )

    available_versions = {p.stem for p in version_files}
    for v in group_to_visit.values():
        if v not in available_versions:
            raise ValueError(f"Mapped version does not exist under confirmed dir: {v}")

    rows: list[dict[str, str]] = []
    for gid in sorted(group_to_visit):
        visit = group_to_visit[gid]
        sub_file = answer_groups_dir / f"{gid}_sublist.txt"
        subs = [x.strip() for x in sub_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        for s in subs:
            rows.append({"subject_id": s, "referred_visit": visit})

    df_map = pd.DataFrame(rows).sort_values("subject_id")
    df_map.to_csv(target_dir / "subject_referred_visit.csv", index=False, encoding="utf-8")

    mapping_meta = {
        "schema": "subject_referred_visit_mapping_v1",
        "source_answer_groups_dir": str(answer_groups_dir).replace("\\", "/"),
        "group_to_visit": group_to_visit,
        "n_subjects": int(df_map.shape[0]),
    }
    (target_dir / "subject_referred_visit_meta.json").write_text(
        json.dumps(mapping_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"run_dir: {run_dir}")
    print(f"confirmed_dir: {target_dir}")
    print(f"versions: {len(version_files)}")
    print(f"subject_referred_visit_rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
