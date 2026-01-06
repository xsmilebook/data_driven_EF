from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


DEFAULT_GROUP_COL = "正式阶段正确答案"


@dataclass(frozen=True)
class WorkbookStimProfile:
    subject_id: str
    excel_path: Path
    sheet_signatures: dict[str, str]
    sheet_count: int
    note: str | None = None


def _normalize_group_cell(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    s = s.replace("\\", "/")
    if "/" in s:
        s = s.split("/")[-1]
    s = s.strip().lower()
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"):
        if s.endswith(ext):
            s = s[: -len(ext)]
            break
    if s.isdigit():
        s2 = s.lstrip("0")
        s = s2 if s2 else "0"
    return s or None


def _normalize_header(value: object) -> str:
    s = str(value).strip()
    return "".join(s.split())


def _find_group_column(columns: list[object], *, target: str) -> str | None:
    target_norm = _normalize_header(target)
    candidates: list[str] = []
    for c in columns:
        c_str = str(c)
        c_norm = _normalize_header(c_str)
        if c_norm == target_norm:
            return c_str
        candidates.append(c_str)

    for c_str in candidates:
        c_norm = _normalize_header(c_str)
        if "正式阶段正确答案" in c_norm:
            return c_str
    return None


def _extract_subject_id_from_filename(excel_path: Path) -> str:
    stem = excel_path.stem
    stem = stem.replace("GameData", "").rstrip("_- ")
    parts = [p for p in stem.split("_") if p]
    if len(parts) >= 4:
        return "_".join(parts[:4])
    return stem


def _sheet_items(
    df: pd.DataFrame,
    *,
    group_column_name: str,
    compare_mode: str,
) -> list[str]:
    if "任务" in df.columns and len(df) == 97:
        # Known app export anomaly: SST sheet may contain 97 rows where the last row is invalid.
        # Do not include the last row in grouping comparisons.
        task_name = str(df["任务"].iloc[0]).strip()
        if task_name.upper() == "SST":
            df = df.iloc[:-1].copy()
    col = _find_group_column(list(df.columns), target=group_column_name)
    if not col:
        raise KeyError(f"missing grouping column {group_column_name!r}")
    raw = df[col].tolist()
    items = [v for v in (_normalize_group_cell(x) for x in raw) if v is not None]
    if compare_mode == "set":
        return sorted(set(items))
    return items


def _compute_signature(
    excel_path: Path,
    *,
    group_column_name: str,
    compare_mode: str,
) -> tuple[dict[str, str], int, str | None]:
    xl = pd.ExcelFile(excel_path)
    sheet_sigs: dict[str, str] = {}
    missing: list[str] = []
    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet, dtype="object")
            items = _sheet_items(
                df,
                group_column_name=group_column_name,
                compare_mode=compare_mode,
            )
            canonical = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
            sheet_sigs[str(sheet)] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        except Exception:
            missing.append(str(sheet))

    note = f"missing_group_column_in_sheets={','.join(missing)}" if missing else None
    return sheet_sigs, len(xl.sheet_names), note


@dataclass
class StimulusGroup:
    constraints: dict[str, str]
    members: list[WorkbookStimProfile]

    def can_accept(self, profile: WorkbookStimProfile, *, require_overlap: bool) -> bool:
        overlap = set(self.constraints).intersection(profile.sheet_signatures)
        if require_overlap and not overlap:
            return False
        for sheet in overlap:
            if self.constraints[sheet] != profile.sheet_signatures[sheet]:
                return False
        return True

    def add(self, profile: WorkbookStimProfile) -> None:
        for sheet, sig in profile.sheet_signatures.items():
            if sheet not in self.constraints:
                self.constraints[sheet] = sig
        self.members.append(profile)


def _resolve_app_data_dir(dataset_cfg: dict, roots: dict[str, Path]) -> Path:
    behavioral_cfg = dataset_cfg.get("behavioral", {})
    rel = behavioral_cfg.get("app_data_dir")
    if not rel:
        raise ValueError("Missing dataset.behavioral.app_data_dir in configs/paths.yaml.")
    return roots["raw_root"] / str(rel)


def _resolve_interim_behavior_dir(dataset_cfg: dict, roots: dict[str, Path]) -> Path:
    behavioral_cfg = dataset_cfg.get("behavioral", {})
    rel = behavioral_cfg.get("interim_preprocess_dir") or "behavioral_preprocess"
    return roots["interim_root"] / str(rel)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Group subjects by identical app-task sequences across all Excel sheets "
            f"(column: {DEFAULT_GROUP_COL})."
        )
    )
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    ap.add_argument("--dataset-config", dest="dataset_config", type=str, default=None)
    ap.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override input dir (absolute or repo-relative). Default: <raw_root>/<dataset.behavioral.app_data_dir>.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Override output dir (absolute or repo-relative). Default: "
            "<interim_root>/<dataset.behavioral.interim_preprocess_dir>/stimulus_groups."
        ),
    )
    ap.add_argument("--group-column", type=str, default=DEFAULT_GROUP_COL)
    ap.add_argument(
        "--compare-mode",
        choices=["sequence", "set"],
        default="sequence",
        help="Compare per-sheet items as full sequences (default) or as unique sets (order ignored).",
    )
    ap.add_argument(
        "--allow-empty-overlap",
        action="store_true",
        help=(
            "Allow grouping when two workbooks have zero common sheets. "
            "Not recommended (can merge unrelated subjects)."
        ),
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional: process only the first N workbooks (sorted) for a quick smoke test.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(paths_cfg, dataset_config_path=args.dataset_config, repo_root=repo_root)

    if args.input_dir:
        p = Path(args.input_dir)
        input_dir = p if p.is_absolute() else (repo_root / p)
    else:
        input_dir = _resolve_app_data_dir(dataset_cfg, roots)

    if args.out_dir:
        p = Path(args.out_dir)
        out_dir = p if p.is_absolute() else (repo_root / p)
    else:
        out_dir = _resolve_interim_behavior_dir(dataset_cfg, roots) / "stimulus_groups"

    excel_files = sorted([p for p in input_dir.glob("*.xlsx") if not p.name.startswith("~$")])
    if not excel_files:
        raise FileNotFoundError(f"No .xlsx files found under: {input_dir}")
    if args.max_files is not None:
        if args.max_files <= 0:
            raise ValueError("--max-files must be a positive integer.")
        excel_files = excel_files[: args.max_files]

    profiles: list[WorkbookStimProfile] = []
    for fp in excel_files:
        subject_id = _extract_subject_id_from_filename(fp)
        sheet_sigs, sheet_count, note = _compute_signature(
            fp,
            group_column_name=args.group_column,
            compare_mode=args.compare_mode,
        )
        profiles.append(
            WorkbookStimProfile(
                subject_id=subject_id,
                excel_path=fp,
                sheet_signatures=sheet_sigs,
                sheet_count=sheet_count,
                note=note,
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    groups: list[StimulusGroup] = []
    require_overlap = not args.allow_empty_overlap

    for profile in sorted(profiles, key=lambda p: (-len(p.sheet_signatures), p.subject_id)):
        placed = False
        for g in groups:
            if g.can_accept(profile, require_overlap=require_overlap):
                g.add(profile)
                placed = True
                break
        if not placed:
            g = StimulusGroup(constraints=dict(profile.sheet_signatures), members=[profile])
            groups.append(g)

    manifest_rows: list[dict[str, object]] = []
    sorted_groups = sorted(
        groups,
        key=lambda x: (-len(x.members), sorted(m.subject_id for m in x.members)[0]),
    )
    for idx, g in enumerate(sorted_groups, start=1):
        out_group_id = f"group_{idx:03d}"
        subject_ids = sorted(m.subject_id for m in g.members)
        (out_dir / f"{out_group_id}_sublist.txt").write_text("\n".join(subject_ids) + "\n", encoding="utf-8")
        manifest_rows.append(
            {
                "group_id": out_group_id,
                "n_subjects": len(subject_ids),
                "subjects": ";".join(subject_ids),
                "n_sheets_constraints": len(g.constraints),
                "example_subject": subject_ids[0] if subject_ids else "",
                "note": g.members[0].note if g.members else None,
            }
        )

    pd.DataFrame(manifest_rows).to_csv(out_dir / "groups_manifest.csv", index=False, encoding="utf-8")
    (out_dir / "groups_manifest.json").write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Input:  {input_dir}")
    print(f"Output: {out_dir}")
    print(f"Found {len(excel_files)} workbooks -> {len(groups)} groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
