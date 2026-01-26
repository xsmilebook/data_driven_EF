from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


@dataclass(frozen=True)
class GroupAudit:
    kind: str  # answer/item
    n_groups: int
    total_lines: int
    unique_subjects: int
    duplicate_lines: int
    duplicate_subject_ids: list[str]
    missing_in_raw: list[str]
    group_sizes: list[tuple[str, int]]


def _parse_subject_date(subject_id: str):
    parts = subject_id.split("_")
    if len(parts) < 2:
        return None
    try:
        return datetime.strptime(parts[1], "%Y%m%d").date()
    except Exception:
        return None


def _read_group_sublists(groups_dir: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for p in sorted(groups_dir.glob("group_*_sublist.txt")):
        sids = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        groups[p.stem] = sids
    return groups


def _collect_raw_subject_ids(raw_app_dir: Path) -> set[str]:
    sids: set[str] = set()
    for fp in raw_app_dir.glob("THU_*_GameData.xlsx"):
        parts = fp.stem.split("_")
        if len(parts) >= 4:
            sids.add("_".join(parts[:4]))
    return sids


def _audit_groups(*, kind: str, groups_dir: Path, raw_subject_ids: set[str]) -> GroupAudit:
    groups = _read_group_sublists(groups_dir)
    all_lines: list[str] = []
    for sids in groups.values():
        all_lines.extend(sids)

    dup_counts = Counter(all_lines)
    dup_sids = sorted([sid for sid, n in dup_counts.items() if n > 1])
    unique = set(all_lines)
    missing_in_raw = sorted(list(unique - raw_subject_ids))
    sizes = sorted(((g, len(sids)) for g, sids in groups.items()), key=lambda x: (-x[1], x[0]))

    return GroupAudit(
        kind=kind,
        n_groups=len(groups),
        total_lines=len(all_lines),
        unique_subjects=len(unique),
        duplicate_lines=len(all_lines) - len(unique),
        duplicate_subject_ids=dup_sids,
        missing_in_raw=missing_in_raw,
        group_sizes=sizes,
    )


def _render_report(
    *,
    dataset: str,
    raw_app_dir: Path,
    answer: GroupAudit,
    item: GroupAudit,
    out_path: Path,
) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        raw_app_display = raw_app_dir.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        raw_app_display = raw_app_dir.as_posix()

    def fmt_sizes(sizes: list[tuple[str, int]]) -> str:
        return ", ".join([f"{g}:{n}" for g, n in sizes])

    lines: list[str] = []
    lines.append("# APP grouping outputs audit")
    lines.append("")
    lines.append("本报告用于检查 `data/interim/behavioral_preprocess/groups_by_answer/` 与 `.../groups_by_item/`")
    lines.append("中由 `temp/group_app_stimulus_groups.py` 生成的分组结果是否与当前原始数据目录一致。")
    lines.append("")
    lines.append(f"- dataset: `{dataset}`")
    lines.append(f"- raw app dir: `{raw_app_display}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| kind | n_groups | total_lines | unique_subjects | duplicate_lines | missing_in_raw |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    lines.append(
        f"| answer | {answer.n_groups} | {answer.total_lines} | {answer.unique_subjects} | {answer.duplicate_lines} | {len(answer.missing_in_raw)} |"
    )
    lines.append(
        f"| item | {item.n_groups} | {item.total_lines} | {item.unique_subjects} | {item.duplicate_lines} | {len(item.missing_in_raw)} |"
    )
    lines.append("")

    lines.append("## Details")
    lines.append("")
    lines.append(f"- answer group sizes: {fmt_sizes(answer.group_sizes)}")
    lines.append(f"- item group sizes: {fmt_sizes(item.group_sizes)}")
    lines.append("")
    if answer.duplicate_subject_ids or item.duplicate_subject_ids:
        lines.append("- duplicated subject_id lines (should be 0):")
        for sid in sorted(set(answer.duplicate_subject_ids + item.duplicate_subject_ids)):
            lines.append(f"  - {sid}")
        lines.append("")
    if answer.missing_in_raw:
        lines.append("- subject_ids present in grouping outputs but missing from current raw dir:")
        for sid in answer.missing_in_raw:
            d = _parse_subject_date(sid)
            d_str = d.isoformat() if d else "NA"
            lines.append(f"  - {sid} (date={d_str})")
        lines.append("")

    lines.append("备注：若 `missing_in_raw>0`，通常意味着分组结果是在去重/移除重复工作簿之前生成的，")
    lines.append("需要在当前 `data/raw/behavior_data/cibr_app_data/` 基础上重新生成分组结果。")
    lines.append("")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit existing app grouping outputs under data/interim.")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    ap.add_argument("--dataset-config", dest="dataset_config", type=str, default=None)
    ap.add_argument(
        "--write-report",
        action="store_true",
        help="Write a markdown report under docs/reports/ (in addition to printing summary).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(paths_cfg, dataset_config_path=args.dataset_config, repo_root=repo_root)

    behavioral_cfg = dataset_cfg.get("behavioral", {})
    raw_app_rel = behavioral_cfg.get("app_data_dir")
    interim_rel = behavioral_cfg.get("interim_preprocess_dir", "behavioral_preprocess")
    if not raw_app_rel:
        raise ValueError("Missing dataset.behavioral.app_data_dir in config.")

    raw_app_dir = roots["raw_root"] / raw_app_rel
    interim_dir = roots["interim_root"] / interim_rel

    answer_dir = interim_dir / "groups_by_answer"
    item_dir = interim_dir / "groups_by_item"

    raw_subject_ids = _collect_raw_subject_ids(raw_app_dir)
    answer = _audit_groups(kind="answer", groups_dir=answer_dir, raw_subject_ids=raw_subject_ids)
    item = _audit_groups(kind="item", groups_dir=item_dir, raw_subject_ids=raw_subject_ids)

    print(f"raw_subjects={len(raw_subject_ids)}")
    print(f"answer: groups={answer.n_groups} total_lines={answer.total_lines} unique={answer.unique_subjects} dup_lines={answer.duplicate_lines} missing_in_raw={len(answer.missing_in_raw)}")
    print(f"item:   groups={item.n_groups} total_lines={item.total_lines} unique={item.unique_subjects} dup_lines={item.duplicate_lines} missing_in_raw={len(item.missing_in_raw)}")

    if args.write_report:
        docs_root = Path(paths_cfg.get("docs_reports_root", "docs/reports"))
        docs_root = docs_root if docs_root.is_absolute() else (repo_root / docs_root)
        out_path = docs_root / "app_grouping_audit.md"
        out_path.write_text(
            _render_report(
                dataset=args.dataset,
                raw_app_dir=raw_app_dir,
                answer=answer,
                item=item,
                out_path=out_path,
            ),
            encoding="utf-8",
        )
        print(f"wrote_report={out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
