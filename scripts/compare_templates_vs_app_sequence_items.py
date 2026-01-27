from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


def _canonical_task_name(raw: str) -> str:
    s = str(raw).strip()
    if not s:
        return s
    s = s.strip("_- ")
    rules = {
        "EmotionStoop": "EmotionStroop",
        "LG": "EmotionSwitch",
        "STROOP": "ColorStroop",
        "SpatialNBack": "Spatial2Back",
        "Emotion2Backformal": "Emotion2Back",
        "Number2Backformal": "Number2Back",
        "Spatial1Backformal": "Spatial1Back",
    }
    if s in rules:
        return rules[s]
    if "formal" in s.lower():
        s = re.sub("formal", "", s, flags=re.IGNORECASE).strip()
    return s


def _normalize_cell(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    low = s.lower()
    if low in {"nan", "none"}:
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


def _resolve_default_paths(
    *,
    dataset: str,
    paths_config: str,
    dataset_config: str | None,
    run_name: str,
    visit: str,
) -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    paths_cfg = load_paths_config(paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=dataset)
    ds_cfg = load_dataset_config(paths_cfg, dataset_config_path=dataset_config, repo_root=repo_root)

    behavioral_cfg = ds_cfg.get("behavioral", {})
    corrected_rel = behavioral_cfg.get("corrected_app_excel_dir")
    docs_reports_rel = paths_cfg.get("docs_reports_root", "docs/reports")
    if not corrected_rel:
        raise ValueError("Missing dataset.behavioral.corrected_app_excel_dir in config.")

    seq_dir = roots["processed_root"] / corrected_rel / run_name / "sequence_library"
    templates_path = seq_dir / f"{visit}_item_group1_templates.json"
    appseq_path = seq_dir / f"{visit}_from_app_sequence.json"

    docs_reports_root = Path(docs_reports_rel)
    if not docs_reports_root.is_absolute():
        docs_reports_root = repo_root / docs_reports_root
    report_path = docs_reports_root / f"{visit}_items_template_vs_app_sequence.md"
    return templates_path, appseq_path, report_path


@dataclass(frozen=True)
class TaskCompareResult:
    task: str
    template_len_all: int
    template_len_nonnull: int
    appseq_len: int
    exact_match: bool | None
    match_ratio: float | None
    first_mismatches: list[tuple[int, str | None, str | None]]
    note: str | None = None


def _compare_sequences(template_items: list[Any] | None, appseq_items: list[Any] | None) -> TaskCompareResult:
    tmpl_all = template_items or []
    app_all = appseq_items or []

    tmpl_norm = [_normalize_cell(x) for x in tmpl_all]
    app_norm = [_normalize_cell(x) for x in app_all]

    tmpl_nonnull = [x for x in tmpl_norm if x is not None]
    app_nonnull = [x for x in app_norm if x is not None]

    note = None
    if not tmpl_all and not app_all:
        return TaskCompareResult(
            task="",
            template_len_all=0,
            template_len_nonnull=0,
            appseq_len=0,
            exact_match=None,
            match_ratio=None,
            first_mismatches=[],
            note="both_empty",
        )

    if not app_nonnull:
        note = "app_sequence_items_empty_or_unparsed"
    elif not tmpl_nonnull:
        note = "template_items_empty_or_missing"

    n = min(len(tmpl_nonnull), len(app_nonnull))
    if n == 0:
        return TaskCompareResult(
            task="",
            template_len_all=len(tmpl_all),
            template_len_nonnull=len(tmpl_nonnull),
            appseq_len=len(app_nonnull),
            exact_match=False if (tmpl_nonnull or app_nonnull) else None,
            match_ratio=None,
            first_mismatches=[],
            note=note,
        )

    mismatches: list[tuple[int, str | None, str | None]] = []
    matches = 0
    for i in range(n):
        a = tmpl_nonnull[i]
        b = app_nonnull[i]
        if a == b:
            matches += 1
        elif len(mismatches) < 10:
            mismatches.append((i, a, b))

    exact = (len(tmpl_nonnull) == len(app_nonnull)) and (matches == n)
    ratio = matches / n if n else None
    return TaskCompareResult(
        task="",
        template_len_all=len(tmpl_all),
        template_len_nonnull=len(tmpl_nonnull),
        appseq_len=len(app_nonnull),
        exact_match=exact,
        match_ratio=ratio,
        first_mismatches=mismatches,
        note=note,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonicalize_keys(obj: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    collisions: dict[str, list[str]] = {}
    for k, v in obj.items():
        ck = _canonical_task_name(k)
        if ck in out:
            collisions.setdefault(ck, []).append(k)
        else:
            out[ck] = v
    if collisions:
        msg = ", ".join([f"{ck}<-{[ck]+orig}" for ck, orig in collisions.items()])
        raise ValueError(f"Task name collisions after canonicalization: {msg}")
    return out


def _render_report(visit: str, results: list[TaskCompareResult]) -> str:
    ok = [r for r in results if r.exact_match is True]
    bad = [r for r in results if r.exact_match is False or (r.note is not None and r.exact_match is not True)]

    lines: list[str] = []
    lines.append(f"# {visit} items: group templates vs app_sequence")
    lines.append("")
    lines.append("本报告对比 `*_item_group1_templates.json` 与 `*_from_app_sequence.json` 的 **items** 是否一致。")
    lines.append("注意：仅对比 items；answers 后续应通过推断与人工复核确定，不在本报告中作为一致性标准。")
    lines.append("")
    lines.append(f"- exact_match tasks: {len(ok)}/{len(results)}")
    lines.append(f"- needs_attention tasks: {len(bad)}/{len(results)}")
    lines.append("")

    lines.append("## Task summary")
    lines.append("")
    lines.append("| task | template_nonnull_len | app_sequence_len | exact_match | match_ratio(on overlap) | note |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in sorted(results, key=lambda x: x.task):
        exact = "" if r.exact_match is None else ("YES" if r.exact_match else "NO")
        ratio = "" if r.match_ratio is None else f"{r.match_ratio:.3f}"
        note = r.note or ""
        lines.append(f"| {r.task} | {r.template_len_nonnull} | {r.appseq_len} | {exact} | {ratio} | {note} |")

    if bad:
        lines.append("")
        lines.append("## Details (first mismatches)")
        lines.append("")
        for r in sorted(bad, key=lambda x: x.task):
            lines.append(f"### {r.task}")
            lines.append("")
            lines.append(f"- template_len_all={r.template_len_all}, template_len_nonnull={r.template_len_nonnull}")
            lines.append(f"- app_sequence_len={r.appseq_len}")
            if r.note:
                lines.append(f"- note: {r.note}")
            if r.first_mismatches:
                lines.append("")
                lines.append("| idx (nonnull) | template_item | app_sequence_item |")
                lines.append("| --- | --- | --- |")
                for idx, a, b in r.first_mismatches:
                    lines.append(f"| {idx} | {a} | {b} |")
            lines.append("")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare group templates vs app_sequence items for a given visit.")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    ap.add_argument("--dataset-config", dest="dataset_config", type=str, default=None)
    ap.add_argument("--run-name", type=str, default="run_corrected_v2")
    ap.add_argument("--visit", type=str, default="visit1", help="e.g., visit1/visit2/visit3/visit4")
    ap.add_argument("--templates", type=str, default=None, help="Override templates JSON path.")
    ap.add_argument("--app-seq", type=str, default=None, help="Override app_sequence JSON path.")
    ap.add_argument("--out", type=str, default=None, help="Write a markdown report to this path.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    default_templates, default_appseq, default_report = _resolve_default_paths(
        dataset=args.dataset,
        paths_config=args.paths_config,
        dataset_config=args.dataset_config,
        run_name=args.run_name,
        visit=args.visit,
    )

    templates_path = Path(args.templates) if args.templates else default_templates
    appseq_path = Path(args.app_seq) if args.app_seq else default_appseq
    out_path = Path(args.out) if args.out else default_report

    templates = _load_json(templates_path)
    appseq = _load_json(appseq_path)

    if not isinstance(templates, dict) or not isinstance(appseq, dict):
        raise ValueError("Expected both inputs to be JSON objects keyed by task name.")

    templates = _canonicalize_keys(templates)
    appseq = _canonicalize_keys(appseq)

    tasks = sorted(set(templates.keys()) | set(appseq.keys()))
    results: list[TaskCompareResult] = []
    for task in tasks:
        tmpl_items = (templates.get(task) or {}).get("items")
        app_items = (appseq.get(task) or {}).get("items")
        r = _compare_sequences(tmpl_items, app_items)
        results.append(
            TaskCompareResult(
                task=task,
                template_len_all=r.template_len_all,
                template_len_nonnull=r.template_len_nonnull,
                appseq_len=r.appseq_len,
                exact_match=r.exact_match,
                match_ratio=r.match_ratio,
                first_mismatches=r.first_mismatches,
                note=r.note,
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_report(args.visit, results), encoding="utf-8")

    ok = sum(1 for r in results if r.exact_match is True)
    print(f"tasks={len(results)} exact_match={ok} report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
