from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.behavioral_preprocess.metrics.efny.io import normalize_columns
from src.behavioral_preprocess.metrics.efny.main import load_task_config, normalize_task_name
from src.behavioral_preprocess.metrics.efny.metrics import _conflict_condition
from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


@dataclass(frozen=True)
class TaskTrialSummary:
    sheet_name: str
    task_name: str
    task_type: str | None
    n_raw: int
    n_included: int | None
    counts: dict[str, int]
    note: str | None = None
    mixed_from: float | None = None


def _to_markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for r in rows:
        body.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    return "\n".join([header, sep] + body)


def _resolve_default_excel_path(dataset_cfg: dict, roots: dict[str, Path]) -> Path:
    behavioral_cfg = dataset_cfg.get("behavioral", {})
    app_data_rel = behavioral_cfg.get("app_data_dir")
    ref_name = behavioral_cfg.get("reference_game_data_file")
    if not app_data_rel:
        raise ValueError("Missing dataset.behavioral.app_data_dir in configs/paths.yaml.")
    if not ref_name:
        raise ValueError("Missing dataset.behavioral.reference_game_data_file in configs/paths.yaml.")
    return roots["raw_root"] / app_data_rel / ref_name


def _resolve_reports_root(paths_cfg: dict, repo_root: Path) -> Path:
    rel = paths_cfg.get("docs_reports_root") or "docs/reports"
    p = Path(str(rel))
    return p if p.is_absolute() else (repo_root / p)


def _safe_report_basename(excel_path: Path) -> str:
    name = excel_path.stem
    parts = name.split("_")
    safe_parts: list[str] = []
    for p in parts:
        if any(ord(ch) > 127 for ch in p):
            continue
        if p.lower() in {"gamedata"}:
            continue
        safe_parts.append(p)
    if not safe_parts:
        safe_parts = ["behavior_data", "eda"]
    return "behavior_data_eda_" + "_".join(safe_parts).lower() + ".md"


def _summarize_gonogo(df: pd.DataFrame) -> tuple[int, dict[str, int], str | None]:
    if "answer" not in df.columns:
        return 0, {}, "missing answer column"
    ans = df["answer"].astype(str).str.strip().str.lower()
    go_mask = ans.isin(["true", "1", "yes"])
    nogo_mask = ans.isin(["false", "0", "no"])
    unknown = int((~(go_mask | nogo_mask)).sum())
    counts = {"go": int(go_mask.sum()), "nogo": int(nogo_mask.sum())}
    note = f"unclassified={unknown}" if unknown else None
    return int(len(df)), counts, note


def _summarize_sst(df: pd.DataFrame, cfg: dict) -> tuple[int, dict[str, int], str | None]:
    if df is None:
        return 0, {}, "empty sheet"
    d = df.copy()
    note_parts: list[str] = []
    if len(d) >= 96:
        d = d.iloc[:96].copy()
        note_parts.append("truncated_to_96")

    ssd_var = cfg.get("ssd_var", "SSRT")
    if ssd_var not in d.columns:
        return int(len(d)), {}, f"missing {ssd_var} column"
    ssd = pd.to_numeric(d[ssd_var], errors="coerce")
    is_stop = ssd.notna()
    counts = {"go": int((~is_stop).sum()), "stop": int(is_stop.sum())}
    note = ";".join(note_parts) if note_parts else None
    return int(len(d)), counts, note


def _summarize_conflict(df: pd.DataFrame, cfg: dict, task_name: str) -> TaskTrialSummary:
    if df is None or len(df) == 0:
        return TaskTrialSummary(
            sheet_name=task_name,
            task_name=task_name,
            task_type="conflict",
            n_raw=int(len(df)),
            n_included=None,
            counts={},
            note="empty sheet",
        )

    if "item" not in df.columns:
        return TaskTrialSummary(
            sheet_name=task_name,
            task_name=task_name,
            task_type="conflict",
            n_raw=int(len(df)),
            n_included=None,
            counts={},
            note="missing item column",
        )

    cond = _conflict_condition(task_name, df["item"])
    keep = cond.notna()
    d2 = df.loc[keep].copy()
    d2["cond"] = cond.loc[keep]
    counts = d2["cond"].value_counts().to_dict()
    return TaskTrialSummary(
        sheet_name=task_name,
        task_name=task_name,
        task_type="conflict",
        n_raw=int(len(df)),
        n_included=int(len(d2)),
        counts={str(k): int(v) for k, v in counts.items()},
        note=None,
    )


def _summarize_switch(df: pd.DataFrame, cfg: dict, task_name: str) -> TaskTrialSummary:
    if df is None or len(df) == 0:
        return TaskTrialSummary(
            sheet_name=task_name,
            task_name=task_name,
            task_type="switch",
            n_raw=int(len(df)),
            n_included=None,
            counts={},
            note="empty sheet",
        )
    d = df.copy()
    if "item" not in d.columns:
        return TaskTrialSummary(
            sheet_name=task_name,
            task_name=task_name,
            task_type="switch",
            n_raw=int(len(df)),
            n_included=None,
            counts={},
            note="missing item column",
        )

    if "trial_index" in d.columns:
        d["_trial_index_num"] = pd.to_numeric(d["trial_index"], errors="coerce")
        d = d.sort_values("_trial_index_num")

    tn = str(task_name)
    rule = pd.Series(np.nan, index=d.index, dtype="object")
    if tn.upper() == "DCCS":
        rule = d["item"].astype(str).str.slice(0, 1)
    elif tn.upper() == "DT":
        rule = np.where(d["item"].astype(str).str.contains("[Tt]"), "TN", "CN")
        rule = pd.Series(rule, index=d.index, dtype="object")
    elif tn.upper() == "EMOTIONSWITCH":
        num = d["item"].astype(str).str.extract(r"(\d+)")[0]
        num = pd.to_numeric(num, errors="coerce")
        rule = pd.Series(np.nan, index=d.index, dtype="object")
        em = num.notna() & num.between(1, 4)
        ge = num.notna() & num.between(5, 8)
        rule.loc[em] = "emotion"
        rule.loc[ge] = "gender"

    prev_rule = pd.Series(rule).shift(1)
    switch_type = pd.Series(np.nan, index=d.index, dtype="object")
    same = (rule == prev_rule) & prev_rule.notna() & pd.Series(rule).notna()
    diff = (rule != prev_rule) & prev_rule.notna() & pd.Series(rule).notna()
    switch_type.loc[same] = "repeat"
    switch_type.loc[diff] = "switch"

    d["switch_type"] = switch_type
    d = d[d["switch_type"].notna()].copy()
    mixed_from = cfg.get("mixed_from", None)
    if mixed_from is not None and "_trial_index_num" in d.columns:
        d = d[d["_trial_index_num"] >= float(mixed_from)]

    counts = d["switch_type"].value_counts().to_dict()
    return TaskTrialSummary(
        sheet_name=task_name,
        task_name=task_name,
        task_type="switch",
        n_raw=int(len(df)),
        n_included=int(len(d)),
        counts={str(k): int(v) for k, v in counts.items()},
        note=None,
        mixed_from=float(mixed_from) if mixed_from is not None else None,
    )


def _summarize_nback(df: pd.DataFrame, cfg: dict, task_name: str) -> TaskTrialSummary:
    n_back = int(cfg.get("n_back", 1))
    if df is None or len(df) == 0:
        return TaskTrialSummary(
            sheet_name=task_name,
            task_name=task_name,
            task_type="nback",
            n_raw=int(len(df)),
            n_included=None,
            counts={},
            note="empty sheet",
        )
    d = df
    if "item" not in d.columns:
        return TaskTrialSummary(
            sheet_name=task_name,
            task_name=task_name,
            task_type="nback",
            n_raw=int(len(df)),
            n_included=None,
            counts={},
            note="missing item column",
        )

    stim = d["item"]
    lag = stim.shift(n_back)
    target_mask = stim.notna() & lag.notna() & (stim == lag)
    nontarget_mask = stim.notna() & ~target_mask
    trial_type = pd.Series(np.nan, index=d.index, dtype="object")
    trial_type.loc[nontarget_mask] = "no_match"
    trial_type.loc[target_mask] = "match"

    d2 = d.copy()
    d2["trial_type"] = trial_type
    d2 = d2[d2["trial_type"].notna()]

    counts = d2["trial_type"].value_counts().to_dict()
    return TaskTrialSummary(
        sheet_name=task_name,
        task_name=task_name,
        task_type="nback",
        n_raw=int(len(df)),
        n_included=int(len(d2)),
        counts={str(k): int(v) for k, v in counts.items()},
        note=None,
    )


def build_report_markdown(
    *,
    excel_path: Path,
    sheet_summaries: list[TaskTrialSummary],
) -> str:
    lines: list[str] = []
    lines.append("# 行为数据探索性分析：试次数量与条件分布（单被试）")
    lines.append("")
    lines.append("## 数据来源")
    lines.append(f"- 工作簿：`{excel_path.as_posix()}`")
    lines.append(f"- Sheet 数：{len(sheet_summaries)}")
    lines.append("")
    lines.append("## 口径说明")
    lines.append("- 本报告用于探索性试次统计与建模可行性评估：**不进行反应时过滤/修剪**（避免单被试过滤后分布失真）。")
    lines.append("- `Included` 表示可被当前规则归类的试次数量（例如切换任务需可判定 repeat/switch；SST 可能截断到前 96 行）。")
    lines.append("")

    by_type: dict[str, list[TaskTrialSummary]] = {}
    for s in sheet_summaries:
        key = s.task_type or "other"
        by_type.setdefault(key, []).append(s)

    def _task_table(task_type: str, columns: list[str], rows: list[dict]) -> None:
        lines.append(f"## {task_type}")
        lines.append(_to_markdown_table(rows, columns))
        lines.append("")

    if "gonogo" in by_type:
        rows = []
        for s in by_type["gonogo"]:
            rows.append(
                {
                    "Task": s.task_name,
                    "Total": s.n_raw,
                    "Go": s.counts.get("go", ""),
                    "NoGo": s.counts.get("nogo", ""),
                    "Note": s.note or "",
                }
            )
        _task_table("Go/NoGo（基于 answer 的 true/false）", ["Task", "Total", "Go", "NoGo", "Note"], rows)

    if "sst" in by_type:
        rows = []
        for s in by_type["sst"]:
            rows.append(
                {
                    "Task": s.task_name,
                    "Total": s.n_raw,
                    "Included": s.n_included if s.n_included is not None else "",
                    "Go": s.counts.get("go", ""),
                    "Stop": s.counts.get("stop", ""),
                    "Note": s.note or "",
                }
            )
        _task_table(
            "SST：Go/Stop（基于 SSD/SSRT 列是否为空）",
            ["Task", "Total", "Included", "Go", "Stop", "Note"],
            rows,
        )

    if "switch" in by_type:
        rows = []
        for s in by_type["switch"]:
            rows.append(
                {
                    "Task": s.task_name,
                    "Total": s.n_raw,
                    "Included": s.n_included if s.n_included is not None else "",
                    "Repeat": s.counts.get("repeat", ""),
                    "Switch": s.counts.get("switch", ""),
                    "Note": s.note or "",
                }
            )
        _task_table("Repeat/Switch（按规则变更；含 mixed_from 截断）", ["Task", "Total", "Included", "Repeat", "Switch", "Note"], rows)

    if "conflict" in by_type:
        rows = []
        for s in by_type["conflict"]:
            rows.append(
                {
                    "Task": s.task_name,
                    "Total": s.n_raw,
                    "Included": s.n_included if s.n_included is not None else "",
                    "Congruent": s.counts.get("congruent", ""),
                    "Incongruent": s.counts.get("incongruent", ""),
                    "Note": s.note or "",
                }
            )
        _task_table("Congruent/Incongruent（基于 item 解析）", ["Task", "Total", "Included", "Congruent", "Incongruent", "Note"], rows)

    if "nback" in by_type:
        rows = []
        for s in by_type["nback"]:
            rows.append(
                {
                    "Task": s.task_name,
                    "Total": s.n_raw,
                    "Included": s.n_included if s.n_included is not None else "",
                    "Match": s.counts.get("match", ""),
                    "NoMatch": s.counts.get("no_match", ""),
                    "Note": s.note or "",
                }
            )
        _task_table("N-back（Match/NoMatch；基于 item 的 n-back lag）", ["Task", "Total", "Included", "Match", "NoMatch", "Note"], rows)

    other = by_type.get("other", [])
    if other:
        rows = []
        for s in sorted(other, key=lambda x: x.task_name):
            rows.append({"Task": s.task_name, "Total": s.n_raw})
        _task_table("其他任务（仅总试次）", ["Task", "Total"], rows)

    lines.append("## Drift model（DDM）适用性建议")
    lines.append(
        "以下判断以“二选一反应（2AFC）+ 试次级 RT + 充分试次数量”为基本前提；"
        "若任务存在抑制/停止机制或大量无反应试次，应优先考虑扩展模型（如 go/no-go DDM、stop-signal race）。"
    )
    lines.append("")

    def _has_type(t: str) -> bool:
        return t in by_type and len(by_type[t]) > 0

    if _has_type("conflict"):
        tasks = ", ".join(s.task_name for s in by_type["conflict"])
        lines.append(f"- **优先候选（冲突类）**：{tasks}（条件划分清晰，通常可按 congruent/incongruent 分层拟合）。")
    if _has_type("switch"):
        sw = by_type["switch"]
        sw_ok = [s for s in sw if (s.n_included or 0) >= 50]
        sw_low = [s for s in sw if (s.n_included or 0) < 50]
        mixed_info = []
        for s in sorted(sw, key=lambda x: x.task_name):
            if s.mixed_from is None:
                continue
            mixed_info.append(f"{s.task_name}: trial_index≥{int(s.mixed_from)}")
        if mixed_info:
            lines.append(f"- **切换类口径**：仅使用 mixed block（{', '.join(mixed_info)}；pure block 不参与 repeat/switch 统计与建模）。")
        if sw_ok:
            parts = []
            for s in sorted(sw_ok, key=lambda x: x.task_name):
                rep = s.counts.get("repeat", 0)
                swc = s.counts.get("switch", 0)
                parts.append(f"{s.task_name}(Repeat={rep}, Switch={swc})")
            lines.append(f"- **可用候选（切换类）**：{', '.join(parts)}（建议多被试层级 DDM；单被试分层拟合不稳定）。")
        if sw_low:
            tasks = ", ".join(f"{s.task_name}(Included={s.n_included})" for s in sw_low)
            lines.append(f"- **谨慎使用（切换类试次偏少）**：{tasks}（分层后单条件试次可能不足，参数估计不稳定）。")
    if _has_type("nback"):
        nb = by_type["nback"]
        tasks = ", ".join(s.task_name for s in nb)
        low_match = [s for s in nb if (s.counts.get('match', 0) < 15)]
        lines.append(
            f"- **可探索（N-back）**：{tasks}（可按 match/no-match 分层；需注意 match 试次往往较少，可能不足以稳定估计条件差异）。"
        )
        if low_match:
            tasks2 = ", ".join(f"{s.task_name}(Match={s.counts.get('match', 0)})" for s in low_match)
            lines.append(f"  - 本工作簿中 match 试次偏少：{tasks2}。")
    if _has_type("gonogo"):
        tasks = ", ".join(s.task_name for s in by_type["gonogo"])
        lines.append(
            f"- **不建议直接用标准 2AFC DDM（Go/NoGo）**：{tasks}（no-go 通常缺失 RT；若要建模可考虑 go/no-go DDM 或 race 模型）。"
        )
    if _has_type("sst"):
        tasks = ", ".join(s.task_name for s in by_type["sst"])
        lines.append(
            f"- **不建议直接用标准 2AFC DDM（SST）**：{tasks}（停止机制更符合 stop-signal race/抑制控制模型；可在此任务上开展 SSRT/race 类建模）。"
        )
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def summarize_workbook(*, excel_path: Path, metrics_config_path: str) -> list[TaskTrialSummary]:
    task_cfg = load_task_config(metrics_config_path)
    xl = pd.ExcelFile(excel_path)
    out: list[TaskTrialSummary] = []

    for sheet_name in xl.sheet_names:
        task_name = normalize_task_name(sheet_name)
        cfg = task_cfg.get(task_name)
        t = cfg.get("type") if isinstance(cfg, dict) else None
        df_raw = xl.parse(sheet_name, dtype="object")
        df = normalize_columns(df_raw)

        if t == "gonogo":
            n_included, counts, note = _summarize_gonogo(df)
            out.append(
                TaskTrialSummary(
                    sheet_name=sheet_name,
                    task_name=task_name,
                    task_type="gonogo",
                    n_raw=int(len(df)),
                    n_included=n_included,
                    counts=counts,
                    note=note,
                )
            )
            continue

        if t == "sst":
            n_included, counts, note = _summarize_sst(df, cfg)
            out.append(
                TaskTrialSummary(
                    sheet_name=sheet_name,
                    task_name=task_name,
                    task_type="sst",
                    n_raw=int(len(df)),
                    n_included=n_included,
                    counts=counts,
                    note=note,
                )
            )
            continue

        if t == "conflict":
            out.append(_summarize_conflict(df, cfg, task_name=task_name))
            continue

        if t == "switch":
            out.append(_summarize_switch(df, cfg, task_name=task_name))
            continue

        if t == "nback":
            out.append(_summarize_nback(df, cfg, task_name=task_name))
            continue

        out.append(
            TaskTrialSummary(
                sheet_name=sheet_name,
                task_name=task_name,
                task_type=None,
                n_raw=int(len(df)),
                n_included=None,
                counts={},
            )
        )

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Exploratory trial-count report for a single app behavioral Excel workbook."
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    parser.add_argument("--dataset-config", dest="dataset_config", type=str, default=None)
    parser.add_argument(
        "--metrics-config",
        dest="metrics_config",
        type=str,
        default="configs/behavioral_metrics.yaml",
        help="Behavioral metrics config with compute_tasks/use_metrics.",
    )
    parser.add_argument(
        "--excel-file",
        dest="excel_file",
        type=str,
        default=None,
        help="Excel workbook path (absolute or repo-relative). When omitted, uses configs/paths.yaml dataset.behavioral.reference_game_data_file under dataset.behavioral.app_data_dir.",
    )
    parser.add_argument(
        "--report-out",
        dest="report_out",
        type=str,
        default=None,
        help="Report output path (absolute or repo-relative). Default: <docs_reports_root>/<auto-name>.md",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(paths_cfg, dataset_config_path=args.dataset_config, repo_root=repo_root)

    if args.excel_file:
        p = Path(args.excel_file)
        excel_path = p if p.is_absolute() else (repo_root / p)
    else:
        excel_path = _resolve_default_excel_path(dataset_cfg, roots)

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel workbook not found: {excel_path}")

    reports_root = _resolve_reports_root(paths_cfg, repo_root=repo_root)
    if args.report_out:
        rp = Path(args.report_out)
        report_path = rp if rp.is_absolute() else (repo_root / rp)
    else:
        report_path = reports_root / _safe_report_basename(excel_path)

    summaries = summarize_workbook(excel_path=excel_path, metrics_config_path=args.metrics_config)
    md = build_report_markdown(excel_path=excel_path, sheet_summaries=summaries)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
