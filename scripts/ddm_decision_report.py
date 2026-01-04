from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from src.behavioral_preprocess.ddm.hssm_fit import FitConfig, fit_ddm
from src.behavioral_preprocess.ddm.trials import DDMExtractionConfig, extract_trials_from_directory
from src.behavioral_preprocess.metrics.efny.main import load_task_config
from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


def _resolve_reports_root(paths_cfg: dict, repo_root: Path) -> Path:
    rel = paths_cfg.get("docs_reports_root") or "docs/reports"
    p = Path(str(rel))
    return p if p.is_absolute() else (repo_root / p)


def _md_table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def _format_beta(beta: dict[str, float] | None) -> str:
    if not beta:
        return ""
    return f"{beta['mean']:.3f} [{beta['hdi_2.5']:.3f}, {beta['hdi_97.5']:.3f}], P(>0)={beta['p_gt_0']:.2f}"


def _format_loo(loo: az.ELPDData | None) -> str:
    if loo is None:
        return ""
    if "elpd_loo" in loo:
        return f"elpd_loo={float(loo['elpd_loo']):.1f}, p_loo={float(loo.get('p_loo', np.nan)):.1f}"
    return ""


def _decision_matrix_markdown() -> str:
    rows = [
        {
            "Task": "FLANKER",
            "纳入试次": "全部 (96; 48C/48I)",
            "层级HDDM": "可选（推荐做统一管线）",
            "模型（主推）": "HDDM/层级DDM：v ~ congruency（必要时 a ~ congruency）",
            "备注/风险点": "条件均衡，最稳定；也可做单被试分层拟合（不推荐作为主结论）。",
        },
        {
            "Task": "ColorStroop",
            "纳入试次": "全部 (96; 24C/72I)",
            "层级HDDM": "是",
            "模型（主推）": "HDDM/层级DDM：v ~ congruency（不建议单被试分层拟合）",
            "备注/风险点": "条件极不均衡；congruent 少，必须用 predictor 思路。",
        },
        {
            "Task": "EmotionStroop",
            "纳入试次": "全部 (96; 24C/72I)",
            "层级HDDM": "是",
            "模型（主推）": "HDDM/层级DDM：v ~ congruency（不建议单被试分层拟合）",
            "备注/风险点": "同上。",
        },
        {
            "Task": "DT",
            "纳入试次": "pure 64 + mixed 64 = 128；switch 仅 mixed",
            "层级HDDM": "是（强烈建议）",
            "模型（主推）": "两层策略：① Mixing-DDM：v ~ block_type(pure/mixed)（用 128）；②（辅）Switch-DDM：v ~ trial_type(R/S)（仅 mixed）",
            "备注/风险点": "更适合“完整 DDM 分解”：mixing 主打，switch 可做补充。",
        },
        {
            "Task": "EmotionSwitch",
            "纳入试次": "pure 64 + mixed 64 = 128；switch 仅 mixed",
            "层级HDDM": "是（强烈建议）",
            "模型（主推）": "两层策略：① Mixing-DDM：v ~ block_type(pure/mixed)（用 128）；②（辅）Switch-DDM：v ~ trial_type(R/S)（仅 mixed）",
            "备注/风险点": "mixed 内 switch 较少；建议只让 v 随条件变，避免多参数一起变。",
        },
        {
            "Task": "DCCS",
            "纳入试次": "pure 20 + mixed 20 = 40；mixed 内 10R/10S",
            "层级HDDM": "否（或仅探索）",
            "模型（主推）": "不建议做 switch-DDM；如一定要做，仅做整体 DDM 或 v ~ block_type",
            "备注/风险点": "mixed 内每格 10 太少；switch-DDM 基本不可识别。",
        },
        {
            "Task": "SST",
            "纳入试次": "主模型用 go+stop（race/SSRT）；DDM 仅用 go RT",
            "层级HDDM": "①否（标准 DDM）②可选（Go-DDM）",
            "模型（主推）": "Stop-signal race/SSRT（integration 等）；可选补充：Go-DDM（HDDM）仅用 go=72",
            "备注/风险点": "抑制过程更适合 race/SSRT；Go-DDM 解释“go 决策”，仅作补充。",
        },
        {
            "Task": "GNG / CPT",
            "纳入试次": "Go/NoGo（NoGo 无 RT）",
            "层级HDDM": "否",
            "模型（主推）": "不做标准 DDM；用 SDT(d'+c) + commission/omission（可加 time-on-task）",
            "备注/风险点": "单键 Go/NoGo 不符合标准 2AFC-RT DDM；NoGo 无 RT。",
        },
        {
            "Task": "N-back 系列",
            "纳入试次": "60（match 很少）",
            "层级HDDM": "否（不建议）",
            "模型（主推）": "SDT(d'+c) + RT/ACC + 负荷/域比较",
            "备注/风险点": "match 太少；分层 DDM 不可行，整体 DDM 解释弱。",
        },
    ]
    cols = ["Task", "纳入试次", "层级HDDM", "模型（主推）", "备注/风险点"]
    return _md_table(rows, cols)


def main():
    parser = argparse.ArgumentParser(description="Generate DDM/HDDM decision report with PyMC-based hierarchical fits.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    parser.add_argument("--dataset-config", dest="dataset_config", type=str, default=None)
    parser.add_argument("--metrics-config", dest="metrics_config", type=str, default="configs/behavioral_metrics.yaml")
    parser.add_argument("--pattern", type=str, default="*.xlsx")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for quick pilot runs.")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--tune", type=int, default=200)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--report-out", type=str, default=None, help="Default: docs/reports/ddm_decision.md")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing report file (legacy script; prefer scripts.fit_ssm_task / scripts.fit_race4_task).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(paths_cfg, dataset_config_path=args.dataset_config, repo_root=repo_root)

    behavioral_cfg = dataset_cfg.get("behavioral", {})
    app_data_rel = behavioral_cfg.get("app_data_dir")
    if not app_data_rel:
        raise ValueError("Missing dataset.behavioral.app_data_dir in configs/paths.yaml.")
    app_dir = roots["raw_root"] / app_data_rel

    task_cfg = load_task_config(args.metrics_config)
    mixed_from_by_task = {}
    for t in ("DCCS", "DT", "EmotionSwitch"):
        if t in task_cfg and "mixed_from" in task_cfg[t]:
            mixed_from_by_task[t] = int(task_cfg[t]["mixed_from"])
    extract_cfg = DDMExtractionConfig(mixed_from_by_task=mixed_from_by_task)

    all_trials = extract_trials_from_directory(
        app_dir,
        task_config=task_cfg,
        cfg=extract_cfg,
        pattern=args.pattern,
        max_files=args.max_files,
    )

    fit_cfg = FitConfig(
        sampler="nuts_numpyro",
        draws=int(args.draws),
        tune=int(args.tune),
        chains=int(args.chains),
        progress_bar=False,
        random_seed=int(args.seed),
    )

    results_rows: list[dict] = []

    # Conflict: v ~ congruency
    conflict = all_trials["conflict"]
    for task in ("FLANKER", "ColorStroop", "EmotionStroop"):
        d = conflict[conflict["task"] == task].copy()
        if d.empty:
            continue
        d["congruency"] = d["congruency"].astype(int)
        v_formula = "v ~ 1 + congruency + (1|subject)"
        res = fit_ddm(
            d,
            v_formula=v_formula,
            group="subject",
            fit_cfg=fit_cfg,
            beta_varname="v_congruency",
            hierarchical_v_only=True,
        )
        results_rows.append(
            {
                "Task": task,
                "Model": "HDDM (hier): v ~ congruency",
                "N_subjects": int(d["subject"].nunique()),
                "N_trials": int(len(d)),
                "Beta(v_congruency)": _format_beta(res.beta_summary),
                "LOO": _format_loo(res.loo),
            }
        )

    # Switch mixing: v ~ block_mixed (pure=0, mixed=1) using all trials
    switch_mixing = all_trials["switch_mixing"]
    for task in ("DT", "EmotionSwitch"):
        d = switch_mixing[switch_mixing["task"] == task].copy()
        if d.empty:
            continue
        d["block_mixed"] = d["block_mixed"].astype(int)
        v_formula = "v ~ 1 + block_mixed + (1|subject)"
        res = fit_ddm(
            d,
            v_formula=v_formula,
            group="subject",
            fit_cfg=fit_cfg,
            beta_varname="v_block_mixed",
            hierarchical_v_only=True,
        )
        results_rows.append(
            {
                "Task": task,
                "Model": "HDDM (hier): v ~ block_mixed",
                "N_subjects": int(d["subject"].nunique()),
                "N_trials": int(len(d)),
                "Beta(v_block_mixed)": _format_beta(res.beta_summary),
                "LOO": _format_loo(res.loo),
            }
        )

    # Switch-only: v ~ is_switch (repeat=0, switch=1) mixed only
    switch_only = all_trials["switch_only"]
    for task in ("DT", "EmotionSwitch"):
        d = switch_only[switch_only["task"] == task].copy()
        if d.empty:
            continue
        d["is_switch"] = d["is_switch"].astype(int)
        v_formula = "v ~ 1 + is_switch + (1|subject)"
        res = fit_ddm(
            d,
            v_formula=v_formula,
            group="subject",
            fit_cfg=fit_cfg,
            beta_varname="v_is_switch",
            hierarchical_v_only=True,
        )
        results_rows.append(
            {
                "Task": task,
                "Model": "HDDM (hier): v ~ is_switch (mixed only)",
                "N_subjects": int(d["subject"].nunique()),
                "N_trials": int(len(d)),
                "Beta(v_is_switch)": _format_beta(res.beta_summary),
                "LOO": _format_loo(res.loo),
            }
        )

    report_lines: list[str] = []
    report_lines.append("# DDM/HDDM 决策文档（PyMC-based）")
    report_lines.append("")
    report_lines.append("## 输入与范围")
    report_lines.append(f"- 输入目录：`{app_dir.as_posix()}`")
    report_lines.append(f"- 文件模式：`{args.pattern}`")
    report_lines.append(f"- 本次运行文件数：{int(args.max_files) if args.max_files else 'ALL'}")
    report_lines.append("")
    report_lines.append("## 决策矩阵（依据任务表）")
    report_lines.append(_decision_matrix_markdown())
    report_lines.append("")
    report_lines.append("## 本次模型计算设置")
    report_lines.append("- 后端：HSSM（PyMC-based hierarchical SSM） + `nuts_numpyro` 采样")
    report_lines.append(f"- draws={fit_cfg.draws}, tune={fit_cfg.tune}, chains={fit_cfg.chains}, seed={fit_cfg.random_seed}")
    report_lines.append("- 层级结构：仅对 drift rate (`v`) 引入被试随机截距（`(1|subject)`）；`a`/`t` 采用群体固定效应（计算可行性优先）。")
    report_lines.append("- 说明：仅将结果嵌入本文档，不写出 CSV。")
    report_lines.append("")
    report_lines.append("## 计算结果摘要（效果评估）")
    if results_rows:
        cols = ["Task", "Model", "N_subjects", "N_trials", "Beta(v_congruency)", "Beta(v_block_mixed)", "Beta(v_is_switch)", "LOO"]
        report_lines.append(_md_table(results_rows, cols))
    else:
        report_lines.append("（未生成任何模型结果：请检查数据目录、sheet 名称与依赖安装。）")
    report_lines.append("")
    report_lines.append("## 复现方式")
    report_lines.append("```bash")
    cmd = "python -m scripts.ddm_decision_report --dataset EFNY --config configs/paths.yaml"
    if args.max_files:
        cmd += f" --max-files {args.max_files}"
    cmd += f" --draws {args.draws} --tune {args.tune} --chains {args.chains} --seed {args.seed}"
    report_lines.append(cmd)
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("## 全样本运行建议（规划）")
    report_lines.append("- 先完成小规模 pilot（例如 `--max-files 10`），确认依赖与模型可跑通、参数符号与数量级合理。")
    report_lines.append("- 全样本层级 HDDM 计算耗时长，建议在可用计算节点上运行，并逐步提高采样长度（如 draws≥500, tune≥500, chains≥2）。")
    report_lines.append("```bash")
    report_lines.append("python -m scripts.ddm_decision_report --dataset EFNY --config configs/paths.yaml --draws 500 --tune 500 --chains 2 --seed 1")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("## 风险与注意事项（规划）")
    report_lines.append("- 层级 HDDM 对计算资源敏感：建议先用 `--max-files` 进行 pilot，确认模型可运行后再扩展到全样本。")
    report_lines.append("- 对条件极不均衡的任务（Color/Emotion Stroop），应避免单被试分层拟合，优先层级回归形式。")
    report_lines.append("- 对 switch 任务：主分析建议优先 Mixing-DDM（pure vs mixed），Switch-DDM 作为补充（仅 mixed）。")
    report_lines.append("")

    reports_root = _resolve_reports_root(paths_cfg, repo_root=repo_root)
    out_path = Path(args.report_out) if args.report_out else (reports_root / "ddm_decision.md")
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        raise SystemExit(
            f"Refusing to overwrite existing report: {out_path} (use --force, or run scripts.fit_ssm_task / scripts.fit_race4_task)."
        )
    out_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
