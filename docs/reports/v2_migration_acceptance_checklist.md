# v2 迁移验收清单

本清单用于判断是否可以从 v1 切换到 v2 主流程。

## A. 版本冻结（必须）

- [ ] 已执行 `python -m scripts.archive_legacy_snapshot --archive-name <name> --execute`。
- [ ] 已确认分支存在：`archive/<name>`。
- [ ] 已确认 tag 存在：`archive/<name>`。
- [ ] 已记录冻结点（分支、tag、commit hash）到会话日志。

## B. 数据契约（必须）

- [ ] 已固定 v2 输入行为表路径与版本（含 `run_id` / manifest）。
- [ ] 已固定主键字段（`subject_code` 或 `subid`）并文档化。
- [ ] 缺失值策略、异常值策略、过滤规则已明确且可复用。

## C. 功能最小闭环（必须）

- [ ] `python -m scripts.v2_run_pipeline --dataset EFNY --config configs/paths.yaml --dry-run` 成功。
- [ ] 非 dry-run 可生成 `outputs/results/v2/<run-id>/run.json`。
- [ ] profile 中样本量、列数、缺失率与预期一致。

## D. 与 v1 对照（建议）

- [ ] 抽样被试在 v1/v2 的关键字段一致（ID、任务映射、核心指标列）。
- [ ] 差异项已形成可解释清单（规则变化、历史错误修复、字段替换）。
- [ ] 至少 1 份差异报告落地到 `docs/reports/`。

## E. 切换门槛（发布前）

- [ ] 关键脚本入口已切到 `python -m scripts.<entry>` 形式。
- [ ] `docs/workflow.md`、`README.md`、`PROGRESS.md` 已同步更新。
- [ ] 迁移回滚路径已明确（回退到 `archive/<name>` 对应 tag）。
- [ ] 团队确认后再将 v2 设为默认执行路径。

