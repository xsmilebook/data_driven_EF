# Plan（APP 数据问题分析与推断修正）

针对 `data/raw/behavior_data/cibr_app_data/` 的 APP 行为工作簿，系统化盘点“刺激序列/正确答案序列来源不一致、序列多次更换、后台下载覆盖、崩溃期参照被试替换”等问题，并通过“版本库 + 相似度匹配 + 时间先验 + 试次级一致性诊断”的推断流程，生成一个**刺激序列与正确答案均被更正**的新数据集（新建输出目录，不覆盖原始文件），同时产出可复现实验记录与异常统计报告。

## Scope
- In: 对 APP 工作簿的结构/内容异常进行全样本统计（含任务级与试次级），形成 `docs/reports/` 报告与机器可读的 audit 表。
- In: 构建 visit1–visit4（含 visit3 子版本：`2025/07/09` FZSS、`2025/07/17` spatial/emotion-nback、`2025/07/29` Emotion1Back 答案修正）的序列版本库，并对每个被试×任务推断“实际使用的刺激/答案版本”。
- In: 基于推断结果生成“更正后的刺激序列与正确答案序列”数据集（新建目录保存；保留原始字段以便可追溯），并给出明确的修正/替换/标记规则与置信度。
- Out: 覆盖或回写 `data/raw/` 原始工作簿；与 APP 数据清洗无关的模型/影像流程改动；对历史序列的主观改写（无证据支持时仅标记为不确定而不强行更正）。

## Action items
[ ] 盘点并量化全样本异常：解析全部工作簿与所有 sheet，统计列缺失/列名变体/行数异常（含 SST 97 行）、空行/重复行、任务缺失等；输出 audit 表（CSV/JSON）与图表/摘要到 `docs/reports/`（报告需给出“发现了多少例、占比多少、集中在哪些时间段/设备批次”的具体数字）。
[ ] 将序列更换时间线固化为配置：新增 `configs/` 下的时间线配置（按你提供的确认时间点：项目初期 visit1；`2025/07/08` visit3；`2025/07/09` FZSS；`2025/07/17` spatial/emotion-nback；`2025/07/29` Emotion1Back 答案；`2025/11/28` visit2；`2025/12/16` visit1；`2026/01/13` visit4），并明确哪些变更是“全任务一致” vs “单任务子版本”。
[ ] 构建“候选序列版本库”：基于现有分组（`data/interim/behavioral_preprocess/groups_by_item/` 与 `.../groups_by_answer/`）与时间线先验，抽取高置信代表被试，分别为每个任务构建候选的刺激序列与答案序列版本（保存为可追溯的 JSON/CSV 版本库，并记录来源被试与生成规则）。
[ ] 实现版本推断器：对每个被试×任务同时计算（a）刺激列序列与版本库的相似度，（b）答案列序列与版本库的相似度，并结合被试日期（从 `THU_YYYYMMDD_...` 推断）作为先验；输出 `subject×task` 级别的 `item_version`/`answer_version`/置信度/冲突类型。
[ ] 将 4 类导出问题显式可判定化并量化：对你描述的 1–4 类情形定义可复现的判别规则（如：刺激版本与答案版本不一致=疑似后台覆盖；崩溃期=刺激列在多个被试间高度相同且与日期先验不符；同日多设备差异=同一时间窗内出现多个答案版本等），并给出全样本数量与示例被试列表。
[ ] 推断并更正刺激序列：默认以“设备端序列”作为目标（由答案版本/日期先验推断得到），当检测到后台覆盖/参照替换时用版本库中的对应刺激序列替换刺激列；保留原刺激列为 `*_raw` 并写入 `*_corrected`，同时记录替换原因与证据（匹配分数/时间先验）。
[ ] 推断并更正正确答案（含试次级修正）：以版本库答案序列为基准，对“疑似少量试次答案配置错误”的情况，结合被试按键与群体正确率分布定位异常试次；对可确定的错误进行修正，对不可确定的试次标记 `answer_uncertain=1`（不强行修改），并输出异常试次清单。
[ ] 生成更正后数据集（新目录、不覆盖）：定义输出数据规范（建议长表：`subject_id, task, trial, item_raw, answer_raw, item_corrected, answer_corrected, version_item, version_answer, flags...`；或同时输出“与原 Excel 同结构”的更正版工作簿），写入 `data/processed/` 下新建目录（路径由 `configs/paths.yaml` 配置），并生成总 manifest（含每位被试的推断结果与异常概览）。
[ ] 验证与回归：在更正后数据集上重新执行“按刺激分组/按答案分组”，验证分组一致性显著提升；对关键案例做 spot-check（如 SST 97 行个案、后台覆盖个案、崩溃期个案），并在报告中给出“更正前后”的量化对比（组数、冲突率、异常试次数等）。
[ ] 文档化与后续正式脚本路线图：在 `docs/reports/` 补充更清晰的阐述（问题机制图、诊断逻辑、推断策略、风险与边界条件、人工复核入口），并明确下一步要实现的“正式统一清洗脚本”（放置位置、CLI 入口、与配置的关系、以及如何复现当前推断版数据集）。

## Open questions
- 是否存在“每个 visit 的官方刺激序列与标准答案”的原始配置文件/后台导出快照（可作为版本库的 ground truth），还是需要完全从现有工作簿中推断构建版本库？
- 更正后数据集的目标输出形式希望是什么：保持每被试一个 Excel（最大兼容性）还是统一长表 CSV/Parquet（最利于下游分析与建模）？
- 崩溃期“参照被试”具体是哪一位（或参照文件名模式）？以及崩溃期的更精确时间范围（起止日期）是否可以提供以提高判别准确性？

---

# Legacy Plan (DDM/HSSM)

Update the EFNY decision+compute pipeline so each task is modeled with an assumption-consistent choice model (2AFC DDM only when valid), run hierarchical PyMC-based fits on the full sample via SLURM, and write back a single evidence-backed decision report under `docs/reports/`. Centralize all I/O via `configs/paths.yaml`, persist posterior traces for reproducibility, and save downloadable result tables under `data/processed/table/metrics/`.

## Scope
- In: DDM decision document (Chinese) + runnable scripts to compute the table’s model summaries + lightweight evaluation metrics (fit quality + condition effects) and report integration.
- In: True hierarchical PyMC-based models (HSSM), including random effects for `v/a/t/z` (DDM) and model comparisons (null vs effect) with LOO/WAIC.
- In: SLURM submission scripts (user submits) for multi-task parallel runs and multi-core single-task runs; posterior traces saved (netcdf).
- Out: Refactors unrelated to behavioral modeling, changes to repository structure, modifications to runtime artifacts already under `data/`/`outputs/` beyond the new results produced by these scripts.

## Action items
[ ] Audit task choice-sets on the full sample and enforce model eligibility (2AFC DDM only after recoding; multi-choice tasks use LBA), updating `docs/reports/ddm_decision.md`.
[ ] Implement 2-choice recoding datasets for `DT` (axis: `left/right` vs `down/up`, with `rule=axis`) and `EmotionSwitch` (dimension: `female/male` vs `happy/sad`, with `rule=dimension`), including logging/excluding cross-axis/dimension responses as `lapse/outlier`.
[ ] For `DT` and `EmotionSwitch`, fit two parallel hierarchical DDMs and compare them (Model A vs Model B), with condition effects on `v/a/t0`:
[ ] - Model A (Mixing): `v ~ block + rule + block:rule`, `a ~ block + rule + block:rule`, `t0 ~ block + rule + block:rule` (pure+mixed).
[ ] - Model B (Switch): `v ~ trial_type + rule + trial_type:rule`, `a ~ trial_type + rule + trial_type:rule`, `t0 ~ trial_type + rule + trial_type:rule` (mixed only).
[ ] Implement 4-choice multi-choice SSM for `ColorStroop` and `EmotionStroop` (HSSM: `race_no_z_4` as a practical substitute for LBA4), and generate report-level collapsed choice rates: Target / Word / Other.
[ ] Implement `SST` go-only 2AFC hierarchical DDM (and keep SSRT/race modeling explicitly out-of-scope for the DDM table unless separately requested).
[ ] Standardize hierarchical model specification: subject random effects for `v/a/t/z` (DDM); condition effects on `v` by default, plus task-specific extensions (DT/EmotionSwitch: `a` and `t0` also vary by condition; FLANKER optional sensitivity: `a ~ congruency`); consistent priors/sampling settings in `configs/analysis.yaml`.
[ ] Add per-task model comparison (null vs effect; mixing+switch joint model where applicable) using LOO/WAIC and posterior predictive checks; write compact metrics tables for downstream plotting.
[ ] Persist outputs for cluster-to-local workflow: save `InferenceData` posterior traces (`.nc`) and lightweight summaries under `data/processed/table/metrics/` (paths resolved via `configs/paths.yaml`).
[ ] Add SLURM scripts patterned after `scripts/submit_hpc_real.sh` to run an array over `task×model` (and optionally `axis/dimension`), controlling threads/cores and writing logs to `outputs/logs/`.
[ ] Regenerate `docs/reports/ddm_decision.md` so it embeds the full-sample results tables and clearly flags any tasks still blocked by missing stimulus-to-choice mappings.
[ ] Update `docs/workflow.md`, `PROGRESS.md`, and add a dated session log under `docs/sessions/` describing how to run locally vs on SLURM and where artifacts are saved.

## Open questions
- For `EmotionStroop`, where is the trial-level mapping for “target category” vs “word-induced category” stored (extra columns in Excel vs an external stimulus dictionary)?
- For `DT/EmotionSwitch`, should cross-axis/dimension responses be excluded (recommended) or represented via a separate multi-choice model as a robustness check?
- What is the intended full-sample sampling budget on SLURM (draws/tune/chains; per-task walltime), and should we prioritize more chains or longer chains?
