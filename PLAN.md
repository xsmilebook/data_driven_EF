# Plan

## 实现 THU app SSM/DDM 阶段

目标：在现有 `check_format -> clean -> metrics` 行为预处理之后，新增独立
SSM/DDM 建模阶段。该阶段读取 `app_trials_clean.csv`，按
`docs/reports/ddm_decision.md` 构造各模型入模 trial 表，调用 HSSM 拟合并导出
posterior trace、summary 和 QC。

执行项：

- 使用 `uv` 新增 HSSM 相关依赖；若 Python 3.12 下失败，则将项目 `.venv`、
  `.python-version`、`pyproject.toml` 和 `uv.lock` 切换到 Python 3.11。
- 在 `configs/behavioral_metrics.yaml` 中新增 `ssm_models`，集中记录模型清单、
  HSSM 模型名、公式、`p_outlier` 和 pilot/full 采样参数。
- 新增 `src/behavior/app/ssm.py`，实现 trial 构造、模型规格读取、HSSM 拟合、
  posterior summary、trial QC 和 trace 写出。
- 新增 `scripts/behavior/app_ssm.py`，提供 `--model`、`--mode`、`--max-subjects`、
  `--draws`、`--tune`、`--chains`、`--seed` 参数。
- DCCS 同时实现整体模型和 block 模型；N-back 按 number/spatial/emotion
  三组建模且不估计 `target_type`；DT/EmotionSwitch 跨规则响应保留，并用
  先验均值为 `0.05` 的 `p_outlier ~ Beta(5, 95)` 估计异常响应比例。
- 更新 README、`docs/workflow.md`、`docs/reports/ddm_decision.md` 和会话日志。
- 增加单元测试覆盖 DCCS block、N-back domain、CPT response trials 与
  DT 跨轴响应保留。
- 使用 `temp/smoke_app_behavior` 做非持久化冒烟验证，并清理临时产物。

约束：

- v1 仅实现 THU `app_data`，不扩展 XY、BNU、inventory、demography 或 task-fMRI。
- 不自动运行全样本长采样；全量模型需通过 `--mode full` 显式触发。
- 不修改 `data/` 或 `outputs/` 下运行产物。
