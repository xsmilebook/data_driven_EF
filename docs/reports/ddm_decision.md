# DDM/HDDM 决策文档（PyMC-based）

## 输入与范围
- 输入目录：`D:/projects/data_driven_EF/data/raw/behavior_data/cibr_app_data`
- 文件模式：`*.xlsx`
- 本次运行文件数：10

## 决策矩阵（依据任务表）
| Task | 纳入试次 | 层级HDDM | 模型（主推） | 备注/风险点 |
| --- | --- | --- | --- | --- |
| FLANKER | 全部 (96; 48C/48I) | 可选（推荐做统一管线） | HDDM/层级DDM：v ~ congruency（必要时 a ~ congruency） | 条件均衡，最稳定；也可做单被试分层拟合（不推荐作为主结论）。 |
| ColorStroop | 全部 (96; 24C/72I) | 是 | HDDM/层级DDM：v ~ congruency（不建议单被试分层拟合） | 条件极不均衡；congruent 少，必须用 predictor 思路。 |
| EmotionStroop | 全部 (96; 24C/72I) | 是 | HDDM/层级DDM：v ~ congruency（不建议单被试分层拟合） | 同上。 |
| DT | pure 64 + mixed 64 = 128；switch 仅 mixed | 是（强烈建议） | 两层策略：① Mixing-DDM：v ~ block_type(pure/mixed)（用 128）；②（辅）Switch-DDM：v ~ trial_type(R/S)（仅 mixed） | 更适合“完整 DDM 分解”：mixing 主打，switch 可做补充。 |
| EmotionSwitch | pure 64 + mixed 64 = 128；switch 仅 mixed | 是（强烈建议） | 两层策略：① Mixing-DDM：v ~ block_type(pure/mixed)（用 128）；②（辅）Switch-DDM：v ~ trial_type(R/S)（仅 mixed） | mixed 内 switch 较少；建议只让 v 随条件变，避免多参数一起变。 |
| DCCS | pure 20 + mixed 20 = 40；mixed 内 10R/10S | 否（或仅探索） | 不建议做 switch-DDM；如一定要做，仅做整体 DDM 或 v ~ block_type | mixed 内每格 10 太少；switch-DDM 基本不可识别。 |
| SST | 主模型用 go+stop（race/SSRT）；DDM 仅用 go RT | ①否（标准 DDM）②可选（Go-DDM） | Stop-signal race/SSRT（integration 等）；可选补充：Go-DDM（HDDM）仅用 go=72 | 抑制过程更适合 race/SSRT；Go-DDM 解释“go 决策”，仅作补充。 |
| GNG / CPT | Go/NoGo（NoGo 无 RT） | 否 | 不做标准 DDM；用 SDT(d'+c) + commission/omission（可加 time-on-task） | 单键 Go/NoGo 不符合标准 2AFC-RT DDM；NoGo 无 RT。 |
| N-back 系列 | 60（match 很少） | 否（不建议） | SDT(d'+c) + RT/ACC + 负荷/域比较 | match 太少；分层 DDM 不可行，整体 DDM 解释弱。 |

## 本次模型计算设置
- 后端：HSSM（PyMC-based hierarchical SSM） + `nuts_numpyro` 采样
- draws=40, tune=40, chains=1, seed=1
- 层级结构：仅对 drift rate (`v`) 引入被试随机截距（`(1|subject)`）；`a`/`t` 采用群体固定效应（计算可行性优先）。
- 说明：仅将结果嵌入本文档，不写出 CSV。

## 计算结果摘要（效果评估）
| Task | Model | N_subjects | N_trials | Beta(v_congruency) | Beta(v_block_mixed) | Beta(v_is_switch) | LOO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FLANKER | HDDM (hier): v ~ congruency | 10 | 960 | -0.257 [-0.501, -0.106], P(>0)=0.00 |  |  | elpd_loo=495.2, p_loo=17.1 |
| ColorStroop | HDDM (hier): v ~ congruency | 10 | 960 | -0.568 [-0.702, -0.421], P(>0)=0.00 |  |  | elpd_loo=-303.2, p_loo=13.8 |
| EmotionStroop | HDDM (hier): v ~ congruency | 10 | 960 | -0.274 [-0.432, -0.136], P(>0)=0.00 |  |  | elpd_loo=-1339.4, p_loo=12.7 |
| DT | HDDM (hier): v ~ block_mixed | 10 | 1280 |  | -0.348 [-0.463, -0.242], P(>0)=0.00 |  | elpd_loo=-1772.3, p_loo=13.8 |
| EmotionSwitch | HDDM (hier): v ~ block_mixed | 10 | 1280 |  | -0.436 [-0.572, -0.309], P(>0)=0.00 |  | elpd_loo=-1582.0, p_loo=13.8 |
| DT | HDDM (hier): v ~ is_switch (mixed only) | 10 | 640 |  |  | -0.179 [-0.329, -0.028], P(>0)=0.03 | elpd_loo=-1016.4, p_loo=16.6 |
| EmotionSwitch | HDDM (hier): v ~ is_switch (mixed only) | 10 | 640 |  |  | -0.320 [-0.473, -0.140], P(>0)=0.00 | elpd_loo=-911.9, p_loo=14.3 |

## 复现方式
```bash
python -m scripts.ddm_decision_report --dataset EFNY --config configs/paths.yaml --max-files 10 --draws 40 --tune 40 --chains 1 --seed 1
```

## 全样本运行建议（规划）
- 先完成小规模 pilot（例如 `--max-files 10`），确认依赖与模型可跑通、参数符号与数量级合理。
- 全样本层级 HDDM 计算耗时长，建议在可用计算节点上运行，并逐步提高采样长度（如 draws≥500, tune≥500, chains≥2）。
```bash
python -m scripts.ddm_decision_report --dataset EFNY --config configs/paths.yaml --draws 500 --tune 500 --chains 2 --seed 1
```

## 风险与注意事项（规划）
- 层级 HDDM 对计算资源敏感：建议先用 `--max-files` 进行 pilot，确认模型可运行后再扩展到全样本。
- 对条件极不均衡的任务（Color/Emotion Stroop），应避免单被试分层拟合，优先层级回归形式。
- 对 switch 任务：主分析建议优先 Mixing-DDM（pure vs mixed），Switch-DDM 作为补充（仅 mixed）。
