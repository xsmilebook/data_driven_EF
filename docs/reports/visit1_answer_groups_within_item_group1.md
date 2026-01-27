# 基于 item group_001 的 visit1 答案分组检查

目的：在 item group_001（作为 visit1 刺激序列的基线候选）内部，检查这些被试被分配到了哪些 answer group，并对这些 answer group 的代表被试做任务级答案序列对比（以 answer group_001 为基准）。

说明：
- 这里只做 **answer 序列类型差异定位**，不直接断言哪一组一定是“正确答案”。
- 对比仅使用代表被试（每个 answer group 在 item group_001 内取 1 名代表），用于快速定位“差异集中在哪些任务”。

## item group_001 内的 answer group 分布

| answer_group | n_subjects_in_item_group1 | representative_subject_id |
| --- | --- | --- |
| group_001 | 308 | THU_20231203_144_LSC |
| group_003 | 27 | THU_20231014_131_ZXM |
| group_005 | 31 | THU_20240831_390_ZJS |
| group_006 | 25 | THU_20250120_521_SRL |

## 代表被试的任务级答案差异（相对 answer group_001）

对每个任务，将代表被试的 `正式阶段正确答案` 序列与 answer group_001 的代表被试做逐位比较（对齐到共同长度），计算 `match_ratio`。仅列出 `match_ratio < 1.0` 的任务。

| answer_group | tasks_compared | tasks_mismatch |
| --- | --- | --- |
| group_003 | 16 | 1 |
| group_005 | 18 | 0 |
| group_006 | 18 | 1 |

### 具体差异任务清单（match_ratio < 1.0）

| answer_group | task | match_ratio | n_overlap |
| --- | --- | --- | --- |
| group_003 | EmotionStroop | 0.990 | 96 |
| group_006 | EmotionStroop | 0.990 | 96 |

解释：
- 目前观察到的差异集中在 `EmotionStroop`，且幅度很小（96 个位置中约 1 个位置不同）。
- 这提示：在 item group_001 这一批“刺激序列一致”的被试中，存在少量答案序列类型差异，后续需要回到 trial 级别（按日期窗口、设备、是否 web/txt 导出等）逐个核查。

