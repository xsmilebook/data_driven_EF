# Task-fMRI 行为数据说明（Psychopy CSV，THU/XY）

本文档用于规范说明 **task-fMRI 行为记录（Psychopy CSV）** 的文件组织、命名规则、关键字段与解析口径。该文档面向本仓库的 task 回归量构建脚本 `scripts/build_task_xcpd_confounds.py`，用于解释：

- 如何从 CSV 中识别“trial 行”、block/state 与事件（stimulus/response/banana）。
- SST 的 **120 试次**与 **180 试次**两种版本如何处理。
- THU 与 XY 由于 Psychopy 版本差异导致的表头差异，以及脚本的兼容方式。

## 1. 文件夹与文件命名

### 1.1 THU（Tsinghua）

- 根目录：`data/raw/MRI_data/task_psych/`
- 被试文件夹：`THU_YYYYMMDD_NUM_CODE`（例如 `THU_20231118_133_GYC`）
- CSV 文件名：通常包含任务名与时间戳，例如：
  - `THU_20231118_133_GYC_nback_2023-11-17_20h12.59.438.csv`
  - `THU_20230910_126_WYA_SST_2023-09-10_13h02.00.690.csv`

### 1.2 XY（Xiangya）

- 根目录：`data/raw/MRI_data/task_psych_xy/`
- 被试文件夹（Psychopy 原始）：`XY_YYYYMMDD_NUM_CODE`（例如 `XY_20240719_168_CTY`；末段 `CODE` 可能含下划线扩展，如 `XY_20240724_173_CY_156`）
- CSV 文件名：与文件夹一致，包含任务名与时间戳，例如：
  - `XY_20240719_168_CTY_nback_2024-07-19_18h36.55.518.csv`
  - `XY_20240719_168_CTY_SST_2024-07-19_18h50.01.614.csv`

### 1.3 时间戳字段（用于“多份记录取最新”）

当同一被试同一任务存在多个 CSV 时，本仓库的默认策略为：**按文件名中的时间戳选择最新**（`YYYY-MM-DD_HHhMM.SS.mmm`），而不是按文件修改时间（mtime）。

## 2. 共同前提：如何识别“trial 行”

Psychopy CSV 为宽表输出，除 trial 行外还可能包含 instruction、等待触发、block cue、结束 fixation 等行。为保证批处理一致性，本项目使用“任务关键刺激时间列非空”来识别 trial 行：

- **N-back / Switch**：`Trial_text.started` 非空 ⇒ 该行视为一次 trial（数字呈现时刻）。
- **SST**：`Trial_image_1.started` 非空 ⇒ 该行视为一次 trial（主要图片呈现时刻）。

注：THU 与 XY 的 Psychopy 版本不同，**XY 的 CSV 可能不包含 `Trial.started`**；因此 `Trial.started` 不应作为唯一判据。本仓库脚本已按上述列进行兼容。

## 3. 扫描起点与时间对齐（t0）

三个任务的 CSV 中通常包含 `MRI_Signal_s.started`，表示程序检测到扫描触发（或模拟触发）的时间点。本项目在构建 fMRI 回归量时使用以下优先级定义行为时间零点 `t0`：

1) 首个非空的 `MRI_Signal_s.started`；
2) 若缺失，回退到 `Begin_fix.started`；
3) 若仍缺失，使用 0。

## 4. 任务字段与解析口径

### 4.1 N-back

**trial 行判定**：`Trial_text.started` 非空。

常用字段：

- `Trial`：呈现的数字。
- `Ans`：正确按键编码（常见为 `1=左键`，`2=右键`）。
- `key_resp.keys / key_resp.rt`：被试按键与反应时。
- `Trial_loop_list`：trial 列表来源（Excel 列表文件），用于识别 block（最稳健）。

**block/state 识别**：优先使用 `Trial_loop_list` 中包含的关键字（例如 `0back`/`2back`）判定 block 类型；若无法判定则回退为 `state_mixed`。

**事件（event）定义**：

- stimulus onset：`Trial_text.started - t0`
- response onset：`(key_resp.started + key_resp.rt) - t0`（两者均非空时）

### 4.2 Switch

**trial 行判定**：`Trial_text.started` 非空。

常用字段：

- `Cond_img`：规则/边框颜色（如 red/blue）。
- `Font`：字体大小（例如 0.2/0.4）。
- `Trial`：数字。
- `Ans`：正确按键编码。
- `Trial_loop_list`：列表文件名，用于区分 `nonswitch1`、`nonswitch2`、`switch` 三段。

**block/state 识别**：

- `nonswitch1` ⇒ `state_pure_red`
- `nonswitch2` ⇒ `state_pure_blue`
- `switch` ⇒ `state_mixed`

**事件（event）定义**：

- stimulus onset：`Trial_text.started - t0`
- response onset：`(key_resp.started + key_resp.rt) - t0`

### 4.3 SST（Stop-Signal Task）

**trial 行判定**：`Trial_image_1.started` 非空。

常用字段：

- `Position`：刺激/香蕉位置（左/右）。
- `bad`：是否出现“黑香蕉/no-go”（XY 示例中为 `material/Banana_1.png`）。
- `Trial_image_1.started`：主要图片出现时刻（作为 stimulus onset）。
- `Trial_image_3.started`：覆盖层/香蕉组件出现时刻（用于 banana 事件 onset）。
- `key_resp.*`：按键与反应时。

**事件（event）定义**：

- stimulus onset：`Trial_image_1.started - t0`
- banana onset：当 `bad` 字段包含 `banana` 或 `Banana` 时，取 `Trial_image_3.started - t0`
- response onset：`(key_resp.started + key_resp.rt) - t0`

**block/state 版本差异（THU 需显式纳入）**：

- **120 试次版本**：单段（`state_part1`）。
- **180 试次版本**：两段（`state_part1`/`state_part2`）。常见现象为：第 90 与 91 试次之间存在约 15s 的静息注视间隔。
  - 若存在 `Trial_loop_list`，可出现可识别的 loop1/loop2（或等价命名）；部分数据每个 loop 可能包含 1 行“无刺激”记录（不产生 stimulus 事件）。
  - 若缺少清晰 loop 标记，可通过试次数量或中途长间隔作为分段依据。

## 5. THU vs XY：Psychopy 版本差异与脚本兼容

THU 与 XY 的行为 CSV 在“数据语义”上相同（都记录 trial、刺激与反应时序），但由于 Psychopy 版本/程序实现不同，表头存在差异：

- XY 中可能缺失 `Trial.started`，但仍包含 `Trial_text.started`（nback/switch）或 `Trial_image_1.started`（sst）。
- XY 的 N-back 可能缺失 `Task_img`，但可用 `Trial_loop_list` 识别 0-back/2-back block。
- XY 的 SST 可能缺失 `Trial_loop_list`，但 `Trial_loop.thisTrialN` 与刺激时间列可支持试次统计与分段推断。

本项目脚本 `scripts/build_task_xcpd_confounds.py` 采用 “任务关键时间列非空” 的方式识别 trial 行，并用任务相关的 start/stop 列推断 block 起止时间，从而实现 THU/XY 共用同一套处理逻辑。

## 6. 批处理建议与常见检查

建议在批处理前做以下检查（可用于快速定位格式异常/重复记录）：

- 同一被试同一任务是否存在多个 CSV：可用 `temp/find_repeated_task_psych_csvs.py` 输出汇总表。
- trial 行数量是否符合预期（例如 nback=120、switch=144、sst=120 或 180）。
- `MRI_Signal_s.started` 是否存在且位置合理（用于 `t0`）。
