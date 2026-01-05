# Task fMRI 的 XCP-D 处理流程（当前实现）

本文档说明本仓库中 **task-fMRI** 的 xcp-d（`xcp_d-0.10.0`）后处理流程：在完成 fMRIPrep 的前提下，执行去噪与滤波，并额外回归任务诱发信号（block/state + event），以获得更接近 resting-state 的残差时序用于后续分析。

## 1. 目标与总体思路

### 1.1 目标

对 task-fMRI 的每个被试（每个任务）：

1) 执行 xcp-d 的标准去噪（以 `36P` 为核心策略）。  
2) 在去噪同时，将任务诱发共激活信号作为额外 confounds 回归，从而降低 task-evoked HRF 对时序的影响。  

### 1.2 实现路径

本仓库当前实现采用 `xcp_d-0.10.0` 支持的 **custom confounds dataset + YAML 配置** 机制：

- 先从 Psychopy 行为 CSV 生成一份与该 run 的 fMRIPrep confounds TSV **同名**的自定义 confounds TSV（每个 TR 一行，每列一个任务回归量）。
- 再调用 xcp-d：
  - `--datasets custom=<folder>`：挂载自定义 confounds 的 BIDS derivative dataset；
  - `--nuisance-regressors <yaml>`：使用 YAML confounds 配置（36P + task regressors）。

## 2. 依赖与输入

### 2.1 依赖

- xcp-d 容器：`xcp_d-0.10.0`
- 前置处理：fMRIPrep 已完成（task-fMRI 对应的 derivatives）
- Psychopy 行为记录：`data/raw/MRI_data/task_psych/`
- FreeSurfer subjects dir（可选但建议）：`/ibmgpfs/cuizaixu_lab/liyang/BrainProject25/Tsinghua_data/freesurfer`（包含 `sub-*/surf,label,mri,...`）

### 2.2 输入目录与配置

任务数据集的外部输入通过配置集中管理：

- 配置文件：`configs/dataset_tsinghua_taskfmri.yaml`
- 关键字段：
  - `external_inputs.task_psych_dir`：Psychopy 行为 CSV 根目录（相对 `data/raw/` 或绝对路径）
  - `external_inputs.fmriprep_task_dirs.{nback,sst,switch}`：各任务的 fMRIPrep 根目录（通常为集群绝对路径）

## 3. 脚本入口与执行流程

### 3.1 单被试单任务（SLURM job）

- 脚本：`src/imaging_preprocess/xcpd_task_36p_taskreg.sh`
- 调用方式（示例）：

```bash
sbatch src/imaging_preprocess/xcpd_task_36p_taskreg.sh THU20231119141SYQ nback
```

此外，为便于在同一集群环境中快速分享“拿来就跑”的版本，本仓库提供了写死路径的等价脚本：

- `temp/run_xcpd_task_direct.sh`（不依赖 `configs/` 与 `scripts.render_paths`，直接内联当前配置的路径）
- `temp/batch_run_xcpd_task_direct.sh`（批量提交版本，内部调用 `temp/run_xcpd_task_direct.sh`）

该脚本包含两个关键阶段：

1) **构建 task 回归量 confounds**：调用 `python -m scripts.build_task_xcpd_confounds`。  
2) **运行 xcp-d**：通过 `--datasets custom=...` + `--nuisance-regressors confounds_config.yml` 注入 task regressors，并在 `--mode none` 下显式指定滤波与运动参数。

### 3.2 批量提交

- 脚本：`src/imaging_preprocess/batch_run_xcpd_task.sh`
- 示例：

```bash
bash src/imaging_preprocess/batch_run_xcpd_task.sh data/processed/table/sublist/taskfmri_sublist.txt "nback sst switch"
```

该脚本将对被试列表×任务列表进行笛卡尔组合提交 SLURM jobs，并把 stdout/stderr 写入 `outputs/logs/<DATASET>/xcpd_task/`。

## 4. task 回归量（custom confounds）如何生成

### 4.1 输出文件与命名约束（关键）

`xcp_d-0.10.0` 的 custom confounds dataset 机制要求：

- `<folder>` 本身是一个 BIDS-derivatives dataset（包含 `dataset_description.json`）；
- TSV 位于 `<folder>/sub-<label>/func/`；
- TSV 文件名必须与该 run 的 fMRIPrep confounds TSV 的 **basename 完全一致**（例如 `sub-XXX_task-sst_desc-confounds_timeseries.tsv`）。

当前生成脚本：

- `scripts/build_task_xcpd_confounds.py`

其输出为：

- `<out-root>/dataset_description.json`
- `<out-root>/confounds_config.yml`（36P + task 列名）
- `<out-root>/sub-<label>/func/<basename>`：与 fMRIPrep `*_desc-confounds_timeseries.tsv` 同名的 TSV，包含 task regressors（每 TR 一行）。

### 4.2 时间对齐与 TR/volume 读取

生成脚本的关键对齐要点：

- TR：从 fMRIPrep 的 BOLD sidecar JSON 读取 `RepetitionTime`。
- volume 数：由对应 `*_desc-confounds_timeseries.tsv` 的行数决定。
- 行为时间零点：优先使用 Psychopy CSV 的首个 `MRI_Signal_s.started` 作为扫描起点；缺失时回退到 `Begin_fix.started`，再回退到 0。

### 4.3 block/state 回归量（canonical HRF）

block/state 回归量旨在回归段落级别（状态级别）的任务诱发信号，计算方式为：

1) 构建 boxcar：处于对应 block/state 时为 1，否则为 0；  
2) 与 canonical HRF（SPM HRF）卷积；  
3) 下采样到 TR 网格（每 TR 取均值）。

当前任务的 block/state 定义：

- NBACK：
  - `state_pure_0back`：通过 `Trial_loop_list`/`Task_img` 中的 `0back` 标识推断
  - `state_pure_2back`：通过 `2back` 标识推断
  - 若出现混合段：`state_mixed`
- SWITCH：
  - `state_pure_red`：`nonswitch1`
  - `state_pure_blue`：`nonswitch2`
  - `state_mixed`：`switch`
- SST：
  - 若行为日志为 120 试次：仅生成单个 block（`state_part1`）。
  - 若行为日志为 180 试次：生成两个 block（`state_part1`/`state_part2`），且在第 90 与 91 试次之间通常存在约 15 s 的注视间隔。
  - 若存在 `Trial_loop_list`：优先使用 `loop1/loop2`（或等价命名）识别两个 block；每个 loop 可能包含 1 行“无刺激”记录（不产生 stimulus 事件，但会影响 block 持续时间）。

### 4.4 event 回归量（FIR）

event regressors 用于回归事件相关的诱发成分，采用 FIR（finite impulse response）实现：

- 将每个事件 onset 落到其对应 TR bin 上，以“事件计数序列”为起点；
- 生成 `delay_0..delay_K` 的离散滞后列，覆盖从 0 到 `fir_window_seconds` 的响应窗口；
- `K` 由 `fir_window_seconds / TR` 决定（向下取整并包含 0）。

当前事件类型：

- 共通：
  - `stimulus`：刺激呈现 onset（SST 使用 `Trial_image_1.started`；NBACK/SWITCH 使用 `Trial_text.started`）
  - `response`：按键 onset = `key_resp.started + key_resp.rt`
- SST 额外：
  - `banana`：当 `bad` 列包含 `banana` 时，以 `Trial_image_3.started` 作为该事件 onset

## 5. xcp-d 参数（当前设置）

当前 task-fMRI 的 xcp-d 调用核心参数为（概念上）：

- `--input-type fmriprep`
- `--mode none`
- `--datasets custom=/custom_confounds`
- `--nuisance-regressors /custom_confounds/confounds_config.yml`
- `--despike`
- `--lower-bpf=0.01 --upper-bpf=0.1`
- `--motion-filter-type lp --band-stop-min 6`
- `--fd-thresh -1`（禁用基于 FD 的 scrubbing）

## 6. 输出目录结构

### 6.1 xcp-d 输出

- `data/interim/MRI_data/xcpd_task/<task>/`（xcp-d 标准 derivatives 输出结构）

### 6.2 自定义 confounds（task regressors）

- `data/interim/MRI_data/xcpd_task/custom_confounds/<task>/sub-<label>/`
  - `<basename>.tsv`：task regressors（必须与 fMRIPrep confounds TSV 同名）

## 7. 常见报错与诊断

### 7.1 `dataset_description.json is missing from project root`

原因：xcp-d/pybids 会把输入目录当作 BIDS dataset root；若缺失 `dataset_description.json` 会直接报错。  
处理：`xcpd_task_36p_taskreg.sh` 会在工作目录创建临时 wrapper（仅包含 `dataset_description.json`），并将 `sub-<label>` bind 到 wrapper 下以通过 BIDS 校验，从而避免向原始 fMRIPrep 目录写入文件。

### 7.2 自定义 confounds 不生效（或整列几乎全 0）

优先检查：

1) 自定义 confounds TSV 是否位于 `<out-root>/sub-<label>/func/`，且文件名是否与 fMRIPrep confounds TSV basename 完全一致；  
2) TR 与 volume 数是否与 fMRIPrep 对齐；  
3) 行为时间零点（`MRI_Signal_s.started`）是否对应真实扫描起点；  
4) 是否选择了正确的行为 CSV（同一被试同任务可能存在多个记录）。
