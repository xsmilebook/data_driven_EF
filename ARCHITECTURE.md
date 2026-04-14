# Project Structure

```
data_driven_EF/                 # 项目根目录
├── README.md                   # 项目简介与快速入口
├── AGENTS.md                   # AI 协作与修改约束
├── PLAN.md                     # 计划与任务拆解
├── PROGRESS.md                 # 进度记录
├── ARCHITECTURE.md             # 结构说明（本文）
├── configs/                    # 项目配置文件
│   ├── paths.yaml              # 路径与数据集相关设置（单一数据集 section）
│   ├── analysis.yaml           # 建模与评估默认参数
│   └── behavioral_metrics.yaml # 行为指标计算与下游使用指标配置
│
├── docs/                       # 方法、流程与决策记录
│   ├── data_dictionary.md      # 数据字典（数据说明）
│   ├── workflow.md             # 可复现流程
│   ├── methods.md              # 方法学细节
│   ├── reports/                # 研究计划与阶段性总结
│   └── sessions/               # 会话记录
│
├── src/                        # 可复用模块（不含硬编码路径）
│   ├── common.py               # 薄通用工具：读写表格、ID 规范化、列名清洗、基础校验
│   ├── imaging/                # 影像预处理与影像指标提取
│   │   ├── fmriprep/           # fmriprep 提交、产物收集、QC
│   │   ├── xcpd/               # xcpd 提交、产物收集、QC
│   │   └── connectivity/       # 功能连接等影像指标提取与后处理
│   └── behavior/               # 行为、量表与人口学预处理
│       ├── app/                # app 行为任务预处理
│       │   ├── check_format.py # 检查原始表格格式、列名与关键字段
│       │   ├── clean.py        # 试次级/被试级清洗
│       │   ├── metrics.py      # 指标计算
│       │   ├── task_registry.py# task 名称、文件匹配与调度
│       │   └── tasks/          # 18 个 task 的特异规则
│       ├── task_fmri/          # task fMRI 行为日志预处理
│       │   ├── check_format.py # 检查 Psychopy 等原始记录格式
│       │   ├── clean.py        # 试次级/被试级清洗
│       │   ├── metrics.py      # 指标计算
│       │   └── tasks/          # 各 task 的特异规则
│       ├── inventory/          # 行为量表预处理
│       │   ├── check_format.py # 检查量表原始表结构
│       │   ├── clean.py        # 清洗与统一编码
│       │   ├── score.py        # 量表计分与因子计算
│       │   └── scales/         # 各量表的计分规则（如 BRIEF）
│       ├── demography/         # 人口学信息预处理
│       │   ├── check_format.py # 检查字段完整性与编码
│       │   ├── clean.py        # 清洗与统一编码
│       │   └── derive.py       # 派生变量（如年龄分组）
│       └── merge.py            # 合并行为、量表、人口学结果
│
├── scripts/                    # 执行入口（本地/HPC/主流程脚本）
│   ├── imaging/
│   │   ├── submit_fmriprep.py  # 提交 fmriprep 任务
│   │   ├── collect_fmriprep.py # 汇总 fmriprep 产物与日志
│   │   ├── submit_xcpd.py      # 提交 xcpd 任务
│   │   ├── collect_xcpd.py     # 汇总 xcpd 产物与日志
│   │   └── extract_connectivity.py
│   └── behavior/
│       ├── app_check_format.py
│       ├── app_clean.py
│       ├── app_metrics.py
│       ├── task_fmri_check_format.py
│       ├── task_fmri_clean.py
│       ├── task_fmri_metrics.py
│       ├── inventory_check_format.py
│       ├── inventory_clean.py
│       ├── inventory_score.py
│       ├── demography_check_format.py
│       ├── demography_clean.py
│       └── build_analysis_table.py
│
├── data/                       # 项目分析数据（不纳入版本控制），用于存放数据型对象：预处理结果、connectivity matrix、analysis-ready tables
│   ├── external/               # 外部第三方输入
│   ├── raw/                    # 原始输入（非脚本产出）
│   │   ├── THU/
│   │   │   ├── app_data/       # 原始的行为任务数据
│   │   │   ├── inventory/      # 原始的量表数据
│   │   │   ├── bids/
│   │   │   └── metadata/
│   │   ├── XY/
│   │   │   ├── app_data/       # 原始的行为任务数据
│   │   │   ├── inventory/      # 原始的量表数据
│   │   │   ├── bids/
│   │   │   └── metadata/
│   │   └── BNU/
│   │       ├── app_data/       # 原始的行为任务数据
│   │       ├── inventory/      # 原始的量表数据
│   │       ├── bids/
│   │       └── metadata/
│   ├── interim/                # 中间衍生结果
│   │   ├── THU/
│   │   │   ├── fmriprep/
│   │   │   ├── xcpd/
│   │   │   └── qc/
│   │   ├── XY/
│   │   │   ├── fmriprep/
│   │   │   ├── xcpd/
│   │   │   └── qc/
│   │   └── BNU/
│   │       ├── fmriprep/
│   │       ├── xcpd/
│   │       └── qc/
│   └── processed/              # 可复用清洗结果
│       ├── THU/
│       │   ├── functional_connectivity/
│       │   ├── behavioral_metrics/
│       │   └── demography/
│       ├── XY/
│       │   ├── functional_connectivity/
│       │   ├── behavioral_metrics/
│       │   └── demography/
│       └── BNU/
│           ├── functional_connectivity/
│           ├── behavioral_metrics/
│           └── demography/
│
├── outputs/                    # 运行产物（不纳入版本控制），用于存放展示型对象：figures、tables、reports、logs
│   ├── figures/                # 图表输出
│   ├── logs/                   # 运行日志（脚本/SLURM 日志）
│   ├── results/                # 结果文件
│   └── tables/                 # 表格输出
│
├── tests/                      # 项目测试脚本（不纳入版本控制）
└── notebooks/                  # 探索性分析（不纳入版本控制）
```
