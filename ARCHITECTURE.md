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
├── docs/                       # 方法、流程与决策记录（索引见 docs/README.md）
│   ├── README.md               # 文档集合索引
│   ├── data_dictionary.md      # 数据字典（数据说明）
│   ├── workflow.md             # 可复现流程
│   ├── methods.md              # 方法学细节
│   ├── reports/                # 研究计划与阶段性总结
│   ├── sessions/               # 会话记录（yy/mm/dd.md）
│
├── src/                        # 可复用模块（不含硬编码路径）
│
├── scripts/                    # 执行入口（HPC/主流程脚本）
│
├── data/                       # 项目分析数据（不纳入版本控制），用于存放数据型对象：预处理结果，connectivity matrix，analysis-ready tables
│   ├── external/               # 外部第三方输入
│   ├── raw/                    # 原始输入（非脚本产出）
│   │   ├── THU/
│   │   │   ├── app_data/       # 原始的行为任务数据
│   │   │   ├── inventory/       # 原始的量表数据
│   │   │   ├── bids/
│   │   │   └── metadata/
│   │   ├── XY/
│   │   │   ├── app_data/       # 原始的行为任务数据
│   │   │   ├── inventory/       # 原始的量表数据
│   │   │   ├── bids/
│   │   │   └── metadata/
│   │   └── BNU/
│   │       ├── app_data/       # 原始的行为任务数据
│   │       ├── inventory/       # 原始的量表数据
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
│   └── processed/                            # 可复用清洗结果
│       ├── THU/
│       │   ├── functional_connectivity/      # 功能连接矩阵
│       │   ├── behavioral_metrics/
│       │   └── demography/
│       ├── XY/
│       │   ├── functional_connectivity/      # 功能连接矩阵
│       │   ├── behavioral_metrics/
│       │   └── demography/
│       └── BNU/
│           ├── functional_connectivity/      # 功能连接矩阵
│           ├── behavioral_metrics/
│           └── demography/  
│
├── outputs/                    # 运行产物（不纳入版本控制），用于存放展示型对象：figures、tables、reports、logs
│   ├── figures/                # 图表输出
│   ├── logs/                   # 运行日志（脚本/SLURM日志）
│   ├── results/                # 结果文件
│   └── tables/                 # 表格输出
│
├── tests/                      # 项目测试脚本（不纳入版本控制）
│
└── notebooks/                  # 探索性分析（不纳入版本控制）
```
