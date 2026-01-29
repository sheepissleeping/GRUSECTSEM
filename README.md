Regulation of Emotional Dynamics by Big Five Personalities: An Empirical Analysis of TFace-Bi-GRU-SE and CTSEM
Study Introduction
This study investigated the moderating effects of the Big Five personality traits, age and gender on the dynamics of emotional pleasure (P) and arousal (A).
Data & Code
Deep Learning Model: TFace-Bi-GRU-SE (Accuracy: 63.5%)
Dynamic Modeling: Continuous Time Structural Equation Model (CTSEM)
Sample: 30 subjects, 19,262 observations
File structure
GRUSECTSEM/
│
├── data/                                  数据文件
│   ├── README.md                          数据说明
│   ├── sample_data.csv                    示例数据（匿名化）
│   └── PAD_mapping.csv                    PAD映射表
│
├── scripts/                               分析脚本
│   ├── 02_preprocessing/
│   │   └── PAD_mapping.R                  PAD映射脚本
│   ├── 03_ctsem_analysis/
│   │   └── ctsem_full_analysis.R          CTSEM完整分析
│   └── 04_sensitivity_analysis/
│       └── sensitivity_analysis.R          敏感性分析
│
├── results/                               结果文件
│   ├── figures/                           图表
│   └── tables/                            表格
│
└── README.md                              本文件
Get started quickly

PAD mapping (converts 7 categories of sentiment to P and A continuous values)

rsource("scripts/02_preprocessing/PAD_mapping.R")

CTSEM analysis

rsource("scripts/03_ctsem_analysis/ctsem_full_analysis.R")

Sensitivity analysis

rsource("scripts/04_sensitivity_analysis/sensitivity_analysis.R")
Environmental requirements
R package

ctsem (>= 3.9.0)
dplyr
ggplot2
readxl
tidyr
zoo

Python package 
torch==1.9.0
opencv-python

Data availability
Due to privacy protection and ethical requirements, the raw data is not publicly available.
Example data: data/sample_data.csv (anonymized mock data for demonstration code)
Real data: Available upon reasonable academic request (data use agreement required)
Contact information: [MZ.L.mingzhengli@mail.ustc.edu.cn]
citation
If you used this code, please quote:
[Moderating Roles of the Big Five in Valence-Arousal Dynamics: A TFace-Bi-GRU-SE and CTSEM Study]
license
Apache-2.0 License
Contact information
For questions, please contact: [MZ.L.mingzhengli@mail.ustc.edu.cn]

Last updated: January 2025
GitHub仓库:https://github.com/sheepissleeping/GRUSECTSEM
