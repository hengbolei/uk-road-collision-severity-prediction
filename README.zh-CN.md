# 英国道路碰撞严重程度预测

[English](README.md) | **简体中文**

## 项目概述

本项目分析英国交通部 2021—2025 年的 513,801 条道路碰撞记录，并预测已报告碰撞是否造成死亡或重伤。项目结合时间、道路、环境和空间模式的描述性分析，并使用严格的时间外验证比较五种分类模型。

KSI 指 **Killed or Seriously Injured（死亡或重伤）**。死亡和重伤碰撞编码为 `KSI=1`，轻伤碰撞编码为 `KSI=0`。

## 研究问题

1. 碰撞数量和 KSI 占比如何随时间及道路环境变化？
2. 碰撞发生时已知的哪些信息最能区分 KSI 与轻伤碰撞？
3. 哪种候选模型在未来验证年度上的泛化表现最好？
4. 入选模型在独立测试年度上表现如何，哪些特征贡献最大？

## 数据与泄漏控制

数据覆盖 2021—2025 年已报告的人员伤亡道路碰撞，来源为英国政府发布的 [Road safety open data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data)。

模型排除了伤情结果衍生字段、碰撞标识符、精确位置标识符、伤亡人数和警员是否到场等字段。道路类型、照明、天气、地方行政区等编码字段按类别属性处理；CatBoost 通过原生类别特征接口接收这些字段。

## 分析与评估设计

工作流程分为四个阶段：

1. 验证原始字段、处理重复记录并形成碰撞级数据集。
2. 分析严重程度构成，以及时间、环境、道路和空间模式。
3. 在 2021—2023 年内部使用扩展年度窗口调优 LightGBM，并在 2024 年比较五个候选模型。
4. 只使用 2024 年选择模型和分类阈值，再在独立的 2025 年测试集上进行一次评估。

候选模型包括 Dummy 基线、类别加权逻辑回归、ExtraTrees、CatBoost 和调参后的 LightGBM。由于 KSI 是少数类别，模型选择以 Average Precision 为主指标，并使用 ROC-AUC 和 Brier score 补充分析排序能力与概率误差。

## 主要描述性结果

- KSI 占已报告碰撞的 24.2%，轻伤碰撞占 75.8%。
- 月度碰撞量总体趋稳，但平滑后的 KSI 占比从约 21.9% 上升至 26.2%。
- 简化后的核心属性中，限速与城乡环境的关联最强（0.68）。
- 无道路照明的黑暗环境对应的 KSI 占比明显高于日间。
- 碰撞密度热点与高严重程度热点并不稳定重合。
- 上述模式描述已报告碰撞，不等于按交通暴露量标准化后的道路风险。

## 模型对照

所有模型使用相同的 2021—2023 年训练集和 2024 年验证集。

| 排名 | 模型 | 验证 AP | ROC-AUC | Brier score | 训练时间 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | LightGBM | 0.3798 | 0.6558 | 0.2311 | 5.9 秒 |
| 2 | CatBoost | 0.3792 | 0.6557 | 0.2296 | 90.0 秒 |
| 3 | ExtraTrees | 0.3670 | 0.6443 | 0.2242 | 28.6 秒 |
| 4 | 逻辑回归 | 0.3244 | 0.5894 | 0.2431 | 7.0 秒 |
| 5 | Dummy 基线 | 0.2484 | 0.5000 | 0.1869 | 2.5 秒 |

LightGBM 与 CatBoost 的验证排序表现几乎相同，但 LightGBM 训练明显更快，因此按照预先确定的 Average Precision 标准保留 LightGBM。Dummy 基线的 Brier score 较低主要来自类别比例预测，并不表示它具有有效的 KSI 排序能力。

## 2025 年独立测试结果

入选的 LightGBM 在 101,525 条测试记录上的结果如下：

| 指标 | 结果 |
| --- | ---: |
| ROC-AUC | 0.6393 |
| Average Precision | 0.3819 |
| Brier score | 0.2349 |
| Balanced accuracy | 0.5950 |
| KSI precision | 0.3280 |
| KSI recall | 0.7004 |
| KSI F1 | 0.4468 |
| 验证集确定的阈值 | 0.46 |

在该阈值下，模型识别出 18,662 条 KSI 碰撞，漏掉 7,982 条。在 56,889 条 KSI 预测中，38,227 条为误报。因此，它更适合作为偏重召回的风险筛查工具，而不是自动碰撞分类系统。模型还会高估实际 KSI 占比，预测概率在按字面解释前需要进一步校准。

排列重要性显示，涉及车辆数量、限速、道路类型和纬度是预测排序的主要贡献特征。这些是模型贡献，不代表因果影响。

## 复现分析

创建环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

将原始 CSV 放在 `configs/default.yaml` 指定的位置，然后运行：

```powershell
python scripts/01_raw_analysis_and_processing.py
python scripts/02_processed_analysis_and_visualisation.py
python scripts/tune_lightgbm.py
python scripts/03_model_training_and_visualisation.py
python scripts/04_additional_visual_analysis.py
python -m pytest -q
```

生成的模型、表格和非核心图表由 Git 忽略。训练脚本将模型对照表保存到 `reports/tables/model_comparison_validation.csv`，模型图保存到 `reports/figures/model/`，入选管线和测试指标保存到 `models/test/`。

## 解读边界

数据只包含已报告碰撞，没有交通暴露量。因此，KSI 占比描述的是已记录碰撞中的严重程度，而不是每次出行、每公里或每位道路使用者的碰撞风险。跨年度变化还可能受到严重程度记录方式变化的影响。描述性差异、相关性和模型重要性都应理解为关联，而非因果关系。

## 报告与 Notebook

- [英文核心图表报告](reports/figure_story.md)
- [中文核心图表报告](reports/figure_story_zh-CN.md)
- [英文分析 Notebook](notebooks/uk_road_collision_analysis.ipynb)
- [中文分析 Notebook](notebooks/uk_road_collision_analysis_zh.ipynb)
