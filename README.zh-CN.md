# UK Road Collision Severity Prediction

[English](README.md) | **简体中文**

一个可复现的数据分析与机器学习项目。本项目使用英国交通部（DfT）近五年道路碰撞记录，研究哪些事故发生前已知的道路、时间、环境和位置因素与严重伤害风险相关，并构建预测 **KSI（Killed or Seriously Injured，死亡或重伤）** 风险的模型。

所有图表、指标和特征解释均可由脚本重建，方便审阅、复现和后续沟通。

## 研究问题与目标

1. 五年间事故数与 KSI 风险如何随时间、地点和道路环境变化？
2. 在事故发生时可得的信息中，哪些因素最能区分 KSI 与轻伤事故？
3. 一个面向未来年份验证的模型，是否优于“始终预测轻伤”的简单基准？
4. 如何把模型结果转化成针对高风险场景的可行动安全建议？

### 目标变量

默认任务为二分类：`collision_severity` 为 1（fatal）或 2（serious）记为 `KSI=1`；3（slight）记为 `KSI=0`。这是比三分类更适合风险筛查和不平衡数据表达的第一版任务。配置文件也为日后多分类实验保留了接口。

为避免数据泄漏，`enhanced_severity_collision`、`collision_injury_based`、两项 `collision_adjusted_*` 结果字段，以及事故 ID/精确位置 ID 都不会进入模型。请勿将事后确认的伤害结果当成预测特征。

## 仓库结构

```text
configs/                 可版本控制的实验设置
configs/collision_schema.yaml  DfT字段契约、允许编码和特殊编码语义
configs/lightgbm_best_params.yaml  一次性寻优后固定使用的 LightGBM 参数
data/raw/                原始 CSV（本地存放，不纳入 Git）
data/processed/          可再生的处理后数据（被 Git 忽略）
src/road_severity/       可复用的数据和建模代码
models/                  模型、指标、重要性（被 Git 忽略）
reports/figures/         PNG 图表（被 Git 忽略）
reports/tables/          数据质量、分析明细与图表目录（被 Git 忽略）
scripts/                 三阶段执行脚本及一次性 LightGBM 寻优脚本
tests/                   数据契约、特征泄漏、时间切分、建模与评估测试
```

## 快速开始

要求：Python 3.11+。建议在仓库根目录创建虚拟环境后安装依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

原始 CSV 应置于 `data/raw/`（本工作区已放置在该位置）后即可执行默认配置。原始数据不纳入版本控制；代码和配置纳入版本控制；模型、生成图表及可重建的处理后数据由 `.gitignore` 排除。若数据在别处，使用 `--data <path>` 覆盖默认路径；使用 `--max-rows 0` 在完整数据上运行。

在仓库根目录按顺序执行：

```powershell
python scripts/01_raw_analysis_and_processing.py
python scripts/02_processed_analysis_and_visualisation.py
python scripts/03_model_training_and_visualisation.py
```

第一部分默认使用全部原始数据，并生成全量处理数据。需要快速调试时，可以显式传入 `--max-rows 250000`，此时按“年份 × 严重程度”分层抽样。共享的数据处理、分析和建模逻辑位于 `src/road_severity/`。

第三部分会直接读取已保存的最优 LightGBM 参数 `configs/lightgbm_best_params.yaml`，正常运行不会重复进行超参数搜索。如果需要在数据、特征或搜索空间发生实质变化后重新寻优，才单独执行：

```powershell
python scripts/tune_lightgbm.py
```

该脚本会覆盖最优参数文件，并将全部候选结果写入 `reports/tables/lightgbm_tuning_results.csv`。调参是一次性步骤，不应放入日常三阶段执行流程。

### 数据来源

数据来自英国政府发布的 [Road safety open data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data?utm_source=chatgpt.com)。下载所需的近五年 collision CSV 后，将文件保存为 `data/raw/dft-road-casualty-statistics-collision-last-5-years.csv`。请在使用或再发布数据前遵守该页面说明的数据许可与使用条件。

## 可复现流程

1. 第一部分先审计全量且未经转换的数据，再验证字段契约、官方分类编码、日期、范围和一致性；按官方数据指南区分 `-1`、`9`、`99` 的字段级含义，在抽样前处理完全重复并拦截冲突重复，生成带来源标记的统一规范字段，并输出处理数据、质量对比、字段决策与可复现元数据。任何 error 级验证失败都会在覆盖上一版有效处理数据前停止。
2. 第二部分只读取通过第一部分契约与全量校验的处理后数据。它输出类别不平衡、年度/小时趋势、天气/路面/月/星期、三个统一字段、道路类型与道路等级的 KSI 分层表；所有比例图均给出样本量和 95% Wilson 置信区间，未知值保留在 CSV、但不参与结论图。
3. 第二部分还比较原始 KSI、DfT 调整后 KSI 与伤情录入方法占比，生成“速度 × 城乡”和“时间 × 光照”交互热力图，以及碰撞密度与达到 200 条样本门槛后的空间 KSI 六边形图。`reports/tables/figure_catalog.csv` 标记了建议用于汇报的核心图，避免把全部诊断图堆入演示文稿。
4. LightGBM 超参数寻优独立运行一次。搜索只使用正式训练期数据，并采用扩展年度窗口：2021 年训练、2022 年验证；随后 2021–2022 年训练、2023 年验证。候选参数按两折平均 Average Precision 排序，2024 和 2025 数据不会参与调参。
5. 第三部分按年份切分：2021–2023 年为训练集、2024 年为验证集、2025 年为最终测试集；比较 Dummy 基线、类别加权逻辑回归和读取固定最优参数的 LightGBM。
6. 最佳模型按 2024 验证集 Average Precision 选择，决策阈值也只在验证集按 KSI F1 选择。2025 测试集仅用于一次最终评估。
7. 最终输出 ROC-AUC、Average Precision、Brier score、balanced accuracy、KSI precision/recall/F1 和完整混淆矩阵计数；生成 PR、ROC、混淆矩阵、校准、阈值指标及置换重要性图。
8. `number_of_casualties`、警员是否到场和伤害结果字段默认不进入模型，避免使用事故后的信息。

## LightGBM 最优参数

当前参数来自 20 组候选的一次性随机搜索，滚动时间验证平均 AP 为 0.3709：

```yaml
max_iter: 400
learning_rate: 0.03
num_leaves: 63
min_child_samples: 20
max_depth: 6
l1_regularization: 1.0
l2_regularization: 5.0
feature_fraction: 1.0
bagging_fraction: 0.85
```

普通第三部分运行只读取这些参数。只有主动再次执行 `scripts/tune_lightgbm.py` 或手动修改参数文件时，后续模型配置才会改变。

## 当前模型结果

在隔离的 2025 测试集（101,525 条记录）上，调参后 LightGBM 的结果为：

| 指标 | 结果 |
| --- | ---: |
| ROC-AUC | 0.6396 |
| Average Precision | 0.3819 |
| Brier score | 0.2348 |
| Balanced accuracy | 0.5932 |
| KSI precision | 0.3240 |
| KSI recall | 0.7236 |
| KSI F1 | 0.4476 |
| 验证集选择阈值 | 0.45 |

在该阈值下，模型识别出 19,280 条 KSI，漏掉 7,364 条 KSI，同时产生 40,225 条误报。该模型适合解释为偏重召回的风险筛查模型，而不是高精度事故判定器。

模型与指标写入 `models/test/`；评估图写入 `reports/figures/model/`；验证集模型比较写入 `reports/tables/model_comparison_validation.csv`。这些都是可重建产物，默认不纳入 Git。

## 建议的结果叙事

| 部分 | 关键视觉/问题 |
| --- | --- |
| 问题定义 | 安全问题、KSI 定义、决策问题、分析流程 |
| 数据探索 | 覆盖范围与质量、类别不平衡、年度/小时趋势、地图或道路环境分层 |
| 建模设计 | 时间外验证设计、泄漏控制、一次性滚动时间调参、模型比较、特征工程 |
| 模型评估 | PR/ROC 曲线、混淆矩阵、校准、阈值曲线、错误切片、置换重要性 |
| 洞察与行动 | 高风险场景及优先行动、限制与下一步 |

不要把相关性写成因果关系；需在结论中说明：记录型事故数据只包含已发生的事故，不能直接替代暴露量（交通流量）或完整的风险率估计。

## Git 提交范围

建议提交代码、配置、测试和 README；不要提交原始数据、模型二进制、渲染图表、可重建的处理后数据或个人信息。README 已记录官方下载地址与本地文件名，便于其他协作者复现。

## 后续工作清单

- [x] 确认数据字典、来源和字段编码，给图表补充可读标签。
- [x] 在完整数据上运行三部分流程，比较逻辑回归基准、树模型和阈值策略。
- [x] 使用滚动年度验证完成一次性 LightGBM 超参数寻优，并固定保存最佳参数。
- [x] 输出 PR、ROC、混淆矩阵、校准、阈值指标与置换重要性图。
- [ ] 检查按警区、城乡、光照条件等分层的公平性和误差。
- [x] 选择最有决策价值的图表，并记录分析假设和生成方法。
