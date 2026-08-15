# UK Road Collision Severity Prediction

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
data/raw/                原始 CSV（本地存放，不纳入 Git）
data/processed/          可再生的处理后数据（被 Git 忽略）
src/road_severity/       可复用的数据和建模代码
models/                  模型、指标、重要性（被 Git 忽略）
reports/figures/         PNG 图表（被 Git 忽略）
reports/tables/          数据质量表（被 Git 忽略）
notebooks/               临时探索；最终逻辑必须回迁到 src/scripts
tests/                   防止特征泄漏和时间特征回归的测试
```

## 快速开始

要求：Python 3.11+。建议在仓库根目录创建虚拟环境后安装依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m jupyter lab
```

原始 CSV 应置于 `data/raw/`（本工作区已放置在该位置）后即可执行默认配置。原始数据不纳入版本控制；代码和配置纳入版本控制；模型、生成图表及可重建的处理后数据由 `.gitignore` 排除。若数据在别处，使用 `--data <path>` 覆盖默认路径；使用 `--max-rows 0` 在完整数据上运行。

在 JupyterLab 中按顺序打开并执行：

1. `notebooks/01_exploratory_analysis.ipynb`：数据质量、类别不平衡、时间趋势和风险分层。
2. `notebooks/02_model_training_evaluation.ipynb`：时间外训练、验证、测试和特征重要性。

Notebook 用于交互式探索与叙事；共享的数据处理和建模逻辑位于 `src/road_severity/`，这样测试与 Notebook 使用相同的实现。

### 数据来源

数据来自英国政府发布的 [Road safety open data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data?utm_source=chatgpt.com)。下载所需的近五年 collision CSV 后，将文件保存为 `data/raw/dft-road-casualty-statistics-collision-last-5-years.csv`。请在使用或再发布数据前遵守该页面说明的数据许可与使用条件。

## 可复现流程

1. `01_exploratory_analysis.ipynb` 输出类别不平衡、年度趋势、小时 KSI 率和缺失率表。
2. `02_model_training_evaluation.ipynb` 按年份切分：训练集为 2024 年之前、验证集为 2024 年、最终测试集为 2025 年。这比随机切分更接近真实的未来部署。
3. 处理流程解析日期/时间并生成周期性特征；缺失值在管线内部处理，避免测试数据影响训练转换。
4. 梯度提升模型输出 ROC-AUC、Average Precision、分类报告及置换重要性，并保存到本地的 `models/validation/` 与 `models/test/`。
5. 仅在确认最终模型与叙事后，再将选定的 PNG 和指标用于报告或演示；每张图需有清晰标题、单位/分母、色盲友好配色和一句可执行的结论。

## 建议的结果叙事

| 部分 | 关键视觉/问题 |
| --- | --- |
| 问题定义 | 安全问题、KSI 定义、决策问题、分析流程 |
| 数据探索 | 覆盖范围与质量、类别不平衡、年度/小时趋势、地图或道路环境分层 |
| 建模设计 | 时间外验证设计、泄漏控制、模型比较/调参曲线、特征工程 |
| 模型评估 | PR/ROC 曲线、混淆矩阵、校准、错误切片、置换或 SHAP 解释 |
| 洞察与行动 | 高风险场景及优先行动、限制与下一步 |

不要把相关性写成因果关系；需在结论中说明：记录型事故数据只包含已发生的事故，不能直接替代暴露量（交通流量）或完整的风险率估计。

## Git 提交范围

建议提交代码、配置、测试和 README；不要提交原始数据、模型二进制、渲染图表、可重建的处理后数据或个人信息。README 已记录官方下载地址与本地文件名，便于其他协作者复现。

## 后续工作清单

- [ ] 确认数据字典、来源和字段编码，给图表补充可读标签。
- [ ] 跑完整数据，比较逻辑回归基准、树模型和阈值策略。
- [ ] 检查按警区、城乡、光照条件等分层的公平性和误差。
- [ ] 选择最有决策价值的图表，并记录分析假设和生成方法。
