# UK Road Collision Severity Prediction

**English** | [简体中文](README.zh-CN.md)

## Project overview

This project analyses 513,801 UK Department for Transport road collision records from 2021–2025 and predicts whether a reported collision resulted in death or serious injury. It combines descriptive analysis of temporal, road, environmental and spatial patterns with an out-of-time comparison of five classifiers.

KSI means **Killed or Seriously Injured**. Fatal and serious collisions are encoded as `KSI=1`; slight collisions are encoded as `KSI=0`.

## Research questions

1. How do collision volume and KSI prevalence vary across time and road environments?
2. Which information known at collision time best distinguishes KSI from slight-injury collisions?
3. Which candidate model generalises best to a future validation year?
4. How well does the selected model perform on a held-out test year, and which features contribute most?

## Data and leakage controls

The data covers reported personal-injury road collisions from 2021–2025 and is published through the UK government's [Road safety open data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data).

Outcome-derived fields, collision identifiers, exact location identifiers, casualty counts and police-attendance information are excluded from model features. Coded fields such as road type, lighting, weather and local authority are treated as categorical attributes; CatBoost receives these attributes through its native categorical-feature interface.

## Analysis and evaluation design

The workflow has four stages:

1. Validate the source schema, resolve duplicates and prepare the collision-level dataset.
2. Analyse severity composition and temporal, environmental, road and spatial patterns.
3. Tune LightGBM with expanding annual windows inside 2021–2023 and compare five candidate models on 2024.
4. Select the model and decision threshold using 2024 only, then evaluate once on held-out 2025 data.

The five candidates are Dummy baseline, class-weighted logistic regression, ExtraTrees, CatBoost and tuned LightGBM. Average Precision is the primary selection metric because KSI is the minority class. ROC-AUC and Brier score provide complementary ranking and probability-error diagnostics.

## Main descriptive findings

- KSI collisions account for 24.2% of reported collisions; 75.8% are recorded as slight.
- Monthly collision volume broadly stabilises, while the smoothed KSI share rises from about 21.9% to 26.2%.
- Speed limit and urban–rural context have the strongest association among the simplified core attributes (0.68).
- Unlit darkness has a markedly higher recorded KSI share than daylight.
- Collision-density hotspots do not consistently coincide with high-severity hotspots.
- These patterns describe reported collisions and do not measure exposure-standardised road risk.

## Model comparison

All models use the same 2021–2023 training data and 2024 validation data.

| Rank | Model | Validation AP | ROC-AUC | Brier score | Training time |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | LightGBM | 0.3798 | 0.6558 | 0.2311 | 5.9 s |
| 2 | CatBoost | 0.3792 | 0.6557 | 0.2296 | 90.0 s |
| 3 | ExtraTrees | 0.3670 | 0.6443 | 0.2242 | 28.6 s |
| 4 | Logistic regression | 0.3244 | 0.5894 | 0.2431 | 7.0 s |
| 5 | Dummy baseline | 0.2484 | 0.5000 | 0.1869 | 2.5 s |

LightGBM and CatBoost have nearly identical validation ranking performance, but LightGBM trains substantially faster. LightGBM is retained under the predefined Average Precision criterion. The Dummy baseline's low Brier score reflects prevalence prediction rather than useful KSI ranking.

## Held-out 2025 results

The selected LightGBM model is evaluated on 101,525 test records.

| Metric | Result |
| --- | ---: |
| ROC-AUC | 0.6393 |
| Average Precision | 0.3819 |
| Brier score | 0.2349 |
| Balanced accuracy | 0.5950 |
| KSI precision | 0.3280 |
| KSI recall | 0.7004 |
| KSI F1 | 0.4468 |
| Validation-selected threshold | 0.46 |

At this threshold, the model identifies 18,662 KSI collisions and misses 7,982. Of 56,889 KSI predictions, 38,227 are false positives. It is therefore more suitable for recall-oriented risk screening than automatic collision classification. Predicted probabilities also overestimate observed KSI shares and require calibration before literal risk interpretation.

Permutation importance identifies number of vehicles, speed limit, road type and latitude as the leading contributors to predictive ranking. These are model contributions, not causal effects.

## Reproduce the analysis

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the source CSV at the path specified in `configs/default.yaml`, then run:

```powershell
python scripts/01_raw_analysis_and_processing.py
python scripts/02_processed_analysis_and_visualisation.py
python scripts/tune_lightgbm.py
python scripts/03_model_training_and_visualisation.py
python scripts/04_additional_visual_analysis.py
python -m pytest -q
```

Generated models, tables and non-core figures are ignored by Git. The training script writes the comparison table to `reports/tables/model_comparison_validation.csv`, model diagnostics to `reports/figures/model/`, and the selected pipeline and test metrics to `models/test/`.

## Interpretation boundaries

The data contains reported collisions rather than traffic exposure. KSI shares describe severity among recorded collisions, not collision risk per journey, kilometre or road user. Temporal changes may reflect changes in severity-recording practice. Descriptive differences, correlations and model importance are associations rather than causal effects.

## Reports and notebooks

- [English figure narrative](reports/figure_story.md)
- [中文核心图表报告](reports/figure_story_zh-CN.md)
- [English analysis notebook](notebooks/uk_road_collision_analysis.ipynb)
- [中文分析 Notebook](notebooks/uk_road_collision_analysis_zh.ipynb)
