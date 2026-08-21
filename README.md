# UK Road Collision Severity Prediction

**English** | [简体中文](README.zh-CN.md)

A reproducible data analysis and machine-learning project based on the latest five years of UK Department for Transport (DfT) road collision records. It investigates which road, time, environmental, and location factors known at the time of a collision are associated with serious harm, and builds a model to predict **KSI (Killed or Seriously Injured)** risk.

All figures, metrics, and feature interpretations can be regenerated from scripts for review, reproducibility, and future communication.

## Research questions

1. How do collision counts and KSI prevalence vary over time, location, and road environment?
2. Which information available at collision time best distinguishes KSI from slight-injury collisions?
3. Does a model evaluated on future years outperform a simple majority-class baseline?
4. How can model findings support practical road-safety prioritisation?

### Target

The default task is binary classification: `collision_severity` 1 (fatal) or 2 (serious) is encoded as `KSI=1`; 3 (slight) is encoded as `KSI=0`.

To reduce leakage, outcome-derived fields such as `enhanced_severity_collision`, `collision_injury_based`, and both `collision_adjusted_*` fields are excluded. Collision identifiers, exact location identifiers, `number_of_casualties`, and whether a police officer attended are also excluded from model features.

## Repository structure

```text
configs/                         Version-controlled experiment settings
configs/collision_schema.yaml    DfT schema, allowed codes, and special-code meanings
configs/lightgbm_best_params.yaml Persisted parameters from one-off LightGBM tuning
data/raw/                        Local source CSV files; ignored by Git
data/processed/                  Reproducible processed data; ignored by Git
models/                          Models, metrics, and importance; ignored by Git
reports/figures/                 Generated PNG figures; ignored by Git
reports/tables/                  Quality reports and analysis tables; ignored by Git
scripts/                         Three-stage pipeline and one-off tuning script
src/road_severity/               Reusable data, analysis, and modelling code
tests/                           Data, leakage, temporal split, model, and metric tests
```

## Quick start

Python 3.11 or later is required. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

Download the DfT collision CSV to:

```text
data/raw/dft-road-casualty-statistics-collision-last-5-years.csv
```

Then run the reproducible pipeline in order:

```powershell
python scripts/01_raw_analysis_and_processing.py
python scripts/02_processed_analysis_and_visualisation.py
python scripts/03_model_training_and_visualisation.py
```

Stage 1 uses the complete dataset by default. For faster development, pass `--max-rows 250000` to use a sample stratified by year and severity. Use `--max-rows 0` for the complete dataset.

### Data source

The data comes from the UK government's [Road safety open data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data). Review the licensing and usage terms on that page before using or redistributing the data.

## Reproducible workflow

1. Stage 1 audits the untouched source data, validates the schema and official codes, resolves exact duplicates, reports conflicting identifiers, standardises fields with provenance, and writes processed data and reproducibility metadata. Error-level validation failures stop the pipeline before a previously valid dataset is overwritten.
2. Stage 2 reads only validated processed data. It creates KSI summaries by year, hour, weather, surface, month, weekday, road type, road class, junction, crossing, and hazard. Proportion plots include sample sizes and 95% Wilson confidence intervals.
3. Stage 2 also compares recorded and DfT-adjusted KSI definitions, produces speed-by-area and time-by-light interaction heatmaps, and creates spatial collision-density and KSI hexbin figures subject to a minimum sample threshold.
4. LightGBM tuning is performed once using expanding-year validation: train on 2021 and validate on 2022, then train on 2021–2022 and validate on 2023. Candidates are ranked by mean Average Precision. Data from 2024 and 2025 is excluded from tuning.
5. Stage 3 trains on 2021–2023, uses 2024 for model and threshold selection, and reserves 2025 for final testing. It compares a Dummy baseline, class-weighted logistic regression, and LightGBM with persisted tuned parameters.
6. The final report includes ROC-AUC, Average Precision, Brier score, balanced accuracy, KSI precision/recall/F1, and confusion-matrix counts. It generates PR, ROC, confusion-matrix, calibration, threshold-metric, and permutation-importance figures.

## One-off LightGBM tuning

Normal Stage 3 runs load `configs/lightgbm_best_params.yaml` and do not repeat tuning. Only rerun tuning after a material change to the data, features, or search space:

```powershell
python scripts/tune_lightgbm.py
```

This overwrites the persisted best-parameter file and writes all candidate results to `reports/tables/lightgbm_tuning_results.csv`.

The current best parameters were selected from 20 sampled candidates and achieved a mean rolling-validation AP of 0.3709:

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

## Current model results

Results for tuned LightGBM on the isolated 2025 test set of 101,525 records:

| Metric | Result |
| --- | ---: |
| ROC-AUC | 0.6396 |
| Average Precision | 0.3819 |
| Brier score | 0.2348 |
| Balanced accuracy | 0.5932 |
| KSI precision | 0.3240 |
| KSI recall | 0.7236 |
| KSI F1 | 0.4476 |
| Validation-selected threshold | 0.45 |

At this threshold, the model identifies 19,280 KSI collisions, misses 7,364, and produces 40,225 false positives. It should therefore be interpreted as a recall-oriented risk-screening model, not a high-precision collision classifier.

Models and metrics are written to `models/test/`, evaluation figures to `reports/figures/model/`, and validation model comparisons to `reports/tables/model_comparison_validation.csv`. These reproducible outputs are ignored by Git.

## Interpretation and limitations

Predictive association and permutation importance do not establish causality. Recorded collision data includes only collisions that occurred and does not directly measure traffic exposure. Consequently, observed KSI shares are not complete estimates of road-user risk.

Recommended reporting structure:

| Section | Key evidence |
| --- | --- |
| Problem definition | Safety question, KSI definition, decision context, workflow |
| Data exploration | Coverage, quality, imbalance, trends, spatial and road context |
| Modelling design | Out-of-time evaluation, leakage controls, one-off tuning, feature engineering |
| Model evaluation | PR/ROC, confusion matrix, calibration, threshold trade-offs, importance |
| Action and limitations | High-risk contexts, prioritisation, limitations, next steps |

## Version-control scope

Commit source code, configuration, tests, and documentation. Do not commit source data, model binaries, rendered figures, reproducible processed data, or personal information.

## Remaining work

- [x] Validate the DfT schema, code meanings, and readable labels.
- [x] Run all three stages on the complete dataset.
- [x] Complete one-off LightGBM tuning with rolling-year validation.
- [x] Generate PR, ROC, confusion-matrix, calibration, threshold, and importance figures.
- [ ] Evaluate subgroup error and fairness across police force, urban/rural context, lighting, and other relevant slices.
- [x] Select decision-relevant figures and document assumptions and generation methods.
