# UK Road Collision Severity Prediction

**English** | [简体中文](README.zh-CN.md)

## Project overview

This project analyses 513,801 UK Department for Transport road collision records from 2021–2025 and models whether a reported collision resulted in death or serious injury. It focuses on the relationship between collision severity and time, lighting, speed limit, urban–rural context, road environment and location.

## Research questions

1. How do collision volume and KSI prevalence vary across time and road environments?
2. Which information known at collision time best distinguishes KSI from slight-injury collisions?
3. How well does the selected model generalise to a later, held-out year?
4. Which features contribute most to the model's predictive ranking?

## Outcome and data

KSI means **Killed or Seriously Injured**. Fatal and serious collisions are encoded as `KSI=1`; slight collisions are encoded as `KSI=0`.

The dataset covers reported personal-injury road collisions from 2021–2025. It is published through the UK government's [Road safety open data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data).

Outcome-derived fields, collision identifiers, exact location identifiers, casualty counts and police-attendance information are excluded from model features to reduce target leakage and avoid relying on information unavailable at collision time.

## Analysis design

The analysis has three stages:

1. Validate the source schema, resolve duplicate records, standardise fields and prepare the collision-level dataset.
2. Describe severity composition and temporal, environmental, road and spatial patterns using counts, proportions and 95% Wilson confidence intervals.
3. Compare a majority-class baseline, class-weighted logistic regression and tuned LightGBM using an out-of-time evaluation.

LightGBM tuning uses expanding annual windows within 2021–2023. The final comparison trains on 2021–2023, selects the model and decision threshold on 2024, and evaluates once on the held-out 2025 data.

## Main descriptive findings

- KSI collisions account for 24.2% of reported collisions; 75.8% are recorded as slight.
- Annual collision volume is comparatively stable, while injury-based severity reporting rises substantially across the period.
- Collision volume peaks in the afternoon, but the highest recorded KSI share occurs around midnight.
- Unlit darkness has a markedly higher recorded KSI share than daylight.
- Speed-limit patterns differ between urban and rural roads; rural groups have higher KSI shares across the displayed limits.
- Spatial collision-density hotspots do not coincide consistently with high-severity hotspots.

## Model results

Results for tuned LightGBM on the held-out 2025 test set of 101,525 records:

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

At this threshold, the model identifies 19,280 KSI collisions and misses 7,364. Of 59,505 KSI predictions, 40,225 are false positives. The model therefore has stronger recall than precision.

Permutation importance shows that the number of vehicles contributes the largest decrease in Average Precision when shuffled, followed by speed limit, longitude and road type. These values describe predictive contribution within this model and do not establish causal effects.

## Interpretation boundaries

The data contains reported collisions rather than traffic exposure. KSI shares therefore describe severity among recorded collisions, not collision risk per journey, kilometre or road user. Temporal changes may also reflect changes in severity-recording practice. Descriptive differences and model importance should be interpreted as associations, not causal effects.

## Report materials

- [Core figure narrative](reports/figure_story.md)
- [English analysis notebook](notebooks/uk_road_collision_analysis.ipynb)
- [中文分析 Notebook](notebooks/uk_road_collision_analysis_zh.ipynb)
