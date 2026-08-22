# UK Road Collision Severity: Core Figures and Visual Story

**English** | [简体中文](figure_story_zh-CN.md)

> Data source: UK Department for Transport (DfT) road collision data, 2021–2025.
> Population analysed: reported personal-injury road collisions. KSI means Killed or Seriously Injured.
> Note: KSI shares in these figures are proportions among reported collisions, not road-risk measures standardised by traffic volume, population or vehicle mileage.

## Story overview

This project examines which temporal, road, environmental and spatial conditions are associated with more severe collision outcomes, and whether machine learning can use information known at collision time to identify higher KSI risk. The central story is that contexts with more collisions are not necessarily the most severe; severity varies with time, lighting, speed, urban–rural context and location; and the model identifies most KSI collisions at the selected threshold but also produces many false positives.

---

## 1. KSI is a minority but consequential collision outcome

![Collision severity composition](figures/processed/01_severity_composition.png)

### What the figure shows

A percentage stacked bar presents the composition of fatal, serious and slight collisions, with direct labels for shares and sample sizes.

### What it represents

Across five years, 513,801 collisions were recorded: 389,435 slight collisions (75.8%), 116,813 serious collisions (22.7%) and 7,553 fatal collisions (1.5%). Fatal and serious collisions total 124,366, giving a KSI share of 24.2%.

### Interpretation boundary

The figure describes the distribution of outcomes among reported collisions. It does not represent the probability that an ordinary journey results in a collision or a KSI outcome.

---

## 2. Collision volume is stable, but recording practice changes substantially

![Annual collision volume and recording-method trend](figures/processed/02_annual_reporting_sensitivity.png)

### What the figure shows

Bars show annual collision volume on a shared time axis, while the orange line shows the share recorded using injury-based severity reporting. The year with the highest collision volume is highlighted in muted blue and labelled directly; the remaining bars are grey.

### What it represents

Annual collision volume ranges from 100,927 to 106,004, peaking at 106,004 in 2022 and varying relatively little across the five years. Injury-based reporting rises from 49.6% in 2021 to 86.5% in 2025, an increase of 36.9 percentage points. The increase from 2024 to 2025 alone is approximately 27.8 percentage points.

### Interpretation boundary

The line describes recording practice rather than road risk. It cannot establish that roads became safer or more dangerous over the period.

---

## 3. The peaks in collision volume and severity do not coincide

![Hourly collision volume and KSI share](figures/processed/03_hourly_volume_and_ksi.png)

### What the figure shows

Bars show the number of reported collisions by hour, while the blue line shows KSI share with a 95% Wilson confidence interval on the same time axis.

### What it represents

Collision volume peaks at 17:00 with 44,792 collisions, when the KSI share is approximately 23.1%. KSI share is highest at 00:00, at 30.9%, and lowest at 08:00, at 19.4%. The difference is 11.5 percentage points, and the midnight KSI share is approximately 1.59 times the 08:00 share.

### Interpretation boundary

The data is not standardised by hourly traffic flow or vehicle mileage. The figure therefore does not measure individual travel risk or establish specific causes.

---

## 4. Unlit darkness corresponds to a higher KSI share

![KSI share by lighting condition](figures/processed/06_ksi_by_light.png)

### What the figure shows

The chart compares KSI share, collision count and 95% Wilson confidence intervals across lighting conditions, highlighting the category with the highest share.

### What it represents

KSI accounts for 34.9% of collisions in darkness with no road lighting (n=27,438), compared with 23.2% in daylight (n=368,193), a difference of 11.8 percentage points. The former is approximately 1.51 times the latter. Darkness with street lights lit has a KSI share of 25.1%, between the two.

### Interpretation boundary

This is an unadjusted association. Unlit roads are also more likely to be rural or in higher-speed environments, so the entire difference cannot be attributed to lighting.

---

## 5. The pattern associated with speed limits depends on urban–rural context

![Speed-limit and urban–rural interaction heatmap](figures/processed/17_speed_by_area_heatmap.png)

### What the figure shows

A red sequential heatmap compares KSI shares across combinations of urban–rural context and speed limit. Every cell directly labels the share and collision count.

### What it represents

Rural 60 mph roads have the highest KSI share, at 34.8% (n=60,693), compared with 25.0% on urban 60 mph roads (n=2,000), a difference of 9.8 percentage points. The urban–rural gap at 50 mph is similarly pronounced, with the rural share 9.9 percentage points higher. Rural KSI shares exceed urban shares in all six displayed speed groups, although the gaps range from 1.9 to 9.9 percentage points. The 70 mph group does not have the highest KSI share.

### Interpretation boundary

The cells show KSI share after a collision has occurred, not collision risk per road kilometre. They also do not establish that higher speed limits cause the full observed difference.

---

## 6. Collision-density hotspots differ from severity hotspots

![Collision density and spatial KSI pattern](figures/processed/19_spatial_hex_analysis.png)

### What the figure shows

Collision coordinates are aggregated in one hexagonal spatial view. Grey intensity represents collision-count density, while red hexagons represent KSI share in cells meeting a minimum threshold of 200 collisions.

### What it represents

A total of 403 cells meet the 200-collision threshold, with KSI shares ranging from approximately 13.6% to 52.6%. Across these cells, the Spearman correlation between collision count and KSI share is approximately -0.41, indicating that denser collision cells do not tend to have higher severity shares at the same time.

### Interpretation boundary

This is not a traffic-exposure risk map and does not control for population, road length or traffic volume. Blank areas or areas without red overlays do not indicate an absence of risk.

---

## 7. Vehicle count, speed and road context provide the most predictive information

![Held-out-year permutation importance](figures/model/permutation_importance.png)

### What the figure shows

Permutation importance measures the decrease in model Average Precision after each feature is randomly shuffled. Error bars show uncertainty across repeated estimates, and the largest importance value is labelled directly.

### What it represents

Number of vehicles contributes the largest importance: shuffling it reduces Average Precision by 0.0297 on average (standard deviation 0.0017), approximately 2.9 times the second-ranked speed limit value of 0.0102. Longitude has an importance of 0.0091 with a wider error bar, while road type has an importance of 0.0086 with a comparatively stable estimate.

### Interpretation boundary

Permutation importance measures contribution to predictions from the current model, not causal influence. Correlated features may also divide importance between them.

---

## 8. The model exceeds the baseline, but discrimination remains limited

![Precision–recall curve](figures/model/precision_recall_curve.png)

### What the figure shows

The precision–recall curve shows the trade-off between KSI precision and recall across thresholds in the held-out test year. KSI prevalence provides the no-skill baseline.

### What it represents

The model achieves an Average Precision of 0.382, which is 0.120 above the KSI prevalence baseline of 0.262 and approximately 1.46 times that baseline. The model ranks more KSI collisions toward the front of the prediction order, but the curve approaches the baseline quickly when high recall is required, indicating limited overall discrimination.

### Interpretation boundary

Average Precision is not prediction accuracy. The result describes generalisation under the current temporal split and feature set.

---

## 9. The selected threshold favours recall but produces many false positives

![Held-out-year confusion matrix](figures/model/confusion_matrix.png)

### What the figure shows

A purple sequential heatmap presents counts and within-row shares for actual and predicted classes in the test year, using the threshold of 0.45 selected on the validation year.

### What it represents

The model correctly identifies 19,280 KSI collisions and misses 7,364, giving a KSI recall of 72.4% and a false-negative rate of 27.6%. It produces 59,505 KSI alerts, of which 40,225 are false positives, giving a precision of 32.4%. At the same threshold, 53.7% of actual slight collisions are classified as KSI.

### Interpretation boundary

The confusion matrix depends on the chosen threshold. Changing the threshold changes recall, precision and the number of false positives simultaneously.

---

## 10. Predicted risk probabilities remain miscalibrated

![Held-out-year calibration curve](figures/model/calibration_curve.png)

### What the figure shows

The test records are divided into ten equal-sized groups by predicted risk. The chart compares each group's mean predicted KSI probability with its observed KSI share; the diagonal represents perfect calibration.

### What it represents

All ten risk groups fall below the ideal calibration line, showing that the model systematically overestimates KSI probabilities in the 2025 test data. The largest absolute gap is 26.6 percentage points. In the highest predicted-risk group, the mean predicted probability is 71.2%, while the observed KSI share is 45.5%. The model retains ranking information, but its probability values should not be interpreted literally.

### Interpretation boundary

The calibration curve depends on the grouping method and sample distribution. The ten-group result is a diagnostic summary.

---

## Overall conclusions

1. KSI is a minority but consequential target that requires class-sensitive evaluation metrics.
2. Severity-recording practice changes substantially across the period, affecting cross-year interpretation.
3. Collision volume and severity do not coincide consistently across time or space.
4. Darkness, speed, urban–rural context, road structure and spatial context are associated with collision severity.
5. LightGBM extracts multidimensional predictive signal, but its discrimination remains moderate.
6. At the selected threshold, the model is recall-oriented rather than a high-precision collision classifier.
