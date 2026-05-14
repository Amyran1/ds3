# EDA Summary — run_01_smoke (civic_shout_action_rate_increase)

## Headline
| metric | value |
|---|---|
| outcome_positive_rate | 0.09267986054810362 |
| n_findings_total | 22 |
| n_findings_info | 19 |
| n_findings_warning | 3 |
| n_findings_error | 0 |
| feature_feature_max_abs_pearson | 0.9147275734781335 |
| cohort_drift_max_relative_lift | 1.668143163426224 |
| elapsed_seconds | 26.42 |

## Method: outcome_distribution
- elapsed: 0.04s
- n_findings: 2
![outcome_distribution.png](outcome_distribution.png)

## Method: univariate_numeric
- elapsed: 0.53s
- n_findings: 6
![univariate_numeric.png](univariate_numeric.png)

## Method: univariate_categorical
- elapsed: 0.06s
- n_findings: 2
![univariate_categorical.png](univariate_categorical.png)

## Method: feature_outcome_correlation
- elapsed: 21.13s
- n_findings: 6
![feature_outcome_correlation.png](feature_outcome_correlation.png)

## Method: feature_feature_correlation
- elapsed: 0.09s
- n_findings: 2
![feature_feature_correlation.png](feature_feature_correlation.png)

## Method: missingness_pattern
- elapsed: 4.50s
- n_findings: 0
![missingness_pattern.png](missingness_pattern.png)

## Method: row_alignment_audit
- elapsed: 0.02s
- n_findings: 1

## Method: eda_cohort_baseline
- elapsed: 0.05s
- n_findings: 3
![eda_cohort_baseline.png](eda_cohort_baseline.png)

## Findings
- [INFO] `outcome_distribution_positive_rate`: Positive rate: 0.0927
  n_rows=1290194, n_positives=119575, n_negatives=1170619
- [INFO] `outcome_distribution_class_imbalance`: Class imbalance ratio: 9.79
  max_class=1170619, min_class=119575
- [INFO] `univariate_numeric_actioned_last_1`: actioned_last_1: mean=0.092, std=0.290, missingness=0.000
  p1=0.000, p25=0.000, p50=0.000, p75=0.000, p99=1.000, skew=2.814
- [INFO] `univariate_numeric_actioned_last_3`: actioned_last_3: mean=0.273, std=0.644, missingness=0.000
  p1=0.000, p25=0.000, p50=0.000, p75=0.000, p99=3.000, skew=2.546
- [INFO] `univariate_numeric_actioned_last_5`: actioned_last_5: mean=0.451, std=0.983, missingness=0.000
  p1=0.000, p25=0.000, p50=0.000, p75=0.000, p99=4.000, skew=2.525
- [INFO] `univariate_numeric_actioned_last_10`: actioned_last_10: mean=0.886, std=1.806, missingness=0.000
  p1=0.000, p25=0.000, p50=0.000, p75=1.000, p99=8.000, skew=2.542
- [INFO] `univariate_numeric_sends_since_last_action`: sends_since_last_action: mean=37.466, std=54.675, missingness=0.000
  p1=0.000, p25=4.000, p50=18.000, p75=52.000, p99=261.000, skew=3.861
- [INFO] `univariate_numeric_lifetime_actions_prior`: lifetime_actions_prior: mean=23.524, std=49.724, missingness=0.000
  p1=0.000, p25=0.000, p50=3.000, p75=20.000, p99=254.000, skew=3.580
- [INFO] `univariate_categorical_action_in_last_5`: action_in_last_5: cardinality=2, missingness=0.000
  Top values: [(False, '0.763'), (True, '0.237')]
- [INFO] `univariate_categorical_is_in_action_streak`: is_in_action_streak: cardinality=2, missingness=0.000
  Top values: [(False, '0.980'), (True, '0.020')]
- [INFO] `feature_outcome_correlation_actioned_last_1`: actioned_last_1: pearson=0.333, spearman=0.804, mi=0.0403
  Pearson=0.3331, Spearman=0.8044, MI=0.040291
- [INFO] `feature_outcome_correlation_actioned_last_3`: actioned_last_3: pearson=0.446, spearman=0.687, mi=0.0716
  Pearson=0.4455, Spearman=0.6873, MI=0.071624
- [INFO] `feature_outcome_correlation_actioned_last_5`: actioned_last_5: pearson=0.480, spearman=0.603, mi=0.0847
  Pearson=0.4805, Spearman=0.6030, MI=0.084721
- [INFO] `feature_outcome_correlation_actioned_last_10`: actioned_last_10: pearson=0.510, spearman=0.519, mi=0.0924
  Pearson=0.5098, Spearman=0.5186, MI=0.092414
- [INFO] `feature_outcome_correlation_sends_since_last_action`: sends_since_last_action: pearson=-0.187, spearman=-0.212, mi=0.0734
  Pearson=-0.1865, Spearman=-0.2116, MI=0.073360
- [INFO] `feature_outcome_correlation_lifetime_actions_prior`: lifetime_actions_prior: pearson=0.385, spearman=0.237, mi=0.0628
  Pearson=0.3850, Spearman=0.2373, MI=0.062758
- [WARNING] `feature_feature_correlation_actioned_last_3_actioned_last_5`: High correlation: actioned_last_3 ↔ actioned_last_5 |r|=0.912
  Pearson r=0.9119 exceeds threshold 0.85
- [WARNING] `feature_feature_correlation_actioned_last_5_actioned_last_10`: High correlation: actioned_last_5 ↔ actioned_last_10 |r|=0.915
  Pearson r=0.9147 exceeds threshold 0.85
- [INFO] `row_alignment_audit_pass`: Row alignment audit passed
  Row keys unique; no nulls in row keys or outcome.
- [INFO] `eda_cohort_baseline_2024`: Cohort 2024: rate=0.0716, n_rows=398261
  relative_lift=1.000
- [INFO] `eda_cohort_baseline_2025`: Cohort 2025: rate=0.0974, n_rows=701799
  relative_lift=1.359
- [WARNING] `eda_cohort_baseline_2026`: Cohort 2026: rate=0.1195, n_rows=190134
  relative_lift=1.668
