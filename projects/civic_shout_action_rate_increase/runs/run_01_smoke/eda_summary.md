# EDA Summary — run_01_smoke (civic_shout_action_rate_increase)

## Headline
| metric | value |
|---|---|
| outcome_positive_rate | 0.09267986054810362 |
| n_findings_total | 8 |
| n_findings_info | 7 |
| n_findings_warning | 1 |
| n_findings_error | 0 |
| feature_feature_max_abs_pearson | None |
| cohort_drift_max_relative_lift | 1.668143163426224 |
| elapsed_seconds | 4.69 |

## Method: outcome_distribution
- elapsed: 0.03s
- n_findings: 2
![outcome_distribution.png](outcome_distribution.png)

## Method: univariate_numeric
- elapsed: 0.00s
- n_findings: 0

## Method: univariate_categorical
- elapsed: 0.06s
- n_findings: 2
![univariate_categorical.png](univariate_categorical.png)

## Method: feature_outcome_correlation
- elapsed: 0.00s
- n_findings: 0

## Method: feature_feature_correlation
- elapsed: 0.00s
- n_findings: 0

## Method: missingness_pattern
- elapsed: 4.52s
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
- [INFO] `univariate_categorical_action_in_last_5`: action_in_last_5: cardinality=2, missingness=0.000
  Top values: [(False, '0.763'), (True, '0.237')]
- [INFO] `univariate_categorical_is_in_action_streak`: is_in_action_streak: cardinality=2, missingness=0.000
  Top values: [(False, '0.980'), (True, '0.020')]
- [INFO] `row_alignment_audit_pass`: Row alignment audit passed
  Row keys unique; no nulls in row keys or outcome.
- [INFO] `eda_cohort_baseline_2024`: Cohort 2024: rate=0.0716, n_rows=398261
  relative_lift=1.000
- [INFO] `eda_cohort_baseline_2025`: Cohort 2025: rate=0.0974, n_rows=701799
  relative_lift=1.359
- [WARNING] `eda_cohort_baseline_2026`: Cohort 2026: rate=0.1195, n_rows=190134
  relative_lift=1.668
