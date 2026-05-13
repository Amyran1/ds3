# winsorize_heavy_tails Feature Plan

Brainstorm: `BS-2026-05-13-wh008` (autoloop iteration 6, T1, score 0.82).
`source.finding_id`: null (seed item, no upstream finding to cite).

## 1. Why Build This
- Hunch or evidence: brainstorm hypothesis — `AveRooms`, `AveBedrms`,
  `AveOccup`, and `Population` have heavy right tails with extreme outliers
  (e.g., `AveRooms > 100`, `AveOccup > 1000` in a few rows). These high-leverage
  points drag the OLS / Ridge fit. Winsorizing the four columns at the 99th
  percentile is a pure preprocessing transform orthogonal to the ratio / log /
  interaction families already explored, and is expected to lift R² over the
  run_01 raw-feature baseline (R² = 0.6014).
- Decision this could change: whether outlier-capping preprocessing — distinct
  from any prior challenger family — is worth promoting to a comparison-scope
  run on top of (or in place of) the rooms_per_household / population_per_household
  ratio families.
- Prior run, EDA, user request, or external data source: run_01 baseline.
  No upstream `FeatureGapFinding` (planner-seeded brainstorm, finding_id null).
- Success signal before harness evaluation: the four winsorized columns are
  finite, non-NaN, ≤ each column's 99th-percentile cap; counts of rows clipped
  per column are non-zero (proof the preprocessing actually fired).

## 2. Source Data
- Source artifact or data structure: `sklearn.datasets.fetch_california_housing`
  via `projects.california_housing_demo.data.load_data()`.
- Current grain: one row per census block group.
- Expected row count / entity count: 20,640.
- Required source versions or fingerprints: sklearn-pinned via project deps.
- Data access path: `from projects.california_housing_demo.data import load_data`.

## 3. Source-To-Feature Steps
1. Load: `load_data()` returns a polars DataFrame with `AveRooms`, `AveBedrms`,
   `AveOccup`, `Population` (and other raw cols).
2. Filter/project: none — all rows eligible; winsorization is a global transform.
3. Join or align: none — single source.
4. Transform: for each of `AveRooms`, `AveBedrms`, `AveOccup`, `Population`,
   compute the column's 99th-percentile threshold on the input frame, then
   produce a new column `{col}_winz = min({col}, p99)`. The lower tail is left
   uncapped (the documented outlier problem is upper-tail only; no negative or
   near-zero outliers documented for these columns).
5. Aggregate to harness row grain: already at harness grain (one row per
   block group); no aggregation required.
6. Validate row keys, nulls, leakage, and schema:
   - Inputs guaranteed finite by sklearn dataset; assert via polars
     `null_count()` and `is_finite()` on the four raw columns.
   - Output dtype: float64.
   - Each winsorized column satisfies `col_winz <= raw_col` (since lower tail
     uncapped) and `max(col_winz) == p99(raw_col)` by construction.
   - No leakage — preprocessing computed on the input data passed to
     `add_features`. Because `run_06.py` pre-samples to 10% before calling
     `add_features`, the winsorization thresholds are computed on the smoke
     sample, which is the same convention used by every other linear-model
     preprocessing step in this codebase (z-score, etc.) — and a precondition
     for a fair R² lift comparison against the prior smoke runs (run_02–run_05)
     that all sampled then transformed.
   - Threshold is the 99th percentile of the *input frame* (the 10% sample at
     smoke time); `add_features` is therefore pure with respect to its input.

## 4. Feature Identity And Cache Access
- Feature family name: `winsorize_heavy_tails`.
- Feature dictionary path:
  `projects/california_housing_demo/features/winsorize_heavy_tails/dictionary.yaml`.
- Cache module path:
  `projects.california_housing_demo.features.winsorize_heavy_tails.cache`
  exposing
  `add_features(df: pl.DataFrame) -> pl.DataFrame`,
  `FEATURE_COLS: list[str]` (the four `*_winz` names),
  `HEAVY_TAIL_RAW_COLS: list[str]` (the four raw cols that `run_06.py` must
  drop from the feature list when using `*_winz` instead), and
  `WINSORIZE_QUANTILE: float = 0.99`.
- Planned local path: in-memory only (added on-the-fly in `run_06.py`);
  20,640-row build is sub-millisecond. No parquet cache.
- Planned remote path, or reason for local-only: local-only — four derived
  float columns over a 20k-row sklearn-shipped dataset.
- Versioning rule: bump via new `dictionary.yaml` version field if the
  transformation or quantile threshold changes.
- Harness smoke load example:
  ```python
  data = add_features(load_data())
  feature_cols = [c for c in RAW_FEATURE_COLS if c not in HEAVY_TAIL_RAW_COLS] + FEATURE_COLS
  ```
  This swap is the whole point of this family — passing both raw and
  winsorized columns would let Ridge see the outliers again and defeat the
  preprocessing.

## 5. EDA Visualizations
- Outcome distribution: handled at project EDA layer (stub).
- Feature distributions: skipped — EDA harness in this project is a stub.
  Sanity-checked via summary stats in the cache module smoke check: per-column
  `(min, max, p99, mean, null_count, n_clipped)` for raw and winsorized
  versions, asserting `n_clipped > 0` for each of the four columns on the full
  20,640-row dataset.
- Missingness: assert zero nulls on raw and winsorized columns.
- Feature-vs-outcome plots: skipped — toy demo; signal will be picked up by
  the run_06 R² delta.
- Correlation or redundancy checks: rely on harness R² lift over the run_01
  baseline. Winsorized columns substitute for raw columns (not added on top),
  so collinearity vs raw is moot.
- Segment/time/entity plots if relevant: none.

## 6. Patterns And Anti-Patterns
- Similar local patterns reviewed:
  - `projects/california_housing_demo/features/population_per_household/` —
    single vectorized polars transform per block group. This family extends the
    pattern from one derived column to four; the cache shape (cache.py +
    dictionary.yaml + timing_performance.jsonl) mirrors that exactly.
  - `projects/california_housing_demo/features/bedrooms_per_room/` — same shape.
- Patterns to reuse: import `RAW_FEATURE_COLS` and `OUTCOME_VARIABLE` from
  `data.py`; mirror `run_05.py` structure (pre-sample → `add_features` →
  Pydantic `RunResultMetadata` → `run_record(...)`).
- Anti-patterns to avoid:
  - Mutating `data.py` to apply winsorization (forbidden by autoloop; also
    bypasses feature-family ownership).
  - Computing winsorization inside `harness.py` (forbidden by autoloop; also
    violates layer ownership).
  - Passing BOTH the raw and `*_winz` columns to the harness — defeats the
    preprocessing because Ridge would still see the outliers through the raw
    columns.
  - Hardcoding p99 thresholds at module-import time — thresholds must be
    computed from the input frame (the 10% smoke sample at smoke time) inside
    `add_features` to keep the function pure w.r.t. its argument.
- Open design risks:
  - Winsorization on the 10% smoke sample uses sample-derived quantiles, not
    population quantiles. With seed 42 and ~2,064 rows, the p99 estimate has
    sampling variance, so smoke-scope R² is a noisier signal than it would be
    at full scale. Tolerated for smoke scope (the comparison run, if this
    challenger is promoted, would re-compute p99 on the full dataset).
  - Lower-tail uncapped. If any of the four columns has a left-tail outlier
    pattern (not documented in the brainstorm), it would not be addressed.
    Tolerated for v1; revisit if smoke shows no lift.

## 7. Runtime Calibration

### Comparable features (cite `timing_performance.jsonl` rows)
| Feature | Pattern overlap | Scale | rows/s | users/s | Source row |
|---|---|---|---|---|---|
| `population_per_household` | partial — one vectorized polars op vs four here | full (20,640) | 1,127,894 | n/a (no user grain) | `projects/california_housing_demo/features/population_per_household/timing_performance.jsonl:L1` |
| `bedrooms_per_room` | partial — one vectorized polars op vs four here | full (20,640) | 1,503,414 | n/a (no user grain) | `projects/california_housing_demo/features/bedrooms_per_room/timing_performance.jsonl:L1` |

### Predicted throughput for this feature
- Predicted rows/s @ full: ~280K (= comparable × 0.25, because this family
  performs four quantile-computation + clip operations vs the comparable's
  single vectorized division; four ops at ~equal cost ≈ 4× wall time, and the
  quantile call is slightly more expensive than a division).
- n_eligible_rows expected: 20,640.
- **Predicted full runtime: ~0.075 s** (= 20,640 ÷ 280K).

### Sample-tier gates
| Tier | Users | Rows | Expected elapsed | Ceiling | Bail-out trigger |
|---|---|---|---|---|---|
| Full | n/a (no user grain) | 20,640 | ~0.075 s | 5 s | observed >2× ceiling |

Sample-tier gates collapse to a single full-scale row because the entire
dataset is 20,640 rows and the operation is four vectorized polars ops. The
sample-first gate from the `feature` skill will run on a 10% sample
(2,064 rows). Bail-out trigger: observed rows/s < 140K (i.e. >2× slower than
predicted), since that would indicate a Python-loop fallback rather than a
real cost concern.

### Cost / memory notes (only if non-trivial)
- API: none. Storage: none (in-memory). Peak memory: ~640 KB (four float64
  columns × 20,640 rows).

### Re-plan trigger
- If 2,064-row sample observes >2× the predicted runtime (i.e. <140K rows/s):
  pause, investigate. The polars quantile call is expected to be C-backed and
  efficient; a slowdown would indicate something is wrong with the loop.
- Timing-performance handoff required? no — trivial vectorized ops with a
  family of comparable rows already on file.

## 8. Handoff
- Next skill: `feature` (implement `cache.py` + `dictionary.yaml` + smoke
  check + sample-first gate), then `run_06.py` invocation.
- Cache skill needed? no — in-memory only.
- Timing-performance skill needed? no.
- Files expected from implementation:
  - `projects/california_housing_demo/features/winsorize_heavy_tails/cache.py`
  - `projects/california_housing_demo/features/winsorize_heavy_tails/dictionary.yaml`
  - `projects/california_housing_demo/features/winsorize_heavy_tails/__init__.py`
  - `projects/california_housing_demo/features/winsorize_heavy_tails/timing_performance.jsonl`
