# log_median_income Feature Plan

Brainstorm: `BS-2026-05-13-lmi002` (autoloop iteration 2, T1, score 0.65).
`source.finding_id`: null (seed item, no upstream FeatureGapFinding to cite).

## 1. Why Build This
- Hunch or evidence: brainstorm hypothesis — `MedInc` has a long right tail
  (top decile income is many multiples of the median), so a linear coefficient
  on raw `MedInc` cannot fit both the lower-income mid-slope and the
  diminishing-returns bend at the top. `log1p(MedInc)` flattens the tail and
  should improve linearity with `median_house_value` in a linear (Ridge) model.
- Decision this could change: whether a single `log1p(MedInc)` column (added
  alongside the raw `RAW_FEATURE_COLS`) lifts R² over the run_01 baseline
  (`r2 = 0.6014`) on the full-data champion comparison group
  `california_housing_v1`.
- Prior run, EDA, user request, or external data source: run_01 (champion;
  raw 8-feature Ridge) and run_02 (smoke; rooms_per_occupant ratio). No
  upstream EDA or discovery finding cited (seed item; `source.finding_id`
  is null).
- Success signal before harness evaluation: feature is finite, non-NaN
  across all 20,640 rows; broad non-degenerate distribution; correlation
  with `median_house_value` at least as strong as raw `MedInc`.

## 2. Source Data
- Source artifact or data structure:
  `sklearn.datasets.fetch_california_housing` via
  `projects.california_housing_demo.data.load_data()`.
- Current grain: one row per census block group.
- Expected row count / entity count: 20,640.
- Required source versions or fingerprints: sklearn-pinned via project deps.
- Data access path:
  `from projects.california_housing_demo.data import load_data`.

## 3. Source-To-Feature Steps
1. Load: `load_data()` returns a polars DataFrame with `MedInc` (float64).
2. Filter/project: none — all rows eligible.
3. Join or align: none — single source.
4. Transform: `log_medinc = log1p(MedInc)`. `log1p` (rather than `log`) is
   defensive — sklearn guarantees `MedInc > 0` (units are tens of thousands
   of USD), but `log1p` is monotone, identical at scale, and well-defined
   at zero in case future data drift introduces a zero value.
5. Aggregate to harness row grain: already at harness grain (one row per
   block group); no aggregation required.
6. Validate row keys, nulls, leakage, and schema:
   - `MedInc >= 0` over all rows (sklearn guarantee; assert via min).
   - No nulls expected; assert via polars `null_count()`.
   - No leakage — function of pre-outcome input only.
   - Output dtype: float64.

## 4. Feature Identity And Cache Access
- Feature family name: `log_median_income`.
- Feature dictionary path:
  `projects/california_housing_demo/features/log_median_income/dictionary.yaml`.
- Cache module path:
  `projects.california_housing_demo.features.log_median_income.cache`
  exposing `add_features(df: pl.DataFrame) -> pl.DataFrame` and
  `FEATURE_COLS: list[str]`.
- Planned local path: in-memory only (added on-the-fly in `run_03.py`);
  20,640-row build is sub-millisecond. No parquet cache.
- Planned remote path, or reason for local-only: local-only — payload is one
  derived float column over a 20k-row sklearn-shipped dataset.
- Versioning rule: bump via new `dictionary.yaml` version field if the
  transformation changes (e.g., switching `log1p` → `log`, or adding a
  shift constant).
- Harness smoke load example: `data = add_features(load_data())`, then
  `feature_cols = RAW_FEATURE_COLS + FEATURE_COLS`.

## 5. EDA Visualizations
- Outcome distribution: handled at project EDA layer (stub).
- Feature distributions: skipped — single derived column; project EDA harness
  is a stub. Sanity-checked via min/max/mean/null_count in the cache module
  smoke test (mirrors rooms_per_household precedent).
- Missingness: assert zero nulls in the implementation gate.
- Feature-vs-outcome plots: skipped — toy demo; signal will be picked up by
  the run_03 R² delta vs run_01 champion.
- Correlation or redundancy checks: rely on the harness R² lift over the
  run_01 baseline.
- Segment/time/entity plots if relevant: none.

## 6. Patterns And Anti-Patterns
- Similar local patterns reviewed: `rooms_per_household` (autoloop iter 1) —
  a single vectorized polars derivation on the same 20,640-row sklearn
  dataset, added in-memory via `add_features(df) -> df` and concatenated
  onto `RAW_FEATURE_COLS` at run time. Direct structural analogue.
- Patterns to reuse:
  - `add_features(df) -> df` signature with module-level `FEATURE_COLS`.
  - `RAW_FEATURE_COLS + FEATURE_COLS` concatenation in the run script.
  - In-memory feature (no parquet cache) for trivial vectorized ops.
  - Smoke run with `sample_frac=0.10` and seed 42, mirroring run_02.
- Anti-patterns to avoid:
  - Mutating `data.py` to add `log_medinc` (forbidden by autoloop;
    bypasses feature-family ownership).
  - Computing `log1p` inside `harness.py` (forbidden by autoloop; violates
    layer ownership).
  - Writing a parquet cache for a 20k-row sub-ms compute.
- Open design risks: none material.

## 7. Runtime Calibration

### Comparable features (cite `timing_performance.jsonl` rows)
| Feature | Pattern overlap | Scale | rows/s | users/s | Source row |
|---|---|---|---|---|---|
| rooms_per_household | single vectorized polars op on same 20,640-row sklearn dataset | smoke (2,064 rows; projects to full) | 1,148,687 | n/a (no user grain) | `projects/california_housing_demo/features/rooms_per_household/timing_performance.jsonl:1` |

### Predicted throughput for this feature
- Predicted rows/s @ full: ≥1.1M (= rooms_per_household × 1.0, because
  `pl.col("MedInc").log1p()` is the same vectorized-arithmetic class as
  `AveRooms / AveOccup` — one elementwise op over one float64 column on the
  same dataset).
- n_eligible_rows expected: 20,640.
- **Predicted full runtime: <0.02 s** (= 20,640 ÷ 1.1M).

### Sample-tier gates
| Tier | Users | Rows | Expected elapsed | Ceiling | Bail-out trigger |
|---|---|---|---|---|---|
| Full | n/a (no user grain) | 20,640 | <0.02 s | 5 s | observed >2× ceiling |

Sample-tier gates collapse to a single full-scale row because the entire
dataset is 20,640 rows and the operation is a single vectorized `log1p`.
No `timing-performance` handoff required; trivial compute by inspection
and by direct analogy to `rooms_per_household`.

### Cost / memory notes (only if non-trivial)
- API: none. Storage: none (in-memory). Peak memory: ~160 KB (one float64
  column × 20,640 rows).

### Re-plan trigger
- If full build observed > 5 s: pause, investigate (likely indicates an
  unintended Python-loop fallback).
- Timing-performance handoff required? no — trivial vectorized op with a
  direct comparable.

## 8. Handoff
- Next skill: `feature` (implement `cache.py` + `dictionary.yaml` + smoke
  check), then write `run_03.py` and invoke via the `run` skill pattern.
- Cache skill needed? no — in-memory only.
- Timing-performance skill needed? no.
- Files expected from implementation:
  - `projects/california_housing_demo/features/log_median_income/cache.py`
  - `projects/california_housing_demo/features/log_median_income/dictionary.yaml`
  - `projects/california_housing_demo/features/log_median_income/__init__.py`
