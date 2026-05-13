# bedrooms_per_room Feature Plan

Brainstorm: `BS-2026-05-13-bpr003` (autoloop iteration 3, T1, score 0.6).
`source.finding_id`: null (seed item, no upstream finding to cite).

## 1. Why Build This
- Hunch or evidence: brainstorm hypothesis — `AveBedrms / AveRooms` is a
  denser proxy for housing quality than the two raw columns separately.
  Both columns are already per-household means in the sklearn California
  Housing dataset, so the ratio is directly interpretable as "share of
  rooms that are bedrooms" (a classic housing-quality proxy — higher
  ratios indicate smaller, denser units).
- Decision this could change: whether the share-of-bedrooms ratio lifts
  R² over the run_01 raw-feature baseline (R² = 0.6014) and over the
  run_02 / run_03 challengers.
- Prior run, EDA, user request, or external data source: run_01 baseline.
  Seed brainstorm item; no `source.finding_id` upstream.
- Success signal before harness evaluation: feature is finite, non-NaN
  across all 20,640 rows; ratio is bounded in `[0, 1]` for the vast
  majority of rows (with a small upper tail above 1 from imputation
  artifacts in the public dataset, tolerated as raw).

## 2. Source Data
- Source artifact or data structure: `sklearn.datasets.fetch_california_housing`
  via `projects.california_housing_demo.data.load_data()`.
- Current grain: one row per census block group.
- Expected row count / entity count: 20,640.
- Required source versions or fingerprints: sklearn-pinned via project deps.
- Data access path: `from projects.california_housing_demo.data import load_data`.

## 3. Source-To-Feature Steps
1. Load: `load_data()` returns a polars DataFrame with `AveRooms`, `AveBedrms`.
2. Filter/project: none — all rows eligible.
3. Join or align: none — single source.
4. Transform: `bedrooms_per_room = AveBedrms / AveRooms`.
5. Aggregate to harness row grain: already at harness grain (one row per
   block group); no aggregation required.
6. Validate row keys, nulls, leakage, and schema:
   - `AveRooms > 0` over all rows (sklearn guarantees this; assert before
     division).
   - No nulls expected; assert via polars `null_count()`.
   - No leakage — function of pre-outcome inputs only.
   - Output dtype: float64.

## 4. Feature Identity And Cache Access
- Feature family name: `bedrooms_per_room`.
- Feature dictionary path:
  `projects/california_housing_demo/features/bedrooms_per_room/dictionary.yaml`.
- Cache module path: `projects.california_housing_demo.features.bedrooms_per_room.cache`
  exposing `add_features(df: pl.DataFrame) -> pl.DataFrame` and
  `FEATURE_COLS: list[str]`.
- Planned local path: in-memory only (added on-the-fly in `run_04.py`);
  20,640-row build is sub-millisecond. No parquet cache.
- Planned remote path, or reason for local-only: local-only — payload is one
  derived float column over a 20k-row sklearn-shipped dataset.
- Versioning rule: bump via new `dictionary.yaml` version field if the
  transformation changes.
- Harness smoke load example: `data = add_features(load_data())`, then
  `feature_cols = RAW_FEATURE_COLS + FEATURE_COLS`.

## 5. EDA Visualizations
- Outcome distribution: handled at project EDA layer (stub).
- Feature distributions: skipped — single derived column, EDA harness in this
  project is a stub. Sanity-checked via summary stats in the cache module
  smoke check (min, max, mean, null_count).
- Missingness: assert zero nulls in the implementation gate.
- Feature-vs-outcome plots: skipped — toy demo; signal will be picked up by
  the run_04 R² delta.
- Correlation or redundancy checks: rely on the harness R² lift over the
  run_01 baseline (raw `AveBedrms` + `AveRooms` already in `RAW_FEATURE_COLS`;
  if the ratio adds signal, lift > 0).
- Segment/time/entity plots if relevant: none.

## 6. Patterns And Anti-Patterns
- Similar local patterns reviewed:
  - `projects/california_housing_demo/features/rooms_per_household/` — same
    pattern (single vectorized polars division per block group). Mirror its
    `cache.py` / `dictionary.yaml` shape exactly.
  - `projects/california_housing_demo/features/log_median_income/` — single
    unary transform; same null/leakage discipline.
- Patterns to reuse: import `RAW_FEATURE_COLS` and `OUTCOME_VARIABLE` from
  `data.py`; mirror `run_02.py` / `run_03.py` structure (Pydantic
  `RunResultMetadata` → `run_record(...)`).
- Anti-patterns to avoid:
  - Mutating `data.py` to add the ratio (forbidden by autoloop; also bypasses
    feature-family ownership).
  - Computing the ratio inside `harness.py` (forbidden by autoloop; also
    violates layer ownership).
  - Writing a parquet cache for a 20k-row sub-ms compute.
- Open design risks: small upper tail of `AveBedrms / AveRooms > 1` from
  public-dataset imputation artifacts. Tolerated as raw — clipping or
  winsorization would be a downstream model decision, not a feature-family
  responsibility.

## 7. Runtime Calibration

### Comparable features (cite `timing_performance.jsonl` rows)
| Feature | Pattern overlap | Scale | rows/s | users/s | Source row |
|---|---|---|---|---|---|
| `rooms_per_household` | identical (single vectorized polars division over the same 20,640-row dataset) | full (20,640) | 1,148,687 | n/a (no user grain) | `projects/california_housing_demo/features/rooms_per_household/timing_performance.jsonl:L1` |

### Predicted throughput for this feature
- Predicted rows/s @ full: ~1.15M (= comparable × 1.0, because the operation
  is byte-identical in shape: one float64 division over the same DataFrame).
- n_eligible_rows expected: 20,640.
- **Predicted full runtime: ~0.018 s** (= 20,640 ÷ 1,148,687).

### Sample-tier gates
| Tier | Users | Rows | Expected elapsed | Ceiling | Bail-out trigger |
|---|---|---|---|---|---|
| Full | n/a (no user grain) | 20,640 | ~0.018 s | 5 s | observed >2× ceiling |

Sample-tier gates collapse to a single full-scale row because the entire
dataset is 20,640 rows and the operation is a vectorized division. The
sample-first gate from the `feature` skill will still run on a 10% sample
(2,064 rows) and compare to the `rooms_per_household` benchmark; >2× means
something has gone wrong (likely a Python-loop fallback), not a real
runtime concern.

### Cost / memory notes (only if non-trivial)
- API: none. Storage: none (in-memory). Peak memory: ~160 KB (one float64
  column × 20,640 rows).

### Re-plan trigger
- If 2,064-row sample observes >2× the `rooms_per_household` benchmark
  (i.e. <575,000 rows/s): pause, investigate.
- Timing-performance handoff required? no — trivial vectorized op with a
  byte-identical comparable already on file.

## 8. Handoff
- Next skill: `feature` (implement `cache.py` + `dictionary.yaml` + smoke
  check + sample-first gate), then `run_04.py` invocation.
- Cache skill needed? no — in-memory only.
- Timing-performance skill needed? no.
- Files expected from implementation:
  - `projects/california_housing_demo/features/bedrooms_per_room/cache.py`
  - `projects/california_housing_demo/features/bedrooms_per_room/dictionary.yaml`
  - `projects/california_housing_demo/features/bedrooms_per_room/__init__.py`
  - `projects/california_housing_demo/features/bedrooms_per_room/timing_performance.jsonl`
