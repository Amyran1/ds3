# rooms_per_household Feature Plan

Brainstorm: `BS-2026-05-13-r1h001` (autoloop iteration 1, T1, score 0.7).
`source.finding_id`: null (seed item, no upstream finding to cite).

## 1. Why Build This
- Hunch or evidence: brainstorm hypothesis — "rooms / occupants normalizes
  for household composition". `AveRooms` in the sklearn California Housing
  dataset is already rooms-per-household, so the proposed signal is actually
  `AveRooms / AveOccup` = rooms per occupant. The brainstorm-supplied name
  `rooms_per_household` is preserved per autoloop continuity; the computed
  semantic is rooms-per-occupant.
- Decision this could change: whether a per-occupant density ratio lifts
  R² over the raw `AveRooms` + `AveOccup` baseline (run_01 R² = 0.6014).
- Prior run, EDA, user request, or external data source: run_01 baseline.
- Success signal before harness evaluation: feature is finite, non-NaN
  across all 20,640 rows; broad distribution (not constant); correlates
  monotonically with `median_house_value` at a level comparable to MedInc.

## 2. Source Data
- Source artifact or data structure: `sklearn.datasets.fetch_california_housing`
  via `projects.california_housing_demo.data.load_data()`.
- Current grain: one row per census block group.
- Expected row count / entity count: 20,640.
- Required source versions or fingerprints: sklearn-pinned via project deps.
- Data access path: `from projects.california_housing_demo.data import load_data`.

## 3. Source-To-Feature Steps
1. Load: `load_data()` returns a polars DataFrame with `AveRooms`, `AveOccup`.
2. Filter/project: none — all rows eligible.
3. Join or align: none — single source.
4. Transform: `rooms_per_occupant = AveRooms / AveOccup`.
5. Aggregate to harness row grain: already at harness grain (one row per
   block group); no aggregation required.
6. Validate row keys, nulls, leakage, and schema:
   - `AveOccup > 0` over all rows (verify before division; sklearn guarantees
     this since it is derived from population / households where both > 0).
   - No nulls expected; assert via polars `null_count()`.
   - No leakage — function of pre-outcome inputs only.
   - Output dtype: float64.

## 4. Feature Identity And Cache Access
- Feature family name: `rooms_per_household`.
- Feature dictionary path:
  `projects/california_housing_demo/features/rooms_per_household/dictionary.yaml`.
- Cache module path: `projects.california_housing_demo.features.rooms_per_household.cache`
  exposing `add_features(df: pl.DataFrame) -> pl.DataFrame` and
  `FEATURE_COLS: list[str]`.
- Planned local path: in-memory only (added on-the-fly in `run_02.py`);
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
  the run_02 R² delta.
- Correlation or redundancy checks: rely on the harness R² lift over run_01
  baseline (run_01 used raw `AveRooms` + `AveOccup`; if the ratio adds
  signal, lift > 0).
- Segment/time/entity plots if relevant: none.

## 6. Patterns And Anti-Patterns
- Similar local patterns reviewed: none yet — this is the first feature
  family in `california_housing_demo`. run_01 used `RAW_FEATURE_COLS` only.
- Patterns to reuse: import `RAW_FEATURE_COLS` and `OUTCOME_VARIABLE` from
  `data.py`; mirror `run_01.py` structure (Pydantic `RunResultMetadata` →
  `run_record(...)`).
- Anti-patterns to avoid:
  - Mutating `data.py` to add the ratio (forbidden by autoloop; also bypasses
    feature-family ownership).
  - Computing the ratio inside `harness.py` (forbidden by autoloop; also
    violates layer ownership).
  - Writing a parquet cache for a 20k-row sub-ms compute.
- Open design risks: none material.

## 7. Runtime Calibration

### Comparable features (cite `timing_performance.jsonl` rows)
| Feature | Pattern overlap | Scale | rows/s | users/s | Source row |
|---|---|---|---|---|---|
| (none) | first feature family in project | n/a | n/a | n/a | n/a |

### Predicted throughput for this feature
- Predicted rows/s @ full: ≥10M (single vectorized polars division over 20,640
  float64 rows; bounded by interpreter overhead, not arithmetic).
- n_eligible_rows expected: 20,640.
- **Predicted full runtime: <0.01 s** (well below any meaningful gate).

### Sample-tier gates
| Tier | Users | Rows | Expected elapsed | Ceiling | Bail-out trigger |
|---|---|---|---|---|---|
| Full | n/a (no user grain) | 20,640 | <0.01 s | 5 s | observed >2× ceiling |

Sample-tier gates collapse to a single full-scale row because the entire
dataset is 20,640 rows and the operation is a vectorized division. No
`timing-performance` handoff required; trivial compute by inspection.

### Cost / memory notes (only if non-trivial)
- API: none. Storage: none (in-memory). Peak memory: ~160 KB (one float64
  column × 20,640 rows).

### Re-plan trigger
- If full build observed > 5 s: pause, investigate (likely indicates an
  unintended Python-loop fallback).
- Timing-performance handoff required? no — trivial vectorized op.

## 8. Handoff
- Next skill: `feature` (implement `cache.py` + `dictionary.yaml` + smoke
  check), then `run_02.py` invocation.
- Cache skill needed? no — in-memory only.
- Timing-performance skill needed? no.
- Files expected from implementation:
  - `projects/california_housing_demo/features/rooms_per_household/cache.py`
  - `projects/california_housing_demo/features/rooms_per_household/dictionary.yaml`
  - `projects/california_housing_demo/features/rooms_per_household/__init__.py`
