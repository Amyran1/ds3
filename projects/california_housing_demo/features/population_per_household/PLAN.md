# population_per_household Feature Plan

Brainstorm: `BS-2026-05-13-pph004` (autoloop iteration 4, T2, score 0.5).
`source.finding_id`: null (seed item, no upstream finding to cite).

## 1. Why Build This
- Hunch or evidence: brainstorm hypothesis — `Population / AveOccup` measures
  neighborhood density at the household-occupant scale (block-group population
  divided by average occupants-per-household ≈ household count). A denser
  proxy for urban-vs-suburban character than either raw column alone, and
  expected to explain residuals in urban block groups where high `Population`
  with moderate `AveOccup` indicates many small households.
- Decision this could change: whether the population/household density ratio
  lifts R² over the run_01 raw-feature baseline (R² = 0.6014) and over the
  run_02 / run_03 / run_04 challengers, all of which lifted negative on the
  10% smoke sample.
- Prior run, EDA, user request, or external data source: run_01 baseline.
  Seed brainstorm item; no `source.finding_id` upstream.
- Success signal before harness evaluation: feature is finite, non-NaN
  across all 20,640 rows; ratio is strictly positive (`Population > 0`,
  `AveOccup > 0` both guaranteed by sklearn dataset).

## 2. Source Data
- Source artifact or data structure: `sklearn.datasets.fetch_california_housing`
  via `projects.california_housing_demo.data.load_data()`.
- Current grain: one row per census block group.
- Expected row count / entity count: 20,640.
- Required source versions or fingerprints: sklearn-pinned via project deps.
- Data access path: `from projects.california_housing_demo.data import load_data`.

## 3. Source-To-Feature Steps
1. Load: `load_data()` returns a polars DataFrame with `Population`, `AveOccup`.
2. Filter/project: none — all rows eligible.
3. Join or align: none — single source.
4. Transform: `population_per_household = Population / AveOccup`.
5. Aggregate to harness row grain: already at harness grain (one row per
   block group); no aggregation required.
6. Validate row keys, nulls, leakage, and schema:
   - `AveOccup > 0` over all rows (sklearn guarantees this; assert before
     division).
   - No nulls expected; assert via polars `null_count()`.
   - No leakage — function of pre-outcome inputs only.
   - Output dtype: float64.

## 4. Feature Identity And Cache Access
- Feature family name: `population_per_household`.
- Feature dictionary path:
  `projects/california_housing_demo/features/population_per_household/dictionary.yaml`.
- Cache module path:
  `projects.california_housing_demo.features.population_per_household.cache`
  exposing `add_features(df: pl.DataFrame) -> pl.DataFrame` and
  `FEATURE_COLS: list[str]`.
- Planned local path: in-memory only (added on-the-fly in `run_05.py`);
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
  the run_05 R² delta.
- Correlation or redundancy checks: rely on the harness R² lift over the
  run_01 baseline (raw `Population` + `AveOccup` already in
  `RAW_FEATURE_COLS`; if the household-count proxy adds signal, lift > 0).
- Segment/time/entity plots if relevant: none.

## 6. Patterns And Anti-Patterns
- Similar local patterns reviewed:
  - `projects/california_housing_demo/features/bedrooms_per_room/` — same
    pattern (single vectorized polars division per block group). Mirror its
    `cache.py` / `dictionary.yaml` shape exactly.
  - `projects/california_housing_demo/features/rooms_per_household/` — same
    pattern (single vectorized polars division).
- Patterns to reuse: import `RAW_FEATURE_COLS` and `OUTCOME_VARIABLE` from
  `data.py`; mirror `run_03.py` / `run_04.py` structure (Pydantic
  `RunResultMetadata` → `run_record(...)`).
- Anti-patterns to avoid:
  - Mutating `data.py` to add the ratio (forbidden by autoloop; also bypasses
    feature-family ownership).
  - Computing the ratio inside `harness.py` (forbidden by autoloop; also
    violates layer ownership).
  - Writing a parquet cache for a 20k-row sub-ms compute.
- Open design risks: redundancy with raw `Population` and `AveOccup` already
  in the baseline — the linear-Ridge model can in principle recover most of
  the `Population / AveOccup` signal from the two raw columns. Tolerated:
  the autoloop's purpose is to test whether the explicit ratio still adds
  measurable lift on the 10% smoke sample.

## 7. Runtime Calibration

### Comparable features (cite `timing_performance.jsonl` rows)
| Feature | Pattern overlap | Scale | rows/s | users/s | Source row |
|---|---|---|---|---|---|
| `bedrooms_per_room` | identical (single vectorized polars division over the same 20,640-row dataset) | full (20,640) | 1,503,414 | n/a (no user grain) | `projects/california_housing_demo/features/bedrooms_per_room/timing_performance.jsonl:L1` |

### Predicted throughput for this feature
- Predicted rows/s @ full: ~1.5M (= comparable × 1.0, because the operation
  is byte-identical in shape: one float64 division over the same DataFrame).
- n_eligible_rows expected: 20,640.
- **Predicted full runtime: ~0.014 s** (= 20,640 ÷ 1,503,414).

### Sample-tier gates
| Tier | Users | Rows | Expected elapsed | Ceiling | Bail-out trigger |
|---|---|---|---|---|---|
| Full | n/a (no user grain) | 20,640 | ~0.014 s | 5 s | observed >2× ceiling |

Sample-tier gates collapse to a single full-scale row because the entire
dataset is 20,640 rows and the operation is a vectorized division. The
sample-first gate from the `feature` skill will still run on a 10% sample
(2,064 rows) and compare to the `bedrooms_per_room` benchmark; >2× means
something has gone wrong (likely a Python-loop fallback), not a real
runtime concern.

### Cost / memory notes (only if non-trivial)
- API: none. Storage: none (in-memory). Peak memory: ~160 KB (one float64
  column × 20,640 rows).

### Re-plan trigger
- If 2,064-row sample observes >2× the `bedrooms_per_room` benchmark
  (i.e. <751,707 rows/s): pause, investigate.
- Timing-performance handoff required? no — trivial vectorized op with a
  byte-identical comparable already on file.

## 8. Handoff
- Next skill: `feature` (implement `cache.py` + `dictionary.yaml` + smoke
  check + sample-first gate), then `run_05.py` invocation.
- Cache skill needed? no — in-memory only.
- Timing-performance skill needed? no.
- Files expected from implementation:
  - `projects/california_housing_demo/features/population_per_household/cache.py`
  - `projects/california_housing_demo/features/population_per_household/dictionary.yaml`
  - `projects/california_housing_demo/features/population_per_household/__init__.py`
  - `projects/california_housing_demo/features/population_per_household/timing_performance.jsonl`
