# California Housing Demo — Goal

This is the **end-to-end test bed** for the ds3 autoloop wrapper.

**Dataset**: `sklearn.datasets.fetch_california_housing` — 20,640 rows, 8 numeric features,
target `median_house_value` (continuous, $ thousands).

**Task**: regression (predict median_house_value). Primary metric: **R²**.

**Baseline**: `sklearn.linear_model.Ridge(alpha=1.0)` on the 8 raw features yields R² ≈ 0.60.

**Autoloop's job**: discover feature transformations that lift R² toward 0.80+.
Obvious wins it should find autonomously:
- `rooms_per_household` = `AveRooms / AveOccup`
- `bedrooms_per_room` = `AveBedrms / AveRooms`
- `log_median_income` = `log1p(MedInc)`
- `population_per_household` = `Population / AveOccup`
- latitude × longitude interaction
- HouseAge bucketing

**Goal acceptance**: this project exists to validate the autoloop runs end-to-end.
Lift target is secondary; observing 3 clean iterations is the primary deliverable.
