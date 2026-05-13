"""run_03_reverse: Time-Reversal Test shadow run — DEFERRED.

Validates that the forward walk-forward signal is not a temporal artefact by
training on the last 22% of dates and testing on the first 22%.  A model with
genuine signal should perform near-chance when trained on the future and tested
on the past; strong above-chance performance in this direction flags temporal
leakage.

Deferred in v1: the harness uses a forward-only quarterly walk-forward driven by
_quarterly_folds() which is not exposed as a public API.  Implementing the reverse
split inline would require duplicating the confound-score computation, residualization,
bootstrap CI, and secondary-metric aggregation logic — a significant maintenance
surface.

Implementation path for v2:
    1. Add a ``reverse_time: bool = False`` parameter to harness() (or expose a
       public ``build_folds(df, reverse=False)`` helper).
    2. Call harness(..., reverse_time=True) here and record the result under
       ``auc_reverse_time`` on the CivicShoutResultRow.
    3. Remove this stub.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "run_03_reverse requires harness API support for reverse-time splits. "
        "Deferred to v2 — see module docstring for implementation path."
    )


if __name__ == "__main__":
    main()
