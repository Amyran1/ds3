from __future__ import annotations

from typing import Literal, NamedTuple

_Phase = Literal["EXPLORE", "EXPLOIT"]

_WEIGHTS: dict[_Phase, dict[str, float]] = {
    "EXPLORE": {"signal": 0.30, "novelty": 0.40, "cost": 0.10, "similar_success": 0.20},
    "EXPLOIT": {"signal": 0.55, "novelty": 0.10, "cost": 0.10, "similar_success": 0.25},
}

for _p, _w in _WEIGHTS.items():
    assert abs(sum(_w.values()) - 1.0) < 1e-9, f"weights for {_p} do not sum to 1.0"


def phase(
    iter_id: int,
    iterations_max: int,
    explore_fraction: float = 0.4,
) -> _Phase:
    if (iter_id - 1) / iterations_max < explore_fraction:
        return "EXPLORE"
    return "EXPLOIT"


def weights(phase: _Phase) -> dict[str, float]:
    return dict(_WEIGHTS[phase])


def similarity_threshold(phase: _Phase) -> float:
    return 0.85 if phase == "EXPLORE" else 0.94


def tier_eligibility(phase: _Phase) -> set[str]:
    if phase == "EXPLORE":
        return {"T0", "T1", "T2"}
    return {"T0", "T1"}


def score_item(
    signal: float,
    novelty: float,
    cost: float,
    similar_success: float,
    phase: _Phase,
) -> float:
    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    s = _clamp(signal)
    n = _clamp(novelty)
    c = _clamp(cost)
    ss = _clamp(similar_success)
    w = _WEIGHTS[phase]
    raw = w["signal"] * s + w["novelty"] * n + w["cost"] * c + w["similar_success"] * ss
    return max(0.0, min(1.0, raw))


def tier_from_score(score: float) -> Literal["T0", "T1", "T2", "T3"]:
    if score >= 0.80:
        return "T0"
    if score >= 0.60:
        return "T1"
    if score >= 0.40:
        return "T2"
    return "T3"


class CurriculumDecision(NamedTuple):
    phase: _Phase
    weights: dict[str, float]
    threshold: float
    tier_eligibility: set[str]


def decide(
    iter_id: int,
    iterations_max: int,
    explore_fraction: float = 0.4,
) -> CurriculumDecision:
    p = phase(iter_id, iterations_max, explore_fraction)
    return CurriculumDecision(
        phase=p,
        weights=weights(p),
        threshold=similarity_threshold(p),
        tier_eligibility=tier_eligibility(p),
    )
