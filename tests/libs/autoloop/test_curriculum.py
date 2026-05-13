from __future__ import annotations

from libs.autoloop.curriculum import (
    CurriculumDecision,
    decide,
    phase,
    score_item,
    similarity_threshold,
    tier_eligibility,
    tier_from_score,
    weights,
)


def test_phase_early_iter_is_explore() -> None:
    assert phase(1, 70) == "EXPLORE"


def test_phase_boundary_still_explore() -> None:
    assert phase(28, 70) == "EXPLORE"


def test_phase_just_past_boundary_is_exploit() -> None:
    assert phase(29, 70) == "EXPLOIT"


def test_phase_final_iter_is_exploit() -> None:
    assert phase(70, 70) == "EXPLOIT"


def test_weights_explore_sum_to_one() -> None:
    w = weights("EXPLORE")
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_weights_exploit_sum_to_one() -> None:
    w = weights("EXPLOIT")
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_weights_explore_novelty_higher_than_exploit() -> None:
    assert weights("EXPLORE")["novelty"] > weights("EXPLOIT")["novelty"]


def test_weights_exploit_signal_higher_than_explore() -> None:
    assert weights("EXPLOIT")["signal"] > weights("EXPLORE")["signal"]


def test_similarity_threshold_explore() -> None:
    assert similarity_threshold("EXPLORE") == 0.85


def test_similarity_threshold_exploit() -> None:
    assert similarity_threshold("EXPLOIT") == 0.94


def test_tier_eligibility_exploit_excludes_t2_t3() -> None:
    eligible = tier_eligibility("EXPLOIT")
    assert "T2" not in eligible
    assert "T3" not in eligible
    assert "T0" in eligible
    assert "T1" in eligible


def test_score_item_clamps_out_of_range_inputs() -> None:
    score = score_item(signal=2.0, novelty=-1.0, cost=1.5, similar_success=0.5, phase="EXPLORE")
    assert 0.0 <= score <= 1.0


def test_tier_from_score_high_is_t0() -> None:
    assert tier_from_score(0.85) == "T0"


def test_tier_from_score_low_is_t3() -> None:
    assert tier_from_score(0.05) == "T3"


def test_tier_from_score_mid_boundaries() -> None:
    assert tier_from_score(0.80) == "T0"
    assert tier_from_score(0.79) == "T1"
    assert tier_from_score(0.60) == "T1"
    assert tier_from_score(0.59) == "T2"
    assert tier_from_score(0.40) == "T2"
    assert tier_from_score(0.39) == "T3"


def test_decide_returns_curriculum_decision_with_all_fields() -> None:
    decision = decide(iter_id=1, iterations_max=70)
    assert isinstance(decision, CurriculumDecision)
    assert decision.phase in {"EXPLORE", "EXPLOIT"}
    assert isinstance(decision.weights, dict)
    assert isinstance(decision.threshold, float)
    assert isinstance(decision.tier_eligibility, set)
