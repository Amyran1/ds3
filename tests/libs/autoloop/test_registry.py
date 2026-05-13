from __future__ import annotations

from pathlib import Path
from typing import Any


from libs.autoloop.registry import rebuild_registry
from libs.autoloop.state import (
    AutoloopConfig,
    BrainstormItem,
    BrainstormSource,
    ExecutorResult,
    IterationRecord,
    PlannerResult,
    State,
    append_jsonl,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(project: str = "test_proj") -> AutoloopConfig:
    raw: dict[str, Any] = {
        "project": project,
        "outcome_variable": "signed",
        "comparison_group": "cg1",
        "primary_metric": "auc",
    }
    return AutoloopConfig.model_validate(raw)


def _make_state(tmp_path: Path, cfg: AutoloopConfig) -> State:
    state = State(tmp_path, cfg)
    state.state_dir.mkdir(parents=True, exist_ok=True)
    return state


def _make_item(
    item_id: str,
    title: str,
    tier_score: float,
    status: str = "queued",
    tier: str = "T1",
    run_id: str | None = None,
    source_kind: str = "discovery",
    finding_id: str | None = None,
) -> BrainstormItem:
    return BrainstormItem(
        id=item_id,
        ts="2025-01-01T00:00:00+00:00",
        title=title,
        hypothesis="some hypothesis",
        source=BrainstormSource(kind=source_kind, finding_id=finding_id),
        feature_family_name_hint="ff",
        row_grain="user",
        content_hash="abc",
        tier=tier,
        tier_score=tier_score,
        status=status,
        run_id=run_id,
    )


def _write_brainstorm(state: State, items: list[BrainstormItem]) -> None:
    path = state.state_dir / "brainstorm.jsonl"
    for item in items:
        append_jsonl(path, item)


def _write_iteration(state: State, rec: IterationRecord) -> None:
    state.append_iteration(rec)


def _make_iteration_with_lift(
    iteration_id: int, brainstorm_id: str, lift: float | None
) -> IterationRecord:
    executor = ExecutorResult(
        role="executor",
        session_id=None,
        exit_code=0,
        model="claude-opus",
        log_path="",
        wall_seconds=1.0,
        brainstorm_id=brainstorm_id,
        lift=lift,
    )
    planner = PlannerResult(
        role="planner",
        session_id=None,
        exit_code=0,
        model="claude-opus",
        log_path="",
        wall_seconds=1.0,
    )
    return IterationRecord(
        iteration_id=iteration_id,
        ts_start="2025-01-01T00:00:00+00:00",
        planner=planner,
        executor=executor,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rebuild_registry_produces_markdown_file(tmp_path: Path) -> None:
    cfg = _make_cfg()
    state = _make_state(tmp_path, cfg)
    _write_brainstorm(state, [_make_item("id1", "Idea One", 0.8)])

    out_path = rebuild_registry(state)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_rebuild_registry_header_includes_counts(tmp_path: Path) -> None:
    cfg = _make_cfg()
    state = _make_state(tmp_path, cfg)
    _write_brainstorm(
        state,
        [
            _make_item("id1", "Queued Idea", 0.9, status="queued"),
            _make_item("id2", "Done Idea", 0.7, status="run_completed"),
            _make_item("id3", "Failed Idea", 0.5, status="run_failed"),
            _make_item("id4", "Dup Idea", 0.3, status="deduped_rejected"),
        ],
    )

    out_path = rebuild_registry(state)
    content = out_path.read_text()

    assert "Total ideas: 4" in content
    assert "queued: 1" in content
    assert "run_completed: 1" in content
    assert "run_failed: 1" in content
    assert "deduped_rejected: 1" in content


def test_rebuild_registry_sorted_by_tier_score_descending(tmp_path: Path) -> None:
    cfg = _make_cfg()
    state = _make_state(tmp_path, cfg)
    _write_brainstorm(
        state,
        [
            _make_item("id-low", "Low Score", 0.2),
            _make_item("id-high", "High Score", 0.9),
            _make_item("id-mid", "Mid Score", 0.5),
        ],
    )

    out_path = rebuild_registry(state)
    content = out_path.read_text()

    pos_high = content.index("High Score")
    pos_mid = content.index("Mid Score")
    pos_low = content.index("Low Score")

    assert pos_high < pos_mid < pos_low


def test_rebuild_registry_same_score_sorted_by_id(tmp_path: Path) -> None:
    cfg = _make_cfg()
    state = _make_state(tmp_path, cfg)
    _write_brainstorm(
        state,
        [
            _make_item("zzz-id", "Z Idea", 0.5),
            _make_item("aaa-id", "A Idea", 0.5),
        ],
    )

    out_path = rebuild_registry(state)
    content = out_path.read_text()

    pos_a = content.index("A Idea")
    pos_z = content.index("Z Idea")

    assert pos_a < pos_z


def test_rebuild_registry_title_truncated_at_60_chars(tmp_path: Path) -> None:
    cfg = _make_cfg()
    state = _make_state(tmp_path, cfg)
    long_title = "A" * 70
    _write_brainstorm(state, [_make_item("id1", long_title, 0.8)])

    out_path = rebuild_registry(state)
    content = out_path.read_text()

    assert long_title not in content
    assert "A" * 59 + "…" in content


def test_rebuild_registry_lift_from_matching_iteration(tmp_path: Path) -> None:
    cfg = _make_cfg()
    state = _make_state(tmp_path, cfg)
    _write_brainstorm(state, [_make_item("bid1", "Some Idea", 0.8)])
    _write_iteration(state, _make_iteration_with_lift(1, "bid1", 0.042))

    out_path = rebuild_registry(state)
    content = out_path.read_text()

    assert "+0.042" in content


def test_rebuild_registry_lift_dash_when_no_matching_iteration(tmp_path: Path) -> None:
    cfg = _make_cfg()
    state = _make_state(tmp_path, cfg)
    _write_brainstorm(state, [_make_item("bid-no-run", "No Run Idea", 0.8)])

    out_path = rebuild_registry(state)
    content = out_path.read_text()

    assert "| — |" in content


def test_rebuild_registry_output_is_atomic_non_empty(tmp_path: Path) -> None:
    cfg = _make_cfg()
    state = _make_state(tmp_path, cfg)
    _write_brainstorm(state, [_make_item("id1", "Test", 0.5)])

    out_path = rebuild_registry(state)

    assert out_path.exists()
    text = out_path.read_text()
    assert len(text) > 0
    assert "# Brainstorm Idea Registry" in text
