from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import yaml

from libs.autoloop.state import (
    AutoloopConfig,
    BrainstormItem,
    IterationRecord,
    State,
    append_jsonl,
    atomic_write_json,
    read_jsonl_iter,
)

# ---------------------------------------------------------------------------
# BrainstormItem tests
# ---------------------------------------------------------------------------


def _make_brainstorm_item(**overrides: object) -> BrainstormItem:
    defaults: dict[str, object] = {
        "schema": "brainstorm/v1",
        "id": "item-001",
        "ts": "2026-01-01T00:00:00Z",
        "title": "Test idea",
        "hypothesis": "This might work",
        "source": {"kind": "discovery"},
        "feature_family_name_hint": "test_feature",
        "row_grain": "user",
        "content_hash": "abc123",
    }
    defaults.update(overrides)
    return BrainstormItem.model_validate(defaults)


def test_brainstorm_item_round_trips_through_json() -> None:
    item = _make_brainstorm_item()
    serialized = item.model_dump_json(by_alias=True)
    restored = BrainstormItem.model_validate_json(serialized)

    assert restored.id == item.id
    assert restored.schema_version == item.schema_version
    assert restored.title == item.title
    assert restored.content_hash == item.content_hash


def test_brainstorm_item_accepts_schema_alias() -> None:
    data = {
        "schema": "brainstorm/v1",
        "id": "x",
        "ts": "2026-01-01T00:00:00Z",
        "title": "T",
        "hypothesis": "H",
        "source": {"kind": "user"},
        "feature_family_name_hint": "f",
        "row_grain": "user",
        "content_hash": "h",
    }
    item = BrainstormItem.model_validate(data)
    assert item.schema_version == "brainstorm/v1"


def test_brainstorm_item_accepts_schema_version_key() -> None:
    data = {
        "schema_version": "brainstorm/v1",
        "id": "x",
        "ts": "2026-01-01T00:00:00Z",
        "title": "T",
        "hypothesis": "H",
        "source": {"kind": "user"},
        "feature_family_name_hint": "f",
        "row_grain": "user",
        "content_hash": "h",
    }
    item = BrainstormItem.model_validate(data)
    assert item.schema_version == "brainstorm/v1"


def test_brainstorm_item_ignores_unknown_fields() -> None:
    data = {
        "schema": "brainstorm/v1",
        "id": "x",
        "ts": "2026-01-01T00:00:00Z",
        "title": "T",
        "hypothesis": "H",
        "source": {"kind": "user"},
        "feature_family_name_hint": "f",
        "row_grain": "user",
        "content_hash": "h",
        "totally_unknown_field": "ignored",
    }
    item = BrainstormItem.model_validate(data)
    assert item.id == "x"


# ---------------------------------------------------------------------------
# AutoloopConfig.load tests
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, extra: dict | None = None) -> Path:
    config: dict[str, object] = {
        "project": "my_project",
        "outcome_variable": "converted",
        "comparison_group": "cg1",
        "primary_metric": "roc_auc",
        "metric_direction": "higher_is_better",
    }
    if extra:
        config.update(extra)
    cfg_path = tmp_path / "autoloop" / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.dump(config))
    return tmp_path


def test_autoloop_config_load_round_trips(tmp_path: Path) -> None:
    project_root = _write_config(tmp_path)
    cfg = AutoloopConfig.load(project_root)

    assert cfg.project == "my_project"
    assert cfg.outcome_variable == "converted"
    assert cfg.comparison_group == "cg1"
    assert cfg.primary_metric == "roc_auc"
    assert cfg.metric_direction == "higher_is_better"


def test_autoloop_config_load_nested_budget(tmp_path: Path) -> None:
    project_root = _write_config(tmp_path, {"budget": {"iterations_max": 42, "dollars_max": 10.0}})
    cfg = AutoloopConfig.load(project_root)

    assert cfg.budget.iterations_max == 42
    assert cfg.budget.dollars_max == 10.0


def test_autoloop_config_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, OSError)):
        AutoloopConfig.load(tmp_path)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def _make_state(tmp_path: Path, project: str = "proj") -> State:
    cfg = AutoloopConfig.model_validate(
        {
            "project": project,
            "outcome_variable": "y",
            "comparison_group": "cg",
            "primary_metric": "roc_auc",
        }
    )
    return State(tmp_path, cfg)


def _state_dir(tmp_path: Path, project: str = "proj") -> Path:
    return tmp_path / "projects" / project / "autoloop"


def test_next_iteration_id_returns_one_when_no_file(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    assert state.next_iteration_id() == 1


def test_next_iteration_id_returns_max_plus_one(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    sd = _state_dir(tmp_path)
    sd.mkdir(parents=True)
    iterations = sd / "iterations.jsonl"
    rows = [
        {
            "schema": "iteration/v1",
            "iteration_id": 1,
            "ts_start": "2026-01-01T00:00:00Z",
            "ts_end": None,
        },
        {
            "schema": "iteration/v1",
            "iteration_id": 3,
            "ts_start": "2026-01-01T00:00:00Z",
            "ts_end": None,
        },
    ]
    with iterations.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    assert state.next_iteration_id() == 4


def test_next_iteration_id_skips_bad_json(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    sd = _state_dir(tmp_path)
    sd.mkdir(parents=True)
    iterations = sd / "iterations.jsonl"
    with iterations.open("w") as f:
        f.write("{not valid json\n")
        f.write(
            json.dumps(
                {
                    "schema": "iteration/v1",
                    "iteration_id": 5,
                    "ts_start": "2026-01-01T00:00:00Z",
                    "ts_end": None,
                }
            )
            + "\n"
        )

    assert state.next_iteration_id() == 6


def _make_iteration_record(iter_id: int) -> IterationRecord:
    return IterationRecord(
        iteration_id=iter_id,
        ts_start="2026-01-01T00:00:00Z",
    )


def test_append_iteration_is_readable(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    rec = _make_iteration_record(1)
    state.append_iteration(rec)

    path = _state_dir(tmp_path) / "iterations.jsonl"
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["iteration_id"] == 1


def test_append_iteration_multiple_atomic(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    for i in range(1, 4):
        state.append_iteration(_make_iteration_record(i))

    path = _state_dir(tmp_path) / "iterations.jsonl"
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == 3
    ids = [json.loads(line)["iteration_id"] for line in lines]
    assert ids == [1, 2, 3]


def test_recent_iterations_returns_last_n(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    for i in range(1, 7):
        state.append_iteration(_make_iteration_record(i))

    recent = state.recent_iterations(n=3)
    assert len(recent) == 3
    assert [r.iteration_id for r in recent] == [4, 5, 6]


def test_load_brainstorm_event_sourced_last_wins(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    sd = _state_dir(tmp_path)
    sd.mkdir(parents=True)
    bs_path = sd / "brainstorm.jsonl"

    first = _make_brainstorm_item(id="item-1", status="queued", tier_score=0.5)
    second = _make_brainstorm_item(id="item-1", status="run_completed", tier_score=0.9)
    with bs_path.open("w") as f:
        f.write(first.model_dump_json(by_alias=True) + "\n")
        f.write(second.model_dump_json(by_alias=True) + "\n")

    items = state.load_brainstorm()
    assert len(items) == 1
    assert items[0].status == "run_completed"


def test_load_brainstorm_ignores_planner_summary_lines(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    sd = _state_dir(tmp_path)
    sd.mkdir(parents=True)
    bs_path = sd / "brainstorm.jsonl"

    brainstorm = _make_brainstorm_item(id="item-1")
    planner_summary = {
        "schema": "planner_summary/v1",
        "iteration_id": 1,
        "added": 3,
        "reranked": 0,
        "skipped": 0,
        "ran_gap_finder": False,
    }
    executor_summary = {
        "schema": "executor_summary/v1",
        "iteration_id": 1,
        "brainstorm_id": "item-1",
        "run_id": "run_01",
        "metric": 0.75,
        "baseline_metric": 0.70,
        "lift": 0.05,
        "verdict": "completed",
    }
    with bs_path.open("w") as f:
        f.write(brainstorm.model_dump_json(by_alias=True) + "\n")
        f.write(json.dumps(planner_summary) + "\n")
        f.write(json.dumps(executor_summary) + "\n")

    items = state.load_brainstorm()
    assert len(items) == 1
    assert items[0].id == "item-1"


def test_pop_top_queued_returns_highest_tier_score(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    sd = _state_dir(tmp_path)
    sd.mkdir(parents=True)
    bs_path = sd / "brainstorm.jsonl"

    low = _make_brainstorm_item(id="low", tier_score=0.3, status="queued")
    high = _make_brainstorm_item(id="high", tier_score=0.9, status="queued")
    mid = _make_brainstorm_item(id="mid", tier_score=0.6, status="queued")
    with bs_path.open("w") as f:
        for item in [low, high, mid]:
            f.write(item.model_dump_json(by_alias=True) + "\n")

    result = state.pop_top_queued()
    assert result is not None
    assert result.id == "high"


def test_pop_top_queued_returns_none_when_empty(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    assert state.pop_top_queued() is None


def test_mark_consumed_appends_updated_line(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    sd = _state_dir(tmp_path)
    sd.mkdir(parents=True)
    bs_path = sd / "brainstorm.jsonl"

    item = _make_brainstorm_item(id="item-1", status="queued")
    with bs_path.open("w") as f:
        f.write(item.model_dump_json(by_alias=True) + "\n")

    state.mark_consumed("item-1", iteration_id=2, run_id="run_02", status="run_completed")

    lines = [line.strip() for line in bs_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    final = json.loads(lines[-1])
    assert final["status"] == "run_completed"
    assert final["iteration_id"] == 2
    assert final["run_id"] == "run_02"


# ---------------------------------------------------------------------------
# atomic_write_json / append_jsonl / read_jsonl_iter tests
# ---------------------------------------------------------------------------


def test_atomic_write_json_writes_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    atomic_write_json(path, {"key": "value", "num": 42})

    assert path.exists()
    parsed = json.loads(path.read_text())
    assert parsed["key"] == "value"
    assert parsed["num"] == 42


def test_atomic_write_json_no_tmp_lingering(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    atomic_write_json(path, {"x": 1})

    tmp_file = path.with_suffix(".tmp")
    assert not tmp_file.exists()


def test_append_jsonl_produces_valid_lines(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    item1 = _make_brainstorm_item(id="a")
    item2 = _make_brainstorm_item(id="b")
    append_jsonl(path, item1)
    append_jsonl(path, item2)

    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    for line in lines:
        json.loads(line)


def test_read_jsonl_iter_skips_bad_lines(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "data.jsonl"
    with path.open("w") as f:
        f.write("{not valid}\n")
        f.write(json.dumps({"good": True}) + "\n")

    with caplog.at_level(logging.WARNING):
        results = list(read_jsonl_iter(path))

    assert len(results) == 1
    assert results[0]["good"] is True
