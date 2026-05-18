import pytest

from libs.autoloop.dashboard.render import _fmt_scope, _resolve_scope


@pytest.mark.parametrize(
    "run_id,results,expected_scope",
    [
        ("run_01_fast_iter", [{"run_id": "run_01_fast_iter", "scope": "fast_iter"}], "fast_iter"),
        (
            "run_02_comp_fast",
            [{"run_id": "run_02_comp_fast", "scope": "comparison_fast"}],
            "comparison_fast",
        ),
        (
            "run_03_champ",
            [{"run_id": "run_03_champ", "scope": "champion_candidate"}],
            "champion_candidate",
        ),
        ("run_04_smoke", [{"run_id": "run_04_smoke", "scope": "smoke"}], "smoke"),
        (
            "run_05_comparison",
            [{"run_id": "run_05_comparison", "scope": "comparison"}],
            "comparison",
        ),
    ],
)
def test_resolve_scope_explicit(run_id: str, results: list[dict], expected_scope: str) -> None:
    res_by_run = {r["run_id"]: r for r in results}
    assert _resolve_scope(run_id, res_by_run) == expected_scope


@pytest.mark.parametrize(
    "run_id,expected_scope",
    [
        ("run_01_smoke_v1", "smoke"),
        ("run_02_something_smoke_test", "smoke"),
        ("run_03_comparison", "comparison"),
        ("run_04_fast_iter", "comparison"),
    ],
)
def test_resolve_scope_legacy_fallback(run_id: str, expected_scope: str) -> None:
    assert _resolve_scope(run_id, {}) == expected_scope


def test_resolve_scope_no_run_id() -> None:
    assert _resolve_scope("—", {}) == "—"


@pytest.mark.parametrize(
    "run_id,row_scope,expected_scope",
    [
        ("run_01", None, "comparison"),
        ("run_01", "", "comparison"),
    ],
)
def test_resolve_scope_falsy_explicit_falls_back(
    run_id: str, row_scope: str | None, expected_scope: str
) -> None:
    res_by_run = {"run_01": {"run_id": "run_01", "scope": row_scope}}
    assert _resolve_scope(run_id, res_by_run) == expected_scope


@pytest.mark.parametrize(
    "scope,expected_contains",
    [
        ("fast_iter", "⚡"),
        ("comparison_fast", "⚡"),
        ("champion_candidate", "🏆"),
        ("smoke", "💨"),
        ("comparison", "≈"),
        ("reproduction", "♻"),
        ("—", "—"),
        ("unknown_scope", "unknown_scope"),
    ],
)
def test_fmt_scope_icons(scope: str, expected_contains: str) -> None:
    result = _fmt_scope(scope)
    assert expected_contains in result


def test_fmt_scope_no_icon_for_unknown() -> None:
    result = _fmt_scope("some_new_scope")
    assert result == "some_new_scope"
