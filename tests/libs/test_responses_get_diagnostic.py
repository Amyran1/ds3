from __future__ import annotations

from libs.responses import (
    HarnessDataProfile,
    HarnessDiagnostic,
    HarnessPerformance,
    HarnessResponse,
    HarnessSummary,
)
from projects.civic_shout_action_rate_increase.harness import (
    HarnessDiagnostic as ProjectHarnessDiagnostic,
)
from projects.civic_shout_action_rate_increase.harness import (
    HarnessResponse as ProjectHarnessResponse,
)


def _make_response(diagnostics: list[HarnessDiagnostic]) -> HarnessResponse:
    return HarnessResponse(
        summary=HarnessSummary(primary_metric_name="roc_auc"),
        performance=HarnessPerformance(total_seconds=1.0),
        data_profile=HarnessDataProfile(
            n_rows=100, n_rows_source=100, sample_frac=1.0, is_full_run=True, n_features=5
        ),
        diagnostics=diagnostics,
    )


def _diag(name: str, severity: str = "info", message: str = "ok") -> HarnessDiagnostic:
    return HarnessDiagnostic(name=name, severity=severity, message=message)


def test_get_diagnostic_empty_list_returns_none() -> None:
    resp = _make_response([])
    assert resp.get_diagnostic("anything") is None


def test_get_diagnostic_found() -> None:
    diag = _diag("future_leak_audit")
    resp = _make_response([diag])
    result = resp.get_diagnostic("future_leak_audit")
    assert result is not None
    assert result.name == "future_leak_audit"


def test_get_diagnostic_multiple_none_matching() -> None:
    resp = _make_response([_diag("a"), _diag("b"), _diag("c")])
    assert resp.get_diagnostic("z") is None


def test_get_diagnostic_returns_first_on_duplicate_names() -> None:
    first = HarnessDiagnostic(name="dup", severity="info", message="first")
    second = HarnessDiagnostic(name="dup", severity="warning", message="second")
    resp = _make_response([first, second])
    result = resp.get_diagnostic("dup")
    assert result is not None
    assert result.message == "first"


def test_get_diagnostic_works_on_project_subclass() -> None:
    diag = ProjectHarnessDiagnostic(name="test_diag", severity="info", message="ok")
    resp = ProjectHarnessResponse.model_construct(diagnostics=[diag])
    result = resp.get_diagnostic("test_diag")
    assert result is not None
    assert isinstance(result, HarnessDiagnostic)
    assert result.name == "test_diag"
