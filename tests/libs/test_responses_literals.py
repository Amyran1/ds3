import pytest
from pydantic import ValidationError

from libs.responses import RunResultMetadata

VALID_SCOPES = [
    "smoke",
    "reproduction",
    "comparison",
    "champion_candidate",
    "fast_iter",
    "comparison_fast",
]

_BASE = dict(
    run_id="run_01",
    project="test",
    comparison_group="g",
    recorded_at_utc="2026-01-01T00:00:00Z",
)


@pytest.mark.parametrize("scope", VALID_SCOPES)
def test_scope_literal_accepts(scope: str) -> None:
    """All canonical scopes including v3 additions (fast_iter, comparison_fast) validate."""
    meta = RunResultMetadata(**_BASE, scope=scope)
    assert meta.scope == scope


def test_scope_literal_rejects_unknown() -> None:
    """Bogus scope values fail validation."""
    with pytest.raises(ValidationError):
        RunResultMetadata(**_BASE, scope="bogus_scope")


def test_verdict_literal_accepts_ambiguous() -> None:
    """verdict='ambiguous' (added for comparison_fast escalation) validates."""
    meta = RunResultMetadata(**_BASE, scope="comparison_fast", verdict="ambiguous")
    assert meta.verdict == "ambiguous"


def test_verdict_literal_rejects_unknown() -> None:
    """Bogus verdict values fail validation."""
    with pytest.raises(ValidationError):
        RunResultMetadata(**_BASE, scope="smoke", verdict="bogus_verdict")
