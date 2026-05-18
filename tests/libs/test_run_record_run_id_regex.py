import pytest

from libs.run_record import RUN_ID_REGEX

ACCEPTED_IDS = [
    "run_01",
    "run_001",
    "run_01_smoke",
    "run_01_v3_t1a_1pct",
    "run_01_final_lr_5pct_fast",
    "run_AB",
    "run_abc_123",
]

REJECTED_IDS = [
    "",
    "run_",
    "run",
    "bad_prefix_01",
    "RUN_01",
    "run-01",
    "run_01-smoke",
    "run_01.suffix",
    "run_01 smoke",
    "run_фрукт",
    "run_テスト",
    "run_émile",
]


@pytest.mark.parametrize("rid", ACCEPTED_IDS)
def test_run_id_regex_accepts(rid):
    assert RUN_ID_REGEX.fullmatch(rid) is not None, f"expected accept: {rid}"


@pytest.mark.parametrize("rid", REJECTED_IDS)
def test_run_id_regex_rejects(rid):
    assert RUN_ID_REGEX.fullmatch(rid) is None, f"expected reject: {rid}"


def test_unicode_run_id_rejected_under_ascii_flag():
    """re.ASCII flag prevents Unicode word chars from matching."""
    assert RUN_ID_REGEX.fullmatch("run_фрукт") is None
    assert RUN_ID_REGEX.fullmatch("run_テスト") is None
    assert RUN_ID_REGEX.fullmatch("run_émile") is None
