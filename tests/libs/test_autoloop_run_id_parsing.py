from __future__ import annotations

import re


def test_retention_run_id_regex_accepts_civic_shout_ids():
    """retention.py's regex now accepts civic_shout-style IDs."""
    pattern = re.compile(r"\brun_\d\w*\b", re.ASCII)
    match = pattern.search("the champion is run_01_smoke this iteration")
    assert match is not None
    assert match.group() == "run_01_smoke", f"got: {match.group()!r}"


def test_retention_run_id_regex_accepts_legacy_ids():
    pattern = re.compile(r"\brun_\d\w*\b", re.ASCII)
    match = pattern.search("champion: run_01")
    assert match is not None
    assert match.group() == "run_01"


def test_retention_run_id_regex_does_not_match_run_id_key():
    """run_id (a field name) is not matched — requires digit after run_."""
    pattern = re.compile(r"\brun_\d\w*\b", re.ASCII)
    assert (
        pattern.search("**run_id**: run_01_smoke") is None
        or (m := pattern.search("**run_id**: run_01_smoke")).group() != "run_id"
    )


def test_dashboard_run_id_regex_accepts_civic_shout_ids():
    """render.py's regex now accepts civic_shout-style IDs."""
    pattern = re.compile(r"(run_\d\w*)", re.ASCII)
    match = pattern.search("Currently running: run_01_v3_final_lr_5pct iteration 5")
    assert match is not None
    assert match.group(1) == "run_01_v3_final_lr_5pct"


def test_dashboard_run_id_regex_accepts_legacy_ids():
    pattern = re.compile(r"(run_\d\w*)", re.ASCII)
    match = pattern.search("champion: run_01")
    assert match is not None
    assert match.group(1) == "run_01"


def test_supervisor_run_id_regex_accepts_civic_shout_ids():
    """supervisor.py's regex now accepts civic_shout-style IDs."""
    pattern = re.compile(r"run_\d\w*", re.ASCII)
    match = pattern.search("Started run_01_smoke")
    assert match is not None
    assert match.group() == "run_01_smoke"


def test_supervisor_run_id_regex_accepts_legacy_ids():
    pattern = re.compile(r"run_\d\w*", re.ASCII)
    match = pattern.search("Started run_01")
    assert match is not None
    assert match.group() == "run_01"
