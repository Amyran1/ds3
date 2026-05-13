from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from libs.autoloop.state import append_jsonl, read_jsonl_iter
from libs.clients.anthropic import CompletionOptions
from libs.resilience import with_retry

logger = logging.getLogger(__name__)

_HAIKU_MODEL = "claude-haiku-4-5-20251001"
_FILE_TOOLS = {"Edit", "Write", "MultiEdit"}


class FailureMode(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    schema_version: Literal["failure_mode/v1"] = Field(default="failure_mode/v1", alias="schema")
    iteration_id: int
    role: Literal["planner", "executor"]
    ts: str
    exit_code: int
    kill_reason: str | None = None
    crashing_tool: str | None = None
    last_file_attempted: str | None = None
    stderr_tail: str = ""
    summary: str = ""


def _parse_stream_events(session_log_path: Path) -> tuple[str | None, str | None, str]:
    crashing_tool: str | None = None
    last_file_attempted: str | None = None
    error_chunks: list[str] = []

    if not session_log_path.exists():
        return crashing_tool, last_file_attempted, ""

    with session_log_path.open() as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                event: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(
                    "JSONDecodeError on line %d of %s — skipping", lineno, session_log_path
                )
                continue

            event_type = event.get("type", "")

            if event_type == "tool_use":
                name = event.get("name")
                if name:
                    crashing_tool = name
                    if name in _FILE_TOOLS:
                        args = event.get("args") or event.get("input") or {}
                        fp = args.get("file_path")
                        if fp:
                            last_file_attempted = fp

            elif event_type == "assistant":
                content = event.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if "Traceback" in text or "Error" in text or "Exception" in text:
                                error_chunks.append(text)
                elif isinstance(content, str):
                    if "Traceback" in content or "Error" in content or "Exception" in content:
                        error_chunks.append(content)

            elif event.get("is_error"):
                msg = event.get("message") or event.get("error") or str(event)
                error_chunks.append(str(msg))

    combined = "\n".join(error_chunks)
    stderr_tail = combined[-2000:] if len(combined) > 2000 else combined
    return crashing_tool, last_file_attempted, stderr_tail


def _build_haiku_prompt(
    role: str,
    exit_code: int,
    kill_reason: str | None,
    crashing_tool: str | None,
    last_file_attempted: str | None,
    stderr_tail: str,
) -> str:
    return (
        f"A claude -p session in role=`{role}` crashed "
        f"(exit_code={exit_code}, kill_reason={kill_reason}).\n"
        f"Last tool used: {crashing_tool} on file {last_file_attempted}.\n"
        f"Last error context (last ~2000 chars):\n"
        f"```\n{stderr_tail}\n```\n\n"
        "In 1-2 plain sentences, summarize:\n"
        "1. WHAT specifically went wrong (the actual error or constraint)\n"
        "2. WHAT a future session should AVOID doing\n"
        "No preamble, no markdown formatting. Just the two sentences."
    )


async def extract_failure_fingerprint(
    *,
    container: Any,
    iteration_id: int,
    role: Literal["planner", "executor"],
    session_log_path: Path,
    exit_code: int,
    kill_reason: str | None,
    project: str,
) -> FailureMode:
    crashing_tool, last_file_attempted, stderr_tail = _parse_stream_events(session_log_path)

    prompt = _build_haiku_prompt(
        role=role,
        exit_code=exit_code,
        kill_reason=kill_reason,
        crashing_tool=crashing_tool,
        last_file_attempted=last_file_attempted,
        stderr_tail=stderr_tail,
    )

    @with_retry(max_attempts=3, retry_on=(Exception,))
    async def _call_haiku() -> str:
        return await container.anthropic.complete(
            messages=[{"role": "user", "content": prompt}],
            options=CompletionOptions(
                model=_HAIKU_MODEL,
                max_tokens=200,
                cost_key=f"autoloop:{project}:failure_summary",
            ),
        )

    summary: str
    try:
        summary = await _call_haiku()
    except Exception as exc:
        logger.warning("Haiku failure summary failed after retries: %s", exc)
        summary = f"Session crashed (exit_code={exit_code}). Context: {stderr_tail[:200]}"

    return FailureMode(
        iteration_id=iteration_id,
        role=role,
        ts=datetime.now(timezone.utc).isoformat(),
        exit_code=exit_code,
        kill_reason=kill_reason,
        crashing_tool=crashing_tool,
        last_file_attempted=last_file_attempted,
        stderr_tail=stderr_tail,
        summary=summary,
    )


def append_failure_mode(state_dir: Path, fm: FailureMode) -> None:
    path = state_dir / "failure_modes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(path, fm)


def recent_failures(state_dir: Path, *, n: int = 10) -> list[FailureMode]:
    path = state_dir / "failure_modes.jsonl"
    results: list[FailureMode] = []
    for row in read_jsonl_iter(path):
        try:
            results.append(FailureMode.model_validate(row))
        except ValidationError as exc:
            logger.warning("Could not parse FailureMode row — skipping: %s", exc)
    return results[-n:] if len(results) > n else results


def format_for_prompt(failures: list[FailureMode], max_chars: int = 4000) -> str:
    """Format last-N failures as a markdown block for inclusion in planner/executor prompts."""
    if not failures:
        return ""

    lines: list[str] = ["## Past session failures (avoid repeating these)"]
    for fm in failures:
        label = fm.crashing_tool or "unknown tool"
        lines.append(f"- **iter {fm.iteration_id} / {fm.role}** ({label}): {fm.summary}")

    block = "\n".join(lines)
    if len(block) <= max_chars:
        return block

    # Drop older entries until it fits
    while len(block) > max_chars and len(lines) > 1:
        lines.pop(1)
        block = "\n".join(lines)

    return block
