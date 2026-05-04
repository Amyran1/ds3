"""Pre-flight validation for API inputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Result of a pre-flight validation check."""

    valid: bool
    n_total: int
    n_invalid: int
    issues: list[str] = field(default_factory=list)
    invalid_indices: list[int] = field(default_factory=list)


def validate_texts(
    texts: list[str],
    *,
    allow_empty: bool = False,
    max_length: int = 8192,
    placeholder: str = "no description",
    fix: bool = False,
) -> tuple[list[str], ValidationReport]:
    """Validate a list of texts before API calls.

    Checks for: empty/whitespace-only strings, None values, strings exceeding max_length.

    If fix=True: replaces empty/None with placeholder, truncates long texts.
    Returns (possibly fixed texts, validation report).
    """
    issues: list[str] = []
    invalid_indices: list[int] = []
    result = list(texts)

    for i, text in enumerate(result):
        if text is None:  # type: ignore[comparison-overlap]
            invalid_indices.append(i)
            issues.append(f"index {i}: None value")
            if fix:
                result[i] = placeholder
        elif not isinstance(text, str):
            invalid_indices.append(i)
            issues.append(f"index {i}: not a string (got {type(text).__name__})")
            if fix:
                result[i] = str(text)
        elif not text.strip():
            if not allow_empty:
                invalid_indices.append(i)
                issues.append(f"index {i}: empty/whitespace-only string")
                if fix:
                    result[i] = placeholder
        elif len(text) > max_length:
            invalid_indices.append(i)
            issues.append(f"index {i}: exceeds max_length ({len(text)} > {max_length})")
            if fix:
                result[i] = text[:max_length]

    n_invalid = len(invalid_indices)
    report = ValidationReport(
        valid=n_invalid == 0,
        n_total=len(texts),
        n_invalid=n_invalid,
        issues=issues,
        invalid_indices=invalid_indices,
    )
    return result, report


def preflight_check(
    texts: list[str],
    *,
    operation: str = "embed",
    fix: bool = True,
    max_length: int = 8192,
    placeholder: str = "no description",
) -> list[str]:
    """Convenience wrapper: validate, log warnings, fix if possible, raise if unfixable.

    Raises ValueError if fix=False and validation fails.
    """
    fixed_texts, report = validate_texts(
        texts,
        max_length=max_length,
        placeholder=placeholder,
        fix=fix,
    )

    if not report.valid:
        msg = f"Preflight {operation}: {report.n_invalid}/{report.n_total} texts invalid"
        for issue in report.issues:
            msg += f"\n  - {issue}"

        if fix:
            logger.warning("%s (auto-fixed)", msg)
        else:
            raise ValueError(msg)

    return fixed_texts
