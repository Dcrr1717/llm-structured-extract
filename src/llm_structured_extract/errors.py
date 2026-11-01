"""Typed errors.

Everything this package raises derives from :class:`StructuredExtractError`, so a
single ``except StructuredExtractError`` is enough for callers who don't care
about the details. The hierarchy separates three situations that need very
different handling:

* :class:`OutputError` - the model answered, but the answer is unusable
  (not JSON, cut off, or invalid against the schema). These are *repairable*:
  the extractor feeds them back to the model and retries.
* :class:`ProviderError` / :class:`RefusalError` - the call itself failed or was
  declined. Repairing the output cannot help, so these propagate immediately.
* :class:`ExtractionFailedError` - every allowed attempt produced an
  :class:`OutputError`. It carries the full attempt history for debugging.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .result import Attempt

__all__ = [
    "ExtractionFailedError",
    "MissingOutputError",
    "OutputError",
    "OutputParseError",
    "OutputTruncatedError",
    "OutputValidationError",
    "ProviderError",
    "ProviderNotConfiguredError",
    "RefusalError",
    "SchemaError",
    "ScriptExhaustedError",
    "StructuredExtractError",
    "ValidationIssue",
]

ROOT_PATH = "<root>"


class StructuredExtractError(Exception):
    """Base class for every error raised by ``llm_structured_extract``."""


class SchemaError(StructuredExtractError):
    """The output schema could not be built or is unusable for the chosen provider mode."""


# --------------------------------------------------------------------------- #
# Provider-side failures (not repairable)
# --------------------------------------------------------------------------- #


class ProviderError(StructuredExtractError):
    """A provider call failed (network, authentication, rate limit, server error...).

    ``retryable`` is a hint: ``True`` for transient failures (timeouts, 429, 5xx)
    that a caller-level retry with backoff may fix. Official SDKs usually retry
    those already before the error surfaces here.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code


class ProviderNotConfiguredError(ProviderError):
    """The provider is missing credentials or an optional dependency."""

    def __init__(self, message: str, *, provider: str) -> None:
        super().__init__(message, provider=provider, retryable=False)


class ScriptExhaustedError(ProviderError):
    """A :class:`~llm_structured_extract.MockProvider` ran out of scripted responses."""

    def __init__(self, message: str, *, provider: str = "mock") -> None:
        super().__init__(message, provider=provider, retryable=False)


class RefusalError(StructuredExtractError):
    """The model declined the request (for example a safety-classifier refusal).

    Not retried by the repair loop: re-asking the same model rarely changes a
    policy decision. ``category`` and ``explanation`` are passed through from
    the provider when available.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        category: str | None = None,
        explanation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.category = category
        self.explanation = explanation


# --------------------------------------------------------------------------- #
# Output-side failures (repairable)
# --------------------------------------------------------------------------- #


class OutputError(StructuredExtractError):
    """The model produced an answer that could not be turned into a valid value.

    ``raw_text`` is the offending output, kept for debugging and for the repair
    prompt.
    """

    def __init__(self, message: str, *, raw_text: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.raw_text = raw_text


class MissingOutputError(OutputError):
    """The answer contained no output at all (empty text, or no tool call)."""


class OutputTruncatedError(OutputError):
    """The answer was cut off by the output-token limit before the JSON was complete."""


class OutputParseError(OutputError):
    """No JSON value could be parsed from the answer, even with tolerant parsing."""

    def __init__(
        self,
        message: str,
        *,
        raw_text: str = "",
        position: int | None = None,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        super().__init__(message, raw_text=raw_text)
        self.position = position
        self.line = line
        self.column = column

    def __str__(self) -> str:
        if self.line is not None and self.column is not None:
            return f"{self.message} (line {self.line}, column {self.column})"
        return self.message


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One schema violation, normalised across Pydantic and JSON Schema validators.

    ``path`` uses a compact JSONPath-like notation (``line_items[2].amount``);
    errors that concern the value as a whole use ``"<root>"``.
    """

    path: str
    message: str
    kind: str = "invalid"
    input: Any = field(default=None, repr=False, compare=False)

    @property
    def is_root(self) -> bool:
        return self.path == ROOT_PATH

    def describe(self, *, max_input_chars: int = 80) -> str:
        """Human/model-readable one-liner, including a short preview of the bad input.

        The preview is omitted for missing values, for root-level issues, and for
        objects/arrays too long to show whole (a truncated container says little).
        """
        text = f"`{self.path}`: {self.message}"
        if self.is_root or self.kind == "missing":
            return text
        preview = _preview(self.input, max_input_chars)
        if preview is not None:
            text += f" (got: {preview})"
        return text

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def _preview(value: Any, limit: int) -> str | None:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - default=str makes this unlikely
        rendered = repr(value)
    if len(rendered) <= limit:
        return rendered
    if isinstance(value, (dict, list, tuple)):
        return None
    return rendered[: limit - 3] + "..."


def format_path(location: Sequence[str | int]) -> str:
    """Render a location tuple such as ``("items", 2, "price")`` as ``items[2].price``."""
    if not location:
        return ROOT_PATH
    out = ""
    for part in location:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else str(part)
    return out


class OutputValidationError(OutputError):
    """The answer parsed as JSON but violates the output schema."""

    def __init__(
        self,
        issues: Sequence[ValidationIssue],
        *,
        raw_text: str = "",
        schema_name: str | None = None,
    ) -> None:
        self.issues: tuple[ValidationIssue, ...] = tuple(issues)
        self.schema_name = schema_name
        target = f" `{schema_name}`" if schema_name else ""
        count = len(self.issues)
        noun = "issue" if count == 1 else "issues"
        summary = f"Output failed validation against{target} schema ({count} {noun})"
        if self.issues:
            summary += ": " + "; ".join(str(issue) for issue in self.issues[:3])
            if count > 3:
                summary += "; ..."
        super().__init__(summary, raw_text=raw_text)


# --------------------------------------------------------------------------- #
# Terminal failure
# --------------------------------------------------------------------------- #


class ExtractionFailedError(StructuredExtractError):
    """Every allowed attempt failed with an :class:`OutputError`."""

    def __init__(self, attempts: Sequence[Attempt], *, schema_name: str) -> None:
        self.attempts: tuple[Attempt, ...] = tuple(attempts)
        self.schema_name = schema_name
        last = self.last_error
        detail = f" Last error: {last}" if last is not None else ""
        noun = "attempt" if len(self.attempts) == 1 else "attempts"
        super().__init__(
            f"Could not extract a valid `{schema_name}` after {len(self.attempts)} {noun}.{detail}"
        )

    @property
    def last_error(self) -> OutputError | None:
        for attempt in reversed(self.attempts):
            if attempt.error is not None:
                return attempt.error
        return None
