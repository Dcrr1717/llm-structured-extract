"""Results and streaming events returned by the extractor."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Any, Generic, TypeVar, Union

from .errors import OutputError
from .types import Usage

__all__ = [
    "Attempt",
    "ExtractionResult",
    "PartialEvent",
    "RepairEvent",
    "ResultEvent",
    "StreamEvent",
]

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Attempt:
    """One model call within an extraction and how it went."""

    #: 1-based position of this attempt.
    number: int
    raw_text: str
    #: ``None`` when this attempt produced the accepted value.
    error: OutputError | None
    usage: Usage
    stop_reason: str | None = None
    model: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True, slots=True)
class ExtractionResult(Generic[T]):
    """A validated value plus the history of how it was obtained."""

    value: T
    attempts: tuple[Attempt, ...]

    @property
    def usage(self) -> Usage:
        """Token usage summed over every attempt, including failed ones."""
        return reduce(lambda total, attempt: total + attempt.usage, self.attempts, Usage())

    @property
    def raw_text(self) -> str:
        """The raw model output that produced :attr:`value`."""
        return self.attempts[-1].raw_text

    @property
    def model(self) -> str | None:
        """The model that produced the accepted answer, when the provider reports it."""
        return self.attempts[-1].model

    @property
    def repaired(self) -> bool:
        """``True`` when at least one repair round was needed."""
        return len(self.attempts) > 1


@dataclass(frozen=True, slots=True)
class PartialEvent:
    """Best-effort snapshot of the JSON generated so far by the current attempt.

    ``data`` is plain JSON-like data (``dict`` / ``list`` / scalars), not yet
    validated: fields may be missing and the last string may be cut short.
    """

    attempt: int
    data: Any
    text: str


@dataclass(frozen=True, slots=True)
class RepairEvent:
    """An attempt failed; its errors are being sent back to the model."""

    attempt: Attempt

    @property
    def error(self) -> OutputError:
        assert self.attempt.error is not None
        return self.attempt.error


@dataclass(frozen=True, slots=True)
class ResultEvent(Generic[T]):
    """The final, validated result. Always the last event of a successful stream."""

    result: ExtractionResult[T]


#: Union of the events yielded by ``Extractor.stream`` / ``Extractor.astream``.
#: (``Union`` rather than ``|`` so that ``StreamEvent[T]`` can be subscripted on 3.10.)
StreamEvent = Union[PartialEvent, RepairEvent, ResultEvent[T]]  # noqa: UP007
