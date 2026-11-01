"""Value types shared by the extractor and every provider.

These are deliberately small, immutable, provider-neutral dataclasses. A provider
receives a :class:`CompletionRequest` and answers with a :class:`Completion`
(or a stream of :class:`StreamChunk` objects ending in one that carries the
:class:`Completion`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "STOP_END_TURN",
    "STOP_MAX_TOKENS",
    "STOP_REFUSAL",
    "STOP_TOOL_USE",
    "Completion",
    "CompletionRequest",
    "Message",
    "Role",
    "StreamChunk",
    "Usage",
]

Role = Literal["user", "assistant"]

#: Normalised stop reasons. Providers should map their native values onto these
#: where one applies; any other string is passed through untouched.
STOP_END_TURN = "end_turn"
STOP_MAX_TOKENS = "max_tokens"
STOP_REFUSAL = "refusal"
STOP_TOOL_USE = "tool_use"


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one or more model calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
        )

    @property
    def total_tokens(self) -> int:
        """All tokens billed for the call(s): uncached input, cache traffic, output."""
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
            + self.output_tokens
        )


@dataclass(frozen=True, slots=True)
class Message:
    """One conversation turn.

    ``native`` is an opaque, provider-specific payload. A provider may attach it
    to a :class:`Completion` so that, when the extractor replays the model's
    previous answer during a repair round, the provider can send back its exact
    native representation (for example Anthropic content blocks, including
    thinking blocks) instead of a lossy plain-text copy. Other providers ignore it.
    """

    role: Role
    content: str
    native: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """Everything a provider needs to produce one candidate answer."""

    system: str
    messages: tuple[Message, ...]
    schema_name: str
    #: The full JSON Schema of the expected output, exactly as produced by the
    #: schema adapter. Providers with native constrained decoding may transform
    #: it into whatever subset their API accepts.
    json_schema: Mapping[str, Any]
    max_tokens: int = 16_000
    #: 1-based attempt number within the extraction (1 = first try).
    attempt: int = 1


@dataclass(frozen=True, slots=True)
class Completion:
    """A provider's raw answer to a :class:`CompletionRequest`."""

    text: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    model: str | None = None
    native: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """An incremental piece of a streamed completion.

    ``text`` is the newly generated output text (it may be empty). Providers
    should end every stream with a chunk whose ``completion`` is set; it carries
    usage, the stop reason and the authoritative final text. If a stream ends
    without one, the extractor treats the concatenated ``text`` as the answer.
    """

    text: str = ""
    completion: Completion | None = None
