"""The extractor: prompt -> call -> parse -> validate -> repair, until valid or out of attempts."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any, Generic, TypeVar, overload

from .errors import (
    ExtractionFailedError,
    MissingOutputError,
    OutputError,
    OutputTruncatedError,
    RefusalError,
)
from .parsing import parse_json, parse_partial_json
from .prompts import PromptBuilder
from .providers.base import Provider
from .result import (
    Attempt,
    ExtractionResult,
    PartialEvent,
    RepairEvent,
    ResultEvent,
    StreamEvent,
)
from .schema import OutputSchema, as_output_schema
from .types import (
    STOP_MAX_TOKENS,
    STOP_REFUSAL,
    Completion,
    CompletionRequest,
    Message,
    StreamChunk,
    Usage,
)

__all__ = ["Extractor"]

T = TypeVar("T")

logger = logging.getLogger("llm_structured_extract")

AttemptCallback = Callable[[Attempt], None]


class _Session(Generic[T]):
    """State of one extraction: the growing conversation and the attempt history.

    Kept free of I/O so the sync and async front-ends share every decision.
    """

    def __init__(
        self,
        *,
        schema: OutputSchema[T],
        prompts: PromptBuilder,
        text: str,
        instructions: str | None,
        max_attempts: int,
        max_tokens: int,
        provider_name: str,
        on_attempt: AttemptCallback | None,
    ) -> None:
        self.schema = schema
        self.prompts = prompts
        self.max_attempts = max_attempts
        self.max_tokens = max_tokens
        self.provider_name = provider_name
        self.on_attempt = on_attempt
        self.system = prompts.system(schema)
        self.messages: list[Message] = [
            Message(role="user", content=prompts.user(text, schema, instructions=instructions))
        ]
        self.attempts: list[Attempt] = []

    @property
    def attempt_number(self) -> int:
        return len(self.attempts) + 1

    def request(self) -> CompletionRequest:
        return CompletionRequest(
            system=self.system,
            messages=tuple(self.messages),
            schema_name=self.schema.name,
            json_schema=self.schema.json_schema,
            max_tokens=self.max_tokens,
            attempt=self.attempt_number,
        )

    def _accept(self, completion: Completion) -> T:
        text = completion.text
        if completion.stop_reason == STOP_REFUSAL:
            raise RefusalError("The model declined the request.", provider=self.provider_name)
        if completion.stop_reason == STOP_MAX_TOKENS:
            raise OutputTruncatedError(
                "The output token limit was reached before the answer was complete.",
                raw_text=text,
            )
        if not text.strip():
            raise MissingOutputError("the response contained no output", raw_text=text)
        data = parse_json(text)
        return self.schema.validate(data)

    def _record(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)
        if attempt.ok:
            logger.debug("%s: attempt %d succeeded", self.schema.name, attempt.number)
        else:
            logger.info(
                "%s: attempt %d/%d failed: %s",
                self.schema.name,
                attempt.number,
                self.max_attempts,
                attempt.error,
            )
        if self.on_attempt is not None:
            self.on_attempt(attempt)

    def evaluate(self, completion: Completion) -> ExtractionResult[T] | Attempt:
        """Accept ``completion`` or record it as a failed attempt.

        Returns the result on success, or the failed :class:`Attempt` after
        queueing the repair turn. Raises :class:`ExtractionFailedError` when no
        attempts are left, and lets :class:`RefusalError` propagate.
        """
        try:
            value = self._accept(completion)
        except OutputError as error:
            if not error.raw_text:
                error.raw_text = completion.text
            return self.fail(error, completion)
        attempt = Attempt(
            number=self.attempt_number,
            raw_text=completion.text,
            error=None,
            usage=completion.usage,
            stop_reason=completion.stop_reason,
            model=completion.model,
        )
        self._record(attempt)
        return ExtractionResult(value=value, attempts=tuple(self.attempts))

    def fail(self, error: OutputError, completion: Completion | None = None) -> Attempt:
        """Record a failed attempt and queue the repair turn (or give up)."""
        attempt = Attempt(
            number=self.attempt_number,
            raw_text=completion.text if completion is not None else error.raw_text,
            error=error,
            usage=completion.usage if completion is not None else Usage(),
            stop_reason=completion.stop_reason if completion is not None else None,
            model=completion.model if completion is not None else None,
        )
        self._record(attempt)
        if len(self.attempts) >= self.max_attempts:
            raise ExtractionFailedError(self.attempts, schema_name=self.schema.name) from error
        # Replay the model's answer (when there is one) and append the feedback:
        # the conversation only ever grows, it is never edited.
        if completion is not None and (completion.text.strip() or completion.native is not None):
            self.messages.append(
                Message(role="assistant", content=completion.text, native=completion.native)
            )
        self.messages.append(Message(role="user", content=self.prompts.repair(error, self.schema)))
        return attempt


#: Re-parsing the buffer costs O(len), so snapshots are spaced geometrically: a new
#: parse happens once the text has grown by 1/16th since the last one. Short outputs
#: are re-parsed on every chunk; a whole stream stays O(n log n) instead of O(n^2).
_PARSE_SPACING = 16


class _StreamAccumulator:
    """Collects streamed text and emits a partial-JSON snapshot whenever it changes."""

    def __init__(self, attempt: int) -> None:
        self.attempt = attempt
        self.text = ""
        self.completion: Completion | None = None
        self._last: Any = object()
        self._parsed_length = 0
        self._next_parse_at = 0

    def feed(self, chunk: StreamChunk) -> PartialEvent | None:
        if chunk.completion is not None:
            self.completion = chunk.completion
        if not chunk.text:
            return None
        self.text += chunk.text
        if len(self.text) < self._next_parse_at:
            return None
        return self._snapshot()

    def flush(self) -> PartialEvent | None:
        """Snapshot any text that arrived since the last parse (call at end of stream)."""
        if len(self.text) == self._parsed_length:
            return None
        return self._snapshot()

    def _snapshot(self) -> PartialEvent | None:
        length = len(self.text)
        self._parsed_length = length
        self._next_parse_at = length + length // _PARSE_SPACING
        partial = parse_partial_json(self.text)
        if partial is None or partial.value == self._last:
            return None
        self._last = partial.value
        return PartialEvent(attempt=self.attempt, data=partial.value, text=self.text)

    def result(self) -> Completion:
        return self.completion if self.completion is not None else Completion(text=self.text)


class Extractor:
    """Turns unstructured text into validated, typed objects using an LLM.

    Args:
        provider: Any :class:`~llm_structured_extract.Provider`.
        max_retries: Repair rounds allowed after the first attempt (so at most
            ``max_retries + 1`` model calls). ``0`` disables repair.
        max_tokens: Output token budget per call.
        prompts: Customise the wording of every prompt.
        on_attempt: Called after each attempt (success or failure), e.g. for
            logging or metrics.

    Example::

        extractor = Extractor(AnthropicProvider(), max_retries=2)
        result = extractor.extract(invoice_text, Invoice)
        invoice: Invoice = result.value
    """

    def __init__(
        self,
        provider: Provider,
        *,
        max_retries: int = 2,
        max_tokens: int = 16_000,
        prompts: PromptBuilder | None = None,
        on_attempt: AttemptCallback | None = None,
    ) -> None:
        _check_retries(max_retries)
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.provider = provider
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.prompts = prompts or PromptBuilder()
        self.on_attempt = on_attempt

    def _session(
        self,
        text: str,
        schema: Any,
        instructions: str | None,
        max_retries: int | None,
    ) -> _Session[Any]:
        retries = self.max_retries if max_retries is None else max_retries
        _check_retries(retries)
        return _Session(
            schema=as_output_schema(schema),
            prompts=self.prompts,
            text=text,
            instructions=instructions,
            max_attempts=retries + 1,
            max_tokens=self.max_tokens,
            provider_name=self.provider.name,
            on_attempt=self.on_attempt,
        )

    # -- one-shot ---------------------------------------------------------- #

    @overload
    def extract(
        self,
        text: str,
        schema: OutputSchema[T],
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> ExtractionResult[T]: ...
    @overload
    def extract(
        self,
        text: str,
        schema: type[T],
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> ExtractionResult[T]: ...
    @overload
    def extract(
        self,
        text: str,
        schema: Mapping[str, Any],
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> ExtractionResult[Any]: ...
    def extract(
        self,
        text: str,
        schema: Any,
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> ExtractionResult[Any]:
        """Extract one value of ``schema`` from ``text``.

        ``schema`` may be a Pydantic-compatible type, a raw JSON Schema
        ``dict`` or an :class:`~llm_structured_extract.OutputSchema`.
        ``instructions`` adds task-specific guidance to the extraction turn.

        Raises:
            ExtractionFailedError: every attempt produced invalid output.
            RefusalError: the model declined.
            ProviderError: the provider call failed.
        """
        session = self._session(text, schema, instructions, max_retries)
        while True:
            try:
                completion = self.provider.complete(session.request())
            except OutputError as error:
                session.fail(error)
                continue
            outcome = session.evaluate(completion)
            if isinstance(outcome, ExtractionResult):
                return outcome

    @overload
    async def aextract(
        self,
        text: str,
        schema: OutputSchema[T],
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> ExtractionResult[T]: ...
    @overload
    async def aextract(
        self,
        text: str,
        schema: type[T],
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> ExtractionResult[T]: ...
    @overload
    async def aextract(
        self,
        text: str,
        schema: Mapping[str, Any],
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> ExtractionResult[Any]: ...
    async def aextract(
        self,
        text: str,
        schema: Any,
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> ExtractionResult[Any]:
        """Async version of :meth:`extract`."""
        session = self._session(text, schema, instructions, max_retries)
        while True:
            try:
                completion = await self.provider.acomplete(session.request())
            except OutputError as error:
                session.fail(error)
                continue
            outcome = session.evaluate(completion)
            if isinstance(outcome, ExtractionResult):
                return outcome

    # -- streaming --------------------------------------------------------- #

    @overload
    def stream(
        self,
        text: str,
        schema: OutputSchema[T],
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> Iterator[StreamEvent[T]]: ...
    @overload
    def stream(
        self,
        text: str,
        schema: type[T],
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> Iterator[StreamEvent[T]]: ...
    @overload
    def stream(
        self,
        text: str,
        schema: Mapping[str, Any],
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> Iterator[StreamEvent[Any]]: ...
    def stream(
        self,
        text: str,
        schema: Any,
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> Iterator[StreamEvent[Any]]:
        """Like :meth:`extract`, but yields events while the model writes.

        Yields :class:`~llm_structured_extract.PartialEvent` snapshots of the
        JSON parsed so far, a :class:`~llm_structured_extract.RepairEvent`
        whenever an attempt fails validation and is sent back for repair, and
        finally one :class:`~llm_structured_extract.ResultEvent`. Raises the same
        errors as :meth:`extract`.
        """
        session = self._session(text, schema, instructions, max_retries)
        return self._stream(session)

    def _stream(self, session: _Session[Any]) -> Iterator[StreamEvent[Any]]:
        while True:
            accumulator = _StreamAccumulator(session.attempt_number)
            try:
                for chunk in self.provider.stream(session.request()):
                    event = accumulator.feed(chunk)
                    if event is not None:
                        yield event
            except OutputError as error:
                yield RepairEvent(session.fail(error))
                continue
            final_snapshot = accumulator.flush()
            if final_snapshot is not None:
                yield final_snapshot
            outcome = session.evaluate(accumulator.result())
            if isinstance(outcome, ExtractionResult):
                yield ResultEvent(outcome)
                return
            yield RepairEvent(outcome)

    @overload
    def astream(
        self,
        text: str,
        schema: OutputSchema[T],
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> AsyncIterator[StreamEvent[T]]: ...
    @overload
    def astream(
        self,
        text: str,
        schema: type[T],
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> AsyncIterator[StreamEvent[T]]: ...
    @overload
    def astream(
        self,
        text: str,
        schema: Mapping[str, Any],
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> AsyncIterator[StreamEvent[Any]]: ...
    def astream(
        self,
        text: str,
        schema: Any,
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> AsyncIterator[StreamEvent[Any]]:
        """Async version of :meth:`stream`."""
        session = self._session(text, schema, instructions, max_retries)
        return self._astream(session)

    async def _astream(self, session: _Session[Any]) -> AsyncIterator[StreamEvent[Any]]:
        while True:
            accumulator = _StreamAccumulator(session.attempt_number)
            try:
                async for chunk in self.provider.astream(session.request()):
                    event = accumulator.feed(chunk)
                    if event is not None:
                        yield event
            except OutputError as error:
                yield RepairEvent(session.fail(error))
                continue
            final_snapshot = accumulator.flush()
            if final_snapshot is not None:
                yield final_snapshot
            outcome = session.evaluate(accumulator.result())
            if isinstance(outcome, ExtractionResult):
                yield ResultEvent(outcome)
                return
            yield RepairEvent(outcome)


def _check_retries(value: int) -> None:
    if value < 0:
        raise ValueError("max_retries must be >= 0")
