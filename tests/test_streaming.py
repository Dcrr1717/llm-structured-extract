from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from itertools import pairwise
from typing import Any

import pytest
from pydantic import BaseModel

from llm_structured_extract import (
    BaseProvider,
    Completion,
    CompletionRequest,
    ExtractionFailedError,
    Extractor,
    MockProvider,
    OutputParseError,
    PartialEvent,
    RepairEvent,
    ResultEvent,
    SchemaError,
    StreamChunk,
    StreamEvent,
)


class Book(BaseModel):
    title: str
    authors: list[str]
    year: int


GOOD = {"title": "The Mythical Man-Month", "authors": ["Fred Brooks"], "year": 1975}


def collect(events: Iterator[StreamEvent[Any]]) -> list[StreamEvent[Any]]:
    return list(events)


def test_stream_yields_growing_partials_then_result() -> None:
    provider = MockProvider([GOOD], chunk_size=5)
    events = collect(Extractor(provider).stream("...", Book))

    *partials, final = events
    assert all(isinstance(event, PartialEvent) for event in partials)
    assert isinstance(final, ResultEvent)
    assert final.result.value == Book.model_validate(GOOD)

    snapshots = [event.data for event in partials if isinstance(event, PartialEvent)]
    assert snapshots[0] == {}
    assert snapshots[-1] == GOOD
    # Snapshots only grow: once a key appears it never disappears.
    for before, after in pairwise(snapshots):
        assert set(before) <= set(after)
    # Consecutive duplicates are suppressed.
    assert all(a != b for a, b in pairwise(snapshots))
    # Partial strings are visible while they are being written.
    title = str(GOOD["title"])
    assert any(0 < len(str(s.get("title", ""))) < len(title) for s in snapshots)


def test_stream_repair_events() -> None:
    provider = MockProvider([{**GOOD, "year": "nineteen"}, GOOD], chunk_size=50)
    events = collect(Extractor(provider).stream("...", Book))

    kinds = [type(event).__name__ for event in events]
    assert kinds.count("RepairEvent") == 1
    assert kinds[-1] == "ResultEvent"
    repair = next(event for event in events if isinstance(event, RepairEvent))
    assert repair.attempt.number == 1
    assert repair.error.issues[0].path == "year"  # type: ignore[attr-defined]

    attempts = {event.attempt for event in events if isinstance(event, PartialEvent)}
    assert attempts == {1, 2}
    final = events[-1]
    assert isinstance(final, ResultEvent)
    assert len(final.result.attempts) == 2


def test_stream_raises_when_out_of_attempts() -> None:
    provider = MockProvider(["nope"], on_exhausted="repeat_last")
    events: list[StreamEvent[Any]] = []
    stream = Extractor(provider, max_retries=1).stream("...", Book)
    with pytest.raises(ExtractionFailedError):
        events.extend(stream)
    assert sum(isinstance(event, RepairEvent) for event in events) == 1
    assert provider.calls == 2


def test_schema_errors_surface_at_call_time() -> None:
    class NotPydantic:
        pass

    with pytest.raises(SchemaError):
        Extractor(MockProvider()).stream("...", NotPydantic)


class ChunkOnlyProvider(BaseProvider):
    """A stream that never sends a final completion chunk."""

    name = "chunks"

    def complete(self, request: CompletionRequest) -> Completion:  # pragma: no cover - unused
        raise NotImplementedError

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        yield StreamChunk(text='{"title": "T", ')
        yield StreamChunk(text='"authors": [], "year": 2000}')


def test_stream_without_final_completion_uses_accumulated_text() -> None:
    events = collect(Extractor(ChunkOnlyProvider()).stream("...", Book))
    final = events[-1]
    assert isinstance(final, ResultEvent)
    assert final.result.value == Book(title="T", authors=[], year=2000)


class FlakyStreamProvider(BaseProvider):
    """Fails mid-stream with an output error on the first attempt."""

    name = "flaky"

    def complete(self, request: CompletionRequest) -> Completion:  # pragma: no cover - unused
        raise NotImplementedError

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        if request.attempt == 1:
            yield StreamChunk(text='{"title": "T"')
            raise OutputParseError("tool input was not valid JSON")
        text = '{"title": "T", "authors": ["A"], "year": 1}'
        yield StreamChunk(text=text, completion=Completion(text=text))

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        for chunk in self.stream(request):
            yield chunk


def test_output_error_mid_stream_is_repaired() -> None:
    events = collect(Extractor(FlakyStreamProvider()).stream("...", Book))
    assert isinstance(events[0], PartialEvent)
    assert isinstance(events[1], RepairEvent)
    assert isinstance(events[1].error, OutputParseError)
    final = events[-1]
    assert isinstance(final, ResultEvent)
    assert final.result.value.authors == ["A"]


def test_async_stream() -> None:
    provider = MockProvider([{**GOOD, "year": "x"}, GOOD], chunk_size=10)

    async def run() -> list[StreamEvent[Book]]:
        return [event async for event in Extractor(provider).astream("...", Book)]

    events = asyncio.run(run())
    assert isinstance(events[-1], ResultEvent)
    assert events[-1].result.value == Book.model_validate(GOOD)
    assert sum(isinstance(event, RepairEvent) for event in events) == 1


def test_async_stream_repairs_mid_stream_output_errors() -> None:
    async def run() -> list[StreamEvent[Book]]:
        return [event async for event in Extractor(FlakyStreamProvider()).astream("...", Book)]

    events = asyncio.run(run())
    assert [type(e).__name__ for e in events] == [
        "PartialEvent",
        "RepairEvent",
        "PartialEvent",
        "ResultEvent",
    ]


def test_pattern_matching_on_events() -> None:
    provider = MockProvider([GOOD])
    titles: list[str] = []
    for event in Extractor(provider).stream("...", Book):
        match event:
            case PartialEvent(data={"title": str(title)}):
                titles.append(title)
            case ResultEvent(result=result):
                assert result.value.year == 1975
    assert titles[-1] == GOOD["title"]


def test_large_streams_are_not_quadratic() -> None:
    """~40 KB of JSON in 12-character chunks: parse work must stay near-linear."""

    class Catalogue(BaseModel):
        items: list[dict[str, Any]]

    document = {
        "items": [
            {"sku": f"SKU-{i}", "description": f"A fairly long description of item {i}", "qty": i}
            for i in range(450)
        ]
    }
    provider = MockProvider([document], chunk_size=12)
    events = list(Extractor(provider).stream("...", Catalogue))
    partials = [event for event in events if isinstance(event, PartialEvent)]
    # Every chunk re-parsed would be ~3,000 snapshots; geometric spacing needs a few hundred.
    assert len(partials) < 300
    assert partials[-1].data == document  # the final flush makes the last snapshot complete
    assert isinstance(events[-1], ResultEvent)
