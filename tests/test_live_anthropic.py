"""Opt-in end-to-end tests against the real Anthropic API.

These make paid API calls and are skipped unless BOTH are set::

    ANTHROPIC_API_KEY=...  LSE_RUN_LIVE_TESTS=1  pytest -m live

CI never sets them.
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest

import invoice
import research_abstract
from llm_structured_extract import AnthropicProvider, Extractor
from llm_structured_extract.providers.anthropic import Mode

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("LSE_RUN_LIVE_TESTS") == "1"),
        reason="live API tests need ANTHROPIC_API_KEY and LSE_RUN_LIVE_TESTS=1",
    ),
]


@pytest.mark.parametrize("mode", ["json_schema", "tool", "prompt"])
def test_invoice_live(mode: Mode) -> None:
    result = Extractor(AnthropicProvider(mode=mode)).extract(invoice.SAMPLE_TEXT, invoice.Invoice)
    assert result.value.invoice_number == "NW-2026-0142"
    assert result.value.total == Decimal("4392.30")
    assert result.value.currency == "EUR"


def test_research_abstract_live_streaming() -> None:
    result = research_abstract.run_streaming(AnthropicProvider())
    assert result.value.sample_size == 212
    assert result.value.study_design is research_abstract.StudyDesign.RANDOMIZED_CONTROLLED_TRIAL
