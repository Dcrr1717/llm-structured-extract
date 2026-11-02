"""The example scripts run offline and their schemas enforce what they claim."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

import custom_provider
import invoice
import research_abstract
from llm_structured_extract import (
    OutputValidationError,
    ProviderNotConfiguredError,
    PydanticSchema,
)


def test_invoice_example_runs_offline(capsys: pytest.CaptureFixture[str]) -> None:
    assert invoice.main([]) == 0
    out = capsys.readouterr().out
    assert "attempt 1: rejected - OutputValidationError" in out
    assert "attempt 2: valid" in out
    assert "total (EUR)                                   4,392.30" in out


def test_invoice_extraction_value() -> None:
    result = invoice.run(invoice.mock_provider())
    value = result.value
    assert value.issue_date == date(2026, 3, 3)
    assert value.currency == "EUR"
    assert value.total == Decimal("4392.30")
    assert [item.amount for item in value.line_items] == [
        Decimal("1200.00"),
        Decimal("1350.00"),
        Decimal("1080.00"),
    ]
    first_error = result.attempts[0].error
    assert isinstance(first_error, OutputValidationError)
    assert [issue.path for issue in first_error.issues] == [
        "issue_date",
        "currency",
        "line_items[1]",
    ]


def _invoice_issues(**changes: Any) -> list[str]:
    data = {**invoice.CORRECTED_ANSWER, **changes}
    with pytest.raises(OutputValidationError) as info:
        PydanticSchema(invoice.Invoice).validate(data)
    return [issue.message for issue in info.value.issues]


def test_invoice_rules() -> None:
    assert PydanticSchema(invoice.Invoice).validate(invoice.CORRECTED_ANSWER).subtotal == Decimal(
        "3630.00"
    )
    (message,) = _invoice_issues(subtotal=3600.00, total=4362.30)
    assert "line item amounts sum to 3630.00, but subtotal is 3600.00" in message
    (message,) = _invoice_issues(total=4400)
    assert "subtotal + tax = 4392.30, but total is 4400.00" in message
    (message,) = _invoice_issues(due_date="2026-01-01")
    assert "due_date is before issue_date" in message
    (message,) = _invoice_issues(line_items=[])
    assert "at least 1 item" in message
    (message,) = _invoice_issues(tax=762.305)
    assert "decimal places" in message


def test_research_abstract_example_runs_offline(capsys: pytest.CaptureFixture[str]) -> None:
    assert research_abstract.main([]) == 0
    out = capsys.readouterr().out
    assert "design: randomized_controlled_trial, n = 212" in out
    assert "attempts: 2, repaired: True" in out


def test_research_abstract_streaming(capsys: pytest.CaptureFixture[str]) -> None:
    assert research_abstract.main(["--stream"]) == 0
    out = capsys.readouterr().out
    assert "[attempt 1] receiving: title" in out
    assert "[attempt 1] rejected, asking for a repair:" in out
    assert "`sample_size`: Input should be a valid integer" in out
    assert "`key_findings[1].p_value`" in out
    assert "[attempt 2] receiving: keywords" in out


def test_research_abstract_rules() -> None:
    schema = PydanticSchema(research_abstract.ResearchAbstract)
    data = research_abstract.CORRECTED_ANSWER
    assert schema.validate(data).sample_size == 212

    bad_findings = [
        {**data["key_findings"][0], "confidence_interval": {"lower": 41, "upper": 27}},
        {**data["key_findings"][1], "p_value": 1.5},
    ]
    with pytest.raises(OutputValidationError) as info:
        schema.validate({**data, "key_findings": bad_findings, "keywords": ["Sleep", "a", "b"]})
    messages = {issue.path: issue.message for issue in info.value.issues}
    assert (
        "lower bound 41.0 is greater than upper bound 27.0"
        in messages["key_findings[0].confidence_interval"]
    )
    assert "less than or equal to 1" in messages["key_findings[1].p_value"]
    assert "keywords must be lower-case: ['Sleep']" in messages["keywords"]


def test_custom_provider_example(capsys: pytest.CaptureFixture[str]) -> None:
    assert custom_provider.main() == 0
    assert "equal: True, model calls: 1, cache hits: 1" in capsys.readouterr().out


@pytest.mark.parametrize("module", [invoice, research_abstract])
def test_live_flag_requires_a_key(module: Any) -> None:
    # The autouse fixture removes ANTHROPIC_API_KEY, so no paid call can happen.
    with pytest.raises(ProviderNotConfiguredError, match="ANTHROPIC_API_KEY"):
        module.main(["--live"])


def test_streaming_and_one_shot_agree() -> None:
    one_shot = research_abstract.run(research_abstract.mock_provider())
    streamed = research_abstract.run_streaming(research_abstract.mock_provider(chunk_size=3))
    assert one_shot.value == streamed.value
