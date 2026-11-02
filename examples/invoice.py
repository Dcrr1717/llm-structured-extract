"""Extract a typed invoice from free text - with business rules the model must satisfy.

Run offline (deterministic, no API key, no cost)::

    python examples/invoice.py

Run against Claude (needs ANTHROPIC_API_KEY; makes a paid API call)::

    python examples/invoice.py --live

The schema does more than describe shapes. Pydantic validators enforce the
arithmetic of an invoice - every line amount equals quantity x unit price, the
line amounts add up to the subtotal, subtotal + tax equals the total - so a
model that misreads a number gets a precise error back and a chance to fix it.

Offline, a scripted ``MockProvider`` plays a model that makes three classic
mistakes on its first attempt (a non-ISO date, a currency symbol instead of an
ISO code, a mis-copied line amount) and fixes them once the validation errors
are fed back.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any

from pydantic import BaseModel, Field, WithJsonSchema, model_validator

from llm_structured_extract import (
    AnthropicProvider,
    Attempt,
    ExtractionResult,
    Extractor,
    MockProvider,
    Provider,
)

CENT = Decimal("0.01")

#: Exact decimals in Python, advertised to the model as plain JSON numbers.
Money = Annotated[
    Decimal,
    Field(max_digits=14, decimal_places=2),
    WithJsonSchema({"type": "number"}),
]
Quantity = Annotated[Decimal, Field(gt=0), WithJsonSchema({"type": "number"})]
Percent = Annotated[Decimal, Field(ge=0, le=100), WithJsonSchema({"type": "number"})]


class Party(BaseModel):
    """A company or person named on the invoice."""

    name: str = Field(description="Legal or trading name.")
    address: str | None = Field(default=None, description="Postal address on a single line.")
    tax_id: str | None = Field(default=None, description="VAT / tax ID exactly as printed.")
    email: str | None = None


class LineItem(BaseModel):
    description: str
    quantity: Quantity = Field(description="Number of units billed.")
    unit_price: Money = Field(description="Price per unit, excluding tax.")
    amount: Money = Field(description="Line total excluding tax, as printed.")

    @model_validator(mode="after")
    def _amount_is_quantity_times_price(self) -> LineItem:
        expected = (self.quantity * self.unit_price).quantize(CENT, rounding=ROUND_HALF_UP)
        if self.amount != expected:
            raise ValueError(
                f"amount {self.amount:.2f} does not equal quantity x unit_price "
                f"({self.quantity} x {self.unit_price:.2f} = {expected:.2f})"
            )
        return self


class Invoice(BaseModel):
    """An invoice, with its arithmetic checked."""

    invoice_number: str
    issue_date: date = Field(description="ISO 8601 date (YYYY-MM-DD).")
    due_date: date | None = Field(default=None, description="ISO 8601 date (YYYY-MM-DD).")
    currency: str = Field(pattern=r"^[A-Z]{3}$", description="ISO 4217 currency code, e.g. EUR.")
    vendor: Party = Field(description="Who issued the invoice.")
    customer: Party = Field(description="Who is billed.")
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: Money = Field(description="Sum of line amounts, excluding tax.")
    tax_rate_percent: Percent | None = Field(default=None, description="E.g. 21 for 21%.")
    tax: Money
    total: Money = Field(description="Amount due, including tax.")

    @model_validator(mode="after")
    def _totals_add_up(self) -> Invoice:
        problems = []
        items_sum = sum((item.amount for item in self.line_items), Decimal(0))
        if items_sum != self.subtotal:
            problems.append(
                f"line item amounts sum to {items_sum:.2f}, but subtotal is {self.subtotal:.2f}"
            )
        if self.subtotal + self.tax != self.total:
            problems.append(
                f"subtotal + tax = {self.subtotal + self.tax:.2f}, but total is {self.total:.2f}"
            )
        if self.due_date is not None and self.due_date < self.issue_date:
            problems.append("due_date is before issue_date")
        if problems:
            raise ValueError("; ".join(problems))
        return self


SAMPLE_TEXT = """\
NORTHWIND DESIGN STUDIO S.L.
Calle de Alcala 112, 28009 Madrid, Spain - VAT ES-B12345678
billing@northwind-design.example

INVOICE No. NW-2026-0142
Date of issue: 3 March 2026            Payment due: 2 April 2026

Bill to:
Lumen Analytics GmbH
Torstrasse 45, 10119 Berlin, Germany
VAT DE298765432

Description                              Qty    Unit price       Amount
Brand identity workshop (half day)         1      1,200.00     1,200.00
Logo design - three concepts               3        450.00     1,350.00
Website UI mock-ups (per page)             6        180.00     1,080.00

                                         Subtotal              3,630.00
                                         VAT 21%                 762.30
                                         TOTAL DUE (EUR)       4,392.30

Payment by bank transfer within 30 days. Thank you for your business!
"""

_VENDOR = {
    "name": "Northwind Design Studio S.L.",
    "address": "Calle de Alcala 112, 28009 Madrid, Spain",
    "tax_id": "ES-B12345678",
    "email": "billing@northwind-design.example",
}
_CUSTOMER = {
    "name": "Lumen Analytics GmbH",
    "address": "Torstrasse 45, 10119 Berlin, Germany",
    "tax_id": "DE298765432",
    "email": None,
}

#: First attempt of the scripted model: plausible, but wrong in three places.
FLAWED_ANSWER: dict[str, Any] = {
    "invoice_number": "NW-2026-0142",
    "issue_date": "3 March 2026",
    "due_date": "2026-04-02",
    "currency": "€",
    "vendor": _VENDOR,
    "customer": _CUSTOMER,
    "line_items": [
        {
            "description": "Brand identity workshop (half day)",
            "quantity": 1,
            "unit_price": 1200.00,
            "amount": 1200.00,
        },
        {
            "description": "Logo design - three concepts",
            "quantity": 3,
            "unit_price": 450.00,
            "amount": 1530.00,
        },
        {
            "description": "Website UI mock-ups (per page)",
            "quantity": 6,
            "unit_price": 180.00,
            "amount": 1080.00,
        },
    ],
    "subtotal": 3630.00,
    "tax_rate_percent": 21,
    "tax": 762.30,
    "total": 4392.30,
}

#: Second attempt: every reported problem fixed, everything else untouched.
CORRECTED_ANSWER: dict[str, Any] = {
    **FLAWED_ANSWER,
    "issue_date": "2026-03-03",
    "currency": "EUR",
    "line_items": [
        FLAWED_ANSWER["line_items"][0],
        {**FLAWED_ANSWER["line_items"][1], "amount": 1350.00},
        FLAWED_ANSWER["line_items"][2],
    ],
}


def mock_provider() -> MockProvider:
    """A deterministic stand-in for an LLM: one flawed answer, then the repair."""
    return MockProvider([FLAWED_ANSWER, CORRECTED_ANSWER])


def describe_attempt(attempt: Attempt) -> str:
    if attempt.error is None:
        return f"attempt {attempt.number}: valid"
    lines = [f"attempt {attempt.number}: rejected - {type(attempt.error).__name__}"]
    for issue in getattr(attempt.error, "issues", ()):
        lines.append(f"    - {issue.describe()}")
    return "\n".join(lines)


def run(provider: Provider) -> ExtractionResult[Invoice]:
    extractor = Extractor(provider, max_retries=2, on_attempt=lambda a: print(describe_attempt(a)))
    return extractor.extract(SAMPLE_TEXT, Invoice)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--live",
        action="store_true",
        help="use Claude via AnthropicProvider (needs ANTHROPIC_API_KEY; paid API call)",
    )
    args = parser.parse_args(argv)
    provider: Provider = AnthropicProvider() if args.live else mock_provider()

    result = run(provider)
    invoice = result.value
    print()
    print(f"{invoice.invoice_number}: {invoice.vendor.name} -> {invoice.customer.name}")
    for item in invoice.line_items:
        print(f"  {item.quantity} x {item.description:<38} {item.amount:>10,.2f}")
    print(f"  {'total (' + invoice.currency + ')':<43} {invoice.total:>10,.2f}")
    print(f"issued {invoice.issue_date.isoformat()}, due {invoice.due_date}")
    print(f"attempts: {len(result.attempts)}, tokens: {result.usage.total_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
