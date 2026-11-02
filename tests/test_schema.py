from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict  # pydantic needs this flavour on Python < 3.12

from llm_structured_extract import (
    JSONSchema,
    OutputSchema,
    OutputValidationError,
    PydanticSchema,
    SchemaError,
    ValidationIssue,
    as_output_schema,
)


class Item(BaseModel):
    sku: str
    quantity: int = Field(gt=0)


class Order(BaseModel):
    order_id: str
    items: list[Item] = Field(min_length=1)
    note: str | None = None

    @field_validator("order_id")
    @classmethod
    def _prefixed(cls, value: str) -> str:
        if not value.startswith("ORD-"):
            raise ValueError("order_id must start with 'ORD-'")
        return value


def _issues(schema: OutputSchema[Any], data: Any) -> tuple[ValidationIssue, ...]:
    with pytest.raises(OutputValidationError) as info:
        schema.validate(data)
    return info.value.issues


# --------------------------------------------------------------------------- #
# Pydantic adapter
# --------------------------------------------------------------------------- #


def test_pydantic_schema_validates_and_returns_model() -> None:
    schema = PydanticSchema(Order)
    order = schema.validate({"order_id": "ORD-1", "items": [{"sku": "A", "quantity": "2"}]})
    assert order == Order(order_id="ORD-1", items=[Item(sku="A", quantity=2)])
    assert schema.name == "Order"
    assert schema.type is Order
    assert schema.json_schema["type"] == "object"
    assert "Item" in schema.json_schema["$defs"]


def test_pydantic_issues_have_model_readable_paths() -> None:
    issues = _issues(
        PydanticSchema(Order),
        {"order_id": "1", "items": [{"sku": "A", "quantity": 1}, {"sku": "B", "quantity": 0}]},
    )
    by_path = {issue.path: issue for issue in issues}
    assert set(by_path) == {"order_id", "items[1].quantity"}
    assert "must start with 'ORD-'" in by_path["order_id"].message
    assert by_path["items[1].quantity"].kind == "greater_than"
    assert by_path["items[1].quantity"].input == 0
    assert by_path["items[1].quantity"].describe() == (
        "`items[1].quantity`: Input should be greater than 0 (got: 0)"
    )


def test_missing_fields_do_not_show_an_input_preview() -> None:
    (issue,) = _issues(PydanticSchema(Item), {"sku": "A"})
    assert issue.path == "quantity"
    assert issue.kind == "missing"
    assert "(got:" not in issue.describe()


def test_root_level_issue_uses_root_path() -> None:
    (issue,) = _issues(PydanticSchema(Item), ["not", "an", "object"])
    assert issue.path == "<root>"
    assert issue.is_root
    assert "(got:" not in issue.describe()


def test_long_inputs_are_truncated_in_descriptions() -> None:
    (issue,) = _issues(PydanticSchema(Item), {"sku": "A", "quantity": "x" * 500})
    description = issue.describe(max_input_chars=40)
    assert description.endswith("...)")
    assert len(description) < 140


def test_strict_mode_disables_coercion() -> None:
    assert PydanticSchema(Item).validate({"sku": "A", "quantity": "3"}).quantity == 3
    _issues(PydanticSchema(Item, strict=True), {"sku": "A", "quantity": "3"})


def test_generic_and_non_model_types() -> None:
    @dataclass
    class Point:
        x: int
        y: int

    class Movie(TypedDict):
        title: str
        year: int

    points = PydanticSchema(list[Point])
    assert points.name == "list[Point]"
    assert points.validate([{"x": 1, "y": 2}]) == [Point(1, 2)]
    assert PydanticSchema(Movie).validate({"title": "Alien", "year": "1979"}) == {
        "title": "Alien",
        "year": 1979,
    }
    assert PydanticSchema(dict[str, int]).name == "dict[str, int]"


def test_custom_name() -> None:
    assert PydanticSchema(Order, name="PurchaseOrder").name == "PurchaseOrder"


def test_unsupported_type_raises_schema_error() -> None:
    class NotPydantic:
        pass

    with pytest.raises(SchemaError, match="NotPydantic"):
        PydanticSchema(NotPydantic)


def test_validation_error_summary() -> None:
    with pytest.raises(OutputValidationError) as info:
        PydanticSchema(Order).validate({})
    error = info.value
    assert error.schema_name == "Order"
    assert "(2 issues)" in str(error)


# --------------------------------------------------------------------------- #
# Raw JSON Schema adapter
# --------------------------------------------------------------------------- #

CONTACT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Contact",
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "email": {"type": "string", "format": "email"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "met_on": {"type": "string", "format": "date"},
    },
    "required": ["name", "email"],
    "additionalProperties": False,
}


def test_json_schema_validates() -> None:
    schema = JSONSchema(CONTACT_SCHEMA)
    data = {"name": "Ada", "email": "ada@example.com", "tags": ["math"]}
    assert schema.validate(data) is data
    assert schema.name == "Contact"
    assert schema.json_schema == CONTACT_SCHEMA


def test_json_schema_issues() -> None:
    issues = _issues(
        JSONSchema(CONTACT_SCHEMA),
        {"name": "", "email": "not-an-email", "tags": ["ok", 3], "extra": 1, "met_on": "soon"},
    )
    paths = {issue.path for issue in issues}
    assert paths == {"<root>", "name", "email", "tags[1]", "met_on"}
    kinds = {issue.kind for issue in issues}
    assert {"additionalProperties", "minLength", "format", "type"} <= kinds


def test_json_schema_missing_required() -> None:
    (issue,) = _issues(JSONSchema(CONTACT_SCHEMA), {"name": "Ada"})
    assert issue.path == "<root>"
    assert "'email' is a required property" in issue.message


def test_json_schema_name_defaults() -> None:
    assert JSONSchema({"type": "object"}).name == "Output"
    assert JSONSchema({"type": "object"}, name="Thing").name == "Thing"


def test_invalid_json_schema_raises_schema_error() -> None:
    with pytest.raises(SchemaError, match="Invalid JSON Schema"):
        JSONSchema({"type": "not-a-type"})


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #


def test_as_output_schema_dispatch() -> None:
    pydantic_schema = as_output_schema(Order)
    assert isinstance(pydantic_schema, PydanticSchema)
    assert isinstance(as_output_schema(CONTACT_SCHEMA), JSONSchema)
    assert as_output_schema(pydantic_schema) is pydantic_schema
    assert as_output_schema(Order, name="X").name == "X"
    assert repr(pydantic_schema) == "PydanticSchema(name='Order')"
