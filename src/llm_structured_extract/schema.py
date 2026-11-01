"""Output schemas: what the model must produce and how the answer is validated.

An :class:`OutputSchema` bundles three things the extractor needs:

* a ``name`` used in prompts (and, for tool-calling providers, the tool name),
* the ``json_schema`` shown to the model / handed to constrained decoding,
* ``validate()``, which turns parsed JSON into the typed result or raises
  :class:`~llm_structured_extract.errors.OutputValidationError` with
  normalised, model-readable issues.

Two adapters ship with the package:

* :class:`PydanticSchema` - any type Pydantic v2 can validate: ``BaseModel``
  subclasses, ``list[Model]``, dataclasses, ``TypedDict``... Custom field and
  model validators run as part of validation, so business rules ("line items
  must add up to the subtotal") become repairable errors too.
* :class:`JSONSchema` - a raw JSON Schema ``dict``; validated with the
  ``jsonschema`` package (install the ``jsonschema`` extra).

:func:`as_output_schema` picks the right adapter for whatever you pass.
"""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, TypeVar, overload

from pydantic import TypeAdapter, ValidationError
from pydantic.errors import PydanticSchemaGenerationError, PydanticUserError

from .errors import (
    OutputValidationError,
    SchemaError,
    ValidationIssue,
    format_path,
)

__all__ = ["JSONSchema", "OutputSchema", "PydanticSchema", "as_output_schema"]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class OutputSchema(ABC, Generic[T_co]):
    """The contract between the extractor, the provider and your code."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, human-readable name of the expected output (e.g. ``"Invoice"``)."""

    @property
    @abstractmethod
    def json_schema(self) -> dict[str, Any]:
        """JSON Schema of the expected output."""

    @abstractmethod
    def validate(self, data: Any) -> T_co:
        """Validate parsed JSON and return the typed value.

        Raises :class:`~llm_structured_extract.errors.OutputValidationError`.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


def _type_name(tp: Any) -> str:
    origin = typing.get_origin(tp)
    if origin is not None:
        args = ", ".join(_type_name(arg) for arg in typing.get_args(tp))
        return f"{_type_name(origin)}[{args}]"
    name = getattr(tp, "__name__", None)
    return name if isinstance(name, str) else repr(tp)


class PydanticSchema(OutputSchema[T]):
    """Adapter for any type supported by :class:`pydantic.TypeAdapter`.

    Args:
        tp: The target type, e.g. ``Invoice`` or ``list[Invoice]``.
        name: Override the display name (defaults to the type's name).
        strict: Use Pydantic strict mode (no ``"42"`` -> ``42`` coercion).
            Defaults to lax mode, which is usually what you want for LLM output.
    """

    def __init__(self, tp: type[T], *, name: str | None = None, strict: bool = False) -> None:
        try:
            self._adapter: TypeAdapter[T] = TypeAdapter(tp)
            self._json_schema = self._adapter.json_schema()
        except (PydanticSchemaGenerationError, PydanticUserError) as error:
            raise SchemaError(
                f"Cannot build a JSON Schema for {_type_name(tp)}: {error}"
            ) from error
        self._type = tp
        self._name = name or _type_name(tp)
        self._strict = strict

    @property
    def name(self) -> str:
        return self._name

    @property
    def json_schema(self) -> dict[str, Any]:
        return self._json_schema

    @property
    def type(self) -> type[T]:
        return self._type

    def validate(self, data: Any) -> T:
        try:
            return self._adapter.validate_python(data, strict=self._strict)
        except ValidationError as error:
            issues = [
                ValidationIssue(
                    path=format_path(detail["loc"]),
                    message=detail["msg"],
                    kind=detail["type"],
                    input=detail.get("input"),
                )
                for detail in error.errors(include_url=False)
            ]
            raise OutputValidationError(issues, schema_name=self._name) from error


class JSONSchema(OutputSchema[Any]):
    """Adapter for a raw JSON Schema document.

    The validator class is chosen from ``$schema`` (Draft 2020-12 by default) and
    string ``format`` keywords are checked. Requires ``pip install
    'llm-structured-extract[jsonschema]'``.
    """

    def __init__(self, schema: Mapping[str, Any], *, name: str | None = None) -> None:
        try:
            import jsonschema
            from jsonschema.exceptions import SchemaError as InvalidSchema
        except ImportError as error:  # pragma: no cover - exercised only without the extra
            raise SchemaError(
                "Validating against a raw JSON Schema requires the 'jsonschema' package: "
                "pip install 'llm-structured-extract[jsonschema]'"
            ) from error

        self._json_schema = dict(schema)
        validator_cls = jsonschema.validators.validator_for(
            self._json_schema, default=jsonschema.Draft202012Validator
        )
        try:
            validator_cls.check_schema(self._json_schema)
        except InvalidSchema as error:
            raise SchemaError(f"Invalid JSON Schema: {error.message}") from error
        self._validator = validator_cls(
            self._json_schema, format_checker=validator_cls.FORMAT_CHECKER
        )
        title = self._json_schema.get("title")
        self._name = name or (title if isinstance(title, str) and title else "Output")

    @property
    def name(self) -> str:
        return self._name

    @property
    def json_schema(self) -> dict[str, Any]:
        return self._json_schema

    def validate(self, data: Any) -> Any:
        errors = sorted(self._validator.iter_errors(data), key=lambda error: error.json_path)
        if errors:
            issues = [
                ValidationIssue(
                    path=format_path(tuple(error.absolute_path)),
                    message=error.message,
                    kind=str(error.validator),
                    input=error.instance,
                )
                for error in errors
            ]
            raise OutputValidationError(issues, schema_name=self._name)
        return data


@overload
def as_output_schema(spec: OutputSchema[T], *, name: str | None = None) -> OutputSchema[T]: ...
@overload
def as_output_schema(spec: type[T], *, name: str | None = None) -> OutputSchema[T]: ...
@overload
def as_output_schema(spec: Mapping[str, Any], *, name: str | None = None) -> OutputSchema[Any]: ...
def as_output_schema(spec: Any, *, name: str | None = None) -> OutputSchema[Any]:
    """Coerce a schema specification into an :class:`OutputSchema`.

    * an :class:`OutputSchema` is returned unchanged (``name`` is ignored),
    * a ``Mapping`` is treated as a raw JSON Schema (:class:`JSONSchema`),
    * anything else is handed to :class:`PydanticSchema`.
    """
    if isinstance(spec, OutputSchema):
        return spec
    if isinstance(spec, Mapping):
        return JSONSchema(spec, name=name)
    return PydanticSchema(spec, name=name)
