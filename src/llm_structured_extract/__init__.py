"""Typed, schema-enforced structured extraction over LLMs.

Define the output as a Pydantic model (or a JSON Schema), hand the extractor a
document, get back a validated, typed object. Invalid output is repaired by
feeding the validation errors back to the model, up to a configurable limit.

>>> from pydantic import BaseModel
>>> from llm_structured_extract import Extractor, MockProvider
>>> class Person(BaseModel):
...     name: str
...     age: int
>>> provider = MockProvider(['{"name": "Ada", "age": "thirty-six"}', '{"name": "Ada", "age": 36}'])
>>> result = Extractor(provider).extract("Ada Lovelace, 36, mathematician.", Person)
>>> result.value
Person(name='Ada', age=36)
>>> len(result.attempts)
2
"""

from .errors import (
    ExtractionFailedError,
    MissingOutputError,
    OutputError,
    OutputParseError,
    OutputTruncatedError,
    OutputValidationError,
    ProviderError,
    ProviderNotConfiguredError,
    RefusalError,
    SchemaError,
    ScriptExhaustedError,
    StructuredExtractError,
    ValidationIssue,
)
from .extractor import Extractor
from .parsing import PartialJSON, parse_json, parse_partial_json
from .prompts import PromptBuilder
from .providers import (
    AnthropicProvider,
    BaseProvider,
    MockProvider,
    MockResponse,
    Provider,
)
from .result import (
    Attempt,
    ExtractionResult,
    PartialEvent,
    RepairEvent,
    ResultEvent,
    StreamEvent,
)
from .schema import JSONSchema, OutputSchema, PydanticSchema, as_output_schema
from .types import Completion, CompletionRequest, Message, StreamChunk, Usage

__version__ = "0.1.0"

__all__ = [
    "AnthropicProvider",
    "Attempt",
    "BaseProvider",
    "Completion",
    "CompletionRequest",
    "ExtractionFailedError",
    "ExtractionResult",
    "Extractor",
    "JSONSchema",
    "Message",
    "MissingOutputError",
    "MockProvider",
    "MockResponse",
    "OutputError",
    "OutputParseError",
    "OutputSchema",
    "OutputTruncatedError",
    "OutputValidationError",
    "PartialEvent",
    "PartialJSON",
    "PromptBuilder",
    "Provider",
    "ProviderError",
    "ProviderNotConfiguredError",
    "PydanticSchema",
    "RefusalError",
    "RepairEvent",
    "ResultEvent",
    "SchemaError",
    "ScriptExhaustedError",
    "StreamChunk",
    "StreamEvent",
    "StructuredExtractError",
    "Usage",
    "ValidationIssue",
    "__version__",
    "as_output_schema",
    "parse_json",
    "parse_partial_json",
]
