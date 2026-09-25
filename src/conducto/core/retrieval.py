"""Backend-neutral retrieval contracts for governed RAG capabilities."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
)


class RetrievalValidationError(ValueError):
    """Raised when a retrieval query or result violates its public contract."""


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise RetrievalValidationError("Retrieval metadata keys must be strings")
        return MappingProxyType({key: _freeze_metadata(value[key]) for key in sorted(value)})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RetrievalValidationError("Retrieval metadata numbers must be finite")
        return value
    raise RetrievalValidationError(
        f"Retrieval metadata contains unsupported value {type(value).__name__}"
    )


def _thaw_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata(item) for item in value]
    return value


class RetrievedDocument(BaseModel):
    """One serializable document returned by a retriever.

    Attributes:
        text: Document content explicitly returned to the caller.
        source: Stable backend-neutral source identifier.
        citation: Optional human- or machine-readable citation.
        score: Optional finite backend-defined relevance score.
        metadata: JSON-safe document metadata such as permission labels,
            document identifiers, or collection names.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    source: str
    citation: str | None = None
    score: float | None = None
    metadata: Mapping[str, object] = Field(default_factory=dict)

    @field_validator("text", "source")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("citation")
    @classmethod
    def _optional_citation(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("citation must not be empty")
        return value

    @field_validator("score", mode="before")
    @classmethod
    def _reject_boolean_score(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("score must be finite")
        return value

    @field_validator("score")
    @classmethod
    def _finite_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("score must be finite")
        return value

    @field_validator("metadata")
    @classmethod
    def _serializable_metadata(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        return cast(Mapping[str, object], _freeze_metadata(value))

    @field_serializer("metadata")
    def _serialize_metadata(self, value: Mapping[str, object]) -> dict[str, object]:
        return cast(dict[str, object], _thaw_metadata(value))


class RetrievalQuery(BaseModel):
    """Backend-neutral query with optional filtering and pagination.

    Attributes:
        query: Non-empty natural-language or keyword query.
        filters: JSON-safe backend-defined filters.
        limit: Positive maximum number of documents to return.
        cursor: Optional opaque pagination cursor.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    filters: Mapping[str, object] = Field(default_factory=dict)
    limit: int = Field(default=10, ge=1)
    cursor: str | None = None

    @field_validator("query")
    @classmethod
    def _non_empty_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value

    @field_validator("cursor")
    @classmethod
    def _non_empty_cursor(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("cursor must not be empty")
        return value

    @field_validator("filters")
    @classmethod
    def _serializable_filters(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        return cast(Mapping[str, object], _freeze_metadata(value))

    @field_serializer("filters")
    def _serialize_filters(self, value: Mapping[str, object]) -> dict[str, object]:
        return cast(dict[str, object], _thaw_metadata(value))


class RetrievalResult(BaseModel):
    """Deterministic retrieval page and optional continuation cursor.

    Attributes:
        documents: Retrieved documents in backend-defined rank order.
        next_cursor: Opaque cursor for the next page, when available.
        metadata: JSON-safe page metadata that contains no implicit payload.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    documents: tuple[RetrievedDocument, ...] = ()
    next_cursor: str | None = None
    metadata: Mapping[str, object] = Field(default_factory=dict)

    @field_validator("next_cursor")
    @classmethod
    def _non_empty_cursor(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("next_cursor must not be empty")
        return value

    @field_validator("metadata")
    @classmethod
    def _serializable_metadata(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        return cast(Mapping[str, object], _freeze_metadata(value))

    @field_serializer("metadata")
    def _serialize_metadata(self, value: Mapping[str, object]) -> dict[str, object]:
        return cast(dict[str, object], _thaw_metadata(value))


class RetrieverProtocol(Protocol):
    """Structural contract for backend adapters implementing retrieval."""

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult | Awaitable[RetrievalResult]:
        """Retrieve one validated result page.

        Args:
            query: Validated backend-neutral retrieval query.

        Returns:
            A result page, directly or through an awaitable.
        """


@dataclass(frozen=True, slots=True)
class RetrievalProvenance:
    """Payload-free summary of a governed retrieval invocation.

    Attributes:
        retriever_id: Published capability identifier of the retriever.
        document_count: Number of returned documents.
        sources: Stable source identifiers in result order.
        citations: Non-empty citations in result order.
    """

    retriever_id: str
    document_count: int
    sources: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe retrieval provenance without document text."""
        return {
            "retriever_id": self.retriever_id,
            "document_count": self.document_count,
            "sources": list(self.sources),
            "citations": list(self.citations),
        }


def validate_retrieval_output(
    value: object,
    *,
    retriever_id: str,
    citations_required: bool,
) -> tuple[RetrievalResult | list[RetrievedDocument], RetrievalProvenance]:
    """Validate and normalize a retriever result.

    Args:
        value: A ``RetrievalResult`` or sequence of document-like values.
        retriever_id: Published retriever capability identifier.
        citations_required: Whether every returned document needs a citation.

    Returns:
        The normalized public result and payload-free provenance.

    Raises:
        RetrievalValidationError: If the result shape or citation contract is invalid.
    """
    try:
        if isinstance(value, RetrievalResult):
            normalized: RetrievalResult | list[RetrievedDocument] = value
            documents = value.documents
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            normalized_documents = [
                item
                if isinstance(item, RetrievedDocument)
                else RetrievedDocument.model_validate(item)
                for item in value
            ]
            normalized = normalized_documents
            documents = tuple(normalized_documents)
        else:
            raise RetrievalValidationError(
                "Retriever results must be a RetrievalResult or document sequence"
            )
    except RetrievalValidationError:
        raise
    except ValidationError as error:
        raise RetrievalValidationError(f"Invalid retrieved document: {error}") from error

    if citations_required and any(document.citation is None for document in documents):
        raise RetrievalValidationError(
            f"Retriever '{retriever_id}' requires a citation for every document"
        )
    provenance = RetrievalProvenance(
        retriever_id=retriever_id,
        document_count=len(documents),
        sources=tuple(document.source for document in documents),
        citations=tuple(
            document.citation for document in documents if document.citation is not None
        ),
    )
    return normalized, provenance
