"""Tests for backend-neutral retriever and RAG contracts."""

import asyncio
import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from conducto import (
    AgentRegistry,
    BaseAgent,
    RetrievalQuery,
    RetrievalResult,
    RetrievedDocument,
    Runtime,
    retriever,
)
from conducto.core.invocation_results import (
    InvocationFailure,
    InvocationSuccess,
    InvocationValidationFailure,
)
from conducto.security import (
    AuditEmitter,
    AuditEventName,
    InMemoryAuditSink,
    SecurityPipeline,
)


def test_retrieval_contracts_are_immutable_deterministic_and_json_safe() -> None:
    """Contracts preserve rank order while canonicalizing nested metadata."""
    document = RetrievedDocument(
        text="Policy text",
        source="handbook",
        citation="handbook#leave",
        score=0.9,
        metadata={"labels": ["internal"], "document_id": "leave"},
    )
    result = RetrievalResult(
        documents=(document,),
        next_cursor="page-2",
        metadata={"collection": "policies"},
    )

    assert json.loads(result.model_dump_json()) == {
        "documents": [
            {
                "citation": "handbook#leave",
                "metadata": {
                    "document_id": "leave",
                    "labels": ["internal"],
                },
                "score": 0.9,
                "source": "handbook",
                "text": "Policy text",
            }
        ],
        "metadata": {"collection": "policies"},
        "next_cursor": "page-2",
    }
    with pytest.raises(ValidationError, match="query must not be empty"):
        RetrievalQuery(query=" ")
    with pytest.raises(ValidationError, match="score must be finite"):
        RetrievedDocument(text="text", source="source", score=float("nan"))
    with pytest.raises(ValidationError, match="score must be finite"):
        RetrievedDocument(text="text", source="source", score="inf")
    with pytest.raises(ValidationError, match="score must be finite"):
        RetrievedDocument(text="text", source="source", score=Decimal("NaN"))
    with pytest.raises(ValidationError, match="unsupported value object"):
        RetrievedDocument(text="text", source="source", metadata={"bad": object()})
    with pytest.raises(ValidationError, match="Instance is frozen"):
        document.text = "changed"


def test_retriever_is_a_governed_capability_with_payload_free_provenance() -> None:
    """Runtime and registry use the normal capability path for retrieval."""

    class PolicyAgent(BaseAgent):
        """Retrieve policy documents."""

        @retriever(
            name="policy_docs",
            description="Retrieve policy documents.",
            citations=True,
        )
        def policy_docs(self, query: str) -> list[RetrievedDocument]:
            return [
                RetrievedDocument(
                    text=f"Raw payload for {query}",
                    source="employee-handbook",
                    citation="employee-handbook#leave",
                    score=1.0,
                    metadata={"document_id": "leave-policy"},
                )
            ]

    agent = PolicyAgent()
    registry = AgentRegistry()
    registry.register(agent)
    sink = InMemoryAuditSink()
    runtime = Runtime(
        agent_registry=registry,
        security_pipeline=SecurityPipeline(audit_emitter=AuditEmitter(sink)),
    )

    result = asyncio.run(runtime.invoke(agent, "policy_docs", {"query": "leave"}))

    assert isinstance(result, InvocationSuccess)
    assert result.value[0]["citation"] == "employee-handbook#leave"
    assert registry.snapshot().capabilities[0].capability_id == "policy_docs"
    assert result.metadata is not None
    assert result.metadata.retrievals[0].to_dict() == {
        "retriever_id": "policy_docs",
        "document_count": 1,
        "sources": ["employee-handbook"],
        "citations": ["employee-handbook#leave"],
    }
    serialized_metadata = json.dumps(result.metadata.to_dict())
    assert "Raw payload" not in serialized_metadata
    completed = next(
        event for event in sink.events if event.event_name is AuditEventName.EXECUTION_COMPLETED
    )
    assert dict(completed.extensions) == {
        "retrieval_document_count": 1,
        "retrieval_source_count": 1,
    }


def test_retriever_rejects_empty_queries_and_missing_required_citations() -> None:
    """Retriever-specific failures remain typed and never become empty success."""

    class InvalidAgent(BaseAgent):
        """Return deliberately invalid retrieval results."""

        @retriever(name="uncited", citations=True)
        def uncited(self, query: str) -> list[RetrievedDocument]:
            return [RetrievedDocument(text=query, source="memory")]

    agent = InvalidAgent()
    runtime = Runtime()

    empty = asyncio.run(runtime.invoke(agent, "uncited", {"query": " "}))
    uncited = asyncio.run(runtime.invoke(agent, "uncited", {"query": "policy"}))

    assert isinstance(empty, InvocationValidationFailure)
    assert empty.errors[0]["loc"] == ("query",)
    assert isinstance(uncited, InvocationFailure)
    assert uncited.classification == "unsatisfied_output_invariant"
    assert "requires a citation" in uncited.message


def test_retriever_without_citation_requirement_accepts_uncited_documents() -> None:
    """Backends may explicitly opt out of citation enforcement."""

    class SearchAgent(BaseAgent):
        @retriever(name="search", citations=False)
        def search(self, query: RetrievalQuery) -> RetrievalResult:
            return RetrievalResult(
                documents=(RetrievedDocument(text=query.query, source="memory"),)
            )

    result = asyncio.run(
        Runtime().invoke(
            SearchAgent(),
            "search",
            {"query": {"query": "local", "limit": 1}},
        )
    )

    assert isinstance(result, InvocationSuccess)
    assert result.value["documents"][0]["citation"] is None
