"""Run a deterministic in-memory retrieval-augmented flow without network access."""

from __future__ import annotations

import asyncio

from conducto import BaseAgent, RetrievedDocument, Runtime, retriever
from conducto.core.invocation_results import InvocationSuccess


class PolicyKnowledge(BaseAgent):
    """Retrieve a small local policy collection."""

    documents = (
        RetrievedDocument(
            text="Employees receive 25 days of annual leave.",
            source="employee-handbook",
            citation="employee-handbook#annual-leave",
            metadata={"document_id": "leave", "collection": "policies"},
        ),
        RetrievedDocument(
            text="Expense reports are due within 30 days.",
            source="finance-handbook",
            citation="finance-handbook#expenses",
            metadata={"document_id": "expenses", "collection": "policies"},
        ),
    )

    @retriever(name="policy_docs", citations=True)
    def policy_docs(self, query: str) -> list[RetrievedDocument]:
        """Return documents containing every case-insensitive query term."""
        terms = query.lower().split()
        return [
            document
            for document in self.documents
            if all(term in document.text.lower() for term in terms)
        ]


async def main() -> None:
    """Retrieve local context and build a deterministic grounded answer."""
    result = await Runtime().invoke(
        PolicyKnowledge(),
        "policy_docs",
        {"query": "annual leave"},
    )
    if not isinstance(result, InvocationSuccess):
        raise RuntimeError(f"Retrieval failed: {type(result).__name__}")
    documents = result.value
    answer = " ".join(document["text"] for document in documents)
    citations = ", ".join(document["citation"] for document in documents)
    print(f"{answer} Sources: {citations}")


if __name__ == "__main__":
    asyncio.run(main())
