"""HTTP routes for the agentic RAG API.

A thin wrapper over the same pipeline the CLI uses: each request runs one turn
through the compiled graph via invoke_safely, so model/API failures return a
clean JSON body instead of a 500 traceback. The collection must already be
indexed (see `main.py --ingest-only` or `scripts/evaluate.py --pdf`).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from src import registry
from src.agent.graph import invoke_safely
from src.log import get_logger

logger = get_logger(__name__)

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(..., description="The user's question.")
    collection: str = Field(..., description="Indexed collection (doc_id slug) to query.")
    session_id: str = Field("default", description="Conversation session for memory.")


class AskResponse(BaseModel):
    final_answer: str
    is_grounded: bool
    confidence: float
    citations: list[str]


class CollectionItem(BaseModel):
    collection: str
    description: str


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@router.get("/collections", response_model=list[CollectionItem])
def list_collections() -> list[CollectionItem]:
    """Return all registered collections with their descriptions."""
    return [CollectionItem(**c) for c in registry.list_collections()]


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request) -> AskResponse:
    """Answer one question against an indexed collection.

    Reuses the process-wide compiled graph (built at startup) so conversation
    memory persists across requests that share a session_id.
    """
    logger.info("POST /ask | collection=%s | session=%s", req.collection, req.session_id)
    result = invoke_safely(
        request.app.state.graph,
        {"question": req.question, "collection": req.collection, "session_id": req.session_id},
        config={"configurable": {"thread_id": req.session_id}},
    )
    return AskResponse(
        final_answer=result.get("final_answer", ""),
        is_grounded=result.get("is_grounded", False),
        confidence=result.get("confidence", 0.0),
        citations=result.get("citations", []),
    )
