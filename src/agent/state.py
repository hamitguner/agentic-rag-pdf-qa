"""LangGraph state definitions for the agentic RAG pipeline."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.agent.validation import ValidationResult


class InputState(TypedDict):
    """What the caller must provide."""

    question: str
    collection: str  # ChromaDB collection slug (e.g. "llama3_herd")
    session_id: str  # conversation thread; mirrored to config thread_id for memory


class OutputState(TypedDict):
    """What the graph returns to the caller."""

    final_answer: str
    is_grounded: bool
    confidence: float
    citations: list[str]


class AgentState(InputState):
    """Full internal state — superset of InputState and OutputState."""

    messages: Annotated[list[AnyMessage], add_messages]

    # All chunks returned by any tool call, accumulated across retries
    retrieved_chunks: list[dict]

    # Chunk IDs the agent cited in its draft answer
    citations: list[str]

    intent: str | None  # "document" | "chitchat", set by triage_node
    draft_answer: str | None
    retries: int
    validation: ValidationResult | None
    final_answer: str | None
