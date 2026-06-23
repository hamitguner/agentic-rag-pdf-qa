"""LangGraph pipeline: ReAct research agent with grounding validation."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    MAX_RETRIES,
    prepare_node,
    research_node,
    respond_node,
    route_after_triage,
    route_after_validation,
    smalltalk_node,
    triage_node,
    validate_node,
)
from src.agent.state import AgentState, InputState, OutputState
from src.log import get_logger

logger = get_logger(__name__)

# Built once at module scope so the same in-memory store survives across
# graph.invoke() calls within one process — that is what makes a session's
# conversation memory persist between questions.
_checkpointer = InMemorySaver()


def build_graph():
    """Compile and return the agentic RAG pipeline.

    Flow:
        START → prepare → triage
        triage → smalltalk → END           (chit-chat: skip retrieval)
        triage → research → validate
        validate → respond → END           (grounded, or abstained after retries)
        validate → research → ...          (retry, MAX_RETRIES budget)

    prepare runs once per turn (the retry loop re-enters at research), handling
    new-question setup and per-turn state reset. triage classifies the turn so
    greetings and chit-chat skip the retrieval pipeline entirely. The research
    node uses create_agent which runs the full ReAct tool-call loop internally,
    so no separate tools node is needed. respond is the single terminal: it emits
    the grounded answer, or abstains (is_grounded=False) when the retry budget is
    spent. The graph is compiled with an InMemorySaver checkpointer so
    conversation memory persists per session.
    """
    graph = StateGraph(AgentState, input_schema=InputState, output_schema=OutputState)

    graph.add_node("prepare", prepare_node)
    graph.add_node("triage", triage_node)
    graph.add_node("smalltalk", smalltalk_node)
    graph.add_node("research", research_node)
    graph.add_node("validate", validate_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "triage")
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {"research": "research", "smalltalk": "smalltalk"},
    )
    graph.add_edge("smalltalk", END)
    graph.add_edge("research", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {"research": "research", "respond": "respond"},
    )
    graph.add_edge("respond", END)

    compiled = graph.compile(checkpointer=_checkpointer, name="agentic_rag")
    logger.info("Graph compiled | nodes=%s | max_retries=%d", list(graph.nodes), MAX_RETRIES)
    return compiled


def invoke_safely(graph, payload: dict, config: dict) -> dict:
    """Run the graph and degrade gracefully on any terminal failure.

    The catch is intentionally broad and provider-agnostic: the model is chosen
    at runtime via init_chat_model (Anthropic today, OpenAI/Gemini tomorrow), so
    binding error handling to one SDK's exception types would silently break on a
    provider swap. The chat models already retry transient errors internally
    (init_chat_model's max_retries); this boundary turns a *terminal* failure at
    the run boundary — a model/API error (auth, quota, outage, bad request) or an
    unexpected internal error (e.g. a missing index) — into a clean OutputState
    instead of a traceback. The full error is always logged (with traceback) so a
    genuine bug is still visible in the logs, never silently swallowed.

    Args:
        graph: A compiled pipeline (from build_graph()).
        payload: The InputState dict to invoke with.
        config: The invoke config (e.g. thread_id for memory).

    Returns:
        The graph's OutputState on success, or an ungrounded error OutputState.
    """
    try:
        return graph.invoke(payload, config=config)
    except Exception as exc:  # noqa: BLE001 — outermost run boundary, see docstring
        logger.exception("Pipeline run failed (%s)", type(exc).__name__)
        return {
            "final_answer": (
                f"The assistant could not complete this request ({type(exc).__name__}). "
                "This is usually a transient model/API issue or a missing index — "
                "check the logs and try again."
            ),
            "is_grounded": False,
            "confidence": 0.0,
            "citations": [],
        }
