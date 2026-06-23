"""Node functions and routing logic for the RAG graph."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain.agents import create_agent

from src import registry
from src.agent.prompts import EVIDENCE_AGENT_PROMPT
from src.agent.state import AgentState
from src.agent.tools import RAG_TOOLS, init_retriever
from src.agent.triage import classify_intent
from src.agent.validation import ValidationResult, grounding_check
from src.config import settings
from src.log import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 1

# Matches chunk IDs like [my_col_p1_c0] as cited inline by the agent.
# Format: {doc_id}_p{page}_c{chunk_index} — all underscores, inside square brackets.
# Also matches vision citations like [vision_page36] produced by analyze_image.
# Tolerates surrounding whitespace and an optional "chunk_id:" label the model
# sometimes adds (e.g. [chunk_id: doc_p1_c0]) so those citations aren't dropped.
_CHUNK_ID_RE = re.compile(r"\[\s*(?:chunk_id:\s*)?(\w+_p\d+_c\d+|vision_page\d+)\s*\]")

# One agent per collection so each can carry its collection's description in the
# system prompt. Built lazily and cached; the model is injected at invoke time
# via configurable, so it stays swappable without rebuilding.
_research_agents: dict[str, object] = {}


def _get_research_agent(collection: str):
    """Return the research agent for ``collection``, building it once.

    The collection's registry description (when present) is appended to the base
    evidence prompt as a COLLECTION CONTEXT block, so the agent knows what corpus
    it is searching before its first tool call.
    """
    if collection not in _research_agents:
        description = registry.get_description(collection)
        system_prompt = EVIDENCE_AGENT_PROMPT
        if description:
            system_prompt += (
                "\n\n<collection_context>\n"
                f"{description}\n"
                "</collection_context>"
            )
        _research_agents[collection] = create_agent(
            init_chat_model(temperature=0),
            tools=RAG_TOOLS,
            system_prompt=system_prompt,
        )
        logger.info(
            "[research] built agent for collection=%r (description=%s)",
            collection, bool(description),
        )
    return _research_agents[collection]


_PAGE_IN_IMAGE_RE = re.compile(r"page(\d+)\.png$")


def _page_from_image(path: str) -> int:
    """Parse the page number out of a rendered PNG path like '.../page36.png'."""
    m = _PAGE_IN_IMAGE_RE.search(path or "")
    return int(m.group(1)) if m else 0


def _parse_chunks_from_messages(state: AgentState) -> list[dict]:
    """Extract chunk dicts from ToolMessages — including vision results.

    analyze_image returns {"vision_result": ..., "source_image": ...} which has
    no evidence_chunks key, so it was silently dropped before. We fold it in as a
    pseudo-chunk so the grounding judge can verify claims read off a chart or table.
    Without this, every vision-sourced fact is flagged unverified and the agent
    abstains — killing the multimodal demo entirely.

    Vision pseudo-chunk IDs are stable and match the citation regex:
        vision_page36  →  [vision_page36] in the agent's answer.
    """
    chunks: list[dict] = []
    for msg in state["messages"]:
        if getattr(msg, "type", None) != "tool":
            continue
        try:
            data = json.loads(msg.content if isinstance(msg.content, str) else "")
        except (json.JSONDecodeError, TypeError):
            continue

        # Standard retrieval tools — evidence_chunks or retrieved_chunks key.
        for key in ("evidence_chunks", "retrieved_chunks"):
            if key in data:
                chunks.extend(data[key])
                break
        else:
            # Vision tool result — synthesize a verifiable pseudo-chunk.
            if "vision_result" in data:
                src = data.get("source_image", "")
                page = _page_from_image(src)
                chunks.append({
                    "chunk_id": f"vision_page{page}",
                    "page_number": page,
                    "section": "vision",
                    "requires_vision": True,
                    "image_path": src,
                    "text": data["vision_result"],
                })

    return chunks


def _extract_citations(text: str) -> list[str]:
    return list(dict.fromkeys(_CHUNK_ID_RE.findall(text)))


def _conversation_cleanup(state: AgentState, final_answer: str) -> list:
    """Build message updates that collapse the current turn to a clean Q→A pair
    and cap the persisted history to ``settings.memory_max_pairs`` recent pairs.

    Runs at turn end (respond / smalltalk), AFTER validation has read the tool
    messages. It removes this turn's scratchpad (tool calls, chunk dumps, base64
    images) and any history beyond the configured window, then appends one clean
    answer. Net history: [...recent Q/A pairs..., current Q, current A].

    Returns:
        A list of RemoveMessage(...) plus a single AIMessage(final_answer),
        suitable as the ``messages`` value in a node's return dict.
    """
    messages = state["messages"]
    question = state["question"]

    # The current turn's question is the one prepare_node appended — match by
    # content (retry critiques are HumanMessages with different text).
    q_index: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if getattr(m, "type", None) == "human" and m.content == question:
            q_index = i
            break

    if q_index is None:  # defensive: question not found, just record the answer
        return [AIMessage(content=final_answer)]

    updates: list = []

    # Drop the current turn's scratchpad (everything after the question).
    for m in messages[q_index + 1:]:
        updates.append(RemoveMessage(id=m.id))

    # Cap prior history to the last (max_pairs - 1) clean pairs; current turn is
    # the final pair. Each pair is one Human + one AI message.
    keep_prior = 2 * max(settings.memory_max_pairs - 1, 0)
    prior = messages[:q_index]
    if len(prior) > keep_prior:
        for m in prior[: len(prior) - keep_prior]:
            updates.append(RemoveMessage(id=m.id))

    updates.append(AIMessage(content=final_answer))
    return updates


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def prepare_node(state: AgentState) -> dict:
    """One-shot turn setup at START (the retry loop re-enters at research, not here).

    Appends the new question to the conversation and resets per-turn state so a
    persisted session does not inherit the previous turn's retry budget or draft.
    """
    logger.info("[prepare] session=%s | new turn", state.get("session_id", "?"))
    return {
        "messages": [HumanMessage(content=state["question"])],
        "retries": 0,
        "intent": None,
        "validation": None,
        "draft_answer": None,
    }


def _last_ai_text(messages: list) -> str:
    """Return the most recent assistant answer text, or '' if there is none.

    Used to give the triage classifier context so terse follow-ups resolve as
    document questions. Skips the question prepare_node just appended (the last
    message), looking only at prior turns.
    """
    for m in reversed(messages[:-1]):
        if getattr(m, "type", None) == "ai":
            content = m.content
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                )
            return content or ""
    return ""


def triage_node(state: AgentState) -> dict:
    """Classify the turn as a document question or chit-chat (one cheap LLM call).

    For chit-chat the classifier also returns the reply, stored in draft_answer
    so smalltalk_node can emit it without a second call.
    """
    result = classify_intent(
        question=state["question"],
        previous_answer=_last_ai_text(state["messages"]),
        collection_context=registry.get_description(state["collection"]),
    )
    logger.info("[triage] intent=%s", result.intent)
    return {"intent": result.intent, "draft_answer": result.reply}


def smalltalk_node(state: AgentState) -> dict:
    """Emit a chit-chat reply directly, skipping retrieval and validation.

    A greeting makes no document claims, so it is reported as grounded with full
    confidence (same semantic as the validator's absence-declaration rule). The
    turn is pruned like respond so memory stays compact.
    """
    reply = state.get("draft_answer") or ""
    logger.info("[smalltalk] reply_len=%d", len(reply))
    return {
        "final_answer": reply,
        "is_grounded": True,
        "confidence": 1.0,
        "citations": [],
        "messages": _conversation_cleanup(state, reply),
    }


def research_node(state: AgentState) -> dict:
    """Run the full ReAct loop: reason, call retrieval tools, produce a cited draft.

    Uses create_agent so the tool-call/tool-result loop is handled internally.
    The model is injected via configurable so it can be swapped without touching
    this code. The question is already in state["messages"] (added by prepare_node).
    """
    init_retriever(state["collection"])
    agent = _get_research_agent(state["collection"])
    result = agent.invoke(
        {"messages": state["messages"]},
        config={"configurable": {"model": settings.model_name}},
    )
    msgs = result["messages"]
    logger.info(
        "[research] steps=%d | tool_msgs=%d",
        len(msgs),
        sum(1 for m in msgs if getattr(m, "type", None) == "tool"),
    )
    return {"messages": msgs}


def validate_node(state: AgentState) -> dict:
    """Check that every claim in the draft is supported by retrieved chunks.

    On failure: injects a targeted critique so the research node can do a
    focused follow-up retrieval rather than starting over.
    """
    draft = state["messages"][-1].content
    if isinstance(draft, list):
        draft = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in draft)

    chunks = _parse_chunks_from_messages(state)
    citations = _extract_citations(draft)

    result: ValidationResult = grounding_check(
        question=state["question"],
        draft_answer=draft,
        retrieved_chunks=chunks,
    )

    update: dict = {
        "draft_answer": draft,
        "retrieved_chunks": chunks,
        "citations": citations,
        "validation": result,
    }

    if not result.is_grounded:
        update["retries"] = (state.get("retries") or 0) + 1
        update["messages"] = [HumanMessage(
            content=(
                f"These claims could not be verified in the document: "
                f"{result.unverified_claims}.\n"
                "Retrieve specific evidence for each unverified claim, "
                "or remove it from your answer."
            )
        )]
        logger.info(
            "[validate] not grounded | confidence=%.2f | retries→%d",
            result.confidence,
            update["retries"],
        )
    else:
        logger.info("[validate] grounded | confidence=%.2f", result.confidence)

    return update


def respond_node(state: AgentState) -> dict:
    """Single terminal: emit the answer, all OutputState fields, and prune the turn.

    Two outcomes share this exit:
      - grounded  → return the validator's cleaned answer (is_grounded=True).
      - abstained → grounding still failed after the retry budget, so return the
        agent's own last draft flagged is_grounded=False rather than fabricating
        confidence. The agent may already have said "I couldn't find this" naturally.

    Either way the conversation is collapsed to a clean Q→A pair and capped to the
    recent history window so the session's memory stays compact across turns.
    """
    v = state["validation"]
    if v and v.is_grounded:
        answer, grounded, confidence = v.final_answer, True, v.confidence
    else:
        answer, grounded, confidence = (state.get("draft_answer") or ""), False, 0.0
        logger.warning(
            "[respond] abstaining — ungrounded after %d retries", state.get("retries") or 0
        )

    logger.info("[respond] grounded=%s | answer_len=%d", grounded, len(answer))
    return {
        "final_answer": answer,
        "is_grounded": grounded,
        "confidence": confidence,
        "citations": state.get("citations") or [],
        "messages": _conversation_cleanup(state, answer),
    }


# ---------------------------------------------------------------------------
# Routing (pure functions — read state only, no mutations)
# ---------------------------------------------------------------------------


def route_after_triage(state: AgentState) -> Literal["research", "smalltalk"]:
    """Document question → run retrieval. Chit-chat → reply directly."""
    return "research" if state.get("intent") == "document" else "smalltalk"


def route_after_validation(state: AgentState) -> Literal["research", "respond"]:
    """Grounded, or retry budget spent → respond (terminal). Otherwise retry research."""
    grounded = bool(state["validation"] and state["validation"].is_grounded)
    if grounded or (state.get("retries") or 0) > MAX_RETRIES:
        return "respond"
    return "research"