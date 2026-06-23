"""Tests for the agent layer — routing logic, node outputs, and citation extraction.

All tests are offline (no LLM or ChromaDB calls). External dependencies are
mocked where needed so the suite runs fast in CI without API keys.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from src.agent.nodes import (
    MAX_RETRIES,
    _conversation_cleanup,
    _extract_citations,
    _last_ai_text,
    _parse_chunks_from_messages,
    prepare_node,
    respond_node,
    route_after_triage,
    route_after_validation,
    smalltalk_node,
    triage_node,
    validate_node,
)
from src.agent.triage import TriageResult
from src.agent.validation import ValidationResult
from src.config import settings


# ── helpers ───────────────────────────────────────────────────────────────────


def _state(**kwargs) -> dict:
    """Build a minimal AgentState dict with sensible defaults."""
    defaults: dict = {
        "question": "What is the interest rate?",
        "collection": "test_col",
        "session_id": "test_session",
        "messages": [],
        "retrieved_chunks": [],
        "citations": [],
        "intent": None,
        "draft_answer": None,
        "retries": 0,
        "validation": None,
        "final_answer": None,
    }
    return {**defaults, **kwargs}


def _validation(grounded: bool = True, confidence: float = 0.95) -> ValidationResult:
    return ValidationResult(
        is_grounded=grounded,
        confidence=confidence,
        unverified_claims=[] if grounded else ["claim A not found in chunks"],
        final_answer="Rate is 3.5%." if grounded else "Rate is unknown.",
    )


# ── _extract_citations ────────────────────────────────────────────────────────


def test_extract_citations_finds_inline_chunk_ids() -> None:
    text = "The rate is 3.5% [my_col_p1_c0]. See also [my_col_p2_c1] for context."
    result = _extract_citations(text)
    assert "my_col_p1_c0" in result
    assert "my_col_p2_c1" in result
    assert len(result) == 2


def test_extract_citations_deduplicates_repeated_ids() -> None:
    text = "See [doc_p1_c0] again: [doc_p1_c0]."
    result = _extract_citations(text)
    assert result.count("doc_p1_c0") == 1


def test_extract_citations_preserves_insertion_order() -> None:
    text = "[col_p3_c0] first, then [col_p1_c0], then [col_p2_c0]."
    result = _extract_citations(text)
    assert result == ["col_p3_c0", "col_p1_c0", "col_p2_c0"]


def test_extract_citations_returns_empty_for_plain_text() -> None:
    assert _extract_citations("No citations here whatsoever.") == []


def test_extract_citations_ignores_partial_patterns() -> None:
    # Must match full underscore-separated format: word_pN_cN inside brackets
    assert _extract_citations("[p1_c0]") == []          # missing doc prefix
    assert _extract_citations("[doc-p1-c0]") == []      # hyphens, not underscores


# ── _parse_chunks_from_messages ───────────────────────────────────────────────


def test_parse_chunks_extracts_evidence_chunks_key() -> None:
    chunks_data = [{"chunk_id": "col_p1_c0", "text": "Rate is 3.5%."}]
    tool_msg = ToolMessage(
        content=json.dumps({"evidence_chunks": chunks_data}),
        tool_call_id="call_1",
    )
    state = _state(messages=[HumanMessage(content="q"), tool_msg])
    result = _parse_chunks_from_messages(state)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "col_p1_c0"


def test_parse_chunks_extracts_retrieved_chunks_key() -> None:
    chunks_data = [{"chunk_id": "col_p2_c0", "text": "Inflation at 3%."}]
    tool_msg = ToolMessage(
        content=json.dumps({"retrieved_chunks": chunks_data}),
        tool_call_id="call_2",
    )
    state = _state(messages=[tool_msg])
    result = _parse_chunks_from_messages(state)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "col_p2_c0"


def test_parse_chunks_ignores_non_tool_messages() -> None:
    state = _state(messages=[
        HumanMessage(content="What is the rate?"),
        AIMessage(content="The rate is 3.5%."),
    ])
    assert _parse_chunks_from_messages(state) == []


def test_parse_chunks_skips_malformed_tool_message() -> None:
    bad_msg = ToolMessage(content="not json at all", tool_call_id="call_3")
    state = _state(messages=[bad_msg])
    assert _parse_chunks_from_messages(state) == []


# ── route_after_validation ────────────────────────────────────────────────────


def test_route_grounded_goes_to_respond() -> None:
    state = _state(validation=_validation(grounded=True), retries=0)
    assert route_after_validation(state) == "respond"


def test_route_ungrounded_within_budget_retries() -> None:
    state = _state(validation=_validation(grounded=False), retries=0)
    assert route_after_validation(state) == "research"


def test_route_budget_exhausted_goes_to_respond() -> None:
    state = _state(validation=_validation(grounded=False), retries=MAX_RETRIES + 1)
    assert route_after_validation(state) == "respond"


def test_route_no_validation_yet_defaults_to_research() -> None:
    # Happens if route is somehow called before validate_node (shouldn't occur in
    # the normal graph flow, but the function must be safe regardless).
    state = _state(validation=None, retries=0)
    assert route_after_validation(state) == "research"


def test_route_retries_none_treated_as_zero() -> None:
    # retries starts as None when the graph is first invoked (not in InputState)
    state = _state(validation=_validation(grounded=False), retries=None)
    assert route_after_validation(state) == "research"


# ── respond_node (single terminal: grounded answer or abstain) ────────────────


def test_respond_node_grounded_emits_validated_answer() -> None:
    v = _validation(grounded=True, confidence=0.9)
    state = _state(validation=v, citations=["col_p1_c0"])
    result = respond_node(state)

    assert result["final_answer"] == v.final_answer
    assert result["is_grounded"] is True
    assert result["confidence"] == pytest.approx(0.9)
    assert result["citations"] == ["col_p1_c0"]


def test_respond_node_abstains_with_ungrounded_validation() -> None:
    # Budget spent, still ungrounded → return the agent's own draft, flagged.
    v = _validation(grounded=False)
    state = _state(validation=v, draft_answer="I could not find this in the document.", retries=2)
    result = respond_node(state)

    assert result["final_answer"] == "I could not find this in the document."
    assert result["is_grounded"] is False
    assert result["confidence"] == 0.0


def test_respond_node_falls_back_to_draft_when_no_validation() -> None:
    state = _state(validation=None, draft_answer="Fallback answer.")
    result = respond_node(state)
    assert result["final_answer"] == "Fallback answer."
    assert result["is_grounded"] is False
    assert result["confidence"] == 0.0


def test_respond_node_empty_answer_when_both_none() -> None:
    state = _state(validation=None, draft_answer=None)
    result = respond_node(state)
    assert result["final_answer"] == ""


def test_respond_node_abstain_preserves_citations() -> None:
    state = _state(draft_answer="Some answer.", retries=2, citations=["col_p1_c0"])
    result = respond_node(state)
    assert result["is_grounded"] is False
    assert result["citations"] == ["col_p1_c0"]


# ── validate_node (grounding_check mocked) ────────────────────────────────────


def test_validate_node_grounded_sets_validation_no_retry() -> None:
    v = _validation(grounded=True)
    ai_msg = AIMessage(content="Rate is 3.5% [col_p1_c0].")
    state = _state(messages=[HumanMessage(content="q"), ai_msg], retries=0)

    with patch("src.agent.nodes.grounding_check", return_value=v):
        result = validate_node(state)

    assert result["validation"].is_grounded is True
    assert result["draft_answer"] == "Rate is 3.5% [col_p1_c0]."
    assert "retries" not in result  # retries only incremented on failure


def test_validate_node_ungrounded_increments_retries() -> None:
    v = _validation(grounded=False)
    ai_msg = AIMessage(content="Rate is 99%.")
    state = _state(messages=[HumanMessage(content="q"), ai_msg], retries=0)

    with patch("src.agent.nodes.grounding_check", return_value=v):
        result = validate_node(state)

    assert result["retries"] == 1


def test_validate_node_retries_none_treated_as_zero() -> None:
    # retries is None on the first invocation (not in InputState)
    v = _validation(grounded=False)
    ai_msg = AIMessage(content="Rate is 99%.")
    state = _state(messages=[HumanMessage(content="q"), ai_msg], retries=None)

    with patch("src.agent.nodes.grounding_check", return_value=v):
        result = validate_node(state)

    assert result["retries"] == 1  # 0 + 1, not None + 1


def test_validate_node_ungrounded_injects_critique_message() -> None:
    v = _validation(grounded=False)
    ai_msg = AIMessage(content="Unverified claim here.")
    state = _state(messages=[HumanMessage(content="q"), ai_msg], retries=0)

    with patch("src.agent.nodes.grounding_check", return_value=v):
        result = validate_node(state)

    injected = result["messages"]
    assert any(isinstance(m, HumanMessage) for m in injected)
    critique_text = next(m.content for m in injected if isinstance(m, HumanMessage))
    assert "could not be verified" in critique_text


# ── ValidationResult ──────────────────────────────────────────────────────────


def test_validation_result_stores_all_fields() -> None:
    v = ValidationResult(
        is_grounded=True,
        confidence=0.85,
        unverified_claims=[],
        final_answer="Rate is 3.5%.",
    )
    assert v.is_grounded is True
    assert v.confidence == pytest.approx(0.85)
    assert v.unverified_claims == []
    assert v.final_answer == "Rate is 3.5%."


def test_validation_result_ungrounded_preserves_claims() -> None:
    v = ValidationResult(
        is_grounded=False,
        confidence=0.2,
        unverified_claims=["claim X", "claim Y"],
        final_answer="partial answer",
    )
    assert v.is_grounded is False
    assert len(v.unverified_claims) == 2
    assert "claim X" in v.unverified_claims


# ── prepare_node (turn setup) ─────────────────────────────────────────────────


def test_prepare_node_appends_question_message() -> None:
    state = _state(question="What is core PCE?")
    result = prepare_node(state)
    msgs = result["messages"]
    assert len(msgs) == 1
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "What is core PCE?"


def test_prepare_node_resets_retries_to_zero() -> None:
    # Simulates a persisted session where the previous turn left retries=2
    state = _state(retries=2)
    result = prepare_node(state)
    assert result["retries"] == 0


def test_prepare_node_clears_stale_validation_and_draft() -> None:
    state = _state(validation=_validation(grounded=True), draft_answer="old draft")
    result = prepare_node(state)
    assert result["validation"] is None
    assert result["draft_answer"] is None


# ── _conversation_cleanup (turn-end pruning) ──────────────────────────────────


def _turn_messages(question: str) -> list:
    """A history with one prior clean pair + a current turn with scratchpad."""
    return [
        HumanMessage(content="Old question", id="h1"),
        AIMessage(content="Old answer", id="a1"),
        HumanMessage(content=question, id="h2"),          # current question
        AIMessage(content="", id="a2"),                   # scratchpad: tool call
        ToolMessage(content='{"evidence_chunks": []}', tool_call_id="t", id="tm1"),
        AIMessage(content="draft answer", id="a3"),       # scratchpad: draft
    ]


def test_conversation_cleanup_removes_current_scratchpad() -> None:
    state = _state(question="Q now", messages=_turn_messages("Q now"))
    updates = _conversation_cleanup(state, final_answer="Clean answer.")

    removed_ids = {u.id for u in updates if isinstance(u, RemoveMessage)}
    assert removed_ids == {"a2", "tm1", "a3"}  # the three scratchpad messages


def test_conversation_cleanup_appends_clean_answer() -> None:
    state = _state(question="Q now", messages=_turn_messages("Q now"))
    updates = _conversation_cleanup(state, final_answer="Clean answer.")

    appended = [u for u in updates if isinstance(u, AIMessage)]
    assert len(appended) == 1
    assert appended[0].content == "Clean answer."


def test_conversation_cleanup_keeps_prior_pair_within_window() -> None:
    # Default window (5 pairs) is larger than the one prior pair, so it stays.
    state = _state(question="Q now", messages=_turn_messages("Q now"))
    updates = _conversation_cleanup(state, final_answer="Clean answer.")

    removed_ids = {u.id for u in updates if isinstance(u, RemoveMessage)}
    assert "h1" not in removed_ids and "a1" not in removed_ids  # prior pair kept


def test_conversation_cleanup_caps_history_to_window(monkeypatch) -> None:
    # Window of 1 pair: only the current turn survives; the prior pair is dropped.
    monkeypatch.setattr(settings, "memory_max_pairs", 1)
    state = _state(question="Q now", messages=_turn_messages("Q now"))
    updates = _conversation_cleanup(state, final_answer="Clean answer.")

    removed_ids = {u.id for u in updates if isinstance(u, RemoveMessage)}
    assert {"h1", "a1"}.issubset(removed_ids)  # prior pair pruned by window cap


def test_conversation_cleanup_no_matching_question_falls_back() -> None:
    state = _state(question="Unmatched", messages=[AIMessage(content="x", id="a0")])
    updates = _conversation_cleanup(state, final_answer="Answer.")
    assert len(updates) == 1
    assert isinstance(updates[0], AIMessage)
    assert updates[0].content == "Answer."


def test_respond_node_prunes_turn_scratchpad() -> None:
    v = _validation(grounded=True)
    state = _state(question="Q now", validation=v, messages=_turn_messages("Q now"))
    result = respond_node(state)

    # OutputState fields still emitted
    assert result["is_grounded"] is True
    # Plus message pruning: scratchpad removed, one clean answer appended
    removed = [u for u in result["messages"] if isinstance(u, RemoveMessage)]
    appended = [u for u in result["messages"] if isinstance(u, AIMessage)]
    assert {u.id for u in removed} == {"a2", "tm1", "a3"}
    assert len(appended) == 1


# ── intent triage ─────────────────────────────────────────────────────────────


def test_route_after_triage_document_goes_to_research() -> None:
    assert route_after_triage(_state(intent="document")) == "research"


def test_route_after_triage_chitchat_goes_to_smalltalk() -> None:
    assert route_after_triage(_state(intent="chitchat")) == "smalltalk"


def test_route_after_triage_missing_intent_defaults_to_smalltalk() -> None:
    # Defensive: only an explicit "document" routes to retrieval.
    assert route_after_triage(_state(intent=None)) == "smalltalk"


def test_triage_node_sets_intent_and_reply() -> None:
    fake = TriageResult(intent="chitchat", reply="Merhaba! Belge hakkında ne sormak istersiniz?")
    state = _state(question="merhaba", messages=[HumanMessage(content="merhaba", id="h1")])

    with patch("src.agent.nodes.classify_intent", return_value=fake) as mock, patch(
        "src.agent.nodes.registry.get_description", return_value=""
    ):
        result = triage_node(state)

    assert result["intent"] == "chitchat"
    assert result["draft_answer"] == "Merhaba! Belge hakkında ne sormak istersiniz?"
    mock.assert_called_once()


def test_triage_node_passes_previous_answer_context() -> None:
    fake = TriageResult(intent="document", reply="")
    messages = [
        HumanMessage(content="Q1", id="h1"),
        AIMessage(content="Faiz %3.5.", id="a1"),
        HumanMessage(content="peki ya çekirdek PCE?", id="h2"),
    ]
    state = _state(question="peki ya çekirdek PCE?", messages=messages)

    with patch("src.agent.nodes.classify_intent", return_value=fake) as mock, patch(
        "src.agent.nodes.registry.get_description", return_value=""
    ):
        triage_node(state)

    # The prior assistant answer is forwarded as context to the classifier.
    assert mock.call_args.kwargs["previous_answer"] == "Faiz %3.5."


def test_triage_node_forwards_collection_context() -> None:
    fake = TriageResult(intent="document", reply="")
    state = _state(question="q", messages=[HumanMessage(content="q", id="h1")])

    with patch("src.agent.nodes.classify_intent", return_value=fake) as mock, patch(
        "src.agent.nodes.registry.get_description", return_value="The Llama 3 report."
    ):
        triage_node(state)

    assert mock.call_args.kwargs["collection_context"] == "The Llama 3 report."


# ── research agent: per-collection description injection ───────────────────────


def test_research_agent_injects_collection_description() -> None:
    import src.agent.nodes as nodes

    nodes._research_agents.clear()
    captured: dict = {}

    def fake_create_agent(model, tools, system_prompt):
        captured["prompt"] = system_prompt
        return object()

    with patch("src.agent.nodes.create_agent", side_effect=fake_create_agent), patch(
        "src.agent.nodes.registry.get_description", return_value="All about Llama 3."
    ):
        nodes._get_research_agent("llama3")

    assert "<collection_context>" in captured["prompt"]
    assert "All about Llama 3." in captured["prompt"]
    nodes._research_agents.clear()


def test_research_agent_omits_context_block_without_description() -> None:
    import src.agent.nodes as nodes

    nodes._research_agents.clear()
    captured: dict = {}

    def fake_create_agent(model, tools, system_prompt):
        captured["prompt"] = system_prompt
        return object()

    with patch("src.agent.nodes.create_agent", side_effect=fake_create_agent), patch(
        "src.agent.nodes.registry.get_description", return_value=""
    ):
        nodes._get_research_agent("no_desc")

    assert "<collection_context>" not in captured["prompt"]
    nodes._research_agents.clear()


def test_research_agent_cache_builds_once_per_collection() -> None:
    import src.agent.nodes as nodes

    nodes._research_agents.clear()
    calls = {"n": 0}

    def fake_create_agent(model, tools, system_prompt):
        calls["n"] += 1
        return object()

    with patch("src.agent.nodes.create_agent", side_effect=fake_create_agent), patch(
        "src.agent.nodes.registry.get_description", return_value=""
    ):
        a1 = nodes._get_research_agent("col")
        a2 = nodes._get_research_agent("col")  # cached — no rebuild

    assert a1 is a2
    assert calls["n"] == 1
    nodes._research_agents.clear()


def test_smalltalk_node_emits_grounded_reply() -> None:
    state = _state(
        question="merhaba",
        draft_answer="Merhaba! Size nasıl yardımcı olabilirim?",
        messages=[HumanMessage(content="merhaba", id="h1")],
    )
    result = smalltalk_node(state)

    assert result["final_answer"] == "Merhaba! Size nasıl yardımcı olabilirim?"
    assert result["is_grounded"] is True
    assert result["confidence"] == 1.0
    assert result["citations"] == []
    # Records the turn in memory (single clean answer appended)
    appended = [u for u in result["messages"] if isinstance(u, AIMessage)]
    assert len(appended) == 1


def test_last_ai_text_returns_prior_answer() -> None:
    messages = [
        HumanMessage(content="Q1", id="h1"),
        AIMessage(content="A1 answer", id="a1"),
        HumanMessage(content="Q2 current", id="h2"),  # last message ignored
    ]
    assert _last_ai_text(messages) == "A1 answer"


def test_last_ai_text_empty_when_no_prior_answer() -> None:
    messages = [HumanMessage(content="first question ever", id="h1")]
    assert _last_ai_text(messages) == ""


def test_triage_result_stores_both_intents() -> None:
    doc = TriageResult(intent="document", reply="")
    chat = TriageResult(intent="chitchat", reply="Hi there!")
    assert doc.intent == "document" and doc.reply == ""
    assert chat.intent == "chitchat" and chat.reply == "Hi there!"


# ── invoke_safely (graceful API/LLM failure) ──────────────────────────────────


class _FakeGraph:
    """Stand-in compiled graph whose invoke raises a configurable error."""

    def __init__(self, exc: Exception | None = None, result: dict | None = None) -> None:
        self._exc = exc
        self._result = result or {}

    def invoke(self, payload, config=None):
        if self._exc is not None:
            raise self._exc
        return self._result


def test_invoke_safely_returns_result_on_success() -> None:
    from src.agent.graph import invoke_safely

    ok = {"final_answer": "ok", "is_grounded": True, "confidence": 0.9, "citations": []}
    out = invoke_safely(_FakeGraph(result=ok), {"question": "q"}, {})
    assert out == ok


def test_invoke_safely_degrades_on_any_exception() -> None:
    # Provider-agnostic: any terminal error (auth, quota, outage) → clean OutputState.
    from src.agent.graph import invoke_safely

    out = invoke_safely(_FakeGraph(exc=RuntimeError("simulated API outage")), {"question": "q"}, {})
    assert out["is_grounded"] is False
    assert out["confidence"] == 0.0
    assert out["citations"] == []
    assert "could not complete" in out["final_answer"].lower()


def test_invoke_safely_output_has_all_outputstate_keys() -> None:
    from src.agent.graph import invoke_safely

    out = invoke_safely(_FakeGraph(exc=ValueError("boom")), {"question": "q"}, {})
    assert set(out) == {"final_answer", "is_grounded", "confidence", "citations"}
