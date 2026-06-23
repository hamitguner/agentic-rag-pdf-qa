"""Grounding check: verify every claim in a draft answer against retrieved chunks."""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from src.agent.prompts import VALIDATION_SYSTEM_PROMPT
from src.log import get_logger

logger = get_logger(__name__)


class ValidationResult(BaseModel):
    """Output of the LLM-as-judge grounding check."""

    is_grounded: bool
    confidence: float
    unverified_claims: list[str]
    final_answer: str


def grounding_check(
    question: str,
    draft_answer: str,
    retrieved_chunks: list[dict],
    model: str = "claude-haiku-4-5-20251001",
) -> ValidationResult:
    """Verify that every claim in draft_answer is supported by retrieved_chunks.

    Uses provider-native structured output (ProviderStrategy) so the response
    is guaranteed to be a valid ValidationResult — no JSON parsing or fallbacks.

    Honest absence answers ("the document doesn't cover this") are treated as
    grounded — the prompt instructs the judge to pass them straight through.

    Args:
        question: The original user question.
        draft_answer: The agent's proposed answer to verify.
        retrieved_chunks: All chunks accumulated during the retrieval run.
        model: Model identifier passed to init_chat_model.

    Returns:
        ValidationResult with grounding status, confidence, unsupported claims,
        and the final cleaned answer.
    """
    chunk_texts = "\n\n---\n\n".join(
        f"[Chunk {c.get('chunk_id', i)} | Page {c.get('page_number', '?')}]\n{c.get('text', '')}"
        for i, c in enumerate(retrieved_chunks)
    )

    user_content = (
        f"QUESTION:\n{question}\n\n"
        f"DRAFT ANSWER:\n{draft_answer}\n\n"
        f"SOURCE CHUNKS:\n{chunk_texts}"
    )

    logger.info(
        "Grounding check | chunks=%d | draft_len=%d",
        len(retrieved_chunks),
        len(draft_answer),
    )

    llm = init_chat_model(model, temperature=0).with_structured_output(ValidationResult)
    result: ValidationResult = llm.invoke(
        [SystemMessage(content=VALIDATION_SYSTEM_PROMPT), HumanMessage(content=user_content)]
    )

    logger.info(
        "Grounding result | grounded=%s | confidence=%.2f | unverified=%d",
        result.is_grounded,
        result.confidence,
        len(result.unverified_claims),
    )
    return result
