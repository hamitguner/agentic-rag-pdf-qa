"""Intent triage: classify a turn as a document question or chit-chat.

Runs before the retrieval pipeline so greetings, thanks, and gibberish get a
direct reply instead of an expensive (and confusing) document search.
"""

from __future__ import annotations

from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from src.agent.prompts import TRIAGE_SYSTEM_PROMPT
from src.log import get_logger

logger = get_logger(__name__)


class TriageResult(BaseModel):
    """Output of the intent classifier."""

    intent: Literal["document", "chitchat"]
    reply: str  # friendly reply for chit-chat; empty for document questions


def classify_intent(
    question: str,
    previous_answer: str = "",
    collection_context: str = "",
    model: str = "claude-haiku-4-5-20251001",
) -> TriageResult:
    """Classify a turn as a document question or chit-chat.

    Uses provider-native structured output so the response is guaranteed to be a
    valid TriageResult. A single call both classifies and (for chit-chat) writes
    the reply, so the chit-chat path needs no further LLM call.

    The classifier is biased toward "document" when unsure, so a borderline turn
    degrades to a normal search rather than refusing a genuine question.

    Args:
        question: The user's current question.
        previous_answer: The last assistant answer, given as context so terse
            follow-ups ("and how is it quantized?") classify as document questions.
        collection_context: The active collection's description, so the
            classifier knows what the document corpus is about (a question on a
            niche topic in that corpus is a document question, not chit-chat).
        model: Model identifier passed to init_chat_model.

    Returns:
        TriageResult with the intent and, for chit-chat, a reply.
    """
    user_content = (
        (f"COLLECTION TOPIC (context):\n{collection_context}\n\n" if collection_context else "")
        + (f"PREVIOUS ANSWER (context):\n{previous_answer}\n\n" if previous_answer else "")
        + f"CURRENT MESSAGE:\n{question}"
    )

    logger.info("Triage | question=%r | has_context=%s", question[:80], bool(previous_answer))

    llm = init_chat_model(model, temperature=0).with_structured_output(TriageResult)
    result: TriageResult = llm.invoke(
        [SystemMessage(content=TRIAGE_SYSTEM_PROMPT), HumanMessage(content=user_content)]
    )

    logger.info("Triage result | intent=%s | reply_len=%d", result.intent, len(result.reply))
    return result
