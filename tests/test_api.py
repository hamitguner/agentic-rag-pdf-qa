"""Tests for the FastAPI surface — offline (no LLM).

The graph compiles without an API key (the model is injected at invoke time), so
the app starts under TestClient. The pipeline call itself is patched so /ask is
exercised without hitting any provider.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app


def test_health_returns_ok() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_returns_output_shape() -> None:
    fake = {
        "final_answer": "Llama 3's flagship model has 405B parameters.",
        "is_grounded": True,
        "confidence": 0.97,
        "citations": ["llama3_eval_p1_c0"],
    }
    with patch("src.api.routes.invoke_safely", return_value=fake) as mock:
        with TestClient(app) as client:
            resp = client.post(
                "/ask",
                json={"question": "How big is Llama 3?", "collection": "llama3_herd", "session_id": "s1"},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["final_answer"] == fake["final_answer"]
    assert body["is_grounded"] is True
    assert body["confidence"] == 0.97
    assert body["citations"] == ["llama3_eval_p1_c0"]
    # session_id flows through to the checkpointer thread_id
    assert mock.call_args.kwargs["config"]["configurable"]["thread_id"] == "s1"


def test_ask_requires_question_and_collection() -> None:
    with TestClient(app) as client:
        resp = client.post("/ask", json={"question": "only question"})
    assert resp.status_code == 422  # missing required 'collection'
