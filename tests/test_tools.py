"""Tests for RAG tools — focused on analyze_image (offline, vision model mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.tools import analyze_image


def test_analyze_image_missing_file_returns_error() -> None:
    out = json.loads(analyze_image.invoke({"image_path": "does/not/exist.png", "query": "anything"}))
    assert "error" in out


def test_analyze_image_returns_text_description(tmp_path: Path) -> None:
    img = tmp_path / "page1.png"
    img.write_bytes(b"\x89PNG fake-bytes")

    fake_model = MagicMock()
    fake_model.invoke.return_value = MagicMock(content="A chart titled X with two series.")

    with patch("src.agent.tools._get_vision_model", return_value=fake_model):
        out = json.loads(
            analyze_image.invoke({"image_path": str(img), "query": "What does the chart show?"})
        )

    # Agent receives compact TEXT keyed as vision_result — no raw image bytes.
    assert out["vision_result"] == "A chart titled X with two series."
    assert out["source_image"] == str(img)
    assert "base64" not in out

    # Vision model received a proper image content block (not base64-as-text).
    human_message = fake_model.invoke.call_args[0][0][0]
    blocks = human_message.content
    assert any(b.get("type") == "image" and "base64" in b for b in blocks)


def test_analyze_image_query_is_forwarded_to_vision_prompt(tmp_path: Path) -> None:
    img = tmp_path / "page1.png"
    img.write_bytes(b"\x89PNG fake-bytes")

    fake_model = MagicMock()
    fake_model.invoke.return_value = MagicMock(content="PCE headline: 2.6%")

    with patch("src.agent.tools._get_vision_model", return_value=fake_model):
        analyze_image.invoke({"image_path": str(img), "query": "PCE inflation figure"})

    human_message = fake_model.invoke.call_args[0][0][0]
    text_block = next(b for b in human_message.content if b.get("type") == "text")
    assert "PCE inflation figure" in text_block["text"]


def test_analyze_image_vision_failure_propagates(tmp_path: Path) -> None:
    img = tmp_path / "page1.png"
    img.write_bytes(b"fake")

    fake_model = MagicMock()
    fake_model.invoke.side_effect = RuntimeError("vision API down")

    with patch("src.agent.tools._get_vision_model", return_value=fake_model):
        with pytest.raises(RuntimeError, match="vision API down"):
            analyze_image.invoke({"image_path": str(img), "query": "anything"})
