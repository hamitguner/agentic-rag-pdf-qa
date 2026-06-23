"""Tests for src.registry — path composition and the collections CSV registry.

Fully offline: settings.data_dir is redirected to a tmp_path so no real data is touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import registry
from src.config import settings


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    return tmp_path


# ── path helpers ──────────────────────────────────────────────────────────────


def test_path_helpers_compose_nested_layout(data_root: Path) -> None:
    assert registry.collection_dir("My Col") == data_root / "my_col"
    assert registry.chroma_dir("My Col") == data_root / "my_col" / "chroma"
    assert registry.pdf_dir("My Col", "Doc A") == data_root / "my_col" / "doc_a"
    assert registry.pages_dir("My Col", "Doc A") == data_root / "my_col" / "doc_a" / "pages"
    assert (
        registry.outline_path("My Col", "Doc A")
        == data_root / "my_col" / "doc_a" / "outline.json"
    )


def test_path_helpers_slugify_inputs(data_root: Path) -> None:
    # Turkish/space inputs resolve to the same slug everywhere (consistency).
    assert registry.collection_dir("Enflasyonun Dönüşü") == data_root / "enflasyonun_donusu"


# ── CSV registry ──────────────────────────────────────────────────────────────


def test_description_round_trip_and_upsert(data_root: Path) -> None:
    assert registry.get_description("c1") == ""  # absent → empty

    registry.set_description("c1", "first")
    assert registry.get_description("c1") == "first"

    registry.set_description("c1", "updated")  # upsert, not duplicate
    assert registry.get_description("c1") == "updated"
    assert len(registry.list_collections()) == 1


def test_registry_lists_all_collections(data_root: Path) -> None:
    registry.set_description("c1", "one")
    registry.set_description("C2", "two")
    names = {row["collection"] for row in registry.list_collections()}
    assert names == {"c1", "c2"}


def test_description_lookup_is_slug_insensitive(data_root: Path) -> None:
    registry.set_description("My Col", "desc")
    assert registry.get_description("my_col") == "desc"
    assert registry.get_description("My Col") == "desc"
