"""Tests for src.retrieval — slugify, metadata rules, and the dense retrieval path."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.preprocessing.chunker import Chunk
from src.retrieval.indexer import Indexer, _to_metadata, slugify
from src.retrieval.retriever import Retriever


# ── fakes ─────────────────────────────────────────────────────────────────────


class FakeEmbedder:
    """Deterministic bag-of-words embedder — no API calls.

    Each text maps to counts over a fixed vocabulary plus a constant bias
    dimension (so no vector is all-zero, which cosine distance can't handle).
    """

    VOCAB = ("inflation", "interest", "rate", "fruit")

    def embed(self, text: str) -> list[float]:
        return self._vectorize(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(t) for t in texts]

    def _vectorize(self, text: str) -> list[float]:
        low = text.lower()
        return [float(low.count(word)) for word in self.VOCAB] + [1.0]


def _chunk(idx: int, text: str, doc_id: str = "sample_doc", section: str = "Intro") -> Chunk:
    return Chunk(
        doc_id=doc_id,
        page_number=idx + 1,
        chunk_index=idx,
        text=text,
        has_table=False,
        has_figure=False,
        requires_vision=False,
        section=section,
        image_path=f"data/pages/{idx}.png",
    )


# ── slugify ───────────────────────────────────────────────────────────────────


def test_slugify_folds_turkish_chars() -> None:
    assert slugify("Haziran FOMC - Enflasyonun Dönüşü") == "haziran_fomc_enflasyonun_donusu"


def test_slugify_handles_all_turkish_letters() -> None:
    assert slugify("şıöüğç ŞİÖÜĞÇ") == "siougc_siougc"


def test_slugify_is_idempotent() -> None:
    raw = "Haziran FOMC - Enflasyonun Dönüşü"
    once = slugify(raw)
    assert slugify(once) == once


def test_slugify_collapses_separators_and_strips_punctuation() -> None:
    assert slugify("A  B--C!!! (x)") == "a_b_c_x"


def test_slugify_empty_input_falls_back() -> None:
    assert slugify("???") == "document"


# ── _to_metadata ──────────────────────────────────────────────────────────────


def test_to_metadata_has_no_none_values() -> None:
    meta = _to_metadata(_chunk(0, "text"))
    assert all(value is not None for value in meta.values())


def test_to_metadata_uses_none_sentinel_for_empty_section() -> None:
    chunk = _chunk(0, "text", section="")
    meta = _to_metadata(chunk)
    assert meta["section"] == "none"


def test_to_metadata_carries_chunk_own_doc_id() -> None:
    # doc_id comes from the chunk itself — a collection may hold several PDFs.
    meta = _to_metadata(_chunk(0, "text", doc_id="doc_b"))
    assert meta["doc_id"] == "doc_b"


def test_to_metadata_values_are_flat_primitives() -> None:
    meta = _to_metadata(_chunk(0, "text"))
    assert all(isinstance(v, (str, int, bool)) for v in meta.values())
    assert meta["image_path"] == "data/pages/0.png"


# ── dense retrieval path (index -> query) ─────────────────────────────────────


@pytest.fixture
def indexer(tmp_path: Path) -> Indexer:
    return Indexer(persist_directory=str(tmp_path / "chroma"), embedder=FakeEmbedder())


def test_index_chunks_rejects_empty(indexer: Indexer) -> None:
    with pytest.raises(ValueError):
        indexer.index_chunks([])


def test_dense_retrieval_returns_relevant_chunk_first(indexer: Indexer) -> None:
    chunks = [
        _chunk(0, "Inflation rose sharply this quarter."),
        _chunk(1, "The committee held the interest rate steady."),
        _chunk(2, "He bought an apple and other fruit."),
    ]
    collection = indexer.index_chunks(chunks)

    retriever = Retriever(embedder=FakeEmbedder(), collection=collection)
    hits = retriever.retrieve("What about inflation?", top_k=3)

    assert len(hits) == 3
    assert "inflation" in hits[0].text.lower()
    # Scores are sorted descending (most relevant first).
    assert hits[0].score >= hits[-1].score


def test_dense_retrieval_populates_typed_fields(indexer: Indexer) -> None:
    chunks = [_chunk(0, "Inflation outlook for the fruit market.")]
    collection = indexer.index_chunks(chunks)

    retriever = Retriever(embedder=FakeEmbedder(), collection=collection)
    hit = retriever.retrieve("inflation", top_k=1)[0]

    assert hit.doc_id == "sample_doc"
    assert isinstance(hit.page_number, int)
    assert hit.chunk_index == 0
    assert hit.image_path == "data/pages/0.png"
    assert hit.section == "Intro"


def test_index_chunks_is_idempotent(indexer: Indexer) -> None:
    chunks = [_chunk(0, "Inflation."), _chunk(1, "Interest rate.")]
    indexer.index_chunks(chunks)
    collection = indexer.index_chunks(chunks)  # re-ingest same chunks
    assert collection.count() == 2


def test_get_collection_unknown_doc_raises(indexer: Indexer) -> None:
    with pytest.raises(ValueError):
        indexer.get_collection("does not exist")


def test_collection_holds_multiple_pdfs(indexer: Indexer) -> None:
    # Two PDFs (distinct doc_ids) indexed into ONE collection: both are searchable
    # and their chunk ids carry their own doc_id (the de-conflation guarantee).
    doc_a = [_chunk(0, "Inflation rose this quarter.", doc_id="doc_a")]
    doc_b = [_chunk(0, "The committee held the interest rate.", doc_id="doc_b")]

    indexer.index_chunks(doc_a, collection_name="mixed")
    collection = indexer.index_chunks(doc_b, collection_name="mixed")

    # Same page/index in both PDFs would have collided under the old scheme; the
    # doc_id prefix keeps them distinct, so both rows survive.
    assert collection.count() == 2
    ids = collection.get()["ids"]
    assert any(i.startswith("doc_a_") for i in ids)
    assert any(i.startswith("doc_b_") for i in ids)

    retriever = Retriever(embedder=FakeEmbedder(), collection=collection)
    hits = retriever.retrieve("inflation interest rate", top_k=2)
    assert {h.doc_id for h in hits} == {"doc_a", "doc_b"}
