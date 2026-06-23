"""Tests for src.preprocessing.pdf_loader and src.preprocessing.chunker."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from src.preprocessing.chunker import chunk_document
from src.preprocessing.pdf_loader import DocumentError, load_document


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_pdf(path: Path, pages: list[str]) -> None:
    """Write a minimal PDF with one text block per page."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((50, 50), text)
    doc.save(str(path))
    doc.close()


# ── pdf_loader ────────────────────────────────────────────────────────────────


def test_load_document_returns_correct_structure(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    _make_pdf(pdf, ["Page one content.", "Page two content."])

    result = load_document(pdf, pages_dir=tmp_path / "pages", outline_path=tmp_path / "outline.json")

    assert result.doc_id == "sample"
    assert len(result.pages) == 2
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2
    assert "Page one" in result.pages[0].text
    assert Path(result.pages[0].image_path).exists()
    assert Path(result.outline_path).exists()


def test_load_document_corrupt_raises_document_error(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"this is not a pdf")

    with pytest.raises(DocumentError):
        load_document(corrupt, pages_dir=tmp_path / "pages", outline_path=tmp_path / "outline.json")


def test_load_document_missing_file_raises_document_error(tmp_path: Path) -> None:
    with pytest.raises(DocumentError):
        load_document(tmp_path / "nonexistent.pdf", pages_dir=tmp_path / "pages", outline_path=tmp_path / "outline.json")


def test_load_document_no_toc_section_is_none(tmp_path: Path) -> None:
    pdf = tmp_path / "notoc.pdf"
    _make_pdf(pdf, ["Content with no table of contents."])

    result = load_document(pdf, pages_dir=tmp_path / "pages", outline_path=tmp_path / "outline.json")

    assert all(page.section == "none" for page in result.pages)


# ── chunker ───────────────────────────────────────────────────────────────────


def test_chunk_document_skips_empty_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "empty.pdf"
    _make_pdf(pdf, [""])  # single blank page

    doc = load_document(pdf, pages_dir=tmp_path / "pages", outline_path=tmp_path / "outline.json")
    chunks = chunk_document(doc)

    assert chunks == []


def test_chunk_document_global_index_is_contiguous(tmp_path: Path) -> None:
    pdf = tmp_path / "twopages.pdf"
    _make_pdf(pdf, ["Content for page one.", "Content for page two."])

    doc = load_document(pdf, pages_dir=tmp_path / "pages", outline_path=tmp_path / "outline.json")
    chunks = chunk_document(doc)

    assert len(chunks) >= 2
    assert sorted(c.chunk_index for c in chunks) == list(range(len(chunks)))


def test_chunk_document_chunks_are_page_bounded(tmp_path: Path) -> None:
    pdf = tmp_path / "twopages.pdf"
    _make_pdf(pdf, ["Page one text.", "Page two text."])

    doc = load_document(pdf, pages_dir=tmp_path / "pages", outline_path=tmp_path / "outline.json")
    chunks = chunk_document(doc)

    assert all(c.page_number in (1, 2) for c in chunks)
