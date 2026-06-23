"""Collection registry and the canonical on-disk data layout.

A *collection* is a named group of one or more PDFs on a related topic. All of a
collection's chunks live in one ChromaDB store so a single semantic search spans
every PDF in the group; each chunk keeps its own ``doc_id`` so citations remain
traceable to the source PDF.

Layout (rooted at ``settings.data_dir``)::

    data/
      collections.csv              # columns: collection,description
      <collection>/
        chroma/                    # one vector store for the whole collection
        <doc_id>/                  # one folder per source PDF
          <original_name>.pdf
          pages/  page1.png ...
          outline.json             # TOC (the per-PDF "index")

This module owns *both* path composition and the tiny CSV registry that maps a
collection to a human-readable description (injected into the agent prompts so
triage and research know what corpus they are working over).
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.config import settings
from src.log import get_logger
from src.slug import slugify

logger = get_logger(__name__)

_REGISTRY_FILENAME = "collections.csv"
_FIELDNAMES = ["collection", "description"]


# ── Path helpers ────────────────────────────────────────────────────────────
# Every helper slugifies its inputs so callers may pass raw names or clean slugs.


def data_dir() -> Path:
    """Root data directory (created on demand)."""
    root = Path(settings.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def collection_dir(collection: str) -> Path:
    """Folder holding one collection's chroma store and PDF subfolders."""
    return data_dir() / slugify(collection)


def chroma_dir(collection: str) -> Path:
    """Per-collection ChromaDB persist directory."""
    return collection_dir(collection) / "chroma"


def pdf_dir(collection: str, doc_id: str) -> Path:
    """Per-PDF folder inside a collection (PDF copy, pages, outline)."""
    return collection_dir(collection) / slugify(doc_id)


def pages_dir(collection: str, doc_id: str) -> Path:
    """Per-PDF rendered-page PNG directory."""
    return pdf_dir(collection, doc_id) / "pages"


def outline_path(collection: str, doc_id: str) -> Path:
    """Per-PDF TOC/outline JSON path."""
    return pdf_dir(collection, doc_id) / "outline.json"


# ── CSV registry ────────────────────────────────────────────────────────────


def _registry_path() -> Path:
    return data_dir() / _REGISTRY_FILENAME


def list_collections() -> list[dict[str, str]]:
    """Return all registered collections as ``{collection, description}`` rows."""
    path = _registry_path()
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def get_description(collection: str) -> str:
    """Return a collection's description, or '' if it is not registered."""
    slug = slugify(collection)
    for row in list_collections():
        if row.get("collection") == slug:
            return row.get("description", "") or ""
    return ""


def set_description(collection: str, description: str) -> None:
    """Upsert a collection's description in the CSV registry."""
    slug = slugify(collection)
    rows = list_collections()
    for row in rows:
        if row.get("collection") == slug:
            row["description"] = description
            break
    else:
        rows.append({"collection": slug, "description": description})

    with _registry_path().open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Registry: set description for collection %r (%d chars)", slug, len(description))
