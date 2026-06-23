"""LangChain tool definitions for the Evidence Agent's RAG operations."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src import registry
from src.config import settings
from src.log import get_logger
from src.retrieval.embedder import Embedder
from src.retrieval.indexer import Indexer
from src.retrieval.retriever import Retriever

logger = get_logger(__name__)

_shared_retriever: Retriever | None = None

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

_vision_model = None


def _get_vision_model():
    """Lazy singleton — fails at call time, not import time; mirrors _get_retriever()."""
    global _vision_model
    if _vision_model is None:
        _vision_model = init_chat_model(settings.vision_model_name, temperature=0)
    return _vision_model


def init_retriever(collection: str) -> None:
    """
    Initialize the shared retriever for a specific collection.

    Must be called before the graph is invoked. Calling again with a different
    collection switches the active corpus. A collection may span several PDFs,
    all stored in that collection's own ChromaDB instance.
    """
    global _shared_retriever
    logger.info("Initializing retriever for collection=%r", collection)
    embedder = Embedder(model=settings.embedding_model, api_key=settings.openai_api_key)
    indexer = Indexer(persist_directory=str(registry.chroma_dir(collection)), embedder=embedder)
    _shared_retriever = Retriever(
        embedder=embedder,
        collection=indexer.get_collection(collection),
    )


def _get_retriever() -> Retriever:
    if _shared_retriever is None:
        raise RuntimeError(
            "Retriever not initialized. Call init_retriever(collection) before invoking the graph.",
        )
    return _shared_retriever


def _format_chroma_get_results(results: dict[str, Any]) -> str:
    """Format metadata-based direct lookup results as a JSON string."""
    metadatas = results.get("metadatas", []) or []
    documents = results.get("documents", []) or []

    formatted_chunks = []
    for meta, text in zip(metadatas, documents):
        formatted_chunks.append(
            {
                "chunk_id": f"{meta.get('doc_id')}_p{meta.get('page_number')}_c{meta.get('chunk_index')}",
                "chunk_index": meta.get("chunk_index"),
                "page_number": meta.get("page_number"),
                "section": meta.get("section"),
                "requires_vision": meta.get("requires_vision"),
                "image_path": meta.get("image_path"),
                "text": text,
            },
        )

    formatted_chunks.sort(key=lambda x: x.get("chunk_index", 0))
    return json.dumps(
        {"retrieved_chunks": formatted_chunks},
        ensure_ascii=False,
        indent=2,
    )


@tool
def get_outline() -> str:
    """
    Return the document's structural map: every section heading with its page span.

    Do NOT call this first — start with search_semantic, which answers most
    questions directly. Use get_outline only when the question is explicitly about
    the document's structure/sections, or when an initial search came back scattered
    and you need to find which section to scope into. Use the returned `section`
    strings verbatim as the `section` argument to search_semantic, or as the
    `section_name` for fetch_section_content — they are the exact keys the document
    is indexed under, so copy them character-for-character.

    Each entry has:
        section: full breadcrumb path, e.g. 'Vision Experiments > Model Architecture'
        level:   heading depth (1 = top-level)
        start_page / end_page: page range the section spans

    Returns an empty outline for documents with no table of contents — in that
    case, fall back to unscoped search_semantic.
    """
    logger.info("Tool triggered [get_outline]")
    retriever = _get_retriever()

    # Derived from indexed metadata rather than the raw outline.json, so the map
    # the agent sees is always exactly what is searchable (no plumbing, no stale
    # file after a re-ingest).
    results = retriever._collection.get(include=["metadatas"])
    metadatas = results.get("metadatas") or []

    spans: dict[str, dict[str, int]] = {}
    for meta in metadatas:
        section = meta.get("section") or "none"
        if section == "none":
            continue
        page = int(meta.get("page_number", 0))
        span = spans.setdefault(section, {"start_page": page, "end_page": page})
        span["start_page"] = min(span["start_page"], page)
        span["end_page"] = max(span["end_page"], page)

    outline = [
        {
            "section": section,
            "level": section.count(" > ") + 1,
            "start_page": span["start_page"],
            "end_page": span["end_page"],
        }
        for section, span in spans.items()
    ]
    outline.sort(key=lambda o: (o["start_page"], o["section"]))
    logger.info("[get_outline] %d section(s)", len(outline))
    return json.dumps({"outline": outline}, ensure_ascii=False, indent=2)


@tool
def search_semantic(query: str, top_k: int = 3, section: str | None = None) -> str:
    """
    Perform dense semantic vector search for a single query and return the top matching
    document chunks. Call this tool multiple times with different queries if you need to
    explore different angles.

    Args:
        query: The search query to embed and retrieve relevant chunks for.
        top_k: Maximum number of chunks to return.
        section: Optional. An exact section path from get_outline (e.g.
            'Results > Tool Use Performance'). When set, the search is scoped to
            that section only — use it once you know where the answer lives to
            avoid pulling near-duplicate matches from unrelated parts of the
            document. Omit to search the whole document.

    """
    logger.info(
        "Tool triggered [search_semantic] | Query: %r | Section: %r", query, section
    )
    retriever = _get_retriever()
    where = {"section": section} if section else None
    hits = retriever.retrieve(query=query, top_k=top_k, where=where)

    evidence_chunks = [
        {
            "chunk_id": f"{hit.doc_id}_p{hit.page_number}_c{hit.chunk_index}",
            "chunk_index": hit.chunk_index,
            "page_number": hit.page_number,
            "section": hit.section,
            "requires_vision": hit.requires_vision,
            "image_path": hit.image_path,
            "score": float(hit.score),
            "text": hit.text,
        }
        for hit in hits
    ]
    return json.dumps(
        {"evidence_chunks": evidence_chunks},
        ensure_ascii=False,
        indent=2,
    )


@tool
def fetch_section_content(section_name: str) -> str:
    """
    Retrieve all document chunks belonging to a specific structural section or heading.

    Use when the answer is likely distributed across multiple chunks inside the same
    section and ranking would miss part of it. Skip if a single chunk already contains
    the answer, or prefer search_semantic(section=...) when you only need the most
    relevant chunks within the section.

    Args:
        section_name: The exact breadcrumb path from get_outline
            (e.g. 'Vision Experiments > Model Architecture').

    """
    logger.info("Tool triggered [fetch_section_content] | Section: %r", section_name)
    retriever = _get_retriever()
    results = retriever._collection.get(where={"section": section_name})
    return _format_chroma_get_results(results)


@tool
def fetch_chunks_by_index(chunk_indexes: list[int]) -> str:
    """
    Retrieve one or more document chunks by their global indexes in a single call.

    Use when you already know which chunk_indexes you need — e.g. to expand context
    around a result, grab neighbours of multiple chunks, or fetch several non-contiguous
    chunks at once. Always prefer one call with a list over multiple single-index calls.

    Args:
        chunk_indexes: List of global integer chunk indexes to retrieve.

    """
    logger.info("Tool triggered [fetch_chunks_by_index] | Indexes: %s", chunk_indexes)

    invalid = [i for i in chunk_indexes if i < 0]
    if invalid:
        return json.dumps({"error": f"Invalid indexes (must be >= 0): {invalid}"})

    retriever = _get_retriever()
    results = retriever._collection.get(
        where={"chunk_index": {"$in": chunk_indexes}},
    )

    if not results.get("ids"):
        return json.dumps({"error": f"No chunks found for indexes {chunk_indexes}."})

    return _format_chroma_get_results(results)


@tool
def analyze_image(image_path: str, query: str, config: RunnableConfig) -> str:
    """
    Extract specific information from a rendered page image (chart, table, figure).

    Use only when requires_vision is true AND text retrieval cannot answer. Pass a
    focused `query` naming exactly what to extract. Returns a text summary — the image
    never enters the conversation, so calling this does not bloat context.

    Args:
        image_path: Local path to the page PNG.
        query: Precisely what to look for in the image.

    """
    logger.info(
        "Tool triggered [analyze_image] | Path: %s | Query: %r",
        image_path,
        query,
    )
    path = Path(image_path)
    if not path.exists():
        return json.dumps({"error": f"Image not found: {image_path}"})

    base64_data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Extract the information needed to answer the question from this page "
                    "image. Transcribe relevant tables/figures verbatim, with exact numbers. "
                    f"If the answer is not visible, say so.\n\nQuestion: {query}"
                ),
            },
            # langchain 1.0 standard cross-provider image block:
            {"type": "image", "base64": base64_data, "mime_type": mime_type},
        ],
    )
    # Pass the injected run config so the vision call nests under this tool's
    # span in tracing instead of starting a separate top-level trace.
    result = _get_vision_model().invoke([message], config=config)
    text = (
        result.content
        if isinstance(result.content, str)
        else " ".join(b.get("text", "") for b in result.content if isinstance(b, dict))
    )
    return json.dumps(
        {"vision_result": text, "source_image": image_path},
        ensure_ascii=False,
    )


RAG_TOOLS = [
    get_outline,
    search_semantic,
    fetch_section_content,
    fetch_chunks_by_index,
    analyze_image,
]
