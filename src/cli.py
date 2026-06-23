"""CLI entry point — ingest a PDF and optionally run a question through the agent."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src import registry
from src.config import settings
from src.log import get_logger, setup_logging
from src.slug import slugify

setup_logging(settings.log_level)
logger = get_logger(__name__)


def _wipe_collection(collection: str) -> None:
    """Delete one collection's folder (chroma store, page images, outlines)."""
    p = registry.collection_dir(collection)
    if p.exists():
        shutil.rmtree(p)
        logger.info("Deleted %s", p)
    else:
        logger.debug("Nothing to delete at %s", p)


def ingest(
    pdf_path: str,
    collection: str | None = None,
    description: str | None = None,
    reset: bool = False,
) -> str:
    """Preprocess a PDF and upsert all chunks into a collection's vector store.

    A collection may hold several PDFs on a related topic; pass the same
    ``collection`` for each to group them in one searchable store.

    Args:
        pdf_path: Path to the PDF file.
        collection: Collection name (slugified). Defaults to a slug derived from
            the PDF filename — a lone-PDF collection named after the document.
        description: Optional human-readable description, written to the registry
            and injected into the agent prompts for this collection.
        reset: When True, wipe this collection's folder before indexing.

    Returns:
        The canonical collection slug used for indexing.
    """
    from src.preprocessing.chunker import chunk_document
    from src.preprocessing.pdf_loader import DocumentError, load_document
    from src.retrieval.embedder import Embedder
    from src.retrieval.indexer import Indexer

    logger.info("=== INGESTION: %s ===", pdf_path)

    src = Path(pdf_path)
    doc_id = slugify(src.stem)
    collection_slug = slugify(collection) if collection else doc_id

    if reset:
        logger.info("--reset: wiping collection %r before ingestion", collection_slug)
        _wipe_collection(collection_slug)

    pages_dir = registry.pages_dir(collection_slug, doc_id)
    outline_path = registry.outline_path(collection_slug, doc_id)

    try:
        doc = load_document(src, pages_dir=pages_dir, outline_path=outline_path)
    except DocumentError as exc:
        logger.error("Could not load PDF: %s", exc)
        sys.exit(1)

    # Keep the source PDF alongside its derived data so the collection is self-contained.
    pdf_copy = registry.pdf_dir(collection_slug, doc_id) / src.name
    if src.resolve() != pdf_copy.resolve():
        shutil.copy2(src, pdf_copy)

    chunks = chunk_document(
        doc,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    logger.info("Preprocessed: %d pages → %d chunks", len(doc.pages), len(chunks))

    embedder = Embedder(model=settings.embedding_model, api_key=settings.openai_api_key)
    indexer = Indexer(persist_directory=str(registry.chroma_dir(collection_slug)), embedder=embedder)
    col = indexer.index_chunks(chunks, collection_name=collection_slug)

    registry.set_description(collection_slug, description or "")

    logger.info(
        "Indexed %d chunks (doc_id=%s) into collection '%s' — use this name to ask questions",
        len(chunks), doc_id, col.name,
    )
    return col.name


def ask(question: str, collection: str, session_id: str = "default") -> None:
    """Run a question through the agentic RAG pipeline and print the answer.

    The session_id maps to the checkpointer's thread_id, so repeated calls with
    the same session_id share conversation memory.
    """
    from src.agent.graph import build_graph, invoke_safely

    logger.info("=== AGENT: %r (collection=%s, session=%s) ===", question, collection, session_id)

    # The retriever is initialized inside research_node from state["collection"],
    # so a chit-chat turn never needs the collection to exist.
    graph = build_graph()
    result = invoke_safely(
        graph,
        {"question": question, "collection": collection, "session_id": session_id},
        config={"configurable": {"thread_id": session_id}},
    )

    # The final answer is the program's user-facing RESULT, not a log line — it is
    # written to stdout via print() on purpose, so it stays clean and parseable even
    # when logs are silenced or redirected. Diagnostics still go through get_logger.
    print("\n" + "=" * 60)
    print("ANSWER:")
    print(result["final_answer"])
    print("-" * 60)
    grounded = result.get("is_grounded", False)
    confidence = result.get("confidence", 0.0)
    citations = result.get("citations", [])
    print(f"Grounded: {grounded}  |  Confidence: {confidence:.0%}")
    if citations:
        print("Citations:", ", ".join(citations))
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic RAG — PDF ingestion and question answering"
    )
    parser.add_argument("--pdf", help="Path to the PDF file to ingest")
    parser.add_argument("--question", "-q", help="Question to answer from the indexed document")
    parser.add_argument(
        "--collection",
        help="Collection name to use (default: slug derived from the PDF filename)",
    )
    parser.add_argument(
        "--description",
        help="Human-readable description of the collection; stored in the registry "
        "and injected into the agent prompts as collection context",
    )
    parser.add_argument(
        "--session-id",
        default="default",
        help="Conversation session for memory; reuse it across questions to keep context",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the target collection's folder before ingesting (other collections survive)",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Ingest the PDF without running a question",
    )
    args = parser.parse_args()

    if not args.pdf and not args.question:
        parser.print_help()
        sys.exit(1)

    collection: str | None = None
    if args.pdf:
        collection = ingest(
            args.pdf,
            collection=args.collection,
            description=args.description,
            reset=args.reset,
        )

    if args.question and not args.ingest_only:
        active_collection = collection or args.collection
        if not active_collection:
            logger.error("--collection is required when --pdf is not specified")
            sys.exit(1)
        ask(args.question, active_collection, session_id=args.session_id)


if __name__ == "__main__":
    main()
