"""ChromaDB dense index: embed chunks and upsert into a persistent collection.

This module owns the *dense* write path only. A sparse (BM25) index can be
added alongside later without touching this code: the slug/metadata helpers
below are framework-agnostic, and the Indexer keeps the embedding+Chroma
concern isolated behind ``index_chunks`` / ``get_collection``.
"""

from __future__ import annotations

import chromadb

from src.log import get_logger
from src.preprocessing.chunker import Chunk
from src.retrieval.embedder import Embedder
from src.slug import slugify  # re-exported for callers that import it from here

logger = get_logger(__name__)

__all__ = ["Indexer", "slugify"]


def _to_metadata(chunk: Chunk) -> dict[str, str | int | bool]:
    """Build a flat, ChromaDB-safe metadata dict for a chunk.

    Rules enforced here:
    - Flat primitives only (str / int / bool) — no None, lists, or nested objects.
    - ``doc_id`` is the chunk's own source-PDF slug (a collection may hold several).
    - ``section`` uses the 'none' sentinel string when absent, never None.
    - ``image_path`` is stored verbatim as written by pdf_loader.
    """
    return {
        "doc_id": chunk.doc_id,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "has_table": chunk.has_table,
        "has_figure": chunk.has_figure,
        "requires_vision": chunk.requires_vision,
        "section": chunk.section or "none",
        "image_path": chunk.image_path,
    }


class Indexer:
    """Builds and accesses persistent ChromaDB collections, one per document."""

    def __init__(self, persist_directory: str, embedder: Embedder) -> None:
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._embedder = embedder
        self._persist_directory = persist_directory
        logger.debug("Indexer initialized: persist_directory=%s", persist_directory)

    def index_chunks(
        self,
        chunks: list[Chunk],
        collection_name: str | None = None,
    ) -> chromadb.Collection:
        """Embed chunks and upsert them into a collection.

        A collection may hold chunks from several PDFs: the collection name is
        independent of each chunk's ``doc_id``, and chunk IDs are prefixed with
        the chunk's own ``doc_id`` so two PDFs in one collection never collide.

        Idempotent: chunk ids are deterministic, so re-ingesting the same PDF
        overwrites its rows, while a new PDF appends to the same collection.

        Args:
            chunks: Non-empty list of chunks (all from a single PDF).
            collection_name: ChromaDB collection to upsert into. Defaults to the
                slug of the chunks' ``doc_id`` (lone-PDF collection).

        Returns:
            The populated ChromaDB collection.

        Raises:
            ValueError: If ``chunks`` is empty.
        """
        if not chunks:
            raise ValueError("chunks must not be empty")

        collection_slug = slugify(collection_name) if collection_name else slugify(chunks[0].doc_id)
        collection = self._client.get_or_create_collection(
            name=collection_slug,
            metadata={"hnsw:space": "cosine"},
        )

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_batch(texts)
        # Per-PDF chunk_id form ({doc_id}_p{page}_c{index}) that tools render and
        # the validator's citation regex expects — unique across PDFs in a collection.
        ids = [f"{c.doc_id}_p{c.page_number}_c{c.chunk_index}" for c in chunks]
        metadatas = [_to_metadata(c) for c in chunks]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(
            "Indexed %d chunks (doc_id=%s) into collection '%s' (persist: %s)",
            len(chunks), chunks[0].doc_id, collection_slug, self._persist_directory,
        )
        return collection

    def reset(self) -> None:
        """Delete all collections from the ChromaDB instance."""
        names = [c.name for c in self._client.list_collections()]
        for name in names:
            self._client.delete_collection(name)
        logger.info("Deleted %d collection(s) from ChromaDB", len(names))

    def get_collection(self, collection: str) -> chromadb.Collection:
        """Return an existing collection by name (raw name or slug).

        Args:
            collection: Collection name; slugified internally so callers may
                pass either a raw name or an already-clean slug.

        Returns:
            The ChromaDB collection.

        Raises:
            ValueError: If no index exists for this collection.
        """
        slug = slugify(collection)
        try:
            return self._client.get_collection(slug)
        except Exception as exc:
            raise ValueError(
                f"No index found for collection='{slug}'. Run index_chunks first."
            ) from exc
