"""Query-time retrieval over a ChromaDB collection.

Only the *dense* (embedding similarity) path is implemented. Sparse (BM25)
retrieval and reciprocal-rank fusion are stubbed below as explicit extension
points — see ``_sparse_retrieve`` and ``_reciprocal_rank_fusion``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb

from src.log import get_logger
from src.retrieval.embedder import Embedder

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A single retrieval hit with its score and source metadata."""

    chunk_id: str
    text: str
    score: float  # cosine similarity in [-1, 1]; higher is more relevant
    doc_id: str
    page_number: int
    chunk_index: int
    section: str
    has_table: bool
    has_figure: bool
    requires_vision: bool
    image_path: str


class Retriever:
    """Retrieves the most relevant chunks for a query from one collection."""

    def __init__(self, embedder: Embedder, collection: chromadb.Collection) -> None:
        self._embedder = embedder
        self._collection = collection

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k chunks most relevant to ``query``.

        Currently dense-only. When sparse retrieval lands, this method becomes
        the fusion entry point: run dense + sparse, then merge via
        ``_reciprocal_rank_fusion``.

        Args:
            query: Natural-language query.
            top_k: Maximum number of chunks to return.
            where: Optional ChromaDB metadata filter applied *before* ranking,
                so the search is scoped to a subset (e.g. one section path or a
                page range) rather than the whole collection. None = no filter.

        Returns:
            Ranked list of RetrievedChunk, most relevant first.
        """
        logger.debug(
            "Retrieve (dense): top_k=%d where=%s query=%r", top_k, where, query[:80]
        )
        hits = self._dense_retrieve(query, top_k, where)
        logger.info("Retrieved %d chunk(s) for query", len(hits))
        return hits

    def _dense_retrieve(
        self,
        query: str,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Embed the query and pull the nearest neighbours from ChromaDB.

        ``where`` is forwarded straight to Chroma's query; passing None applies
        no filter, so the unscoped path is unchanged.
        """
        query_embedding = self._embedder.embed(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns one inner list per query; we only sent one query.
        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        hits: list[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            hits.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=1.0 - float(distance),  # cosine distance -> similarity
                    doc_id=str(meta.get("doc_id", "")),
                    page_number=int(meta.get("page_number", 0)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    section=str(meta.get("section", "none")),
                    has_table=bool(meta.get("has_table", False)),
                    has_figure=bool(meta.get("has_figure", False)),
                    requires_vision=bool(meta.get("requires_vision", False)),
                    image_path=str(meta.get("image_path", "")),
                )
            )
        return hits

    # ── Deferred: sparse retrieval + fusion ───────────────────────────────────
    # Kept as explicit extension points so the dense path above stays untouched
    # when BM25 is added. Wire both into ``retrieve`` and merge with RRF.

    def _sparse_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """TODO: BM25 keyword retrieval over the same chunk corpus."""
        raise NotImplementedError("Sparse (BM25) retrieval not implemented yet")

    @staticmethod
    def _reciprocal_rank_fusion(
        rankings: list[list[RetrievedChunk]],
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """TODO: merge dense + sparse rankings via reciprocal rank fusion."""
        raise NotImplementedError("Reciprocal rank fusion not implemented yet")
