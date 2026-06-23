"""Page-atomic text chunking with tiktoken-based length measurement."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.log import get_logger
from src.preprocessing.pdf_loader import DocumentData

logger = get_logger(__name__)


@dataclass
class Chunk:
    doc_id: str
    page_number: int
    chunk_index: int  # global, 0-indexed across the whole document
    text: str
    has_table: bool
    has_figure: bool
    requires_vision: bool
    section: str
    image_path: str  # verbatim PNG path of the source page (from pdf_loader)


def chunk_document(
    doc: DocumentData,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """
    Split a document into token-bounded, page-atomic chunks.

    Chunks never cross page boundaries. Each chunk inherits its page's
    structural flags (has_table, has_figure, requires_vision, section). A page's
    visual content is signalled by those flags and read on demand from
    image_path by the vision tool.

    Args:
        doc: Output of pdf_loader.load_document.
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Token overlap between consecutive chunks on the same page.

    Returns:
        Ordered list of Chunk objects with a global chunk_index.

    """
    encoding = tiktoken.encoding_for_model("gpt-4o")

    def _token_len(text: str) -> int:
        return len(encoding.encode(text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_len,
    )

    logger.info(
        "Chunking '%s': %d pages, chunk_size=%d, overlap=%d",
        doc.doc_id, len(doc.pages), chunk_size, chunk_overlap,
    )

    chunks: list[Chunk] = []
    global_index = 0

    for page in doc.pages:
        if not page.text.strip():
            logger.debug("Page %d is empty — skipped", page.page_number)
            continue

        page_chunks = splitter.split_text(page.text)

        logger.debug(
            "Page %d → %d chunk(s) | table=%s figure=%s section=%r",
            page.page_number, len(page_chunks),
            page.has_table, page.has_figure, page.section,
        )

        for text in page_chunks:
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    page_number=page.page_number,
                    chunk_index=global_index,
                    text=text,
                    has_table=page.has_table,
                    has_figure=page.has_figure,
                    requires_vision=page.requires_vision,
                    section=page.section,
                    image_path=page.image_path,
                ),
            )
            global_index += 1

    logger.info("Produced %d chunks from '%s'", len(chunks), doc.doc_id)
    return chunks
