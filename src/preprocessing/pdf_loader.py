"""PDF loading, page rendering, and structural metadata extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from src.log import get_logger
from src.slug import slugify

logger = get_logger(__name__)


class DocumentError(Exception):
    """Raised when a PDF cannot be opened or parsed."""


@dataclass
class PageData:
    page_number: int  # 1-indexed
    text: str
    image_path: str  # path to rendered PNG
    has_table: bool  # find_tables() returned at least one table
    has_figure: bool  # embedded raster image present
    requires_vision: bool  # has_table OR has_figure
    section: str  # TOC-derived breadcrumb path, or 'none'


@dataclass
class DocumentData:
    doc_id: str
    pages: list[PageData]
    outline_path: str  # path to written outline.json


def _build_section_index(toc: list[list]) -> list[tuple[int, str]]:
    """Turn a PyMuPDF TOC into ordered (page, breadcrumb_path) entries.

    PyMuPDF's get_toc() returns [[level, title, page], ...] in document order.
    We collapse each heading into a *fully-qualified* path of its active
    ancestors — e.g. 'Vision Experiments > Model Architecture' — so headings
    that share a leaf title across the language / vision / speech halves of a
    paper (Model Architecture, Reward Modeling, Data, ...) become unique keys.
    Without this, a bare-title 'section' silently merges unrelated pages in
    fetch_section_content and section-scoped search.

    A level stack tracks the current ancestors; entering level L drops every
    deeper heading. The sort is stable and keyed on page only, so multiple
    headings starting on the same page keep their original reading order.

    Returns:
        (page, path) entries sorted ascending by page, enabling the O(n)
        early-break lookup in load_document.
    """
    toc_sorted = sorted(toc, key=lambda entry: entry[2])
    stack: dict[int, str] = {}
    index: list[tuple[int, str]] = []
    for level, title, page in toc_sorted:
        stack[level] = title.strip()
        for deeper in [lvl for lvl in stack if lvl > level]:
            del stack[deeper]
        path = " > ".join(stack[lvl] for lvl in sorted(stack))
        index.append((page, path))
    return index


def load_document(
    pdf_path: str | Path,
    pages_dir: str | Path,
    outline_path: str | Path,
) -> DocumentData:
    """
    Open a PDF and extract per-page text, rendered PNG, and structural flags.

    Args:
        pdf_path: Path to the PDF file.
        pages_dir: Directory for rendered page PNGs (one per page).
        outline_path: File path for the written TOC/outline JSON.

    Returns:
        DocumentData with one PageData per page.

    Raises:
        DocumentError: If the PDF cannot be opened or is corrupt.

    """
    pdf_path = Path(pdf_path)
    pages_dir = Path(pages_dir)
    outline_path = Path(outline_path)
    pages_dir.mkdir(parents=True, exist_ok=True)
    outline_path.parent.mkdir(parents=True, exist_ok=True)

    # Slugged so it is safe in chunk ids, ChromaDB metadata, and file names —
    # source stems may contain spaces or Turkish characters.
    doc_id = slugify(pdf_path.stem)

    logger.info("Loading document: %s (doc_id=%s)", pdf_path, doc_id)

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.error("Failed to open '%s': %s", pdf_path, exc)
        raise DocumentError(f"Cannot open '{pdf_path}': {exc}") from exc

    toc = doc.get_toc()  # [[level, title, page_num], ...]
    section_index = _build_section_index(toc)

    # Write outline.json in the same {section, level, start_page, end_page} shape
    # that get_outline() returns at query time, so the file is human-readable and
    # matches what the agent sees without any translation.
    total_pages = len(doc)
    outline_entries = []
    for i, (start_page, path) in enumerate(section_index):
        end_page = section_index[i + 1][0] - 1 if i + 1 < len(section_index) else total_pages
        outline_entries.append({
            "section": path,
            "level": path.count(" > ") + 1,
            "start_page": start_page,
            "end_page": end_page,
        })
    outline_path.write_text(
        json.dumps({"outline": outline_entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("TOC: %d entries → %s", len(section_index), outline_path)

    pages: list[PageData] = []
    for idx in range(len(doc)):
        page_number = idx + 1
        page = doc[idx]

        text = page.get_text("text")

        # Folder already namespaces by PDF, so a bare page name is enough.
        image_path = pages_dir / f"page{page_number}.png"
        page.get_pixmap(dpi=150).save(str(image_path))

        has_table = bool(page.find_tables().tables)

        # Figure detection via embedded raster images only. get_drawings() is
        # intentionally omitted: vector primitives (rules, borders, cell lines)
        # produce heavy false positives that would over-trigger vision calls.
        # Revisit with a size/area heuristic — see DESIGN.md trade-offs.
        has_figure = len(page.get_images()) > 0
        requires_vision = has_table or has_figure

        # Section = breadcrumb path of the last TOC heading starting on or
        # before this page. Empty section_index → loop is a no-op → 'none'.
        #
        # KNOWN LIMITATION: granularity is page-level. A page holding two
        # sections (one ending, another beginning) inherits only the later
        # heading, so part of the page is mislabelled. Correct per-chunk
        # attribution needs position-aware chunking; deferred for the MVP
        # (documented as a trade-off in DESIGN.md).
        section = "none"
        for toc_page, path in section_index:
            if toc_page <= page_number:
                section = path
            else:
                break

        logger.debug(
            "Page %d | chars=%d table=%s figure=%s section=%r",
            page_number,
            len(text),
            has_table,
            has_figure,
            section,
        )

        pages.append(
            PageData(
                page_number=page_number,
                text=text,
                image_path=str(image_path),
                has_table=has_table,
                has_figure=has_figure,
                requires_vision=requires_vision,
                section=section,
            ),
        )

    doc.close()
    logger.info("Loaded %d pages from '%s'", len(pages), doc_id)
    return DocumentData(doc_id=doc_id, pages=pages, outline_path=str(outline_path))
