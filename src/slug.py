"""Canonical slug for document/collection ids: ASCII-only, lowercase, underscores.

Shared by preprocessing (doc_id from filename) and retrieval (collection name and
chunk ids) so the two layers can never derive different slugs for the same input.
"""

from __future__ import annotations

import re
import unicodedata

# Fold Turkish characters first (NFKD mangles İ/ı), then strip remaining accents.
_TR_FOLD = str.maketrans(
    {
        "ş": "s", "Ş": "s",
        "ı": "i", "İ": "i",
        "ö": "o", "Ö": "o",
        "ü": "u", "Ü": "u",
        "ğ": "g", "Ğ": "g",
        "ç": "c", "Ç": "c",
    }
)


def slugify(name: str) -> str:
    """Derive a clean, ASCII-only slug from a filename stem or collection name.

    Folds Turkish characters, strips remaining accents via NFKD, lowercases, and
    collapses every run of non-alphanumeric characters to a single underscore.
    Idempotent: ``slugify(slugify(x)) == slugify(x)``.

    Args:
        name: Raw name (e.g. "Haziran FOMC - Enflasyonun Dönüşü").

    Returns:
        Slug such as "haziran_fomc_enflasyonun_donusu". Falls back to "document"
        if the input has no usable characters.
    """
    text = name.translate(_TR_FOLD)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return text.strip("_") or "document"
