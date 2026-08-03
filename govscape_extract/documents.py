"""Loading OCR documents and slicing them down to a usable extraction window.

Full OCR text can run to tens of thousands of tokens, which is more than
either backend should be fed: it's wasted cost for the LLM, and silently
truncated by GLiNER2's encoder if it's over that model's limit. Title,
authors, agency, and dates are overwhelmingly a front-matter concern, so we
extract from the first few pages only.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MAX_PAGES = 3
DEFAULT_MAX_CHARS = 9000


def load_document(path: Path) -> dict:
    return json.loads(path.read_text())


def extraction_window(
    doc: dict,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Return the leading slice of a document's OCR text used for extraction.

    Uses `attributes.pdf_page_numbers` (a list of `[char_start, char_end,
    page_number]` triples) to cut cleanly at a page boundary when available,
    falling back to a flat character cap otherwise.
    """
    text = doc.get("text", "")
    pages = doc.get("attributes", {}).get("pdf_page_numbers")

    if pages:
        page_end = 0
        for _start, end, page_num in pages:
            if page_num > max_pages:
                break
            page_end = end
        if page_end:
            text = text[:page_end]

    return text[:max_chars]
