from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from govscape_extract.schema import DocumentMetadata


class MetadataExtractor(ABC):
    """Common interface for pulling `DocumentMetadata` out of document text."""

    # Populated by `extract()` on backends that have a notion of token usage /
    # finish reason (currently just LLMExtractor); left as None otherwise so
    # callers can read these uniformly off any backend.
    last_usage: Optional[dict] = None
    last_finish_reason: Optional[str] = None

    @abstractmethod
    def extract(self, text: str) -> DocumentMetadata: ...
