from __future__ import annotations

from abc import ABC, abstractmethod

from govscape_extract.schema import DocumentMetadata


class MetadataExtractor(ABC):
    """Common interface for pulling `DocumentMetadata` out of document text."""

    @abstractmethod
    def extract(self, text: str) -> DocumentMetadata: ...
