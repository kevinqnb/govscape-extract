"""Single source of truth for the metadata attributes we extract.

Both extraction backends (the LLM prompt in `extractors/llm.py` and the
GLiNER2 schema in `extractors/gliner.py`) are built from the `FIELDS` list
below, so the two never drift out of sync with each other or with
`DocumentMetadata`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# A general-purpose list of document types seen across public government
# archives. Not exhaustive -- "Other" is always a valid fallback.
DOCUMENT_TYPES = [
    "Report",
    "Memorandum",
    "Testimony",
    "Fact Sheet",
    "Press Release",
    "Policy or Guidance Document",
    "Regulation or Rule",
    "Executive Order",
    "Congressional Record",
    "Technical or Research Report",
    "Presentation or Briefing",
    "Form",
    "Manual or Handbook",
    "Strategic Plan",
    "Budget Document",
    "Other",
]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    description: str
    dtype: Literal["str", "list"] = "str"
    choices: Optional[list[str]] = None


FIELDS: list[FieldSpec] = [
    FieldSpec("title", "The document's title."),
    FieldSpec(
        "authors",
        "Named individual author(s) of the document, if any are credited.",
        dtype="list",
    ),
    FieldSpec(
        "publication_date",
        "The date the document was published or released, as written in the "
        "text.",
    ),
    FieldSpec(
        "government_agency",
        "The government agency, office, or program that published or funded "
        "the work.",
    ),
    FieldSpec(
        "document_type",
        "The kind of document this is.",
        choices=DOCUMENT_TYPES,
    ),
]


class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    publication_date: Optional[str] = None
    government_agency: Optional[str] = None
    document_type: Optional[str] = None

    @field_validator("authors", mode="before")
    @classmethod
    def _coerce_authors(cls, v):
        # Extractors occasionally return a single string instead of a list.
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v
