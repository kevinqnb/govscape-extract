"""Single source of truth for the metadata attributes we extract.

Both extraction backends (the LLM prompt in `extractors/llm.py` and the
GLiNER2 schema in `extractors/gliner.py`) are built from the `FIELDS` list
below, so the two never drift out of sync with each other or with
`DocumentMetadata`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# A general-purpose set of document types seen across public government
# archives, as label -> definition. Not exhaustive -- "other" is always a
# valid fallback. The definitions are what the LLM prompt shows alongside
# each label; GLiNER2 only ever sees the labels themselves.
DOCUMENT_TYPES = {
    "technical_report":
        "Reports of research, engineering, monitoring, or analysis, "
        "including contractor-prepared studies.",
    "environmental_review":
        "Environmental impact statements, environmental assessments, "
        "biological opinions, and similar statutory reviews.",
    "guidance":
        "Manuals, handbooks, protocols, standard operating procedures, "
        "and recommended practices.",
    "regulation":
        "Rules, proposed rules, and other regulatory text.",
    "legislation":
        "Bills, resolutions, statutes, and enacted laws.",
    "hearing":
        "Transcripts of hearings and submitted testimony.",
    "oversight_report":
        "Audits, inspector general reports, program evaluations, and "
        "investigative findings.",
    "budget_document":
        "Budget requests and justifications, appropriations materials, "
        "and financial statements.",
    "plan":
        "Strategic plans, management plans, and performance plans.",
    "statistical_report":
        "Compilations of statistics, survey results, and data summaries "
        "presented as a periodic or standalone publication.",
    "dataset_documentation":
        "Codebooks, data dictionaries, metadata records, and other "
        "documentation describing a dataset.",
    "legal_decision":
        "Court opinions, administrative decisions, and adjudications.",
    "public_information":
        "Fact sheets, brochures, press releases, and other outreach "
        "material for a general audience.",
    "correspondence":
        "Memoranda, letters, and internal communications.",
    "other":
        "A government document that does not fit any category above.",
}


# Same label -> definition shape as DOCUMENT_TYPES above.
JURISDICTION_LEVELS = {
    "federal":
        "The national government of the United States.",
    "state":
        "A U.S. state, territory, or the District of Columbia.",
    "tribal":
        "A federally or state recognized tribal government.",
    "regional":
        "A multi-state or multi-county body such as an interstate "
        "compact, river basin commission, council of governments, or "
        "metropolitan planning organization.",
    "local":
        "A county, municipality, township, or special district.",
    "international":
        "An intergovernmental organization such as the UN, OECD, World "
        "Bank, or EU.",
    "foreign_national":
        "The national government of a country other than the United "
        "States.",
    "other":
        "A governmental body that does not fit any category above.",
}


@dataclass(frozen=True)
class FieldSpec:
    name: str
    description: str
    dtype: Literal["str", "list"] = "str"
    # Closed label set for the field, as label -> definition. Fields with
    # choices are classified against the labels rather than extracted freely.
    choices: Optional[Mapping[str, str]] = None

    @property
    def choice_values(self) -> Optional[list[str]]:
        """Just the labels -- for consumers (GLiNER2) that take a bare list."""
        return list(self.choices) if self.choices else None


FIELDS: list[FieldSpec] = [
    FieldSpec(
        "title",
        "The document's full title as printed on the cover or title page, "
        "including any subtitle. Do not include the series name or the "
        "issuing agency's name unless they are part of the title itself.",
    ),
    FieldSpec(
        "authors",
        "Named individual people credited as authors, as printed and in the "
        "order given. Do not include agencies, offices, committees, or other "
        "organizations here. Return an empty list if no individual is "
        "credited.",
        dtype="list",
    ),
    FieldSpec(
        "publication_date",
        "The date the document was published or issued, normalized to "
        "ISO-8601. Use the most precise form supported by the text: "
        "YYYY-MM-DD, YYYY-MM, or YYYY. Prefer the publication or issue date "
        "over an approval, revision, or data-collection date.",
    ),
    FieldSpec(
        "publication_date_raw",
        "The same date, copied verbatim as it appears in the document, with "
        "no reformatting (e.g. 'March 2015', 'Rev. 3/12/09').",
    ),
    FieldSpec(
        "issuing_agency",
        "The government body that issued or published the document. Give the "
        "most specific unit named (office, bureau, or program) together with "
        "its parent department, as printed.",
    ),
    FieldSpec(
        "performing_organization",
        "The outside organization that prepared the document under contract "
        "or grant, if one is credited and it differs from the issuing "
        "agency. Null otherwise.",
    ),
    FieldSpec(
        "document_type",
        "The kind of document this is.",
        choices=DOCUMENT_TYPES,
    ),
    FieldSpec(
        "report_number",
        "The agency-assigned report or publication number, as printed "
        "(e.g. 'EPA/600/R-15/047'). Null if none appears.",
    ),
    FieldSpec(
        "series",
        "The title and number of the series the document belongs to, if it "
        "is part of one. Null otherwise.",
    ),
    FieldSpec(
        "jurisdiction_level",
        "The level of government that issued the document.",
        choices=JURISDICTION_LEVELS,
    ),
    FieldSpec(
        "geographic_coverage",
        "Place names the document's content is specifically about (states, "
        "watersheds, regions, facilities). Empty list if national in scope "
        "or not geographically specific.",
        dtype="list",
    ),
]


class DocumentMetadata(BaseModel):
    """Shared output shape for both backends. Field-for-field with `FIELDS`
    above -- the assert below enforces that. Every field is optional because
    neither backend is guaranteed to find every attribute in every document.
    """

    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    publication_date: Optional[str] = None
    publication_date_raw: Optional[str] = None
    issuing_agency: Optional[str] = None
    performing_organization: Optional[str] = None
    document_type: Optional[str] = None
    report_number: Optional[str] = None
    series: Optional[str] = None
    jurisdiction_level: Optional[str] = None
    geographic_coverage: list[str] = Field(default_factory=list)

    @field_validator("authors", "geographic_coverage", mode="before")
    @classmethod
    def _coerce_list(cls, v):
        # Extractors occasionally return a single string instead of a list.
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v


# Guards against `FIELDS` and `DocumentMetadata` drifting apart -- a field
# added to only one of them would otherwise be silently dropped, since
# pydantic ignores unknown keys by default. (Not `extra="forbid"`: an LLM
# inventing a key shouldn't fail the whole document.)
assert set(DocumentMetadata.model_fields) == {f.name for f in FIELDS}, (
    "DocumentMetadata is out of sync with FIELDS"
)
