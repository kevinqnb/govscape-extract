"""Extraction via GLiNER2 (https://github.com/fastino-ai/GLiNER2).

Unlike the LLM backend, GLiNER2 is a small span-extraction model: it locates
and returns literal substrings from the input rather than generating or
normalizing text. It won't infer an agency that isn't named, or reformat a
date -- it finds spans that match the field's learned meaning. `dtype="str"`
fields with `choices` (`document_type` and `jurisdiction_level`) run as
classification instead, matched against the given label strings.

Two consequences of that for the current schema, both expected rather than
fixable here:

- `builder.field()` takes `choices` as a bare list of labels, so the
  label definitions in schema.py's `DOCUMENT_TYPES`/`JURISDICTION_LEVELS`
  reach the LLM prompt but not this backend, which sees only the labels.
- `publication_date` asks for ISO-8601 normalization, which a span model
  structurally cannot do; expect it to return the date as printed and to
  score accordingly. `publication_date_raw` is the field that actually
  matches this backend's behavior.

Watch the input budget here in a way the LLM backend doesn't need to: the
schema's field names, descriptions, and choice labels are encoded *alongside*
the document text, and the schema recently grew to 11 fields with 23 choice
labels while `documents.DEFAULT_MAX_PAGES` went 2 -> 3. gliner2's `max_len`
is left unset (no explicit truncation), so the effective ceiling is the
encoder's own position limit and anything past it is dropped silently. If
recall on later fields degrades, suspect the window before the model.

Requires the `local` extra (`uv add 'gliner2[local]'`), which pulls in torch
and transformers.
"""

from __future__ import annotations

from govscape_extract.extractors.base import MetadataExtractor
from govscape_extract.schema import FIELDS, DocumentMetadata

STRUCTURE_NAME = "document_metadata"
DEFAULT_MODEL = "fastino/gliner2-base-v1"


class GlinerExtractor(MetadataExtractor):
    def __init__(self, model_name: str = DEFAULT_MODEL, threshold: float = 0.5):
        from gliner2 import GLiNER2

        self.model = GLiNER2.from_pretrained(model_name)
        self.threshold = threshold
        self._schema = self._build_schema()

    def _build_schema(self):
        schema = self.model.create_schema()
        builder = schema.structure(STRUCTURE_NAME)
        for f in FIELDS:
            # choice_values, not choices: the builder wants a list of labels,
            # not the label -> definition mapping schema.py stores.
            builder.field(f.name, dtype=f.dtype, choices=f.choice_values, description=f.description)
        return schema

    def extract(self, text: str) -> DocumentMetadata:
        result = self.model.extract(text, self._schema, threshold=self.threshold)
        instances = result.get(STRUCTURE_NAME, [])
        data = instances[0] if instances else {}
        return DocumentMetadata.model_validate(data)
