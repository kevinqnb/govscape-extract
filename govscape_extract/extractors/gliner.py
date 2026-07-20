"""Extraction via GLiNER2 (https://github.com/fastino-ai/GLiNER2).

Unlike the LLM backend, GLiNER2 is a small span-extraction model: it locates
and returns literal substrings from the input rather than generating or
normalizing text. It won't infer an agency that isn't named, or reformat a
date -- it finds spans that match the field's learned meaning. `dtype="str"`
fields with `choices` (like `document_type`) run as classification instead,
matched against the given label strings.

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
            builder.field(f.name, dtype=f.dtype, choices=f.choices, description=f.description)
        return schema

    def extract(self, text: str) -> DocumentMetadata:
        result = self.model.extract(text, self._schema, threshold=self.threshold)
        instances = result.get(STRUCTURE_NAME, [])
        data = instances[0] if instances else {}
        return DocumentMetadata.model_validate(data)
