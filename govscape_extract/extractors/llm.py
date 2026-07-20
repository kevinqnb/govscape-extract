"""General-purpose extraction via an OpenAI-compatible chat completions API.

Works against any endpoint that speaks the OpenAI protocol: hosted OpenAI,
or a local model served through vLLM's `--api` mode (or any other
OpenAI-compatible server). Swap between them with `base_url`/`model`
(constructor args or the `GOVSCAPE_LLM_*` env vars below) -- no code
changes needed.
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

from govscape_extract.extractors.base import MetadataExtractor
from govscape_extract.schema import DOCUMENT_TYPES, FIELDS, DocumentMetadata

DEFAULT_INSTRUCTIONS = """You are extracting bibliographic metadata from a US government document.

Extract the following fields:
{field_descriptions}

Rules:
- If a field can't be determined from the text, use null (or [] for authors).
- document_type must be the closest match from this list: {doc_types}. If nothing fits, use "Other".
- Respond with a single JSON object with exactly these keys: {field_names}. No prose, no markdown fences."""

DEFAULT_QUERY = (
    "Please extract the attributes defined in the instructions from the "
    "document above, and return them as a single JSON object."
)

PROMPT_TEMPLATE = """{instructions}

DOCUMENT TEXT:
\"\"\"
{document_text}
\"\"\"

{query}"""


def default_instructions() -> str:
    return DEFAULT_INSTRUCTIONS.format(
        field_descriptions="\n".join(f"- {f.name} ({f.dtype}): {f.description}" for f in FIELDS),
        doc_types=", ".join(DOCUMENT_TYPES),
        field_names=", ".join(f.name for f in FIELDS),
    )


def build_prompt(document_text: str, instructions: str | None = None, query: str | None = None) -> str:
    return PROMPT_TEMPLATE.format(
        instructions=instructions or default_instructions(),
        document_text=document_text,
        query=query or DEFAULT_QUERY,
    )


class LLMExtractor(MetadataExtractor):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        instructions: str | None = None,
        query: str | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        extra_body: dict | None = None,
    ):
        self.model = model or os.environ.get("GOVSCAPE_LLM_MODEL", "gpt-4o-mini")
        self.instructions = instructions
        self.query = query
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.extra_body = extra_body
        self.client = OpenAI(
            base_url=base_url or os.environ.get("GOVSCAPE_LLM_BASE_URL"),
            # vLLM and other local servers ignore the key but the SDK requires a non-empty string.
            api_key=api_key or os.environ.get("GOVSCAPE_LLM_API_KEY", "EMPTY"),
        )

    def extract(self, text: str) -> DocumentMetadata:
        prompt = build_prompt(text, self.instructions, self.query)
        kwargs = {}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=self.temperature,
            **kwargs,
        )
        self.last_usage = response.usage.model_dump() if response.usage else None
        self.last_finish_reason = response.choices[0].finish_reason
        data = json.loads(response.choices[0].message.content)
        return DocumentMetadata.model_validate(data)
