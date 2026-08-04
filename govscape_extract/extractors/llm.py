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
from govscape_extract.schema import FIELDS, DocumentMetadata

DEFAULT_INSTRUCTIONS = """You are extracting bibliographic metadata from a US government document.

Extract the following fields:

{field_descriptions}

Rules:
- Extract only what the document itself supports. Do not guess a value that the text doesn't state or clearly imply.
- If a field can't be determined from the text, use null -- or an empty list for {list_fields}.
- For fields marked "one of", answer with exactly one label from that field's list, spelled exactly as shown. Use "other" if nothing fits.
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


def _field_block(field) -> str:
    """One field's entry in the instructions.

    Fields with `choices` get their labels *and* the definition of each one
    listed underneath, indented -- the labels alone ("guidance",
    "public_information") aren't self-explanatory enough to classify against
    reliably, and the definitions are already written in schema.py.
    """
    block = f"- {field.name} ({field.dtype}): {field.description}"
    if field.choices:
        block += "\n  One of:"
        block += "".join(f"\n    - {label}: {meaning}" for label, meaning in field.choices.items())
    return block


def default_instructions() -> str:
    return DEFAULT_INSTRUCTIONS.format(
        field_descriptions="\n".join(_field_block(f) for f in FIELDS),
        list_fields=" and ".join(f.name for f in FIELDS if f.dtype == "list"),
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
        # None => omit the parameter entirely. Some hosted reasoning models
        # reject any explicit temperature, including the 0.0 we'd otherwise
        # send for determinism. Default stays 0.0 so existing callers are
        # unaffected.
        temperature: float | None = 0.0,
        seed: int | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        extra_body: dict | None = None,
        max_retries: int | None = None,  # None => the SDK's own default (2)
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
            # `or None` matters: GOVSCAPE_LLM_BASE_URL is present-but-empty in
            # .env (that's how you select hosted OpenAI), and an empty string
            # would be passed through to the SDK as a real base URL.
            base_url=base_url or os.environ.get("GOVSCAPE_LLM_BASE_URL") or None,
            # vLLM and other local servers ignore the key but the SDK requires a non-empty string.
            api_key=api_key or os.environ.get("GOVSCAPE_LLM_API_KEY") or "EMPTY",
            **({} if max_retries is None else {"max_retries": max_retries}),
        )

    def extract(self, text: str) -> DocumentMetadata:
        prompt = build_prompt(text, self.instructions, self.query)
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
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
            **kwargs,
        )
        self.last_usage = response.usage.model_dump() if response.usage else None
        self.last_finish_reason = response.choices[0].finish_reason
        data = json.loads(response.choices[0].message.content)
        return DocumentMetadata.model_validate(data)
