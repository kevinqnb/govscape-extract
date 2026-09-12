# govscape-extract :bookmark_tabs:

[![attested by humans](https://github.com/kevinqnb/govscape-extract/actions/workflows/git-signoff.yml/badge.svg)](https://github.com/kevinqnb/govscape-extract/actions/workflows/git-signoff.yml)

Information extraction for annotating and metadata-filing [Govscape](https://govscape.net) documents.

## Setup

```bash
uv sync
cp .env.example .env   # fill in AWS credentials (if not already configured) and/or LLM backend config
```

## Data

`data/` has scripts to sample OCR text + source PDFs from the Govscape archive for local
development. See `data/README.md`.

## Metadata extraction

`govscape_extract/` extracts `title`, `authors`, `publication_date`, `agency`,
and `document_type` from OCR text (field definitions in `govscape_extract/schema.py`), via two
interchangeable backends:

```bash
# any OpenAI-compatible chat endpoint -- hosted OpenAI, or a local vLLM/Ollama server
uv run -m govscape_extract.cli --backend llm --base-url http://localhost:8000/v1 --model <model>

# GLiNER2 (https://github.com/fastino-ai/GLiNER2), a local span-extraction model
uv run -m govscape_extract.cli --backend gliner
```

Both read `data/sample_ocr/*.json` by default and write results to `data/extracted/`. The LLM
backend generates/normalizes text (dates, full names); GLiNER2 returns literal spans from the
document instead.
