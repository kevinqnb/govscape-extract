#!/usr/bin/env python3
"""Extract title/authors/dates/agency/document-type metadata from OCR text.

Two interchangeable backends, both producing the same `DocumentMetadata`:

    uv run -m govscape_extract.cli --backend llm
    uv run -m govscape_extract.cli --backend gliner

`--backend llm` talks to any OpenAI-compatible chat completions endpoint --
hosted OpenAI by default, or a local vLLM server via `--base-url`.
`--backend gliner` runs a local GLiNER2 model (downloaded from Hugging Face
on first use).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from govscape_extract.documents import extraction_window, load_document
from govscape_extract.extractors.base import MetadataExtractor

DATA_DIR = Path(__file__).parent.parent / "data"


def build_extractor(args: argparse.Namespace) -> MetadataExtractor:
    if args.backend == "llm":
        from govscape_extract.extractors.llm import LLMExtractor

        return LLMExtractor(model=args.model, base_url=args.base_url, api_key=args.api_key)
    else:
        from govscape_extract.extractors.gliner import GlinerExtractor

        return GlinerExtractor(model_name=args.gliner_model, threshold=args.threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["llm", "gliner"], required=True)
    parser.add_argument("--input-dir", type=Path, default=DATA_DIR / "sample_ocr", help="directory of <digest>.json OCR documents")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "extracted", help="where to write <digest>.json metadata results")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N documents")
    parser.add_argument("--max-pages", type=int, default=3, help="how many leading pages of OCR text to feed the extractor")

    llm_group = parser.add_argument_group("llm backend")
    llm_group.add_argument("--model", help="chat model name (default: $GOVSCAPE_LLM_MODEL or gpt-4o-mini)")
    llm_group.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1 for vLLM (default: $GOVSCAPE_LLM_BASE_URL, or OpenAI's API)")
    llm_group.add_argument("--api-key", help="default: $GOVSCAPE_LLM_API_KEY")

    gliner_group = parser.add_argument_group("gliner backend")
    gliner_group.add_argument("--gliner-model", default="fastino/gliner2-base-v1", help="Hugging Face model repo")
    gliner_group.add_argument("--threshold", type=float, default=0.5, help="span/classification confidence threshold")

    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.json"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No documents found in {args.input_dir}/")

    extractor = build_extractor(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        doc = load_document(path)
        text = extraction_window(doc, max_pages=args.max_pages)
        metadata = extractor.extract(text)

        digest = doc.get("digest", path.stem)
        out_path = args.output_dir / f"{digest}.json"
        out_path.write_text(metadata.model_dump_json(indent=2))
        print(f"  {digest}: {metadata.title!r}")

    print(f"Done. Wrote {len(paths)} results to {args.output_dir}/")


if __name__ == "__main__":
    main()
