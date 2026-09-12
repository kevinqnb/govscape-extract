#!/usr/bin/env python3
"""Download a random sample of Govscape OCR documents and their source PDFs.

Randomly shuffles the list of OCR shard files (each a .jsonl of many
documents) and downloads them one at a time -- parsing out individual
documents as it goes -- until exactly `n_docs` documents have been collected.
Each document is written to `sample_ocr/<digest>.json`, and its source PDF is
downloaded to `sample_pdfs/<digest>.pdf`.

The sampling core lives in `data/sampling.py`, shared with `build_validation.py`.

Usage:
    uv run data/build_sample.py -n 10 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from sampling import build

DATA_DIR = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--n-docs", type=int, default=10, help="number of documents to sample")
    parser.add_argument("--seed", type=int, default=342, help="random seed for reproducible sampling")
    parser.add_argument("--ocr-dir", type=Path, default=DATA_DIR / "sample_ocr")
    parser.add_argument("--pdf-dir", type=Path, default=DATA_DIR / "sample_pdfs")
    parser.add_argument("--refresh-index", action="store_true", help="re-list the OCR bucket instead of using the cached shard index")
    args = parser.parse_args()

    load_dotenv(DATA_DIR / ".env")

    build(
        n_docs=args.n_docs,
        seed=args.seed,
        ocr_dir=args.ocr_dir,
        pdf_dir=args.pdf_dir,
        refresh_index=args.refresh_index,
    )


if __name__ == "__main__":
    main()
