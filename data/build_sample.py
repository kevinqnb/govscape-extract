#!/usr/bin/env python3
"""Download a random sample of Govscape OCR documents and their source PDFs.

Randomly shuffles the list of OCR shard files (each a .jsonl of many
documents) and downloads them one at a time -- parsing out individual
documents as it goes -- until exactly `n_docs` documents have been collected.
Each document is written to `sample_ocr/<digest>.json`, and its source PDF is
downloaded to `sample_pdfs/<digest>.pdf`.

Usage:
    uv run data/build_sample.py -n 10 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from govscape_s3 import (
    document_digest,
    download_pdf,
    iter_shard_documents,
    list_ocr_shard_keys,
    ocr_client,
    pdf_client,
)

DATA_DIR = Path(__file__).parent
SHARD_INDEX_CACHE = DATA_DIR / ".ocr_shard_index.txt"


def cached_shard_keys(refresh: bool = False) -> list[str]:
    """List of OCR shard keys, cached locally.

    The bucket currently holds ~450k shard files, so a full listing takes
    minutes. That cost shouldn't be paid on every run just to sample a
    handful of documents -- cache it and only re-list on request.
    """
    if not refresh and SHARD_INDEX_CACHE.exists():
        return SHARD_INDEX_CACHE.read_text().splitlines()

    print("Listing OCR shards (first run, or --refresh-index) -- this can take a few minutes...")
    keys = list_ocr_shard_keys(ocr_client())
    SHARD_INDEX_CACHE.write_text("\n".join(keys))
    return keys


def sample_documents(n_docs: int, seed: int | None, refresh_index: bool) -> list[dict]:
    import random

    s3 = ocr_client()
    keys = cached_shard_keys(refresh_index)
    random.Random(seed).shuffle(keys)

    docs = []
    for key in keys:
        if len(docs) >= n_docs:
            break
        shard_docs = list(iter_shard_documents(key, s3))
        docs.extend(shard_docs[: n_docs - len(docs)])
        print(f"  {key}: {len(docs)}/{n_docs} documents collected")

    if len(docs) < n_docs:
        raise RuntimeError(
            f"Only found {len(docs)} documents across the entire archive "
            f"(requested {n_docs})."
        )
    return docs


def write_ocr_sample(docs: list[dict], ocr_dir: Path) -> list[str]:
    ocr_dir.mkdir(parents=True, exist_ok=True)
    digests = []
    for doc in docs:
        digest = document_digest(doc)
        doc["digest"] = digest
        (ocr_dir / f"{digest}.json").write_text(json.dumps(doc, indent=2))
        digests.append(digest)
    return digests


def download_pdfs(digests: list[str], pdf_dir: Path) -> None:
    s3 = pdf_client()
    for digest in digests:
        download_pdf(digest, pdf_dir, s3)
        print(f"  downloaded {digest}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--n-docs", type=int, default=10, help="number of documents to sample")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible sampling")
    parser.add_argument("--ocr-dir", type=Path, default=DATA_DIR / "sample_ocr")
    parser.add_argument("--pdf-dir", type=Path, default=DATA_DIR / "sample_pdfs")
    parser.add_argument("--refresh-index", action="store_true", help="re-list the OCR bucket instead of using the cached shard index")
    args = parser.parse_args()

    load_dotenv(DATA_DIR / ".env")

    print(f"Sampling {args.n_docs} OCR documents (seed={args.seed})...")
    docs = sample_documents(args.n_docs, args.seed, args.refresh_index)

    print(f"Writing {len(docs)} documents to {args.ocr_dir}/...")
    digests = write_ocr_sample(docs, args.ocr_dir)

    print(f"Downloading {len(digests)} source PDFs to {args.pdf_dir}/...")
    download_pdfs(digests, args.pdf_dir)

    print("Done.")


if __name__ == "__main__":
    main()
