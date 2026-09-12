"""Shared sampling core for `build_sample.py` and `build_validation.py`.

Both scripts do the same thing -- shuffle the OCR shard list (seeded), pull
documents one shard at a time until an exact count is reached, write each to
`<ocr_dir>/<digest>.json`, and download the matching source PDFs -- so that
logic lives here once rather than being copy-pasted. The only real difference
between the two callers is the output directories, the seed, and (for the
validation set) a set of digests to exclude so it stays disjoint from an
existing sample.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

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


def digests_in_dir(ocr_dir: Path) -> set[str]:
    """Digests already written to an OCR sample directory (files are
    `<digest>.json`, but read the internal digest field too in case a file
    was renamed)."""
    digests: set[str] = set()
    if not ocr_dir.is_dir():
        return digests
    for path in sorted(ocr_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError:
            digests.add(path.stem)
            continue
        digests.add(doc.get("digest", path.stem))
    return digests


def sample_documents(
    n_docs: int,
    seed: int | None,
    refresh_index: bool,
    exclude_digests: Iterable[str] = (),
) -> list[dict]:
    """Collect exactly `n_docs` OCR documents, skipping any whose digest is in
    `exclude_digests` (and skipping duplicates within the sample itself)."""
    exclude = set(exclude_digests)
    s3 = ocr_client()
    keys = cached_shard_keys(refresh_index)
    random.Random(seed).shuffle(keys)

    docs: list[dict] = []
    seen: set[str] = set()
    for key in keys:
        if len(docs) >= n_docs:
            break
        added = 0
        for doc in iter_shard_documents(key, s3):
            if len(docs) >= n_docs:
                break
            digest = document_digest(doc)
            if digest in exclude or digest in seen:
                continue
            seen.add(digest)
            docs.append(doc)
            added += 1
        if added:
            print(f"  {key}: {len(docs)}/{n_docs} documents collected")

    if len(docs) < n_docs:
        raise RuntimeError(
            f"Only found {len(docs)} documents across the entire archive "
            f"(requested {n_docs}, excluding {len(exclude)} digests)."
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


def download_pdfs(digests: list[str], pdf_dir: Path) -> list[str]:
    """Download each digest's source PDF, returning the digests with no matching PDF."""
    s3 = pdf_client()
    missing = []
    for digest in digests:
        if download_pdf(digest, pdf_dir, s3) is None:
            print(f"  no source PDF found for {digest}, skipping")
            missing.append(digest)
        else:
            print(f"  downloaded {digest}.pdf")
    return missing


def build(
    n_docs: int,
    seed: int | None,
    ocr_dir: Path,
    pdf_dir: Path,
    refresh_index: bool = False,
    exclude_digests: Iterable[str] = (),
) -> None:
    """The whole pipeline, shared by both CLI entry points."""
    exclude = set(exclude_digests)
    if exclude:
        print(f"Excluding {len(exclude)} already-sampled digests.")

    print(f"Sampling {n_docs} OCR documents (seed={seed})...")
    docs = sample_documents(n_docs, seed, refresh_index, exclude)

    print(f"Writing {len(docs)} documents to {ocr_dir}/...")
    digests = write_ocr_sample(docs, ocr_dir)

    print(f"Downloading {len(digests)} source PDFs to {pdf_dir}/...")
    missing = download_pdfs(digests, pdf_dir)

    if missing:
        print(
            f"Done. {len(missing)}/{len(digests)} documents have no source PDF "
            f"in the archive (OCR JSON was still kept):"
        )
        for digest in missing:
            print(f"  {digest}")
    else:
        print("Done.")
