"""Shared S3 access helpers for the Govscape OCR/PDF archive.

Two buckets are involved, with different access requirements:

- OCR text (``eot-pdf-archive``) is "requester pays": any authenticated AWS
  account can read it, but that account is billed for the transfer. Reads use
  the caller's default AWS credential chain (env vars, ``~/.aws/credentials``,
  SSO, etc. -- see ``.env.example``).
- Source PDFs (``eota-pdf-archive`` on source.coop) are public open data.
  Reads are unsigned and need no AWS account at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

OCR_BUCKET = "eot-pdf-archive"
OCR_PREFIX = "ai2-olmocr/"

PDF_BUCKET = "us-west-2.opendata.source.coop"
PDF_PREFIX = "govscape/eota-pdf-archive/pdfs/"


def ocr_client():
    return boto3.client("s3")


def pdf_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_ocr_shard_keys(s3=None) -> list[str]:
    """Return every OCR shard (.jsonl) key, sorted for reproducible shuffling."""
    s3 = s3 or ocr_client()
    paginator = s3.get_paginator("list_objects_v2")
    keys = [
        obj["Key"]
        for page in paginator.paginate(
            Bucket=OCR_BUCKET, Prefix=OCR_PREFIX, RequestPayer="requester"
        )
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".jsonl")
    ]
    return sorted(keys)


def iter_shard_documents(key: str, s3=None) -> Iterator[dict]:
    """Yield each document (one JSON object per line) from an OCR shard."""
    s3 = s3 or ocr_client()
    obj = s3.get_object(Bucket=OCR_BUCKET, Key=key, RequestPayer="requester")
    for line in obj["Body"].iter_lines():
        if line.strip():
            yield json.loads(line)


def document_digest(doc: dict) -> str:
    """Extract the archive digest from a document's Source-File path.

    e.g. "s3://ai2-oe-data-acquisition/eot-pdf-archive/PDFs/<DIGEST>.pdf"
    """
    source_file = doc["metadata"]["Source-File"]
    return Path(source_file).stem


def download_pdf(digest: str, dest_dir: Path, s3=None) -> Path | None:
    """Download the source PDF for `digest`, or return None if it's not in the archive.

    The OCR archive (`eot-pdf-archive`) and the public PDF mirror
    (`eota-pdf-archive`) aren't perfectly aligned -- a small fraction of OCR'd
    documents have no corresponding PDF. Treat that as expected (skip, don't
    crash the whole batch) but let any other error (auth, network, etc.)
    propagate.
    """
    s3 = s3 or pdf_client()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{digest}.pdf"
    try:
        s3.download_file(PDF_BUCKET, f"{PDF_PREFIX}{digest}.pdf", str(dest))
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "404":
            return None
        raise
    return dest
