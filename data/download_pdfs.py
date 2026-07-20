#!/usr/bin/env python3
"""Download source PDFs by digest from the public Govscape PDF archive.

No AWS account needed -- this bucket is public open data.

Examples:
    # download PDFs for every parsed document in sample_ocr/ (the default)
    uv run data/download_pdfs.py

    # download PDFs for specific digests
    uv run data/download_pdfs.py N2W76N2BGZ62NJL7PMPGINIXCZYUSKEK 5E37QUBVDLHX5AB424W26JE3B4JQCFFM
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from govscape_s3 import download_pdf, pdf_client

DATA_DIR = Path(__file__).parent


def digests_from_ocr_dir(ocr_dir: Path) -> list[str]:
    digests = []
    for path in sorted(ocr_dir.glob("*.json")):
        doc = json.loads(path.read_text())
        digests.append(doc.get("digest", path.stem))
    return digests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("digests", nargs="*", help="specific digests to download")
    parser.add_argument("--ocr-dir", type=Path, default=DATA_DIR / "sample_ocr", help="fetch digests from every *.json here when no digests are given")
    parser.add_argument("--pdf-dir", type=Path, default=DATA_DIR / "sample_pdfs")
    args = parser.parse_args()

    digests = args.digests or digests_from_ocr_dir(args.ocr_dir)
    if not digests:
        raise SystemExit(f"No digests given and none found in {args.ocr_dir}/")

    s3 = pdf_client()
    for digest in digests:
        download_pdf(digest, args.pdf_dir, s3)
        print(f"  downloaded {digest}.pdf")


if __name__ == "__main__":
    main()
