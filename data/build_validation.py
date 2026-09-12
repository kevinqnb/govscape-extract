#!/usr/bin/env python3
"""Build a held-out validation set of Govscape OCR documents.

Identical sampling mechanics to `build_sample.py` (see `data/sampling.py`),
with two deliberate differences:

- a different default seed, and separate output directories
  (`validation_ocr/` / `validation_pdfs/`), so the validation set is its own
  thing;
- `--exclude-dir` (default: `sample_ocr/`), so no document that already
  appears in the development sample is drawn into the validation set.
  Overlap would contaminate any accuracy comparison between a model's runs
  on the two sets.

This set is the input to the frontier-model consensus panel that produces
the ground-truth dataset -- see `experiments/README.md` (fuse.py) and
`experiments/config.py`'s "validation" experiment.

Usage:
    uv run data/build_validation.py -n 100 --seed 771
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from sampling import build, digests_in_dir

DATA_DIR = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--n-docs", type=int, default=100, help="number of documents to sample")
    parser.add_argument("--seed", type=int, default=771, help="random seed for reproducible sampling (distinct from build_sample.py's)")
    parser.add_argument("--ocr-dir", type=Path, default=DATA_DIR / "validation_ocr")
    parser.add_argument("--pdf-dir", type=Path, default=DATA_DIR / "validation_pdfs")
    parser.add_argument(
        "--exclude-dir",
        type=Path,
        action="append",
        default=None,
        help="OCR sample dir whose digests must NOT appear in the validation set "
        "(repeatable; defaults to sample_ocr/)",
    )
    parser.add_argument("--refresh-index", action="store_true", help="re-list the OCR bucket instead of using the cached shard index")
    args = parser.parse_args()

    load_dotenv(DATA_DIR / ".env")

    exclude_dirs = args.exclude_dir if args.exclude_dir is not None else [DATA_DIR / "sample_ocr"]
    exclude_digests: set[str] = set()
    for d in exclude_dirs:
        found = digests_in_dir(d)
        print(f"  {d}: {len(found)} digests to exclude")
        exclude_digests |= found

    build(
        n_docs=args.n_docs,
        seed=args.seed,
        ocr_dir=args.ocr_dir,
        pdf_dir=args.pdf_dir,
        refresh_index=args.refresh_index,
        exclude_digests=exclude_digests,
    )


if __name__ == "__main__":
    main()
