# data/

Scripts for pulling a sample of Govscape OCR text and the PDFs it was
extracted from, for local development and testing.

## Layout

- `govscape_s3.py` -- shared S3 helpers (bucket names, listing, digest parsing).
- `sampling.py` -- the sampling pipeline (seeded shard shuffle, collect exactly
  `n_docs`, write, download PDFs), shared by both build scripts.
- `build_sample.py` -- samples `n_docs` random OCR documents for local dev and
  downloads their source PDFs in one pass.
- `build_validation.py` -- same, but for the held-out validation set:
  different seed, `validation_ocr/` / `validation_pdfs/`, and `--exclude-dir`
  (default `sample_ocr/`) so the two sets stay disjoint. Input to the
  frontier-model consensus panel -- see `experiments/README.md`
  ("Ground-truth dataset").
- `download_pdfs.py` -- downloads just the PDFs for a set of digests (or for
  everything already in `sample_ocr/`).
- `sample_ocr/` , `validation_ocr/` -- one JSON file per document, `<digest>.json`.
- `sample_pdfs/` , `validation_pdfs/` -- one PDF per document, `<digest>.pdf`.

The four `sample_*/` and `validation_*/` directories are gitignored -- run the
scripts to populate your own local copy. (`data/validation_gold/`, the fused
ground-truth labels, *is* committed.)

## Two buckets, two access levels

| | OCR text | Source PDFs |
|---|---|---|
| Bucket | `eot-pdf-archive` (requester pays) | source.coop `govscape/eota-pdf-archive` (public) |
| Credentials needed | Yes -- any authenticated AWS account | No |
| Who pays | You (small S3 transfer cost) | Nobody |

This means `download_pdfs.py` works out of the box for anyone, with zero AWS
setup. `build_sample.py`'s OCR step needs AWS credentials -- see the repo
root `.env.example`.

## Setup

See `.env.example` at the repo root -- only the AWS credentials there are
needed for this directory's scripts (only for `build_sample.py`'s OCR
download step; `download_pdfs.py` needs no credentials at all). `aws
configure` / SSO also work -- boto3 uses whatever it finds.

## Usage

```bash
# sample 10 random documents + their PDFs (reproducible with --seed)
uv run data/build_sample.py -n 10 --seed 42

# build the 100-doc validation set (disjoint from sample_ocr/)
uv run data/build_validation.py -n 100 --seed 771

# re-download PDFs for whatever's already in sample_ocr/
uv run data/download_pdfs.py

# or fetch specific PDFs by digest, no AWS account required
uv run data/download_pdfs.py N2W76N2BGZ62NJL7PMPGINIXCZYUSKEK
```

## How sampling works

OCR text is stored as ~500 shard files (`.jsonl`), each holding a variable
number of documents (one JSON object per line). To hit an exact `n_docs`
count without knowing shard sizes up front, `build_sample.py` shuffles the
full shard list (seeded, for reproducibility) and downloads shards one at a
time -- counting documents as it goes -- stopping as soon as `n_docs` is
reached (only taking as many documents as needed from the final shard).

The bucket holds ~450k shard files, so listing them all takes a few minutes.
`build_sample.py` caches that listing to `.ocr_shard_index.txt` (gitignored)
after the first run -- pass `--refresh-index` to force a re-list if the
bucket has grown and you want to sample from the newest shards.

Each document's `digest` (the ID that links it to its source PDF) isn't a
top-level field in the raw OCR JSON -- it's the filename in
`metadata["Source-File"]`, e.g.:

```
s3://ai2-oe-data-acquisition/eot-pdf-archive/PDFs/5E37QUBVDLHX5AB424W26JE3B4JQCFFM.pdf
```

`build_sample.py` parses this out and adds it as an explicit top-level
`digest` field on each document before writing it to `sample_ocr/`.
