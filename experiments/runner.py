"""CLI: run one configured model's extraction over a document set, with
per-document + run-level timing, writing a fully reproducible run directory.

    uv run -m experiments.runner --model-key gpt-oss-120b  --run-name baseline
    uv run -m experiments.runner --model-key qwen3-0.6b    --run-name baseline
    uv run -m experiments.runner --model-key gliner2-base  --run-name baseline

Reuses govscape_extract.documents.{load_document, extraction_window} and the
existing MetadataExtractor backends unmodified; writes
metadata/<digest>.json in exactly the shape govscape_extract.cli already
writes today.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from govscape_extract.documents import extraction_window, load_document

from experiments.config import (
    DEFAULT_EXPERIMENT,
    EXPERIMENT_REGISTRY,
    MODEL_REGISTRY,
    GlinerModelConfig,
)
from experiments.runtime import RunManifest, git_sha, installed_versions
from experiments.serving import endpoint_for

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEVANT_PACKAGES = ["gliner2", "torch", "transformers", "openai", "httpx", "govscape"]


def _run_id(run_name: str, model_key: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{run_name}__{model_key}__{ts}"


def _percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def _summarize_timing(
    model_key: str,
    backend: str,
    records: list[dict],
    model_load_seconds: Optional[float],
    serving_startup_seconds: Optional[float],
) -> dict:
    ok = [r for r in records if r["error"] is None]
    wall = [r["wall_seconds"] for r in ok]
    prompt = [r["prompt_tokens"] for r in ok if r["prompt_tokens"] is not None]
    completion = [r["completion_tokens"] for r in ok if r["completion_tokens"] is not None]
    total_tok = [r["total_tokens"] for r in ok if r["total_tokens"] is not None]
    total_wall = sum(wall) if wall else 0.0
    return {
        "model_key": model_key,
        "backend": backend,
        "n_documents": len(records),
        "n_errors": len(records) - len(ok),
        "model_load_seconds": model_load_seconds,
        "serving_startup_seconds": serving_startup_seconds,
        "wall_seconds": {
            "mean": statistics.mean(wall) if wall else None,
            "median": statistics.median(wall) if wall else None,
            "p90": _percentile(wall, 90) if wall else None,
            "p99": _percentile(wall, 99) if wall else None,
            "min": min(wall) if wall else None,
            "max": max(wall) if wall else None,
            "stdev": statistics.stdev(wall) if len(wall) > 1 else 0.0,
            "total": total_wall,
        },
        "tokens": {
            "mean_prompt": statistics.mean(prompt) if prompt else None,
            "mean_completion": statistics.mean(completion) if completion else None,
            "mean_total": statistics.mean(total_tok) if total_tok else None,
        },
        "throughput_docs_per_sec": (len(wall) / total_wall) if wall and total_wall > 0 else None,
    }


def _process_documents(
    extractor,
    docs: list[Path],
    max_pages: int,
    max_chars: int,
    metadata_dir: Path,
    timing_dir: Path,
    manifest: RunManifest,
    skip_warmup: bool,
) -> None:
    # One untimed warmup call so cold-start cost (model load / server JIT /
    # first-request overhead) doesn't land on document #1's timing.
    if docs and not skip_warmup:
        try:
            warm_doc = load_document(docs[0])
            warm_text = extraction_window(warm_doc, max_pages=max_pages, max_chars=max_chars)
            extractor.extract(warm_text)
        except Exception as e:
            print(f"  warmup call failed (continuing): {e}")

    digests = []
    n_errors = 0
    timing_records = []
    for path in docs:
        doc = load_document(path)
        digest = doc.get("digest", path.stem)
        digests.append(digest)
        text = extraction_window(doc, max_pages=max_pages, max_chars=max_chars)
        record = {
            "digest": digest,
            "wall_seconds": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "finish_reason": None,
            "error": None,
        }
        t0 = time.monotonic()
        try:
            result = extractor.extract(text)
            record["wall_seconds"] = time.monotonic() - t0
            usage = getattr(extractor, "last_usage", None) or {}
            record["prompt_tokens"] = usage.get("prompt_tokens")
            record["completion_tokens"] = usage.get("completion_tokens")
            record["total_tokens"] = usage.get("total_tokens")
            record["finish_reason"] = getattr(extractor, "last_finish_reason", None)
            (metadata_dir / f"{digest}.json").write_text(result.model_dump_json(indent=2))
            print(f"  {digest}: {result.title!r} ({record['wall_seconds']:.2f}s)")
        except Exception as e:
            # A single bad response shouldn't discard the rest of the run.
            record["wall_seconds"] = time.monotonic() - t0
            record["error"] = str(e)
            n_errors += 1
            (metadata_dir / f"{digest}.json").write_text(json.dumps(None))
            print(f"  {digest}: ERROR {e}")
        timing_records.append(record)
        (timing_dir / f"{digest}.json").write_text(json.dumps(record, indent=2))

    manifest.n_documents = len(docs)
    manifest.digests = digests
    manifest.n_errors = n_errors

    summary = _summarize_timing(
        manifest.model_key,
        manifest.backend,
        timing_records,
        manifest.model_load_seconds,
        manifest.serving_startup_seconds,
    )
    (timing_dir.parent / "timing_summary.json").write_text(json.dumps(summary, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-key", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--run-name", required=True, help="Human label, becomes part of the run_id")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT.name, choices=sorted(EXPERIMENT_REGISTRY))
    parser.add_argument("--input-dir", type=Path, default=None, help="Overrides the experiment's input_dir")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N documents")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--base-url",
        default=None,
        help="Force this OpenAI-compatible endpoint, skipping the model's base_url_env lookup / LocalVLLMServer entirely",
    )
    parser.add_argument("--seed", type=int, default=None, help="Overrides the model config's seed")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and print the manifest preview, run nothing")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    experiment = EXPERIMENT_REGISTRY[args.experiment]

    input_dir = args.input_dir or (REPO_ROOT / experiment.input_dir)
    limit = args.limit if args.limit is not None else experiment.limit
    max_pages = args.max_pages if args.max_pages is not None else experiment.max_pages
    max_chars = args.max_chars if args.max_chars is not None else experiment.max_chars
    output_root = args.output_root or (REPO_ROOT / experiment.output_root)

    model_config = MODEL_REGISTRY[args.model_key]
    backend = "gliner" if isinstance(model_config, GlinerModelConfig) else "llm"

    docs = sorted(input_dir.glob("*.json"))
    if limit is not None:
        docs = docs[:limit]
    if not docs:
        raise SystemExit(f"No documents found in {input_dir}")

    run_id = _run_id(args.run_name, model_config.key)
    run_dir = output_root / run_id
    metadata_dir = run_dir / "metadata"
    timing_dir = run_dir / "timing"

    seed = args.seed if args.seed is not None else getattr(model_config, "seed", None)

    manifest = RunManifest(
        run_id=run_id,
        experiment_name=experiment.name,
        model_key=model_config.key,
        backend=backend,
        started_at=datetime.now(timezone.utc).isoformat(),
        input_dir=str(input_dir),
        max_pages=max_pages,
        max_chars=max_chars,
        seed=seed,
        config={
            "model": dataclasses.asdict(model_config),
            "experiment": dataclasses.asdict(experiment),
        },
    )
    manifest.git_sha, manifest.git_dirty = git_sha(REPO_ROOT)
    manifest.package_versions = installed_versions(RELEVANT_PACKAGES)

    print(f"[{run_id}] {len(docs)} documents, backend={backend}")

    if args.dry_run:
        print(json.dumps(dataclasses.asdict(manifest), indent=2, default=str))
        return

    metadata_dir.mkdir(parents=True, exist_ok=True)
    timing_dir.mkdir(parents=True, exist_ok=True)

    if backend == "gliner":
        from govscape_extract.extractors.gliner import GlinerExtractor

        t0 = time.monotonic()
        extractor = GlinerExtractor(model_name=model_config.model_name, threshold=model_config.threshold)
        manifest.model_load_seconds = time.monotonic() - t0
        _process_documents(extractor, docs, max_pages, max_chars, metadata_dir, timing_dir, manifest, args.skip_warmup)
    else:
        from govscape_extract.extractors.llm import LLMExtractor

        with endpoint_for(model_config, base_url_override=args.base_url) as (base_url, startup_seconds):
            manifest.serving_startup_seconds = startup_seconds
            api_key = os.environ.get(model_config.api_key_env)
            extractor = LLMExtractor(
                model=model_config.model,
                base_url=base_url,
                api_key=api_key,
                temperature=model_config.temperature,
                seed=seed,
                max_tokens=model_config.max_tokens,
                top_p=model_config.top_p,
                extra_body=model_config.extra_body,
            )
            _process_documents(extractor, docs, max_pages, max_chars, metadata_dir, timing_dir, manifest, args.skip_warmup)

    manifest.finished_at = datetime.now(timezone.utc).isoformat()
    (run_dir / "manifest.json").write_text(json.dumps(dataclasses.asdict(manifest), indent=2, default=str))
    print(f"[{run_id}] done: {manifest.n_documents} documents, {manifest.n_errors} errors -> {run_dir}")


if __name__ == "__main__":
    main()
