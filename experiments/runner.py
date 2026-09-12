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

from dotenv import load_dotenv

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


def _write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    """Rewritten after every document, not just at the end -- that's what
    makes a killed run resumable rather than an orphaned directory."""
    (run_dir / "manifest.json").write_text(json.dumps(dataclasses.asdict(manifest), indent=2, default=str))


def _completed_digests(metadata_dir: Path) -> set[str]:
    """Digests that already have a usable result on disk.

    A literal JSON `null` is what _process_documents writes for a failed
    extraction, so those are deliberately *not* counted as complete: a
    resume retries them, which is what you want after a transient API
    failure. A file that doesn't parse at all (truncated by a kill
    mid-write) is likewise treated as not done.
    """
    done = set()
    for path in metadata_dir.glob("*.json"):
        try:
            if json.loads(path.read_text()) is not None:
                done.add(path.stem)
        except json.JSONDecodeError:
            continue
    return done


def _load_timing_records(timing_dir: Path, digests: list[str]) -> list[dict]:
    """Timing for the run's whole scope, read back from disk.

    Reading from disk rather than accumulating in memory is what lets a
    resumed run's summary cover the documents processed by the *earlier*
    invocation too. Restricted to `digests` so a resume with a smaller
    --limit doesn't fold in records that are now out of scope.
    """
    records = []
    for digest in digests:
        path = timing_dir / f"{digest}.json"
        if not path.exists():
            continue
        try:
            records.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return records


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
    run_dir: Path,
    metadata_dir: Path,
    timing_dir: Path,
    manifest: RunManifest,
    skip_warmup: bool,
    completed: set[str],
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

    # Seeded from what a previous invocation left on disk, then kept current
    # in memory: re-globbing metadata/ after every document would be O(n^2)
    # file reads over a 1000-document run.
    done = set(completed)
    scope = set(manifest.digests)

    def sync_counts() -> None:
        # Called per-document *and* once after the loop: on a resume with
        # nothing left to do, every document is skipped and the loop body
        # never runs, which would otherwise leave these at their defaults
        # and report a finished run as 0/N extracted.
        manifest.n_completed = len(done & scope)
        # n_errors counts every in-scope document without a usable result,
        # including ones not yet attempted -- so on a partial run it reads as
        # "not done", which is what `finished_at: null` alongside it means.
        manifest.n_errors = manifest.n_documents - manifest.n_completed
        _write_manifest(run_dir, manifest)

    for path in docs:
        # Fast path: documents are named <digest>.json, so this skips a
        # resumed run's already-done work without reading (potentially
        # large) OCR files back off disk. The post-load check below is the
        # correctness backstop for any file whose stem isn't its digest.
        if path.stem in completed:
            continue
        doc = load_document(path)
        digest = doc.get("digest", path.stem)
        if digest != path.stem:
            # manifest.digests is built from filenames (see main()), and it's
            # evaluate.py's join key -- a document whose internal digest
            # disagrees would be written to a file no evaluation looks for.
            raise SystemExit(
                f"{path}: internal digest {digest!r} != filename stem {path.stem!r}. "
                "The run manifest keys documents by filename; rename the file or "
                "fix the digest field before running."
            )
        if digest in completed:
            continue
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
            done.add(digest)
            print(f"  {digest}: {result.title!r} ({record['wall_seconds']:.2f}s)")
        except Exception as e:
            # A single bad response shouldn't discard the rest of the run.
            record["wall_seconds"] = time.monotonic() - t0
            record["error"] = str(e)
            (metadata_dir / f"{digest}.json").write_text(json.dumps(None))
            done.discard(digest)  # a retried-then-failed document is not done
            print(f"  {digest}: ERROR {e}")
        (timing_dir / f"{digest}.json").write_text(json.dumps(record, indent=2))
        sync_counts()

    sync_counts()


def _write_timing_summary(run_dir: Path, timing_dir: Path, manifest: RunManifest) -> None:
    summary = _summarize_timing(
        manifest.model_key,
        manifest.backend,
        _load_timing_records(timing_dir, manifest.digests),
        manifest.model_load_seconds,
        manifest.serving_startup_seconds,
    )
    (run_dir / "timing_summary.json").write_text(json.dumps(summary, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-key", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--run-name", help="Human label, becomes part of the run_id. Required unless --resume.")
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="Continue an interrupted run in place instead of starting a new one: reuses that "
        "run directory and run_id, and skips documents that already have a non-null result. "
        "Documents whose previous attempt errored are retried.",
    )
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
    # Endpoint URLs and API keys live in the repo-root .env (see .env.example).
    # Without this, os.environ below silently misses them and LLMExtractor
    # falls back to the literal api_key "EMPTY" -> an opaque 401.
    load_dotenv(REPO_ROOT / ".env")

    args = build_arg_parser().parse_args()
    if bool(args.run_name) == bool(args.resume):
        raise SystemExit("Pass exactly one of --run-name (new run) or --resume RUN_ID (continue one).")
    experiment = EXPERIMENT_REGISTRY[args.experiment]

    output_root = args.output_root or (REPO_ROOT / experiment.output_root)
    model_config = MODEL_REGISTRY[args.model_key]
    backend = "gliner" if isinstance(model_config, GlinerModelConfig) else "llm"

    prior_manifest = None
    if args.resume:
        run_id = args.resume
        run_dir = output_root / run_id
        if not run_dir.is_dir():
            raise SystemExit(f"Cannot resume: {run_dir} does not exist.")
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(
                f"Cannot resume: {manifest_path} is missing. Runs started before --resume existed "
                "only wrote their manifest on clean exit, so an interrupted one can't be continued."
            )
        prior_manifest = json.loads(manifest_path.read_text())

        # Scope and windowing are properties of the *run*, so they're read
        # back from its manifest rather than re-derived from flags -- you
        # shouldn't have to remember the original --limit to continue.
        # An explicit flag that contradicts the manifest is an error, not an
        # override: blending two windowings into one run directory would
        # produce a reference set that no evaluation can interpret (the same
        # reasoning as evaluate.py's windowing abort).
        input_dir = Path(prior_manifest["input_dir"])
        max_pages = prior_manifest["max_pages"]
        max_chars = prior_manifest["max_chars"]
        limit = prior_manifest["n_documents"]
        for flag, given, was in [
            ("--model-key", model_config.key, prior_manifest["model_key"]),
            ("--input-dir", None if args.input_dir is None else str(args.input_dir), None if args.input_dir is None else prior_manifest["input_dir"]),
            ("--max-pages", args.max_pages, None if args.max_pages is None else max_pages),
            ("--max-chars", args.max_chars, None if args.max_chars is None else max_chars),
            ("--limit", args.limit, None if args.limit is None else limit),
        ]:
            if given is not None and given != was:
                raise SystemExit(
                    f"Cannot resume {run_id}: it ran with {flag}={was!r}, but this invocation "
                    f"passes {flag}={given!r}. Drop the flag to reuse the run's own setting, "
                    "or start a new run."
                )
        if prior_manifest.get("finished_at"):
            print(f"[{run_id}] note: this run already finished; re-running only its failed documents.")
    else:
        input_dir = args.input_dir or (REPO_ROOT / experiment.input_dir)
        limit = args.limit if args.limit is not None else experiment.limit
        max_pages = args.max_pages if args.max_pages is not None else experiment.max_pages
        max_chars = args.max_chars if args.max_chars is not None else experiment.max_chars
        run_id = _run_id(args.run_name, model_config.key)
        run_dir = output_root / run_id

    docs = sorted(input_dir.glob("*.json"))
    if limit is not None:
        docs = docs[:limit]
    if not docs:
        raise SystemExit(f"No documents found in {input_dir}")
    if prior_manifest:
        # Identity, not just count: a regenerated sample directory can hold
        # exactly as many documents as the original run covered and share
        # none of them. Comparing lengths would pass, and the run would then
        # be rewritten to describe documents it never processed.
        was = set(prior_manifest["digests"])
        now = {p.stem for p in docs}
        if was != now:
            raise SystemExit(
                f"Cannot resume {run_id}: its {len(was)} documents are not the ones currently in "
                f"{input_dir} ({len(now - was)} new, {len(was - now)} gone). The input set changed "
                "underneath the run -- resuming would blend two different document sets into one "
                "run directory. Start a new run instead."
            )

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
        # Populated up front rather than accumulated during processing, so a
        # manifest written mid-run still describes the run's full intended
        # scope. Safe because OCR files are named <digest>.json -- the loop
        # asserts that, rather than trusting it silently.
        n_documents=len(docs),
        digests=[p.stem for p in docs],
    )
    manifest.git_sha, manifest.git_dirty = git_sha(REPO_ROOT)
    manifest.package_versions = installed_versions(RELEVANT_PACKAGES)

    completed: set[str] = set()
    if prior_manifest:
        # The prior manifest is authoritative about what this run *is*: its
        # identity, when it started, the environment it started in, and which
        # documents it covers. Recomputing any of that from current state
        # would let a resume silently redefine the run it's continuing.
        manifest.started_at = prior_manifest["started_at"]
        manifest.digests = prior_manifest["digests"]
        manifest.n_documents = prior_manifest["n_documents"]
        resumed_git_sha, resumed_git_dirty = manifest.git_sha, manifest.git_dirty
        resumed_versions = manifest.package_versions
        manifest.git_sha = prior_manifest.get("git_sha")
        manifest.git_dirty = prior_manifest.get("git_dirty")
        manifest.package_versions = prior_manifest.get("package_versions", {})
        # This invocation's own environment goes in the resume event, so a
        # sha/version drift between the original run and the resume stays
        # visible instead of overwriting the original's record.
        manifest.resume_events = list(prior_manifest.get("resume_events", []))
        manifest.resume_events.append(
            {
                "resumed_at": datetime.now(timezone.utc).isoformat(),
                "git_sha": resumed_git_sha,
                "git_dirty": resumed_git_dirty,
                "package_versions": resumed_versions,
            }
        )
        completed = _completed_digests(metadata_dir) & set(manifest.digests)

    print(f"[{run_id}] {len(docs)} documents, backend={backend}")
    if args.resume:
        print(f"  resuming: {len(completed)} already done, {len(docs) - len(completed)} to go")
        if len(completed) == len(docs):
            print("  nothing left to do.")

    if args.dry_run:
        print(json.dumps(dataclasses.asdict(manifest), indent=2, default=str))
        return

    metadata_dir.mkdir(parents=True, exist_ok=True)
    timing_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(run_dir, manifest)

    if backend == "gliner":
        from govscape_extract.extractors.gliner import GlinerExtractor

        t0 = time.monotonic()
        extractor = GlinerExtractor(model_name=model_config.model_name, threshold=model_config.threshold)
        manifest.model_load_seconds = time.monotonic() - t0
        _process_documents(extractor, docs, max_pages, max_chars, run_dir, metadata_dir, timing_dir, manifest, args.skip_warmup, completed)
    else:
        from govscape_extract.extractors.llm import LLMExtractor

        with endpoint_for(model_config, base_url_override=args.base_url) as (base_url, startup_seconds):
            manifest.serving_startup_seconds = startup_seconds
            api_key = os.environ.get(model_config.api_key_env)
            if not api_key:
                # Self-served vLLM ignores the key, but a hosted API doesn't:
                # fail here with the variable name rather than at request time
                # with a 401 that doesn't say what's missing.
                raise SystemExit(
                    f"{model_config.key}: ${model_config.api_key_env} is not set. "
                    f"Add it to {REPO_ROOT / '.env'} (see .env.example). "
                    "Local vLLM servers ignore the value, so any non-empty "
                    "string works for those."
                )
            print(f"  endpoint: {base_url or 'OpenAI SDK default'}")
            extractor = LLMExtractor(
                model=model_config.model,
                base_url=base_url,
                api_key=api_key,
                temperature=model_config.temperature,
                seed=seed,
                max_tokens=model_config.max_tokens,
                top_p=model_config.top_p,
                extra_body=model_config.extra_body,
                max_retries=model_config.max_retries,
                response_format=model_config.response_format,
            )
            # Read it off the constructed client rather than from `base_url`
            # above: LLMExtractor still falls back to $GOVSCAPE_LLM_BASE_URL,
            # so the yielded value isn't necessarily where requests go.
            manifest.resolved_base_url = str(extractor.client.base_url)
            _process_documents(extractor, docs, max_pages, max_chars, run_dir, metadata_dir, timing_dir, manifest, args.skip_warmup, completed)

    # Written even when nothing was processed (a fully-complete --resume), so
    # the summary and finished_at reflect the run's final state either way.
    _write_timing_summary(run_dir, timing_dir, manifest)
    manifest.finished_at = datetime.now(timezone.utc).isoformat()
    _write_manifest(run_dir, manifest)
    print(
        f"[{run_id}] done: {manifest.n_completed}/{manifest.n_documents} documents extracted, "
        f"{manifest.n_errors} missing/errored -> {run_dir}"
    )
    if manifest.n_errors:
        print(f"  retry just those with: --model-key {model_config.key} --resume {run_id}")


if __name__ == "__main__":
    main()
