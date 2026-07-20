"""CLI: score a candidate model's run against a ground-truth run.

There's no gold-labeled dataset, so accuracy is measured by proxy: treat the
ground-truth model's (gpt-oss-120b) output as truth and fuzzy-match a
candidate model's output against it, per document, per field (see
experiments/similarity.py for the comparators and their rationale).

    uv run -m experiments.evaluate --truth-run <run_id_or_path> --candidate-run <run_id_or_path>

Writes experiments/evaluations/<candidate_run_id>__vs__<truth_run_id>/
{evaluation.json, per_document.csv}.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import statistics
from pathlib import Path
from typing import Optional

from experiments.similarity import FIELD_COMPARATORS, authors_similarity, score_document

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = REPO_ROOT / "experiments" / "runs"
DEFAULT_EVALUATIONS_ROOT = REPO_ROOT / "experiments" / "evaluations"


def _resolve_run_dir(run: str, runs_root: Path) -> Path:
    path = Path(run)
    if path.is_dir():
        return path
    candidate = runs_root / run
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"Run not found: {run!r} (tried {path} and {candidate})")


def _load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{run_dir} has no manifest.json -- is this a run directory?")
    return json.loads(manifest_path.read_text())


def _load_metadata(run_dir: Path, digest: str) -> Optional[dict]:
    """None if the file is missing, or the extraction for this digest errored
    (runner.py writes a literal JSON `null` for failed extractions)."""
    path = run_dir / "metadata" / f"{digest}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _is_present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    return bool(str(value).strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--truth-run", required=True, help="run_id under --runs-root, or a direct path to a run directory")
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_EVALUATIONS_ROOT)
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.75,
        help="Passed through to similarity.authors_similarity (see its docstring for calibration)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    truth_dir = _resolve_run_dir(args.truth_run, args.runs_root)
    candidate_dir = _resolve_run_dir(args.candidate_run, args.runs_root)
    truth_manifest = _load_manifest(truth_dir)
    candidate_manifest = _load_manifest(candidate_dir)

    # Different windowing means the models literally saw different input
    # text -- any accuracy gap would be confounded with window size, not
    # model quality, so this is a hard abort rather than a warning.
    if truth_manifest["max_pages"] != candidate_manifest["max_pages"] or truth_manifest["max_chars"] != candidate_manifest["max_chars"]:
        raise SystemExit(
            f"Windowing mismatch: truth run used max_pages={truth_manifest['max_pages']}, "
            f"max_chars={truth_manifest['max_chars']}; candidate run used "
            f"max_pages={candidate_manifest['max_pages']}, max_chars={candidate_manifest['max_chars']}. "
            f"Re-run one of them with matching --max-pages/--max-chars before comparing."
        )

    truth_digests = set(truth_manifest["digests"])
    candidate_digests = set(candidate_manifest["digests"])

    extra_in_candidate = candidate_digests - truth_digests
    if extra_in_candidate:
        raise SystemExit(
            f"Candidate run has {len(extra_in_candidate)} digest(s) absent from the truth run "
            f"(nothing to compare them against): {sorted(extra_in_candidate)[:5]}..."
        )
    missing_in_candidate = sorted(truth_digests - candidate_digests)
    compared_digests = sorted(truth_digests & candidate_digests)
    coverage = len(compared_digests) / len(truth_digests) if truth_digests else 0.0
    if coverage < 1.0:
        print(f"WARNING: coverage {coverage:.1%} -- {len(missing_in_candidate)} truth digest(s) not scored")

    field_comparators = dict(FIELD_COMPARATORS)
    field_comparators["authors"] = functools.partial(authors_similarity, match_threshold=args.match_threshold)

    per_document = []
    field_names = list(FIELD_COMPARATORS)
    truth_nonnull_scores: dict[str, list[float]] = {f: [] for f in field_names}
    all_scores: dict[str, list[float]] = {f: [] for f in field_names}
    overall_scores = []
    n_truth_failed = 0
    n_candidate_failed = 0

    for digest in compared_digests:
        truth_meta = _load_metadata(truth_dir, digest)
        candidate_meta = _load_metadata(candidate_dir, digest)
        if truth_meta is None:
            n_truth_failed += 1
            continue
        if candidate_meta is None:
            n_candidate_failed += 1
            continue
        doc_score = score_document(digest, truth_meta, candidate_meta, field_comparators=field_comparators)
        per_document.append(doc_score)
        overall_scores.append(doc_score.overall_score)
        for f in field_names:
            all_scores[f].append(doc_score.field_scores[f])
            if _is_present(truth_meta.get(f)):
                truth_nonnull_scores[f].append(doc_score.field_scores[f])

    def _mean(xs: list[float]) -> Optional[float]:
        return statistics.mean(xs) if xs else None

    def _stdev(xs: list[float]) -> Optional[float]:
        return statistics.stdev(xs) if len(xs) > 1 else (0.0 if xs else None)

    evaluation = {
        "truth_run_id": truth_manifest["run_id"],
        "truth_model_key": truth_manifest["model_key"],
        "candidate_run_id": candidate_manifest["run_id"],
        "candidate_model_key": candidate_manifest["model_key"],
        "match_threshold": args.match_threshold,
        "n_truth_documents": len(truth_digests),
        "n_candidate_documents": len(candidate_digests),
        "n_compared": len(per_document),
        "coverage": coverage,
        "missing_in_candidate": missing_in_candidate,
        "n_truth_extraction_failed": n_truth_failed,
        "n_candidate_extraction_failed": n_candidate_failed,
        "per_document": [
            {"digest": ds.digest, "field_scores": ds.field_scores, "overall_score": ds.overall_score}
            for ds in per_document
        ],
        "aggregate": {
            "per_field_mean": {f: _mean(all_scores[f]) for f in field_names},
            "per_field_stdev": {f: _stdev(all_scores[f]) for f in field_names},
            "overall_mean": _mean(overall_scores),
            "overall_stdev": _stdev(overall_scores),
            # Restricted to documents where the *ground-truth* value for that
            # field is non-null -- guards against a model that returns null
            # often looking artificially good under the both-empty->1.0
            # convention used by every similarity.py comparator. Reported
            # alongside, not instead of, the full-coverage numbers above.
            "per_field_mean_truth_nonnull": {f: _mean(truth_nonnull_scores[f]) for f in field_names},
            "per_field_n_truth_nonnull": {f: len(truth_nonnull_scores[f]) for f in field_names},
        },
    }

    output_dir = args.output_root / f"{candidate_manifest['run_id']}__vs__{truth_manifest['run_id']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2))

    with (output_dir / "per_document.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["digest", "field", "score"])
        for ds in per_document:
            for f in field_names:
                writer.writerow([ds.digest, f, ds.field_scores[f]])
            writer.writerow([ds.digest, "overall", ds.overall_score])

    print(
        f"{candidate_manifest['model_key']} vs {truth_manifest['model_key']}: "
        f"{len(per_document)} compared, overall_mean={evaluation['aggregate']['overall_mean']:.3f} "
        f"-> {output_dir}"
    )


if __name__ == "__main__":
    main()
