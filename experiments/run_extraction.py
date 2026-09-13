"""Run the "extraction" experiment type: compare a set of extraction methods
(params.model_keys) over one dataset split, optionally scoring each against a
truth run.

    # Run every model_key in the config sequentially (smoke/local/CPU-only):
    uv run -m experiments.run_extraction experiments/experiment-configs/govscape/extraction/<id>/<id>.yaml

    # Run just one model's slice (what a submitted GPU job invokes):
    uv run -m experiments.run_extraction <config.yaml> --model-key qwen3-0.6b

    # Once every model_keys run exists under out/runs/, score + write the
    # experiment-level out/run.json + out/metrics.json:
    uv run -m experiments.run_extraction <config.yaml> --aggregate

See experiments/model-configs/README.md and experiments/dataset-configs/ for
what params.model_keys / params.dataset / params.split resolve against.
Config shape:

    params:
      experiment_type: extraction
      dataset: govscape
      split: sample
      model_keys: [gpt-5.6-terra, qwen3-0.6b, gliner2-base]
      limit: null           # optional
      truth_run: null       # optional, e.g. data/validation_gold
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from experiments import runner
from experiments.evaluate import evaluate_run
from experiments.utils import (
    REPO_ROOT,
    ExperimentSpec,
    latest_run_for,
    load_experiment_spec,
    load_manifest,
    now_iso,
    snapshot_config,
    write_experiment_manifest,
    write_json,
)

EXPERIMENT_TYPE = "extraction"


def _check_experiment_type(spec: ExperimentSpec) -> None:
    actual = spec.params.get("experiment_type")
    assert actual == EXPERIMENT_TYPE, f"{spec.config_path}: params.experiment_type is {actual!r}, expected {EXPERIMENT_TYPE!r}"


def run_one_model(spec: ExperimentSpec, model_key: str, *, limit_override: Optional[int] = None, skip_warmup: bool = False) -> Path:
    params = spec.params
    return runner.run_model(
        model_key,
        dataset=params["dataset"],
        split=params["split"],
        run_name=spec.id,
        limit=limit_override if limit_override is not None else params.get("limit"),
        output_root_override=spec.runs_dir,
        seed_override=spec.seed,
        skip_warmup=skip_warmup,
    )


def aggregate(spec: ExperimentSpec) -> None:
    started_at = now_iso()
    truth_run = spec.params.get("truth_run")
    truth_dir = (REPO_ROOT / truth_run) if truth_run else None
    if truth_dir is not None and not truth_dir.is_dir():
        raise SystemExit(f"{spec.config_path}: params.truth_run {truth_run!r} is not a directory ({truth_dir})")

    metrics: dict[str, float] = {}
    for model_key in spec.params["model_keys"]:
        candidate_dir = latest_run_for(model_key, spec.runs_dir)
        if candidate_dir is None:
            raise SystemExit(
                f"No run found for {model_key!r} under {spec.runs_dir}. Run it first with "
                f"--model-key {model_key}."
            )
        manifest = load_manifest(candidate_dir)
        metrics[f"{model_key}__n_documents"] = manifest["n_documents"]
        metrics[f"{model_key}__n_errors"] = manifest["n_errors"]
        if truth_dir is not None:
            evaluation = evaluate_run(truth_dir, candidate_dir, output_root=spec.out_dir / "evaluations")
            metrics[f"{model_key}__overall_score"] = evaluation["aggregate"]["overall_mean"]

    write_json(spec.out_dir / "metrics.json", metrics)
    snapshot_config(spec)
    write_experiment_manifest(spec, status="success", started_at=started_at, finished_at=now_iso())
    print(f"[{spec.id}] aggregated {len(spec.params['model_keys'])} model(s) -> {spec.out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="Path to an experiment-configs/.../<id>.yaml")
    parser.add_argument("--model-key", default=None, help="Run only this model_key's slice, instead of every params.model_keys entry")
    parser.add_argument("--aggregate", action="store_true", help="Score + write the experiment-level out/run.json and out/metrics.json")
    parser.add_argument("--limit", type=int, default=None, help="Overrides params.limit for this invocation")
    parser.add_argument("--skip-warmup", action="store_true")
    return parser


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = build_arg_parser().parse_args()
    spec = load_experiment_spec(args.config)
    _check_experiment_type(spec)

    if args.aggregate:
        if args.model_key:
            raise SystemExit("--aggregate and --model-key are mutually exclusive.")
        aggregate(spec)
        return

    model_keys = [args.model_key] if args.model_key else spec.params["model_keys"]
    for model_key in model_keys:
        run_one_model(spec, model_key, limit_override=args.limit, skip_warmup=args.skip_warmup)


if __name__ == "__main__":
    main()
