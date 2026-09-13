"""Run the "ground_truth" experiment type: run a frontier-model panel over a
dataset split, then fuse their outputs into a ground-truth value per
(document, field) by 2-of-N fuzzy agreement (see
govscape_extract/consensus.py). Absorbs what used to be the separate
experiments/fuse.py CLI.

    # Run every panel model in the config sequentially:
    uv run -m experiments.run_ground_truth experiments/experiment-configs/govscape/ground_truth/<id>/<id>.yaml

    # Run just one panel model's slice (what a submitted job invokes):
    uv run -m experiments.run_ground_truth <config.yaml> --model-key claude-sonnet-5

    # Once every params.model_keys run exists under out/runs/, fuse them:
    uv run -m experiments.run_ground_truth <config.yaml> --aggregate

    # Only after reviewing out/fused/fusion_report.json and flagged.csv:
    # promote this experiment's fused set to the committed gold set.
    uv run -m experiments.run_ground_truth <config.yaml> --promote

For every (document, field): if at least 2 of the panel's models agree --
fuzzily, per govscape_extract.similarity.FUSION_COMPARATORS /
AGREEMENT_THRESHOLDS -- the gold value is taken *verbatim* from the
highest-priority model present in the agreeing set (priority =
params.model_keys order). Otherwise the pair is FLAGGED: the gold value is
null/empty and all raw values are recorded in flagged.csv, the worklist for
manual extraction. List fields (authors, geographic_coverage) are fused
element-wise instead of as a whole list.

Config shape:

    params:
      experiment_type: ground_truth
      dataset: govscape
      split: validation
      model_keys: [gpt-5.6-terra, claude-sonnet-5, gemini-3.7-flash]  # priority order
      limit: null            # optional
      allow_unfinished: false
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from govscape_extract.consensus import fuse_list, fuse_scalar, is_present
from govscape_extract.schema import FIELDS, DocumentMetadata
from govscape_extract.similarity import AGREEMENT_THRESHOLDS, FUSION_COMPARATORS

from experiments import runner
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

EXPERIMENT_TYPE = "ground_truth"
DEFAULT_GOLD_DIR = REPO_ROOT / "data" / "validation_gold"

LIST_FIELDS = [f.name for f in FIELDS if f.dtype == "list"]
SCALAR_FIELDS = [f.name for f in FIELDS if f.dtype != "list"]
ALL_FIELDS = [f.name for f in FIELDS]


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


def _status_entry(res: dict, n_present: int) -> dict:
    """One field's entry in status.json. n_agree < n_present (a panel model
    errored on this doc, or the panel is smaller than 3) means the consensus
    rests on fewer than the full panel -- surfaced here so a single scan of
    status.json finds them without cross-referencing fusion_report.json."""
    return {
        "status": res["status"],
        "n_present": n_present,
        "n_agree": len(res.get("agreeing_models") or []),
        "source_model": res.get("source_model"),
    }


def _flag_row(digest, field, reason, values_by_model, priority, extra=None) -> dict:
    row = {"digest": digest, "field": field, "reason": reason, "dropped_elements": ""}
    for k in priority:
        v = values_by_model.get(k, "<no result>")
        row[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else ("" if v is None else str(v))
    if extra and "dropped_elements" in extra:
        row["dropped_elements"] = json.dumps(extra["dropped_elements"], ensure_ascii=False)
    return row


def _load_metadata(run_dir: Path, digest: str) -> Optional[dict]:
    path = run_dir / "metadata" / f"{digest}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def fuse_panel(
    by_model: dict[str, tuple[Path, dict]],
    priority: list[str],
    out_dir: Path,
    *,
    run_id: str,
    split: str,
) -> dict:
    """Fuses `by_model` ({model_key: (run_dir, manifest)}) into a run-shaped
    directory at `out_dir` (manifest.json + metadata/<digest>.json, so it can
    be passed straight to evaluate.py's --truth-run), plus status.json,
    flagged.csv, and fusion_report.json. Returns field_counts, summed across
    all fields, for the experiment's metrics.json.

    This is experiments/fuse.py's old main() body, made reusable and pointed
    at a per-experiment out/fused/ directory instead of always writing
    straight to the committed data/validation_gold/ -- promoting a specific
    fused result to that committed path is now the separate, explicit
    --promote step.
    """
    ground_truth_key = priority[0]

    windows = {(m["max_pages"], m["max_chars"]) for _, m in by_model.values()}
    if len(windows) != 1:
        raise SystemExit(
            "Panel runs used different windowing (max_pages, max_chars): "
            + ", ".join(f"{k}={(m['max_pages'], m['max_chars'])}" for k, (_, m) in by_model.items())
            + ". Re-run so all of them match before fusing."
        )
    max_pages, max_chars = windows.pop()

    digest_sets = {key: set(m["digests"]) for key, (_, m) in by_model.items()}
    common = set.intersection(*digest_sets.values())
    for key, ds in digest_sets.items():
        missing = ds - common
        if missing:
            print(f"WARNING: {key} covers {len(missing)} digest(s) the other runs don't; excluded from the gold set.")
    digests = sorted(common)
    if not digests:
        raise SystemExit("Panel runs share no documents.")

    print(f"Fusing {len(digests)} documents from {len(by_model)} models (priority: {' > '.join(priority)}).")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)

    status: dict[str, dict[str, dict]] = {}
    flagged_rows: list[dict] = []
    per_doc: dict[str, dict] = {}
    field_counts = {f: {"consensus": 0, "consensus_null": 0, "flagged": 0, "partial_list": 0} for f in ALL_FIELDS}
    ground_truth_outvoted: list[dict] = []
    pair_agree = {f"{a}__vs__{b}": [0, 0] for a, b in itertools.combinations(priority, 2)}

    for digest in digests:
        metas = {key: _load_metadata(d, digest) for key, (d, _) in by_model.items()}
        got = {key: mm for key, mm in metas.items() if mm is not None}
        per_doc[digest] = {
            "n_models_present": len(got),
            "models_present": [k for k in priority if k in got],
            "flagged_fields": [],
            "dropped_list_elements": {},
        }
        status[digest] = {}
        gold: dict[str, object] = {}

        for field in SCALAR_FIELDS:
            values = {key: got[key].get(field) for key in got}
            res = fuse_scalar(field, values, priority) if values else {
                "status": "flagged", "value": None, "source_model": None, "agreeing_models": [],
            }
            gold[field] = res["value"]
            status[digest][field] = _status_entry(res, len(got))
            field_counts[field][res["status"]] += 1

            for a, b in itertools.combinations([k for k in priority if k in values], 2):
                if is_present(values[a]) and is_present(values[b]):
                    pair_key = f"{a}__vs__{b}"
                    pair_agree[pair_key][1] += 1
                    if FUSION_COMPARATORS[field](values[a], values[b]) >= AGREEMENT_THRESHOLDS[field]:
                        pair_agree[pair_key][0] += 1

            if res["status"] == "flagged":
                per_doc[digest]["flagged_fields"].append(field)
                reason = "insufficient_models" if len(got) < 2 else "no_consensus"
                flagged_rows.append(_flag_row(digest, field, reason, values, priority))
            elif (
                ground_truth_key in values
                and is_present(values[ground_truth_key])
                and res["status"] == "consensus"
                and ground_truth_key not in res["agreeing_models"]
            ):
                ground_truth_outvoted.append({
                    "digest": digest,
                    "field": field,
                    f"{ground_truth_key}_value": values[ground_truth_key],
                    "consensus_value": res["value"],
                    "consensus_source": res["source_model"],
                })

        for field in LIST_FIELDS:
            lists = {key: got[key].get(field) for key in got}
            res = fuse_list(field, lists, priority) if lists else {
                "status": "flagged", "value": [], "dropped_elements": [],
                "agreeing_models": [], "source_model": None,
            }
            gold[field] = res["value"]
            status[digest][field] = _status_entry(res, len(got))
            field_counts[field][res["status"]] += 1

            if res["status"] == "flagged":
                per_doc[digest]["flagged_fields"].append(field)
                reason = "insufficient_models" if len(got) < 2 else "no_consensus"
                flagged_rows.append(_flag_row(digest, field, reason, lists, priority))
            elif res["status"] == "partial_list":
                per_doc[digest]["dropped_list_elements"][field] = res["dropped_elements"]
                flagged_rows.append(_flag_row(digest, field, "partial_list", lists, priority,
                                              extra={"dropped_elements": res["dropped_elements"]}))

        # schema-validate so the gold file is exactly DocumentMetadata-shaped
        doc_meta = DocumentMetadata.model_validate(gold)
        (out_dir / "metadata" / f"{digest}.json").write_text(doc_meta.model_dump_json(indent=2))

    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "run_id": run_id,
        "experiment_name": split,
        "model_key": "gold-fused",
        "backend": "llm",
        "started_at": now,
        "finished_at": now,
        "input_dir": next(iter(by_model.values()))[1]["input_dir"],
        "max_pages": max_pages,
        "max_chars": max_chars,
        "n_documents": len(digests),
        "digests": digests,
        "n_completed": len(digests),
        "n_errors": 0,
        "seed": None,
        "fusion": {
            "generated_at": now,
            "priority": priority,
            "tie_break": "verbatim value from the highest-priority model in the agreeing set",
            "agreement_thresholds": AGREEMENT_THRESHOLDS,
            "panel": [
                {
                    "model_key": k,
                    "run_id": m["run_id"],
                    "run_dir": str(d),
                    "priority_rank": priority.index(k),
                    "response_format": m.get("config", {}).get("model", {}).get("response_format"),
                    "resolved_base_url": m.get("resolved_base_url"),
                }
                for k, (d, m) in by_model.items()
            ],
            "n_flagged_pairs": sum(1 for r in flagged_rows if r["reason"] != "partial_list"),
            "n_partial_list_pairs": sum(1 for r in flagged_rows if r["reason"] == "partial_list"),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "status.json", status)

    fieldnames = ["digest", "field", "reason", *priority, "dropped_elements"]
    with (out_dir / "flagged.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in sorted(flagged_rows, key=lambda r: (r["digest"], r["field"])):
            w.writerow(row)

    report = {
        "generated_at": now,
        "n_documents": len(digests),
        "priority": priority,
        "panel": manifest["fusion"]["panel"],
        "agreement_thresholds": AGREEMENT_THRESHOLDS,
        "per_field": {
            f: {
                **field_counts[f],
                "resolved_rate": round(
                    (field_counts[f]["consensus"] + field_counts[f]["consensus_null"] + field_counts[f]["partial_list"])
                    / len(digests),
                    4,
                ),
            }
            for f in ALL_FIELDS
        },
        "pairwise_scalar_agreement": {
            k: {"agree": v[0], "comparable": v[1], "rate": round(v[0] / v[1], 4) if v[1] else None}
            for k, v in pair_agree.items()
        },
        f"{ground_truth_key}_outvoted": ground_truth_outvoted,
        "documents": per_doc,
    }
    write_json(out_dir / "fusion_report.json", report)

    n_flag = manifest["fusion"]["n_flagged_pairs"]
    n_partial = manifest["fusion"]["n_partial_list_pairs"]
    print(f"Wrote {out_dir}/ ({len(digests)} docs: {n_flag} flagged, {n_partial} partial-list pairs)")
    if ground_truth_outvoted:
        print(f"  {len(ground_truth_outvoted)} scalar field(s) where {ground_truth_key} was outvoted -- see fusion_report.json")

    return field_counts


def aggregate(spec: ExperimentSpec, *, allow_unfinished: bool = False) -> None:
    started_at = now_iso()
    priority = spec.params["model_keys"]

    by_model: dict[str, tuple[Path, dict]] = {}
    for model_key in priority:
        run_dir = latest_run_for(model_key, spec.runs_dir)
        if run_dir is None:
            raise SystemExit(f"No run found for {model_key!r} under {spec.runs_dir}. Run it first with --model-key {model_key}.")
        manifest = load_manifest(run_dir)
        if not manifest.get("finished_at") and not allow_unfinished:
            raise SystemExit(
                f"Panel run {manifest['run_id']} ({model_key}) never finished "
                f"({manifest.get('n_completed')}/{manifest.get('n_documents')} documents). Finish it with "
                f"--model-key {model_key} (runner.py --resume {run_dir.name}), or pass --allow-unfinished."
            )
        by_model[model_key] = (run_dir, manifest)

    if len(by_model) < 2:
        raise SystemExit("Need at least 2 panel runs to fuse.")
    if len(by_model) < 3:
        print("WARNING: fusing fewer than 3 runs -- the 2-of-3 rule degrades to unanimity (every disagreement is flagged).")

    field_counts = fuse_panel(by_model, priority, spec.out_dir / "fused", run_id=spec.id, split=spec.params["split"])

    metrics = {
        status: sum(counts[status] for counts in field_counts.values())
        for status in ("consensus", "consensus_null", "flagged", "partial_list")
    }
    write_json(spec.out_dir / "metrics.json", {f"n_{k}": v for k, v in metrics.items()})
    snapshot_config(spec)
    write_experiment_manifest(spec, status="success", started_at=started_at, finished_at=now_iso())
    print(f"[{spec.id}] aggregated {len(priority)} panel model(s) -> {spec.out_dir}")


def promote(spec: ExperimentSpec, gold_dir: Path = DEFAULT_GOLD_DIR) -> None:
    """Copies this experiment's out/fused/ over the committed gold set at
    `gold_dir` (default data/validation_gold/). Separate from --aggregate on
    purpose: review out/fused/fusion_report.json and flagged.csv first, so a
    routine or smoke ground-truth run never silently overwrites the
    committed gold set."""
    fused_dir = spec.out_dir / "fused"
    if not (fused_dir / "manifest.json").exists():
        raise SystemExit(f"{fused_dir} has no manifest.json -- run --aggregate first.")
    if gold_dir.exists():
        shutil.rmtree(gold_dir)
    shutil.copytree(fused_dir, gold_dir)
    print(f"[{spec.id}] promoted {fused_dir} -> {gold_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="Path to an experiment-configs/.../<id>.yaml")
    parser.add_argument("--model-key", default=None, help="Run only this model_key's slice, instead of every params.model_keys entry")
    parser.add_argument("--aggregate", action="store_true", help="Fuse the panel's runs + write the experiment-level out/run.json and out/metrics.json")
    parser.add_argument("--promote", action="store_true", help="Copy this experiment's out/fused/ over the committed data/validation_gold/")
    parser.add_argument("--allow-unfinished", action="store_true", help="--aggregate even if a panel run's manifest has finished_at=null")
    parser.add_argument("--limit", type=int, default=None, help="Overrides params.limit for this invocation")
    parser.add_argument("--skip-warmup", action="store_true")
    return parser


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = build_arg_parser().parse_args()
    if sum([bool(args.aggregate), bool(args.promote), bool(args.model_key)]) > 1:
        raise SystemExit("--model-key, --aggregate, and --promote are mutually exclusive.")
    spec = load_experiment_spec(args.config)
    _check_experiment_type(spec)

    if args.promote:
        promote(spec)
        return
    if args.aggregate:
        aggregate(spec, allow_unfinished=args.allow_unfinished)
        return

    model_keys = [args.model_key] if args.model_key else spec.params["model_keys"]
    for model_key in model_keys:
        run_one_model(spec, model_key, limit_override=args.limit, skip_warmup=args.skip_warmup)


if __name__ == "__main__":
    main()
