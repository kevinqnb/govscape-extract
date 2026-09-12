"""CLI: fuse the frontier-model panel's runs into the ground-truth dataset.

    # explicit runs
    uv run -m experiments.fuse \\
        --run gold__gpt-5.6-terra__<ts> \\
        --run gold__claude-sonnet-5__<ts> \\
        --run gold__gemini-3.7-flash__<ts>

    # or let it pick the latest run per experiments.config.VALIDATION_PANEL_KEYS
    uv run -m experiments.fuse

For every (document, field): if at least 2 of the 3 models agree -- fuzzily,
per experiments.similarity.FUSION_COMPARATORS / AGREEMENT_THRESHOLDS -- the
gold value is taken *verbatim* from the highest-priority model present in the
agreeing set (priority = VALIDATION_PANEL_KEYS order, i.e.
gpt-5.6-terra > claude-sonnet-5 > gemini-3.7-flash). Otherwise the pair is
FLAGGED: the gold value is null/empty and all three raw values are recorded
in flagged.csv so the pair can be extracted by hand later.

List fields (authors, geographic_coverage) are fused element-wise: an element
enters the gold list when >= 2 models contribute a fuzzily-matching element.
The field is flagged only if no element reaches 2-of-3; if some elements
agree and some don't, the field keeps the agreed subset and the dropped
elements are still logged (reason "partial_list").

Writes <out>/ (default data/validation_gold/) as a *run-shaped* directory --
manifest.json + metadata/<digest>.json in exactly the shape runner.py writes
-- so it can be passed straight to evaluate.py:

    uv run -m experiments.evaluate --truth-run data/validation_gold \\
        --candidate-run <a candidate run over data/validation_ocr>

Also written:
    status.json       {digest: {field: consensus|consensus_null|flagged|partial_list}}
    flagged.csv       one row per flagged / partial_list (digest, field): the manual worklist
    fusion_report.json per-field agreement rates, gpt-outvoted pairs, per-doc detail
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from govscape_extract.schema import FIELDS, DocumentMetadata

from experiments.config import GROUND_TRUTH_KEY, VALIDATION_EXPERIMENT, VALIDATION_PANEL_KEYS
from experiments.similarity import (
    AGREEMENT_THRESHOLDS,
    FUSION_COMPARATORS,
    list_element_similarity,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = REPO_ROOT / "experiments" / "runs"
DEFAULT_OUT = REPO_ROOT / "data" / "validation_gold"

LIST_FIELDS = [f.name for f in FIELDS if f.dtype == "list"]
SCALAR_FIELDS = [f.name for f in FIELDS if f.dtype != "list"]
ALL_FIELDS = [f.name for f in FIELDS]


# --------------------------------------------------------------------------- #
# run discovery / validation
# --------------------------------------------------------------------------- #
def _resolve_run_dir(run: str, runs_root: Path) -> Path:
    path = Path(run)
    if path.is_dir():
        return path
    candidate = runs_root / run
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"Run not found: {run!r} (tried {path} and {candidate})")


def _load_manifest(run_dir: Path) -> dict:
    mf = run_dir / "manifest.json"
    if not mf.exists():
        raise SystemExit(f"{run_dir} has no manifest.json -- is this a run directory?")
    return json.loads(mf.read_text())


def _latest_run_for(model_key: str, runs_root: Path) -> Optional[Path]:
    best: Optional[tuple[str, Path]] = None
    for d in sorted(runs_root.iterdir()) if runs_root.is_dir() else []:
        if not d.is_dir():
            continue
        mf = d / "manifest.json"
        if not mf.exists():
            continue
        try:
            m = json.loads(mf.read_text())
        except json.JSONDecodeError:
            continue
        if m.get("model_key") != model_key:
            continue
        started = m.get("started_at") or ""
        if best is None or started > best[0]:
            best = (started, d)
    return best[1] if best else None


def _load_metadata(run_dir: Path, digest: str) -> Optional[dict]:
    """DocumentMetadata dict, or None if the model's extraction for this digest
    failed (runner.py writes a literal JSON `null` in that case) or is absent."""
    path = run_dir / "metadata" / f"{digest}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------- #
# fusion core
# --------------------------------------------------------------------------- #
def _present(v) -> bool:
    if v is None:
        return False
    if isinstance(v, list):
        return len(v) > 0
    return bool(str(v).strip())


def _components(keys: list[str], agree) -> list[list[str]]:
    """Connected components of the agreement graph over `keys` (order
    preserved, so each component and the component list stay deterministic)."""
    parent = {k: k for k in keys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in itertools.combinations(keys, 2):
        if agree(a, b):
            parent[find(a)] = find(b)

    comps: dict[str, list[str]] = {}
    for k in keys:
        comps.setdefault(find(k), []).append(k)
    return list(comps.values())


def fuse_scalar(field: str, values: dict[str, object], priority: list[str]) -> dict:
    """values: {model_key: value} for the models that produced *any* metadata
    for this document (value itself may be None). Returns
    {status, value, source_model, agreeing_models}."""
    present = [m for m in priority if m in values]
    comp_fn = FUSION_COMPARATORS[field]
    thr = AGREEMENT_THRESHOLDS[field]

    def agree(a: str, b: str) -> bool:
        return comp_fn(values[a], values[b]) >= thr

    comps = _components(present, agree)
    # largest component wins; ties broken toward the one with the
    # highest-priority member (present is priority-ordered, so comp[0] is that).
    comps.sort(key=lambda c: (-len(c), priority.index(c[0])))
    best = comps[0] if comps else []

    if len(best) >= 2:
        rep = best[0]  # already the highest-priority member
        val = values[rep]
        if _present(val):
            return {"status": "consensus", "value": val, "source_model": rep, "agreeing_models": best}
        return {"status": "consensus_null", "value": None, "source_model": rep, "agreeing_models": best}

    return {"status": "flagged", "value": None, "source_model": None, "agreeing_models": best}


def fuse_list(field: str, lists: dict[str, object], priority: list[str]) -> dict:
    """Element-wise consensus. Returns
    {status, value, dropped_elements, agreeing_models, source_model}."""
    present = [m for m in priority if m in lists]
    norm = {
        m: [e for e in (lists[m] if isinstance(lists[m], list) else [lists[m]] if lists[m] else []) if _present(e)]
        for m in present
    }
    nonempty = [m for m in present if norm[m]]

    if len(present) < 2:
        return {"status": "flagged", "value": [], "dropped_elements": [],
                "agreeing_models": [], "source_model": None}
    if not nonempty:
        return {"status": "consensus_null", "value": [], "dropped_elements": [],
                "agreeing_models": present, "source_model": None}

    thr = AGREEMENT_THRESHOLDS[field]
    items = [(m, i, e) for m in present for i, e in enumerate(norm[m])]
    clusters: list[list[tuple[str, int, str]]] = []
    for it in items:
        for cl in clusters:
            if any(list_element_similarity(it[2], other[2]) >= thr for other in cl):
                cl.append(it)
                break
        else:
            clusters.append([it])

    consensus: list[tuple[tuple[str, int, str], set]] = []
    dropped: list[str] = []
    for cl in clusters:
        models_in = {it[0] for it in cl}
        rep = min(cl, key=lambda it: (priority.index(it[0]), it[1]))
        if len(models_in) >= 2:
            consensus.append((rep, models_in))
        else:
            dropped.append(rep[2])

    if not consensus:
        return {"status": "flagged", "value": [], "dropped_elements": dropped,
                "agreeing_models": [], "source_model": None}

    consensus.sort(key=lambda ce: (priority.index(ce[0][0]), ce[0][1]))
    value = [ce[0][2] for ce in consensus]
    agreeing = sorted({m for _, ms in consensus for m in ms}, key=priority.index)
    status = "partial_list" if dropped else "consensus"
    return {"status": status, "value": value, "dropped_elements": dropped,
            "agreeing_models": agreeing, "source_model": None}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run",
        action="append",
        dest="runs",
        metavar="RUN_ID_OR_PATH",
        help="A panel run directory (repeatable). If omitted, the latest run for each "
        "model in experiments.config.VALIDATION_PANEL_KEYS is used.",
    )
    p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--allow-unfinished",
        action="store_true",
        help="Fuse even if a panel run's manifest has finished_at=null. Documents that "
        "run never got to count as extraction failures for that model.",
    )
    return p


def _resolve_priority(model_keys: list[str]) -> list[str]:
    known = [k for k in VALIDATION_PANEL_KEYS if k in model_keys]
    unknown = [k for k in model_keys if k not in VALIDATION_PANEL_KEYS]
    if unknown:
        print(
            f"WARNING: run model_key(s) {unknown} are not in VALIDATION_PANEL_KEYS "
            f"{VALIDATION_PANEL_KEYS}; ranking them last for tie-breaks, in the order given."
        )
    return known + unknown


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.runs:
        run_dirs = [_resolve_run_dir(r, args.runs_root) for r in args.runs]
    else:
        run_dirs = []
        for key in VALIDATION_PANEL_KEYS:
            d = _latest_run_for(key, args.runs_root)
            if d is None:
                raise SystemExit(
                    f"No run found for panel model {key!r} under {args.runs_root}. "
                    f"Run it first:  uv run -m experiments.runner --model-key {key} "
                    f"--experiment {VALIDATION_EXPERIMENT.name} --run-name gold --skip-warmup"
                )
            run_dirs.append(d)

    if len(run_dirs) < 2:
        raise SystemExit("Need at least 2 panel runs to fuse.")
    if len(run_dirs) < 3:
        print(
            "WARNING: fusing fewer than 3 runs -- the 2-of-3 rule degrades to unanimity "
            "(every field where the models differ is flagged)."
        )

    manifests = [_load_manifest(d) for d in run_dirs]
    model_keys = [m["model_key"] for m in manifests]
    if len(set(model_keys)) != len(model_keys):
        raise SystemExit(f"Two panel runs have the same model_key: {model_keys}")
    priority = _resolve_priority(model_keys)
    by_model = {m["model_key"]: (d, m) for d, m in zip(run_dirs, manifests)}

    off_experiment = sorted(k for k, (_, m) in by_model.items() if m.get("experiment_name") != VALIDATION_EXPERIMENT.name)
    if off_experiment:
        print(
            f"WARNING: run(s) {off_experiment} were not from the '{VALIDATION_EXPERIMENT.name}' experiment; "
            "fusing them anyway, but check they cover the intended document set."
        )

    # Windowing must match: a different window means the models literally read
    # different input text, so "disagreement" would be confounded with it.
    windows = {(m["max_pages"], m["max_chars"]) for m in manifests}
    if len(windows) != 1:
        raise SystemExit(
            "Panel runs used different windowing (max_pages, max_chars): "
            + ", ".join(f"{k}={ (mm['max_pages'], mm['max_chars']) }" for k, (_, mm) in by_model.items())
            + ". Re-run so all three match before fusing."
        )
    max_pages, max_chars = windows.pop()

    for key, (d, m) in by_model.items():
        if not m.get("finished_at") and not args.allow_unfinished:
            raise SystemExit(
                f"Panel run {m['run_id']} ({key}) never finished "
                f"({m.get('n_completed')}/{m.get('n_documents')} documents). Finish it with "
                f"`uv run -m experiments.runner --model-key {key} --resume {d.name}`, "
                f"or pass --allow-unfinished."
            )

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

    # ---- fuse ---------------------------------------------------------------
    gold_dir = args.out
    (gold_dir / "metadata").mkdir(parents=True, exist_ok=True)

    # status[digest][field] = {status, n_present, n_agree, source_model}
    status: dict[str, dict[str, dict]] = {}
    flagged_rows: list[dict] = []
    per_doc: dict[str, dict] = {}
    field_counts = {f: {"consensus": 0, "consensus_null": 0, "flagged": 0, "partial_list": 0} for f in ALL_FIELDS}
    gpt_outvoted: list[dict] = []
    # pairwise agreement over (doc, scalar field) pairs where both models produced a value
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

            # pairwise agreement bookkeeping (both values present, non-null)
            for a, b in itertools.combinations([k for k in priority if k in values], 2):
                if _present(values[a]) and _present(values[b]):
                    key = f"{a}__vs__{b}"
                    pair_agree[key][1] += 1
                    if FUSION_COMPARATORS[field](values[a], values[b]) >= AGREEMENT_THRESHOLDS[field]:
                        pair_agree[key][0] += 1

            if res["status"] == "flagged":
                per_doc[digest]["flagged_fields"].append(field)
                reason = "insufficient_models" if len(got) < 2 else "no_consensus"
                flagged_rows.append(_flag_row(digest, field, reason, values, priority))
            elif (
                GROUND_TRUTH_KEY in values
                and _present(values[GROUND_TRUTH_KEY])
                and res["status"] == "consensus"
                and GROUND_TRUTH_KEY not in res["agreeing_models"]
            ):
                gpt_outvoted.append({
                    "digest": digest,
                    "field": field,
                    f"{GROUND_TRUTH_KEY}_value": values[GROUND_TRUTH_KEY],
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
        (gold_dir / "metadata" / f"{digest}.json").write_text(doc_meta.model_dump_json(indent=2))

    # ---- write sidecars ---------------------------------------------------
    now = datetime.now(timezone.utc).isoformat()
    run_id = f"gold-fused__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    manifest = {
        "run_id": run_id,
        "experiment_name": VALIDATION_EXPERIMENT.name,
        "model_key": "gold-fused",
        "backend": "llm",
        "started_at": now,
        "finished_at": now,
        "input_dir": VALIDATION_EXPERIMENT.input_dir,
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
                    # What each panel model actually sent -- the panel is not
                    # homogeneous (Anthropic's endpoint 400s on json_object), so
                    # record it here to keep data/validation_gold/ reproducible.
                    "response_format": m.get("config", {}).get("model", {}).get("response_format"),
                    "resolved_base_url": m.get("resolved_base_url"),
                }
                for k, (d, m) in by_model.items()
            ],
            "n_flagged_pairs": sum(1 for r in flagged_rows if r["reason"] != "partial_list"),
            "n_partial_list_pairs": sum(1 for r in flagged_rows if r["reason"] == "partial_list"),
        },
    }
    (gold_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (gold_dir / "status.json").write_text(json.dumps(status, indent=2))

    fieldnames = ["digest", "field", "reason", *priority, "dropped_elements"]
    with (gold_dir / "flagged.csv").open("w", newline="") as fh:
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
                # fraction of documents where the field was NOT flagged
                # (includes consensus_null -- models agreeing it's absent --
                # and partial_list -- at least one list element agreed)
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
        f"{GROUND_TRUTH_KEY}_outvoted": gpt_outvoted,
        "documents": per_doc,
    }
    (gold_dir / "fusion_report.json").write_text(json.dumps(report, indent=2))

    n_flag = manifest["fusion"]["n_flagged_pairs"]
    n_partial = manifest["fusion"]["n_partial_list_pairs"]
    total_pairs = len(digests) * len(ALL_FIELDS)
    print(
        f"Wrote {gold_dir}/  ({len(digests)} docs, {total_pairs} (doc,field) pairs: "
        f"{n_flag} flagged, {n_partial} partial-list -- see flagged.csv)"
    )
    if gpt_outvoted:
        print(f"  {len(gpt_outvoted)} scalar field(s) where {GROUND_TRUTH_KEY} was outvoted -- see fusion_report.json")


def _status_entry(res: dict, n_present: int) -> dict:
    """One field's entry in status.json. n_agree < n_present (e.g. 2 of 3) or
    n_present < 3 (a panel model errored on this doc) both mean the consensus
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


if __name__ == "__main__":
    main()
