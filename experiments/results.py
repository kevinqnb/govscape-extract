"""CLI: aggregate runner.py runs + evaluate.py evaluations into a single
summary table and the plots that make the time-vs-accuracy tradeoff legible.

    uv run -m experiments.results \\
        --runs-root experiments/runs --evaluations-root experiments/evaluations \\
        --output-dir experiments/reports

By default, picks the most recent run per model_key found under
--runs-root (by manifest started_at); pass --run <run_id_or_path> (repeatable)
to select specific runs explicitly instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.config import GROUND_TRUTH_KEY, MODEL_REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = REPO_ROOT / "experiments" / "runs"
DEFAULT_EVALUATIONS_ROOT = REPO_ROOT / "experiments" / "evaluations"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "reports"

# Fixed-order categorical palette (dataviz skill's validated default,
# references/palette.md light-mode steps) -- assigned by MODEL_REGISTRY's
# declaration order so a given model always gets the same color across all
# three plots below ("color follows the entity, never its rank").
_CATEGORICAL_HUES = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
_MODEL_COLOR = {key: _CATEGORICAL_HUES[i % len(_CATEGORICAL_HUES)] for i, key in enumerate(MODEL_REGISTRY)}
_INK = "#0b0b0b"
_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"


def _style_axes(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_BASELINE)
    ax.tick_params(colors=_MUTED, labelcolor=_INK)


def _discover_latest_runs(runs_root: Path) -> dict[str, Path]:
    """model_key -> latest run directory (by manifest started_at)."""
    latest: dict[str, tuple[str, Path]] = {}
    if not runs_root.exists():
        return {}
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        model_key = manifest["model_key"]
        started_at = manifest["started_at"]
        if model_key not in latest or started_at > latest[model_key][0]:
            latest[model_key] = (started_at, run_dir)
    return {k: v[1] for k, v in latest.items()}


def _resolve_explicit_runs(runs: list[str], runs_root: Path) -> dict[str, Path]:
    resolved = {}
    for run in runs:
        path = Path(run)
        run_dir = path if path.is_dir() else runs_root / run
        if not run_dir.is_dir():
            raise SystemExit(f"Run not found: {run!r}")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        resolved[manifest["model_key"]] = run_dir
    return resolved


def _raw_wall_seconds(run_dir: Path) -> list[float]:
    values = []
    timing_dir = run_dir / "timing"
    if not timing_dir.exists():
        return values
    for path in sorted(timing_dir.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("error") is None and record.get("wall_seconds") is not None:
            values.append(record["wall_seconds"])
    return values


def build_summary_rows(selected_runs: dict[str, Path], evaluations_root: Path) -> list[dict]:
    manifests = {mk: json.loads((rd / "manifest.json").read_text()) for mk, rd in selected_runs.items()}
    timing_summaries = {mk: json.loads((rd / "timing_summary.json").read_text()) for mk, rd in selected_runs.items()}
    truth_run_id = manifests[GROUND_TRUTH_KEY]["run_id"] if GROUND_TRUTH_KEY in manifests else None

    rows = []
    for model_key, run_dir in selected_runs.items():
        manifest = manifests[model_key]
        summary = timing_summaries[model_key]
        model_cfg = manifest["config"]["model"]
        row = {
            "model_key": model_key,
            "role": model_cfg.get("role", "ground_truth" if model_key == GROUND_TRUTH_KEY else "candidate"),
            "n_documents": summary["n_documents"],
            "wall_seconds_total": summary["wall_seconds"]["total"],
            "wall_seconds_mean": summary["wall_seconds"]["mean"],
            "wall_seconds_stdev": summary["wall_seconds"]["stdev"],
            "overall_similarity_mean": None,
            "overall_similarity_stdev": None,
            "title_similarity": None,
            "authors_similarity": None,
            "publication_date_similarity": None,
            "government_agency_similarity": None,
            "document_type_similarity": None,
        }
        if model_key != GROUND_TRUTH_KEY and truth_run_id:
            eval_path = evaluations_root / f"{manifest['run_id']}__vs__{truth_run_id}" / "evaluation.json"
            if eval_path.exists():
                agg = json.loads(eval_path.read_text())["aggregate"]
                row["overall_similarity_mean"] = agg["overall_mean"]
                row["overall_similarity_stdev"] = agg["overall_stdev"]
                row["title_similarity"] = agg["per_field_mean"]["title"]
                row["authors_similarity"] = agg["per_field_mean"]["authors"]
                row["publication_date_similarity"] = agg["per_field_mean"]["publication_date"]
                row["government_agency_similarity"] = agg["per_field_mean"]["government_agency"]
                row["document_type_similarity"] = agg["per_field_mean"]["document_type"]
        rows.append(row)
    return rows


def plot_efficiency_frontier(rows: list[dict], output_path: Path) -> None:
    """x = mean per-doc wall time (log scale), y = overall similarity to
    ground truth. Ground truth itself is anchored at (its own wall_seconds_mean,
    1.0) as a reference point, not a real self-comparison. Direct
    visualization of the time-vs-accuracy tradeoff -- Pareto-dominated models
    (worse on both axes than another) are visible at a glance."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for row in rows:
        y = 1.0 if row["model_key"] == GROUND_TRUTH_KEY else row["overall_similarity_mean"]
        x = row["wall_seconds_mean"]
        if x is None or y is None:
            continue
        color = _MODEL_COLOR.get(row["model_key"], _MUTED)
        marker = "D" if row["model_key"] == GROUND_TRUTH_KEY else "o"
        ax.scatter([x], [y], s=90, color=color, marker=marker, zorder=3, edgecolors="white", linewidths=0.8)
        label = row["model_key"] + (" (ground truth)" if row["model_key"] == GROUND_TRUTH_KEY else "")
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9, color=_INK)
    ax.set_xscale("log")
    ax.set_xlabel("Mean extraction time per document (s, log scale)", color=_INK)
    ax.set_ylabel("Overall similarity to ground truth", color=_INK)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Time vs. accuracy", color=_INK, fontsize=12, fontweight="bold")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, facecolor="#fcfcfb")
    plt.close(fig)


def plot_latency_distribution(selected_runs: dict[str, Path], output_path: Path) -> None:
    """Box plot of raw per-document wall_seconds per model -- surfaces tail
    behavior a mean/p50/p90 table hides (low mean + long tail is a worse
    production choice than the mean alone suggests)."""
    model_keys = list(selected_runs)
    data = [_raw_wall_seconds(selected_runs[mk]) for mk in model_keys]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    bp = ax.boxplot(data, tick_labels=model_keys, patch_artist=True, widths=0.5)
    for patch, mk in zip(bp["boxes"], model_keys):
        patch.set_facecolor(_MODEL_COLOR.get(mk, _MUTED))
        patch.set_alpha(0.55)
        patch.set_edgecolor(_MODEL_COLOR.get(mk, _MUTED))
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color(_INK)
            line.set_linewidth(1.2)
    ax.set_ylabel("Per-document extraction time (s)", color=_INK)
    ax.set_title("Latency distribution by model", color=_INK, fontsize=12, fontweight="bold")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, facecolor="#fcfcfb")
    plt.close(fig)


def plot_per_field_accuracy(rows: list[dict], output_path: Path) -> None:
    """Grouped bar chart, field x model, mean similarity. overall_similarity_mean
    collapses 5 fields with very different comparison semantics into one
    number; this shows *where* a model is weak (e.g. strong document_type,
    weak authors)."""
    field_cols = [
        ("title_similarity", "title"),
        ("authors_similarity", "authors"),
        ("publication_date_similarity", "publication_date"),
        ("government_agency_similarity", "government_agency"),
        ("document_type_similarity", "document_type"),
    ]
    candidates = [r for r in rows if r["model_key"] != GROUND_TRUTH_KEY and r["overall_similarity_mean"] is not None]
    if not candidates:
        return
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    n_models = len(candidates)
    bar_width = 0.8 / n_models
    x = range(len(field_cols))
    for i, row in enumerate(candidates):
        offsets = [xi + i * bar_width - 0.4 + bar_width / 2 for xi in x]
        heights = [row[col] if row[col] is not None else 0.0 for col, _ in field_cols]
        ax.bar(
            offsets,
            heights,
            width=bar_width,
            color=_MODEL_COLOR.get(row["model_key"], _MUTED),
            label=row["model_key"],
            zorder=3,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([name for _, name in field_cols], rotation=20, ha="right", color=_INK)
    ax.set_ylabel("Mean similarity to ground truth", color=_INK)
    ax.set_ylim(0, 1.08)
    ax.set_title("Per-field accuracy by model", color=_INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, labelcolor=_INK)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, facecolor="#fcfcfb")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--evaluations-root", type=Path, default=DEFAULT_EVALUATIONS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="run_id or path to select explicitly (repeatable); default: latest run per model_key under --runs-root",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    selected_runs = (
        _resolve_explicit_runs(args.run, args.runs_root) if args.run else _discover_latest_runs(args.runs_root)
    )
    if not selected_runs:
        raise SystemExit(f"No runs found under {args.runs_root}")

    rows = build_summary_rows(selected_runs, args.evaluations_root)
    df = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "summary_table.csv"
    df.to_csv(summary_path, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {summary_path}")

    plot_efficiency_frontier(rows, plots_dir / "efficiency_frontier.png")
    plot_latency_distribution(selected_runs, plots_dir / "latency_distribution.png")
    plot_per_field_accuracy(rows, plots_dir / "per_field_accuracy.png")
    print(f"Wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
