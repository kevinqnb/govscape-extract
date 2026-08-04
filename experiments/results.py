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
import math
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.config import GROUND_TRUTH_KEY, MODEL_REGISTRY
from govscape_extract.schema import FIELDS

# ACL-style body font: Nimbus Roman is the URW Times clone LaTeX's PSNFSS
# uses for Times substitution -- the closest open, non-proprietary match to
# what ACL's own LaTeX template renders. Liberation/DejaVu Serif are
# metric-compatible fallbacks if Nimbus isn't installed on whatever machine
# renders this.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = REPO_ROOT / "experiments" / "runs"
DEFAULT_EVALUATIONS_ROOT = REPO_ROOT / "experiments" / "evaluations"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "reports"

# tab10, the standard matplotlib qualitative map, lightly blended toward
# white for a softer print-friendly look, used in its own canonical order
# (blue, orange, green, red, purple, brown, pink, gray, olive, cyan).
_PASTEL_MIX = 0.25  # fraction of white blended into each tab10 hue


def _pastelize(rgb: tuple[float, float, float], mix: float = _PASTEL_MIX) -> tuple[float, float, float]:
    return tuple(c + (1.0 - c) * mix for c in rgb)


_CATEGORICAL_HUES = [_pastelize(c) for c in matplotlib.colormaps["tab10"].colors]
_INK = "#0b0b0b"
_MUTED = "#898781"


def _model_color_map(model_keys) -> dict[str, tuple]:
    """model_key -> tab10 color, assigned as a contiguous run through the
    palette in its own order (index 0 is always blue, 1 orange, 2 green, 3
    red, ...) across whichever models are actually present in model_keys --
    not a fixed slot per model over the full MODEL_REGISTRY, which would
    leave gaps (skip red entirely, etc.) whenever a registered model has no
    run yet to plot. `model_keys` is ordered by MODEL_REGISTRY's declaration
    position among the keys present, so a given set of plotted models still
    gets the same colors call to call ("color follows the entity, never its
    rank") -- it's *which set* of colors that adapts to what's present, not
    the assignment within that set."""
    ordered = [k for k in MODEL_REGISTRY if k in model_keys]
    return {k: _CATEGORICAL_HUES[i % len(_CATEGORICAL_HUES)] for i, k in enumerate(ordered)}

# Per-field summary columns are derived from schema.FIELDS rather than listed
# by hand, so adding a field to the schema flows through to the table and the
# per-field plot without another edit here.
_FIELD_SIMILARITY_COLS = [(f"{f.name}_similarity", f.name) for f in FIELDS]
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"


def _nice_log_ticks(xmin: float, xmax: float) -> list[float]:
    """"Nice" 1-2-5-per-decade tick values spanning [xmin, xmax] -- the
    standard log-axis spacing convention (engineering graph-paper style).
    Set explicitly rather than relying on matplotlib's default LogLocator,
    which mixes scientific notation (4x10^-1) with plain-looking ticks
    (10^0) whenever the data spans a bit more than one decade -- reads as
    cluttered/inconsistent rather than as one coherent scale."""
    lo_exp = math.floor(math.log10(xmin))
    hi_exp = math.ceil(math.log10(xmax))
    ticks = []
    for exp in range(lo_exp, hi_exp + 1):
        for base in (1, 2, 5):
            v = base * (10.0**exp)
            if xmin * 0.95 <= v <= xmax * 1.05:
                ticks.append(v)
    return ticks


def _format_tick(v: float) -> str:
    return f"{v:g}"


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
    """model_key -> latest *finished* run directory (by manifest started_at).
    Unfinished runs (finished_at is None -- runner.py still writing to them,
    or interrupted) are skipped: they have no timing_summary.json yet, which
    build_summary_rows needs, and a run still being written to shouldn't be
    picked over a completed older one just because it started more recently."""
    latest: dict[str, tuple[str, Path]] = {}
    if not runs_root.exists():
        return {}
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("finished_at"):
            continue
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
            # n_compared/coverage: how many truth digests this candidate's
            # similarity means are actually averaged over. A candidate run
            # with extraction failures (n_errors in its own timing_summary)
            # scores over fewer documents than n_documents implies -- without
            # this the table silently compares means with different
            # denominators (see evaluate.py's own n_candidate_extraction_failed).
            "n_compared": None,
            "coverage": None,
            "overall_similarity_mean": None,
            "overall_similarity_stdev": None,
            **{col: None for col, _ in _FIELD_SIMILARITY_COLS},
        }
        if model_key != GROUND_TRUTH_KEY and truth_run_id:
            eval_path = evaluations_root / f"{manifest['run_id']}__vs__{truth_run_id}" / "evaluation.json"
            if eval_path.exists():
                evaluation = json.loads(eval_path.read_text())
                agg = evaluation["aggregate"]
                row["n_compared"] = evaluation["n_compared"]
                row["coverage"] = evaluation["coverage"]
                row["overall_similarity_mean"] = agg["overall_mean"]
                row["overall_similarity_stdev"] = agg["overall_stdev"]
                for col, field_name in _FIELD_SIMILARITY_COLS:
                    # .get, not [...]: evaluations produced before a field was
                    # added to the schema simply won't have a mean for it.
                    row[col] = agg["per_field_mean"].get(field_name)
        rows.append(row)
    return rows


def estimate_concurrent_runtime(
    model_key: str,
    n_documents: int,
    concurrency: int,
    scaling_efficiency: float = 1.0,
    runs_root: Path = DEFAULT_RUNS_ROOT,
) -> dict:
    """Project wall-clock time to process n_documents at a given concurrency,
    extrapolated from this model's *sequential* (concurrency=1) per-document
    timing -- the only kind runner.py currently measures (_process_documents
    is a plain sequential loop; see its docstring discussion for why).

    This is a projection, not a measurement, and rests on one number this
    repo has no data to derive: `scaling_efficiency`, the fraction of ideal
    linear speedup the serving stack actually delivers at that concurrency
    (1.0 = N concurrent requests finish, in aggregate, exactly N times faster
    than one at a time; lower values model GPU compute/memory contention
    under load). Pass a value you've measured against the real server, or a
    deliberately-labeled guess -- don't treat the default of 1.0 as anything
    but the optimistic upper bound. Larger/denser models saturate a GPU's
    batched throughput sooner than small ones, so a realistic
    scaling_efficiency at a given concurrency is typically lower for e.g.
    gemma4-31b-it than for qwen3-0.6b -- that asymmetry is exactly what this
    single-number knob can't capture on its own; run the same concurrency at
    a few efficiency guesses to see the spread (see
    estimate_concurrent_runtime_sweep).
    """
    run_dir = _discover_latest_runs(runs_root).get(model_key)
    if run_dir is None:
        raise SystemExit(f"No finished run found for model_key={model_key!r} under {runs_root}")
    summary = json.loads((run_dir / "timing_summary.json").read_text())
    sequential_docs_per_sec = summary["throughput_docs_per_sec"]
    if not sequential_docs_per_sec:
        raise SystemExit(f"{model_key}: no successful documents in {run_dir} to extrapolate from")

    projected_docs_per_sec = sequential_docs_per_sec * concurrency * scaling_efficiency
    projected_seconds = n_documents / projected_docs_per_sec
    sequential_seconds = n_documents / sequential_docs_per_sec  # concurrency=1 baseline, for comparison

    return {
        "model_key": model_key,
        "n_documents": n_documents,
        "concurrency": concurrency,
        "scaling_efficiency": scaling_efficiency,
        "measured_wall_seconds_mean": summary["wall_seconds"]["mean"],
        "measured_sequential_docs_per_sec": sequential_docs_per_sec,
        "projected_docs_per_sec": projected_docs_per_sec,
        "projected_hours": projected_seconds / 3600,
        "sequential_baseline_hours": sequential_seconds / 3600,
    }


def estimate_concurrent_runtime_sweep(
    model_key: str,
    n_documents: int,
    concurrencies: list[int],
    scaling_efficiencies: list[float],
    runs_root: Path = DEFAULT_RUNS_ROOT,
) -> list[dict]:
    """estimate_concurrent_runtime over a grid of (concurrency,
    scaling_efficiency), so the unavoidable uncertainty in scaling_efficiency
    shows up as a range of hours rather than one falsely-precise number."""
    return [
        estimate_concurrent_runtime(model_key, n_documents, c, e, runs_root)
        for c in concurrencies
        for e in scaling_efficiencies
    ]


def plot_efficiency_frontier(rows: list[dict], output_path: Path, model_color: dict[str, tuple]) -> None:
    """x = mean per-doc wall time (log scale), y = overall similarity to
    ground truth. Ground truth itself is anchored at (its own wall_seconds_mean,
    1.0) as a reference point, not a real self-comparison. Direct
    visualization of the time-vs-accuracy tradeoff -- Pareto-dominated models
    (worse on both axes than another) are visible at a glance."""
    # A hosted-API ground truth (hardware.device == "none") has no local
    # compute cost, so its wall_seconds_mean includes network + provider
    # queueing instead -- not the same axis as the self-served models. Flag
    # it on the plot rather than let it read as a real measurement (see
    # config.py's docstring / README.md's hosted-ground-truth caveat).
    truth_cfg = MODEL_REGISTRY.get(GROUND_TRUTH_KEY)
    truth_is_hosted = bool(truth_cfg) and getattr(truth_cfg, "hardware", None) is not None and truth_cfg.hardware.device == "none"

    xs = [row["wall_seconds_mean"] for row in rows if row["wall_seconds_mean"] is not None]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for row in rows:
        y = 1.0 if row["model_key"] == GROUND_TRUTH_KEY else row["overall_similarity_mean"]
        x = row["wall_seconds_mean"]
        if x is None or y is None:
            continue
        color = model_color.get(row["model_key"], _MUTED)
        marker = "D" if row["model_key"] == GROUND_TRUTH_KEY else "o"
        # Ink edges, not white: pastel fills need a darker outline to stay
        # legible against the near-white axes background.
        ax.scatter([x], [y], s=90, color=color, marker=marker, zorder=3, edgecolors=_INK, linewidths=0.8)
        is_truth = row["model_key"] == GROUND_TRUTH_KEY
        label = row["model_key"] + (" (ground truth)*" if is_truth and truth_is_hosted else " (ground truth)" if is_truth else "")
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9, color=_INK)
    ax.set_xscale("log")
    if xs:
        ticks = _nice_log_ticks(min(xs), max(xs))
        ax.set_xticks(ticks)
        ax.set_xticklabels([_format_tick(t) for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Mean extraction time per document (s, log scale)", color=_INK)
    ax.set_ylabel("Overall similarity to ground truth", color=_INK)
    ax.set_ylim(-0.05, 1.08)
    _style_axes(ax)
    fig.tight_layout()
    if truth_is_hosted:
        fig.subplots_adjust(bottom=0.22)
        fig.text(
            0.01,
            0.02,
            "* ground truth is a hosted API; its time includes network + provider queueing,\n"
            "not comparable to the locally-served models' compute time.",
            fontsize=7.5,
            color=_MUTED,
        )
    fig.savefig(output_path, facecolor="#fcfcfb")
    plt.close(fig)


def plot_latency_distribution(selected_runs: dict[str, Path], output_path: Path, model_color: dict[str, tuple]) -> None:
    """Box plot of raw per-document wall_seconds per model -- surfaces tail
    behavior a mean/p50/p90 table hides (low mean + long tail is a worse
    production choice than the mean alone suggests)."""
    model_keys = list(selected_runs)
    data = [_raw_wall_seconds(selected_runs[mk]) for mk in model_keys]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    bp = ax.boxplot(data, tick_labels=model_keys, patch_artist=True, widths=0.5)
    for patch, mk in zip(bp["boxes"], model_keys):
        patch.set_facecolor(model_color.get(mk, _MUTED))
        patch.set_alpha(0.55)
        # Ink edge, not the same pastel as the fill: a pastel-on-pastel
        # outline nearly disappears against the near-white axes background.
        patch.set_edgecolor(_INK)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color(_INK)
            line.set_linewidth(1.2)
    # Fliers (points beyond the whiskers): low alpha so the tail-latency
    # outliers read as individually-unimportant scatter rather than
    # competing visually with the box itself.
    for flier, mk in zip(bp["fliers"], model_keys):
        flier.set_markerfacecolor(model_color.get(mk, _MUTED))
        flier.set_markeredgecolor(model_color.get(mk, _MUTED))
        flier.set_alpha(0.35)
        flier.set_markersize(4)
    ax.set_ylabel("Per-document extraction time (s)", color=_INK)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, facecolor="#fcfcfb")
    plt.close(fig)


def plot_per_field_accuracy(rows: list[dict], output_path: Path, model_color: dict[str, tuple]) -> None:
    """Grouped bar chart, field x model, mean similarity. overall_similarity_mean
    collapses every field's very different comparison semantics into one
    number; this shows *where* a model is weak (e.g. strong document_type,
    weak authors)."""
    field_cols = _FIELD_SIMILARITY_COLS
    candidates = [r for r in rows if r["model_key"] != GROUND_TRUTH_KEY and r["overall_similarity_mean"] is not None]
    if not candidates:
        return
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(field_cols)), 5), dpi=150)
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
            color=model_color.get(row["model_key"], _MUTED),
            edgecolor=_INK,
            linewidth=0.6,
            label=row["model_key"],
            zorder=3,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([name for _, name in field_cols], rotation=20, ha="right", color=_INK)
    ax.set_ylabel("Mean similarity to ground truth", color=_INK)
    ax.set_ylim(0, 1.08)
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
    model_color = _model_color_map(list(selected_runs))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "summary_table.csv"
    df.to_csv(summary_path, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {summary_path}")

    plot_efficiency_frontier(rows, plots_dir / "efficiency_frontier.png", model_color)
    plot_latency_distribution(selected_runs, plots_dir / "latency_distribution.png", model_color)
    plot_per_field_accuracy(rows, plots_dir / "per_field_accuracy.png", model_color)
    print(f"Wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
