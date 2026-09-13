"""Shared helpers for experiments/run_extraction.py, run_ground_truth.py, and
submit.sh -- reading an experiment-configs/<dataset>/<type>/<id>/<id>.yaml
file, and writing the experiment-level out/ manifest the harness contract
expects (run.json, config.snapshot.yaml, metrics.json). Per-model run
directories under out/runs/ are runner.py's own concern, unchanged by this.
"""

from __future__ import annotations

import json
import shutil
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from experiments.runtime import git_sha

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_CONFIGS_DIR = Path(__file__).resolve().parent / "experiment-configs"


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    project: str
    description: str
    seed: int
    params: dict[str, Any]
    config_path: Path

    @property
    def out_dir(self) -> Path:
        return self.config_path.parent / "out"

    @property
    def runs_dir(self) -> Path:
        return self.out_dir / "runs"


def load_experiment_spec(config_path: Path) -> ExperimentSpec:
    data = yaml.safe_load(config_path.read_text())
    required = {"id", "project", "description", "seed", "params"}
    missing = required - data.keys()
    assert not missing, f"{config_path}: missing required field(s) {sorted(missing)}"
    assert data["id"] == config_path.stem, f"{config_path}: id {data['id']!r} does not match filename"
    return ExperimentSpec(
        id=data["id"],
        project=data["project"],
        description=data["description"],
        seed=data["seed"],
        params=data["params"],
        config_path=config_path,
    )


def find_experiment_config(experiment_id: str, root: Path = EXPERIMENT_CONFIGS_DIR) -> Path:
    """Locates <root>/**/<experiment_id>.yaml -- lets submit.sh and a user
    address a nested config by id alone, the same way the harness's flat
    configs/<id>.yaml convention would."""
    matches = sorted(root.glob(f"**/{experiment_id}.yaml"))
    if not matches:
        raise SystemExit(f"No experiment config found for id {experiment_id!r} under {root}")
    if len(matches) > 1:
        raise SystemExit(f"Ambiguous experiment id {experiment_id!r}: found {matches}")
    return matches[0]


def load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{run_dir} has no manifest.json -- is this a run directory?")
    return json.loads(manifest_path.read_text())


def latest_run_for(model_key: str, runs_dir: Path) -> Optional[Path]:
    """The most recently started run directory under `runs_dir` for
    `model_key`, or None if there isn't one -- used by --aggregate to find a
    model's run without the caller needing to know its exact timestamp."""
    best: Optional[tuple[str, Path]] = None
    for d in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        if not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        if manifest.get("model_key") != model_key:
            continue
        started = manifest.get("started_at") or ""
        if best is None or started > best[0]:
            best = (started, d)
    return best[1] if best else None


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def snapshot_config(spec: ExperimentSpec) -> None:
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(spec.config_path, spec.out_dir / "config.snapshot.yaml")


def write_experiment_manifest(
    spec: ExperimentSpec,
    *,
    status: str,
    started_at: str,
    finished_at: Optional[str] = None,
    job_id: Optional[str] = None,
) -> None:
    """Writes the experiment-level out/run.json the harness contract expects
    (id/git_sha/status/host/job_id/config_path) -- one level above runner.py's
    existing per-model run.json-equivalent (manifest.json)."""
    sha, dirty = git_sha(REPO_ROOT)
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        spec.out_dir / "run.json",
        {
            "id": spec.id,
            "project": spec.project,
            "git_sha": sha,
            "git_dirty": dirty,
            "started_at": started_at,
            "finished_at": finished_at,
            "status": status,
            "host": socket.gethostname(),
            "job_id": job_id,
            "config_path": str(spec.config_path),
        },
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
