"""Runtime-captured reproducibility facts -- the "what actually happened, on
this machine, right now" half of reproducibility (see config.py's module
docstring for the split rationale).

Nothing here is hand-authored; it's all detected fresh at the start of each
run and folded into a RunManifest, written to runs/<run_id>/manifest.json by
experiments/runner.py.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Literal, Optional


@dataclass
class GPUInfo:
    available: bool
    device_type: Literal["cuda", "mps", "cpu", "none"]
    name: Optional[str] = None
    vram_gb: Optional[float] = None
    count: int = 0


def detect_gpu() -> GPUInfo:
    """Best-effort; never raises. On a CPU-only dev machine this degrades to
    device_type="cpu"/"none" rather than crashing the run."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return GPUInfo(
                available=True,
                device_type="cuda",
                name=props.name,
                vram_gb=props.total_memory / 1e9,
                count=torch.cuda.device_count(),
            )
        if torch.backends.mps.is_available():
            return GPUInfo(available=True, device_type="mps", name="Apple Silicon (MPS)")
    except Exception:
        pass
    return GPUInfo(available=False, device_type="cpu")


def git_sha(repo_root: Path) -> tuple[Optional[str], Optional[bool]]:
    """(sha, is_dirty); (None, None) if git isn't available or repo_root
    isn't a git repo -- never raises."""
    try:
        sha = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            .stdout.strip()
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
        return sha, bool(status.strip())
    except Exception:
        return None, None


def installed_versions(packages: list[str]) -> dict[str, str]:
    """Best-effort per-package version lookup; tolerates packages that
    aren't installed (e.g. gliner2/torch when profiling an LLM-only run)."""
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return versions


@dataclass
class RunManifest:
    run_id: str
    experiment_name: str
    model_key: str
    backend: Literal["llm", "gliner"]
    started_at: str  # ISO8601 UTC
    finished_at: Optional[str] = None
    git_sha: Optional[str] = None
    git_dirty: Optional[bool] = None
    hostname: str = field(default_factory=platform.node)
    python_version: str = field(default_factory=platform.python_version)
    gpu: GPUInfo = field(default_factory=detect_gpu)
    package_versions: dict[str, str] = field(default_factory=dict)
    config: dict = field(default_factory=dict)  # resolved model_config + experiment, as dicts
    input_dir: str = ""
    max_pages: int = 2
    max_chars: int = 6000
    n_documents: int = 0
    digests: list[str] = field(default_factory=list)  # evaluate.py's comparability/coverage join key
    # The manifest is rewritten after every document, so `finished_at is None`
    # on a manifest found at rest means the run died partway and can be
    # resumed (`runner.py --resume <run_id>`). n_completed/n_errors are
    # recounted from disk on each write, so they stay true of a partial run.
    n_completed: int = 0
    # One entry per `--resume` invocation: git sha / package versions can
    # differ between the original run and the resume, and for a run whose
    # output is being used as ground truth that difference is provenance,
    # not noise.
    resume_events: list[dict] = field(default_factory=list)
    # Where the request actually went, after --base-url / ENDPOINTS_FILE / env
    # var / SDK-default resolution. config["model"] records only the *intent*,
    # so without this a run overridden with --base-url has no provenance.
    # None means the OpenAI SDK's own default endpoint (a hosted model).
    resolved_base_url: Optional[str] = None
    serving_startup_seconds: Optional[float] = None  # None when serving == "external"
    model_load_seconds: Optional[float] = None  # e.g. GLiNER weight load
    seed: Optional[int] = None
    n_errors: int = 0
