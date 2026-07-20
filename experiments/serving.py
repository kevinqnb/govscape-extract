"""Optional local vLLM server management.

Both models currently in config.py's MODEL_REGISTRY default to
`serving="external"` -- served manually on a remote/persistent GPU box.
Rather than re-exporting a base-url env var by hand every session,
experiments/serve_model.py wraps `vllm serve` and records its endpoint in
ENDPOINTS_FILE (a gitignored dotfile next to this module); endpoint_for()
below checks that file first, before falling back to
LLMModelConfig.base_url_env. This only helps when runner.py and
serve_model.py can see the same ENDPOINTS_FILE -- same machine, or a shared/
synced filesystem; across genuinely separate machines the env var (or
--base-url) remains the way to point at the server. See experiments/README.md.

A cached endpoint is only trusted after a live GET to its /health endpoint
succeeds, not just because the file has an entry -- this is what makes the
mechanism safe against a server that crashed or was killed without cleaning
up its own entry (serve_model.py removes its entry on a clean exit, but a
`kill -9` or an OOM won't run that cleanup).

vLLM itself is intentionally NOT a dependency of this project -- see the
note in pyproject.toml's [project.optional-dependencies] -- so this module
only ever shells out to a `vllm` CLI that must already be on PATH (installed
into its own separate environment, e.g. `uv tool install vllm`).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx

from experiments.config import LLMModelConfig

DEFAULT_STARTUP_TIMEOUT_S = 900.0  # a 120B-class model's weight load can take minutes
DEFAULT_POLL_INTERVAL_S = 2.0
ENDPOINTS_FILE = Path(__file__).resolve().parent / ".endpoints.json"


class ServerStartupError(RuntimeError):
    pass


def read_endpoints() -> dict:
    if not ENDPOINTS_FILE.exists():
        return {}
    try:
        return json.loads(ENDPOINTS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def write_endpoint(model_key: str, base_url: str) -> None:
    endpoints = read_endpoints()
    endpoints[model_key] = {"base_url": base_url, "updated_at": datetime.now(timezone.utc).isoformat()}
    ENDPOINTS_FILE.write_text(json.dumps(endpoints, indent=2))


def clear_endpoint(model_key: str) -> None:
    endpoints = read_endpoints()
    if endpoints.pop(model_key, None) is not None:
        ENDPOINTS_FILE.write_text(json.dumps(endpoints, indent=2))


def _health_url_from_base(base_url: str) -> str:
    return base_url.rsplit("/v1", 1)[0].rstrip("/") + "/health"


def _is_reachable(base_url: str, timeout: float = 2.0) -> bool:
    try:
        return httpx.get(_health_url_from_base(base_url), timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


@dataclass
class LocalVLLMServer:
    model: str
    served_model_name: Optional[str] = None
    port: int = 8000
    extra_args: list[str] = field(default_factory=list)
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S
    log_path: Optional[Path] = None
    # When True (serve_model.py's foreground use), don't redirect vllm's
    # stdout/stderr at all -- let its normal startup logs stream straight to
    # the terminal. Ignored if log_path is set. Runner.py's automated
    # local_vllm path leaves this False (discard to DEVNULL) since nothing
    # is watching a live terminal there.
    inherit_stdio: bool = False

    _proc: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    _log_fh: Any = field(default=None, init=False, repr=False)
    startup_seconds: Optional[float] = field(default=None, init=False)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def _health_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/health"

    def start(self) -> None:
        served_name = self.served_model_name or self.model
        argv = [
            "vllm",
            "serve",
            self.model,
            "--served-model-name",
            served_name,
            "--port",
            str(self.port),
            *self.extra_args,
        ]
        if self.log_path:
            self._log_fh = open(self.log_path, "w")
            stdout, stderr = self._log_fh, subprocess.STDOUT
        elif self.inherit_stdio:
            self._log_fh = None
            stdout, stderr = None, None
        else:
            self._log_fh = subprocess.DEVNULL
            stdout, stderr = subprocess.DEVNULL, subprocess.STDOUT
        t0 = time.monotonic()
        # start_new_session=True puts vLLM's worker processes in their own
        # process group; teardown must kill the *group*, not just this
        # parent process, or a bare terminate() orphans GPU worker
        # subprocesses vLLM spawns.
        self._proc = subprocess.Popen(
            argv,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            self._wait_healthy(t0)
        except Exception:
            self.stop()
            raise
        self.startup_seconds = time.monotonic() - t0

    def _wait_healthy(self, t0: float) -> None:
        deadline = t0 + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise ServerStartupError(
                    f"vllm serve exited early with code {self._proc.returncode} "
                    f"(model={self.model!r}); check {self.log_path or 'stdout (discarded)'}"
                )
            try:
                resp = httpx.get(self._health_url, timeout=5.0)
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(self.poll_interval_s)
        raise ServerStartupError(
            f"vllm serve for {self.model!r} did not become healthy within "
            f"{self.startup_timeout_s}s"
        )

    def stop(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(self._proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        finally:
            if self._log_fh not in (None, subprocess.DEVNULL):
                self._log_fh.close()

    def wait(self) -> None:
        """Block until the underlying process exits (serve_model.py's
        foreground use); Ctrl+C/SIGTERM should be caught by the caller,
        which then calls stop() for a clean process-group teardown."""
        if self._proc is not None:
            self._proc.wait()

    def __enter__(self) -> "LocalVLLMServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


@contextmanager
def endpoint_for(model_config: LLMModelConfig, base_url_override: Optional[str] = None) -> Iterator[tuple[str, Optional[float]]]:
    """Yield (base_url, startup_seconds).

    base_url_override (e.g. runner.py's --base-url) always wins and skips
    everything below. Otherwise:
      - serving == "local_vllm" -> launch+teardown a LocalVLLMServer.
      - serving == "external" (the default for every model in
        config.MODEL_REGISTRY today) -> check ENDPOINTS_FILE for an entry
        written by `serve_model.py` and confirm it's actually reachable
        (live /health check, not just "the file has a row"); if that's
        missing or stale, fall back to os.environ[model_config.base_url_env].
        startup_seconds is None either way since this project doesn't own
        that server's lifecycle.
    """
    if base_url_override:
        yield base_url_override, None
        return
    if model_config.serving == "local_vllm":
        with LocalVLLMServer(model=model_config.model, extra_args=model_config.vllm_args) as server:
            yield server.base_url, server.startup_seconds
        return

    cached = read_endpoints().get(model_config.key)
    if cached and _is_reachable(cached["base_url"]):
        yield cached["base_url"], None
        return

    if not model_config.base_url_env:
        raise ValueError(f"{model_config.key}: serving='external' requires base_url_env to be set")
    base_url = os.environ.get(model_config.base_url_env)
    if not base_url:
        raise ValueError(
            f"{model_config.key}: no reachable endpoint found. Either run "
            f"`uv run -m experiments.serve_model --model-key {model_config.key}` "
            f"(writes {ENDPOINTS_FILE.name}, auto-discovered next time), "
            f"set ${model_config.base_url_env} to the running server's base URL "
            f"(e.g. http://host:8000/v1), or pass --base-url to override."
        )
    yield base_url, None
