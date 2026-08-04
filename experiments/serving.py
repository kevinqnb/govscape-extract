"""Optional local vLLM server management.

Both models currently in config.py's MODEL_REGISTRY default to
`serving="external"` -- served manually on a remote/persistent GPU box (e.g.
one node of an HPC cluster allocated by a batch scheduler), typically from a
different node than wherever runner.py runs. Rather than re-exporting a
base-url env var by hand every session, experiments/serve_model.py wraps
`vllm serve` and records its endpoint in ENDPOINTS_FILE (a gitignored dotfile
next to this module); endpoint_for() below checks that file first, before
falling back to LLMModelConfig.base_url_env. Two independent things have to
actually reach across nodes for this to work, and both are handled below:
  1. The *file* itself -- ENDPOINTS_FILE must be on a filesystem shared
     between the server's node and the runner's node (true by default on
     most HPC clusters' project/scratch storage; if genuinely unshared,
     use the env var / --base-url instead).
  2. The *network* -- LocalVLLMServer binds `--host 0.0.0.0` and advertises
     the server node's real hostname (see host/public_host below), not
     127.0.0.1, so another node can actually open the connection.
See experiments/README.md.

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
import socket
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
    # The base command that starts vllm's server; `model`, `--served-model-name`,
    # `--port`, and `extra_args` are appended after it -- so this must end
    # exactly at a bare `vllm serve`, not a `bash -c "..."` wrapper (appended
    # args would land as bash's positional params, not vllm's). To set env
    # vars inside a container, use the container runtime's own --env flag
    # rather than a shell wrapper, e.g.
    # ["singularity", "exec", "--nv", "--env", "HF_HOME=/cache,FOO=bar",
    #  "/path/to/vllm.sif", "vllm", "serve"].
    vllm_command: list[str] = field(default_factory=lambda: ["vllm", "serve"])
    # Bind address passed to vllm's own --host. "0.0.0.0" (default) listens
    # on every interface, not just loopback -- required on a multi-node
    # cluster where runner.py runs on a different node (login node, another
    # job) than this server. Still reachable at 127.0.0.1 from this same
    # node, so the internal health check below is unaffected.
    host: str = "0.0.0.0"
    # Hostname/IP advertised in base_url -- what *other* nodes use to reach
    # this server. None (default) resolves to socket.gethostname(), which on
    # a batch scheduler (SGE/Slurm/...) is the compute node's own hostname,
    # resolvable from the login node and other nodes on the cluster's
    # internal network. Override if your cluster needs a specific
    # interface's hostname (e.g. an InfiniBand-specific name) instead.
    public_host: Optional[str] = None

    _proc: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    _log_fh: Any = field(default=None, init=False, repr=False)
    startup_seconds: Optional[float] = field(default=None, init=False)

    @property
    def base_url(self) -> str:
        # Externally-advertised address -- what runner.py, possibly on a
        # different node, connects to. NOT 127.0.0.1 (see host/public_host
        # field docs above).
        return f"http://{self.public_host or socket.gethostname()}:{self.port}/v1"

    @property
    def _health_url(self) -> str:
        # Always loopback: this check runs from the same node/process as
        # the server itself, so it doesn't depend on the node's hostname
        # resolving to something reachable (or on any firewall rule beyond
        # loopback being open).
        return f"http://127.0.0.1:{self.port}/health"

    def start(self) -> None:
        served_name = self.served_model_name or self.model
        argv = [
            *self.vllm_command,
            self.model,
            "--served-model-name",
            served_name,
            "--host",
            self.host,
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

    def _output_hint(self) -> str:
        if self.log_path:
            return f"check {self.log_path}"
        if self.inherit_stdio:
            return "see vllm's output above"
        return "stdout/stderr were discarded -- pass log_path=... to capture them"

    def _wait_healthy(self, t0: float) -> None:
        deadline = t0 + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise ServerStartupError(
                    f"vllm serve exited early with code {self._proc.returncode} "
                    f"(model={self.model!r}); {self._output_hint()}"
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
def endpoint_for(model_config: LLMModelConfig, base_url_override: Optional[str] = None) -> Iterator[tuple[Optional[str], Optional[float]]]:
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

    "External" covers a commercial API (hosted OpenAI et al.) as much as it
    covers a self-served vLLM box -- in both cases we don't own the server.
    A hosted model just declares no base_url_env at all, and gets a yielded
    base_url of None, meaning "whatever the OpenAI SDK defaults to". The
    ENDPOINTS_FILE lookup above is harmless for those: it's keyed by model
    key, a hosted model never has an entry, so it falls straight through
    without ever health-probing a URL that has no /health endpoint.

    Note the asymmetry in the two failure modes below, which is deliberate:
    declaring *no* base_url_env is a hosted model, but declaring one and
    leaving it unset in the environment is a misconfigured self-served model,
    and still raises with the "go start your server" message.
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
        yield None, None
        return
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
