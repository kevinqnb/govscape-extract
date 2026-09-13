"""resolve_vllm_command's precedence -- the fix that makes endpoint_for()'s
local_vllm path actually usable on a cluster where vllm runs inside a
Singularity image (see model-configs/README.md's "Serving" section)."""

from __future__ import annotations

from experiments.serving import resolve_vllm_command


def test_override_wins_over_everything(monkeypatch):
    monkeypatch.setenv("GOVSCAPE_VLLM_COMMAND", "singularity exec --nv /env.sif vllm serve")
    assert resolve_vllm_command("apptainer exec --nv /override.sif vllm serve") == [
        "apptainer", "exec", "--nv", "/override.sif", "vllm", "serve",
    ]


def test_env_var_wins_over_default(monkeypatch):
    monkeypatch.setenv("GOVSCAPE_VLLM_COMMAND", "singularity exec --nv /env.sif vllm serve")
    assert resolve_vllm_command() == ["singularity", "exec", "--nv", "/env.sif", "vllm", "serve"]


def test_falls_back_to_bare_vllm_serve(monkeypatch):
    monkeypatch.delenv("GOVSCAPE_VLLM_COMMAND", raising=False)
    assert resolve_vllm_command() == ["vllm", "serve"]
