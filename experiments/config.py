"""Static, hand-authored experiment configuration -- the "what did we intend
to run" half of reproducibility.

This module answers *intent*: which models, with which parameters, on what
hardware. It should never be filled in by introspecting the machine it
happens to run on -- that's experiments/runtime.py's job (git sha, detected
GPU, installed package versions), captured fresh into a RunManifest at the
start of every run and written alongside that run's output, not stored here.

Reproducibility caveat, stated plainly: `seed` + `temperature=0` is
best-effort determinism, not a guarantee. GPU batched inference (vLLM, and
torch under GLiNER) is not bit-reproducible across runs/hardware due to
kernel/batching nondeterminism. The seed and temperature are captured for
traceability, not promised as exact reproduction.

Reasoning-mode caveat: both configured LLMs are "thinking" models by
default, which inflates per-document latency and risks breaking the
structured-JSON response format if left on. `extra_body` below disables/
minimizes it per model:
  - qwen3-0.6b: `chat_template_kwargs={"enable_thinking": False}` fully
    disables thinking.
  - gpt-oss-120b: `reasoning_effort="low"`. Harmony-format gpt-oss models
    always do *some* reasoning -- there is no full "off" switch analogous to
    Qwen3's -- so "low" is the closest available approximation. This means
    gpt-oss-120b's timing/accuracy numbers carry a residual reasoning-token
    cost that qwen3-0.6b's don't; that asymmetry is a known limitation of
    comparing these two specific model families, not something this harness
    can equalize, and should be called out when interpreting results.py's
    output.

Serving: both configured LLMs default to `serving="external"` -- they're
served manually on a remote/persistent GPU box, not launched by the runner.
`base_url_env` names the environment variable (set in a local `.env`, never
hardcoded here) that holds the actual endpoint URL; `vllm_args` doubles as
documentation of the `vllm serve` invocation expected on that remote box,
even though this project doesn't invoke it directly. See experiments/README.md
for the exact commands. A local-launch path (serving="local_vllm") exists
via experiments/serving.py for future convenience/smoke-testing, but is not
the default for either model configured below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union


@dataclass(frozen=True)
class HardwareRequirement:
    """Declarative hardware expectation -- documentation plus a guard the
    runner can check, not an auto-provisioner."""

    device: Literal["cpu", "cuda", "mps", "none"] = "cpu"
    min_vram_gb: Optional[float] = None  # None => no GPU needed
    gpu_count: int = 0
    notes: str = ""


@dataclass(frozen=True)
class LLMModelConfig:
    key: str
    model: str  # passed to LLMExtractor(model=...) / expected `vllm serve <model>` name
    role: Literal["ground_truth", "candidate"] = "candidate"
    serving: Literal["local_vllm", "external"] = "external"
    base_url_env: Optional[str] = None  # env var *name* holding the endpoint URL
    api_key_env: str = "GOVSCAPE_LLM_API_KEY"
    hardware: HardwareRequirement = field(default_factory=HardwareRequirement)
    vllm_args: list[str] = field(default_factory=list)  # local_vllm argv, or documents the expected remote invocation
    temperature: float = 0.0
    seed: Optional[int] = 0
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = None
    extra_body: dict = field(default_factory=dict)  # passthrough chat-completion kwargs (thinking-mode knobs)


@dataclass(frozen=True)
class GlinerModelConfig:
    key: str
    model_name: str = "fastino/gliner2-base-v1"
    threshold: float = 0.5
    hardware: HardwareRequirement = field(
        default_factory=lambda: HardwareRequirement(
            device="cpu", notes="runs on CPU; GPU is an optional speedup, not required."
        )
    )


ModelConfig = Union[LLMModelConfig, GlinerModelConfig]


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-oss-120b": LLMModelConfig(
        key="gpt-oss-120b",
        model="openai/gpt-oss-120b",
        role="ground_truth",
        serving="external",
        base_url_env="GOVSCAPE_GPT_OSS_120B_BASE_URL",
        hardware=HardwareRequirement(
            device="cuda",
            min_vram_gb=80,
            gpu_count=1,
            notes="~120B params / ~5B active MoE; needs 80GB-class VRAM even quantized "
            "(A100/H100 80GB, or multi-GPU tensor-parallel).",
        ),
        vllm_args=["--reasoning-parser", "openai_gptoss"],
        extra_body={"reasoning_effort": "low"},
    ),
    "qwen3-0.6b": LLMModelConfig(
        key="qwen3-0.6b",
        model="Qwen/Qwen3-0.6B",
        role="candidate",
        serving="external",
        base_url_env="GOVSCAPE_QWEN3_0_6B_BASE_URL",
        hardware=HardwareRequirement(
            device="cuda",
            min_vram_gb=4,
            gpu_count=1,
            notes="trivially small; also runs on CPU for smoke testing, just slow.",
        ),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "gliner2-base": GlinerModelConfig(key="gliner2-base"),
}

GROUND_TRUTH_KEY = "gpt-oss-120b"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_keys: list[str]
    input_dir: str = "data/sample_ocr"
    limit: Optional[int] = None
    max_pages: int = 2
    max_chars: int = 6000
    seed: int = 0
    output_root: str = "experiments/runs"


DEFAULT_EXPERIMENT = ExperimentConfig(
    name="baseline",
    model_keys=["gpt-oss-120b", "qwen3-0.6b", "gliner2-base"],
)

EXPERIMENT_REGISTRY: dict[str, ExperimentConfig] = {DEFAULT_EXPERIMENT.name: DEFAULT_EXPERIMENT}
