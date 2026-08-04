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

Reasoning-mode caveat: several of the configured LLMs are "thinking" models,
which inflates per-document latency and risks breaking the structured-JSON
response format if left on. Where an off switch exists it is taken, so the
models are compared as close to like-for-like as their families allow:
  - qwen3-0.6b, qwen3-4b: `chat_template_kwargs={"enable_thinking": False}`
    fully disables thinking.
  - gemma4-12b-it, gemma4-31b-it: also thinking-capable, but their chat
    template's `enable_thinking` already defaults to *false* (verified
    against the published `chat_template.jinja`), and the non-thinking
    branch pre-fills an empty thought channel. Nothing to pass -- left out
    deliberately rather than overlooked.
  - olmo3-7b-instruct: not a thinking model at all. Olmo 3 ships reasoning
    as a separate `Olmo-3-7B-Think` checkpoint; the Instruct one configured
    here has no reasoning mode to disable.
  - gpt-oss-120b: `reasoning_effort="low"`. Harmony-format gpt-oss models
    always do *some* reasoning -- there is no full "off" switch analogous to
    Qwen3's -- so "low" is the closest available approximation. This means
    gpt-oss-120b's timing/accuracy numbers carry a residual reasoning-token
    cost that the others' don't; that asymmetry is a known limitation of
    comparing these specific model families, not something this harness can
    equalize, and should be called out when interpreting results.py's
    output.

Context-length caveat: the four candidates added on 2026-08-03 (qwen3-4b,
olmo3-7b-instruct, gemma4-{12b,31b}-it) each pin `--max-model-len 16384` in
`vllm_args`; gpt-oss-120b and qwen3-0.6b deliberately keep the settings they
were already served and scored with. These are 32K-256K-context models,
and vLLM sizes its KV cache from the model's *declared* maximum, so left
alone gemma4-31b-it's 256K window fails to allocate on an 80GB card before
it serves a single request. 16384 is chosen against measured demand, not
guessed: the largest prompt across the runs on disk is ~6.4K tokens, plus a
1024-token completion budget. Note the failure mode if it were set too low
-- a mid-run 400 on one long document, not a startup error -- which is why
the headroom is deliberate rather than tight.

Serving: the self-served LLMs default to `serving="external"` -- served
manually on a remote/persistent GPU box, not launched by the runner.
`base_url_env` names the environment variable (set in a local `.env`, never
hardcoded here) that holds the actual endpoint URL; `vllm_args` doubles as
documentation of the `vllm serve` invocation expected on that remote box,
even though this project doesn't invoke it directly. See experiments/README.md
for the exact commands. A local-launch path (serving="local_vllm") exists
via experiments/serving.py for future convenience/smoke-testing, but is not
the default for any model configured below.

"External" also covers a commercial API, which is the same situation from
this project's point of view -- somebody else owns the server's lifecycle.
Such a model simply declares no `base_url_env`, and endpoint_for() resolves
it to None, i.e. the OpenAI SDK's own default endpoint. It still needs an
API key in `api_key_env`, and unlike the vLLM boxes that key is real.
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
    temperature: Optional[float] = 0.0  # None => omit the parameter (some hosted models reject it)
    seed: Optional[int] = 0
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = None
    max_retries: Optional[int] = None  # None => the OpenAI SDK's default (2)
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
    "gpt-5.6-terra": LLMModelConfig(
        key="gpt-5.6-terra",
        model="gpt-5.6-terra",
        role="ground_truth",
        serving="external",  # commercial API; no base_url_env => SDK default endpoint
        base_url_env=None,
        hardware=HardwareRequirement(
            device="none",
            notes="hosted API -- no local hardware; wall-clock timings include "
            "network + provider queueing and are NOT comparable to the "
            "locally-served models on results.py's computational-time axis.",
        ),
        # This model rejects `max_tokens` outright ("use max_completion_tokens
        # instead"), so the budget goes through extra_body. 4096 rather than
        # the 1024 the self-served models use: reasoning tokens count against
        # the completion budget, and a truncated response fails json.loads.
        max_tokens=None,
        extra_body={"max_completion_tokens": 4096},
        # Rejects any explicit temperature ("only the default (1) is
        # supported"), so temperature=0 determinism is not available here at
        # all -- see the reproducibility caveat in this module's docstring.
        temperature=None,
        # The API accepts `seed`, but it's meaningless at temperature 1 --
        # None so the manifest doesn't record a seed implying determinism
        # this run doesn't have.
        seed=None,
        # 1000 sequential calls with no resume path: a transient 429/5xx
        # that exhausts retries writes a null for that document and it's
        # only recoverable by re-running the whole set.
        max_retries=6,
    ),
    "gpt-oss-120b": LLMModelConfig(
        key="gpt-oss-120b",
        model="openai/gpt-oss-120b",
        role="candidate",  # demoted: gpt-5.6-terra is the ground truth as of 2026-08-03
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
    # --- candidates added 2026-08-03. Appended rather than slotted in beside
    # the other LLMs on purpose: results.py assigns plot colors by this dict's
    # declaration order, so inserting above gliner2-base would silently
    # recolor it in every previously-generated report.
    "qwen3-4b": LLMModelConfig(
        key="qwen3-4b",
        model="Qwen/Qwen3-4B",
        role="candidate",
        serving="external",
        base_url_env="GOVSCAPE_QWEN3_4B_BASE_URL",
        hardware=HardwareRequirement(
            device="cuda",
            min_vram_gb=16,
            gpu_count=1,
            notes="~4B params, ~8GB of bf16 weights; comfortable on a 24GB card.",
        ),
        vllm_args=["--max-model-len", "16384"],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "olmo3-7b-instruct": LLMModelConfig(
        key="olmo3-7b-instruct",
        model="allenai/Olmo-3-7B-Instruct",
        role="candidate",
        serving="external",
        base_url_env="GOVSCAPE_OLMO3_7B_INSTRUCT_BASE_URL",
        hardware=HardwareRequirement(
            device="cuda",
            min_vram_gb=24,
            gpu_count=1,
            notes="~7.3B params, ~15GB of bf16 weights; a 24GB card leaves enough "
            "for the KV cache at the 16384 context pinned below.",
        ),
        vllm_args=["--max-model-len", "16384"],
        # No thinking knob: reasoning is a separate Olmo-3-7B-Think checkpoint.
    ),
    "gemma4-12b-it": LLMModelConfig(
        key="gemma4-12b-it",
        model="google/gemma-4-12B-it",
        role="candidate",
        serving="external",
        base_url_env="GOVSCAPE_GEMMA4_12B_IT_BASE_URL",
        hardware=HardwareRequirement(
            device="cuda",
            min_vram_gb=48,
            gpu_count=1,
            notes="~12B params, ~24GB of bf16 weights. Gemma4UnifiedForConditionalGeneration "
            "-- an encoder-free multimodal model, so vLLM reserves capacity for image/audio "
            "input this text-only harness never sends; 48GB leaves room for that plus the KV "
            "cache. Needs vLLM >= 0.24 (see experiments/README.md on SIF image choice).",
        ),
        vllm_args=["--max-model-len", "16384"],
        # Thinking-capable, but its chat template's `enable_thinking` already
        # defaults to false -- see this module's reasoning-mode caveat.
    ),
    "gemma4-31b-it": LLMModelConfig(
        key="gemma4-31b-it",
        model="google/gemma-4-31B-it",
        role="candidate",
        serving="external",
        base_url_env="GOVSCAPE_GEMMA4_31B_IT_BASE_URL",
        hardware=HardwareRequirement(
            device="cuda",
            min_vram_gb=80,
            gpu_count=1,
            notes="~32.7B params, ~65GB of bf16 weights -- 80GB-class VRAM (A100/H100 80GB) "
            "or multi-GPU tensor-parallel. Largest model here that isn't MoE, so all 65GB is "
            "active weight, unlike gpt-oss-120b's ~5B active.",
        ),
        vllm_args=["--max-model-len", "16384"],
        # Same as gemma4-12b-it: thinking defaults off in the chat template.
    ),
}

GROUND_TRUTH_KEY = "gpt-5.6-terra"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_keys: list[str]
    input_dir: str = "data/sample_ocr"
    limit: Optional[int] = None
    max_pages: int = 3
    max_chars: int = 9000
    seed: int = 0
    output_root: str = "experiments/runs"


DEFAULT_EXPERIMENT = ExperimentConfig(
    name="baseline",
    model_keys=[
        "gpt-5.6-terra",
        "gpt-oss-120b",
        "qwen3-0.6b",
        "gliner2-base",
        "qwen3-4b",
        "olmo3-7b-instruct",
        "gemma4-12b-it",
        "gemma4-31b-it",
    ],
)

EXPERIMENT_REGISTRY: dict[str, ExperimentConfig] = {DEFAULT_EXPERIMENT.name: DEFAULT_EXPERIMENT}
