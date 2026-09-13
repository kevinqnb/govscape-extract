"""Loads experiment configuration from `experiments/model-configs/*.yaml` and
`experiments/dataset-configs/*.yaml` -- the "what did we intend to run" half of
reproducibility.

This module answers *intent*: which models, with which parameters, on what
hardware, over which dataset split. It should never be filled in by
introspecting the machine it happens to run on -- that's experiments/runtime.py's
job (git sha, detected GPU, installed package versions), captured fresh into a
RunManifest at the start of every run and written alongside that run's output,
not stored here.

Per-model rationale (reasoning-mode switches, context-length derivations,
serving choices) lives in each model's own YAML `notes:` field; cross-cutting
rationale lives in experiments/model-configs/README.md. This module only
defines the schema those files are validated against (HardwareRequirement /
LLMModelConfig / GlinerModelConfig / DatasetConfig) and loads them.

MODEL_REGISTRY's declaration order matters -- results.py assigns plot colors by
it -- so it comes from MODEL_KEY_ORDER (fixed here, not from directory-listing
order, which is not a stable thing to depend on).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_CONFIGS_DIR = Path(__file__).resolve().parent / "model-configs"
DATASET_CONFIGS_DIR = Path(__file__).resolve().parent / "dataset-configs"


@dataclass(frozen=True)
class HardwareRequirement:
    """Declarative hardware expectation -- documentation plus a guard the
    runner/submit.sh can check, not an auto-provisioner."""

    device: Literal["cpu", "cuda", "mps", "none"] = "cpu"
    min_vram_gb: Optional[float] = None  # None => no GPU needed
    gpu_count: int = 0
    gpu_compute_capability: Optional[str] = None  # submit.sh's `-l gpu_c=...`
    max_walltime: str = "24:00:00"  # submit.sh's `-l h_rt=...`
    notes: str = ""


@dataclass(frozen=True)
class LLMModelConfig:
    key: str
    model: str  # passed to LLMExtractor(model=...) / expected `vllm serve <model>` name
    role: Literal["ground_truth", "candidate"] = "candidate"
    serving: Literal["local_vllm", "external"] = "external"
    base_url_env: Optional[str] = None  # env var *name* holding the endpoint URL (serving="external" only)
    api_key_env: str = "GOVSCAPE_LLM_API_KEY"
    hardware: HardwareRequirement = field(default_factory=HardwareRequirement)
    vllm_args: list[str] = field(default_factory=list)  # local_vllm argv, or documents the expected remote invocation
    temperature: Optional[float] = 0.0  # None => omit the parameter (some hosted models reject it)
    seed: Optional[int] = 0
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = None
    max_retries: Optional[int] = None  # None => the OpenAI SDK's default (2)
    extra_body: dict = field(default_factory=dict)  # passthrough chat-completion kwargs (thinking-mode knobs)
    # Structured-output hint sent to the endpoint. `{"type": "json_object"}` for
    # most; None to omit it entirely (Anthropic's OpenAI-compat layer 400s on
    # json_object). JSON parsing does not depend on this either way --
    # LLMExtractor.loads_lenient handles fenced / unstructured output.
    response_format: Optional[dict] = field(default_factory=lambda: {"type": "json_object"})
    notes: str = ""  # model-specific rationale not captured by another field


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
    notes: str = ""


ModelConfig = Union[LLMModelConfig, GlinerModelConfig]

# Fixed, repo-wide: which models exist and the display order results.py colors
# plots by. Adding a model means adding it here *and* a model-configs/<key>.yaml;
# load_model_registry() asserts the two stay in sync.
MODEL_KEY_ORDER: list[str] = [
    "gpt-5.6-terra",
    "gpt-oss-120b",
    "qwen3-0.6b",
    "gliner2-base",
    "qwen3-4b",
    "olmo3-7b-instruct",
    "gemma4-12b-it",
    "gemma4-31b-it",
    "claude-sonnet-5",
    "gemini-3.7-flash",
]

GROUND_TRUTH_KEY = "gpt-5.6-terra"

# The frontier panel experiments.run_ground_truth fuses by 2-of-3 fuzzy
# agreement into data/validation_gold/. Order is the tie-break priority: when a
# field has consensus, the value is taken verbatim from the highest-ranked
# model present in the agreeing set.
VALIDATION_PANEL_KEYS = ["gpt-5.6-terra", "claude-sonnet-5", "gemini-3.7-flash"]


def _hardware_from_dict(data: Optional[dict]) -> HardwareRequirement:
    return HardwareRequirement(**(data or {}))


def _load_model_config(path: Path) -> ModelConfig:
    data = yaml.safe_load(path.read_text())
    kind = data.pop("kind")
    hardware = _hardware_from_dict(data.pop("hardware", None))
    if kind == "llm":
        config = LLMModelConfig(hardware=hardware, **data)
    elif kind == "gliner":
        config = GlinerModelConfig(hardware=hardware, **data)
    else:
        raise ValueError(f"{path}: unknown kind {kind!r} (expected 'llm' or 'gliner')")
    assert config.key == path.stem, f"{path}: key {config.key!r} does not match filename"
    return config


def load_model_registry(directory: Path = MODEL_CONFIGS_DIR, order: list[str] = MODEL_KEY_ORDER) -> dict[str, ModelConfig]:
    loaded = {path.stem: _load_model_config(path) for path in sorted(directory.glob("*.yaml"))}
    missing = set(order) - set(loaded)
    extra = set(loaded) - set(order)
    assert not missing and not extra, (
        f"MODEL_KEY_ORDER out of sync with {directory}: missing={sorted(missing)} extra={sorted(extra)}"
    )
    return {key: loaded[key] for key in order}


@dataclass(frozen=True)
class DatasetSplit:
    input_dir: str


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    splits: dict[str, DatasetSplit]
    max_pages: int = 3
    max_chars: int = 9000


def load_dataset_config(name: str, directory: Path = DATASET_CONFIGS_DIR) -> DatasetConfig:
    path = directory / f"{name}.yaml"
    data = yaml.safe_load(path.read_text())
    splits = {split_name: DatasetSplit(**split) for split_name, split in data.pop("splits").items()}
    return DatasetConfig(splits=splits, **data)


MODEL_REGISTRY: dict[str, ModelConfig] = load_model_registry()
