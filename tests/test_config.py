"""Parity check for experiments/config.py's YAML-backed MODEL_REGISTRY.

MODEL_REGISTRY used to be a hardcoded dict of dataclasses in config.py; it is
now loaded from experiments/model-configs/*.yaml. This test pins the loaded
values against that prior hardcoded registry (see the 2026-09-12 commit that
introduced model-configs/ for the original), so a typo or dropped field in a
YAML file fails loudly instead of silently changing what a model does. The
`serving` overrides map documents the one deliberate behavior change made
during the migration: self-served vLLM candidates flipped from "external" to
"local_vllm" so a submitted job serves and calls a model in one self-contained
step, rather than a separate always-on server plus a separate caller job.
"""

from __future__ import annotations

import dataclasses

from experiments.config import (
    GlinerModelConfig,
    GROUND_TRUTH_KEY,
    HardwareRequirement,
    LLMModelConfig,
    MODEL_KEY_ORDER,
    MODEL_REGISTRY,
    VALIDATION_PANEL_KEYS,
)

# Deliberate serving changes: was "external" (persistent, separately-serve
# vLLM box), now "local_vllm" (one self-contained job serves + calls).
SERVING_OVERRIDES = {
    "gpt-oss-120b": "local_vllm",
    "qwen3-0.6b": "local_vllm",
    "qwen3-4b": "local_vllm",
    "olmo3-7b-instruct": "local_vllm",
    "gemma4-12b-it": "local_vllm",
    "gemma4-31b-it": "local_vllm",
}

def _expected_registry() -> dict:
    return {
        "gpt-5.6-terra": LLMModelConfig(
            key="gpt-5.6-terra",
            model="gpt-5.6-terra",
            role="ground_truth",
            serving="external",
            hardware=HardwareRequirement(device="none"),
            max_tokens=None,
            extra_body={"max_completion_tokens": 4096},
            temperature=None,
            seed=None,
            max_retries=6,
        ),
        "gpt-oss-120b": LLMModelConfig(
            key="gpt-oss-120b",
            model="openai/gpt-oss-120b",
            role="candidate",
            serving="local_vllm",
            hardware=HardwareRequirement(
                device="cuda", min_vram_gb=80, gpu_count=1, gpu_compute_capability="9.0"
            ),
            vllm_args=["--reasoning-parser", "openai_gptoss"],
            extra_body={"reasoning_effort": "low"},
        ),
        "qwen3-0.6b": LLMModelConfig(
            key="qwen3-0.6b",
            model="Qwen/Qwen3-0.6B",
            role="candidate",
            serving="local_vllm",
            hardware=HardwareRequirement(
                device="cuda", min_vram_gb=4, gpu_count=1, gpu_compute_capability="7.0"
            ),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
        "gliner2-base": GlinerModelConfig(key="gliner2-base"),
        "qwen3-4b": LLMModelConfig(
            key="qwen3-4b",
            model="Qwen/Qwen3-4B",
            role="candidate",
            serving="local_vllm",
            hardware=HardwareRequirement(
                device="cuda", min_vram_gb=16, gpu_count=1, gpu_compute_capability="8.0"
            ),
            vllm_args=["--max-model-len", "16384"],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
        "olmo3-7b-instruct": LLMModelConfig(
            key="olmo3-7b-instruct",
            model="allenai/Olmo-3-7B-Instruct",
            role="candidate",
            serving="local_vllm",
            hardware=HardwareRequirement(
                device="cuda", min_vram_gb=24, gpu_count=1, gpu_compute_capability="8.0"
            ),
            vllm_args=["--max-model-len", "16384"],
        ),
        "gemma4-12b-it": LLMModelConfig(
            key="gemma4-12b-it",
            model="google/gemma-4-12B-it",
            role="candidate",
            serving="local_vllm",
            hardware=HardwareRequirement(
                device="cuda", min_vram_gb=48, gpu_count=1, gpu_compute_capability="8.0"
            ),
            vllm_args=["--max-model-len", "16384"],
        ),
        "gemma4-31b-it": LLMModelConfig(
            key="gemma4-31b-it",
            model="google/gemma-4-31B-it",
            role="candidate",
            serving="local_vllm",
            hardware=HardwareRequirement(
                device="cuda", min_vram_gb=80, gpu_count=1, gpu_compute_capability="9.0"
            ),
            vllm_args=["--max-model-len", "16384"],
        ),
        "claude-sonnet-5": LLMModelConfig(
            key="claude-sonnet-5",
            model="claude-sonnet-5",
            role="candidate",
            serving="external",
            base_url_env="GOVSCAPE_ANTHROPIC_BASE_URL",
            api_key_env="GOVSCAPE_ANTHROPIC_API_KEY",
            hardware=HardwareRequirement(device="none"),
            response_format=None,
            temperature=None,
            seed=None,
            max_tokens=8192,
        ),
        "gemini-3.7-flash": LLMModelConfig(
            key="gemini-3.7-flash",
            model="gemini-3.7-flash",
            role="candidate",
            serving="external",
            base_url_env="GOVSCAPE_GEMINI_BASE_URL",
            api_key_env="GOVSCAPE_GEMINI_API_KEY",
            hardware=HardwareRequirement(device="none"),
            seed=None,
            max_tokens=2048,
        ),
    }


# Fields that legitimately differ and are excluded from the equality check
# below rather than hand-duplicated: free-text `notes` (both top-level and
# hardware.notes) migrated from config.py's old docstring/comments, and the
# two new HardwareRequirement fields not present in the old dataclass at all.
IGNORED_FIELDS = {"notes"}
IGNORED_HARDWARE_FIELDS = {"notes", "max_walltime"}


def _strip_ignored(obj):
    data = dataclasses.asdict(obj)
    for key in IGNORED_FIELDS:
        data.pop(key, None)
    hardware = data.get("hardware")
    if isinstance(hardware, dict):
        for key in IGNORED_HARDWARE_FIELDS:
            hardware.pop(key, None)
    return data


def test_model_registry_keys_and_order():
    assert list(MODEL_REGISTRY) == MODEL_KEY_ORDER


def test_model_registry_matches_prior_hardcoded_values():
    expected = _expected_registry()
    assert set(expected) == set(MODEL_REGISTRY)
    for key, expected_config in expected.items():
        actual_config = MODEL_REGISTRY[key]
        assert type(actual_config) is type(expected_config), key
        assert _strip_ignored(actual_config) == _strip_ignored(expected_config), key


def test_serving_overrides_are_exactly_the_local_vllm_candidates():
    local_vllm_keys = {key for key, cfg in MODEL_REGISTRY.items() if getattr(cfg, "serving", None) == "local_vllm"}
    assert local_vllm_keys == set(SERVING_OVERRIDES)


def test_ground_truth_and_panel_keys_exist():
    assert GROUND_TRUTH_KEY in MODEL_REGISTRY
    assert all(key in MODEL_REGISTRY for key in VALIDATION_PANEL_KEYS)
