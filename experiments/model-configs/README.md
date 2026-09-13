# model-configs

One YAML file per model/method used by an experiment, keyed by filename
(`<key>.yaml`, `key:` inside must match). Loaded and validated into the
`HardwareRequirement`/`LLMModelConfig`/`GlinerModelConfig` dataclasses in
`experiments/config.py`; declaration order for display (`results.py`'s plot
colors) comes from `config.py`'s `MODEL_KEY_ORDER`, not directory-listing order.
Per-model rationale that doesn't fit a comment lives in that model's own `notes:`
field. This file carries the cross-cutting rationale that used to live in
`config.py`'s module docstring, before `MODEL_REGISTRY` moved out of Python.

## Reproducibility caveat

`seed` + `temperature=0` is best-effort determinism, not a guarantee. GPU batched
inference (vLLM, and torch under GLiNER) is not bit-reproducible across
runs/hardware due to kernel/batching nondeterminism. The seed and temperature are
captured for traceability, not promised as exact reproduction.

## Reasoning-mode caveat

Several of the configured LLMs are "thinking" models, which inflates
per-document latency and risks breaking the structured-JSON response format if
left on. Where an off switch exists it is taken, so the models are compared as
close to like-for-like as their families allow -- see each model's own `notes:`
field for its specific case (`qwen3-0.6b`, `qwen3-4b`, `gemma4-12b-it`,
`gemma4-31b-it`, `olmo3-7b-instruct`, `gpt-oss-120b`).

## Context-length caveat

The four candidates added on 2026-08-03 (`qwen3-4b`, `olmo3-7b-instruct`,
`gemma4-{12b,31b}-it`) each pin `--max-model-len 16384` in `vllm_args`;
`gpt-oss-120b` and `qwen3-0.6b` deliberately keep the settings they were already
served and scored with. These are 32K-256K-context models, and vLLM sizes its KV
cache from the model's *declared* maximum, so left alone `gemma4-31b-it`'s 256K
window fails to allocate on an 80GB card before it serves a single request. 16384
is chosen against measured demand, not guessed: the largest prompt across the
runs on disk is ~6.4K tokens, plus a 1024-token completion budget. Note the
failure mode if it were set too low -- a mid-run 400 on one long document, not a
startup error -- which is why the headroom is deliberate rather than tight.

## Serving

Every self-served candidate LLM (`gpt-oss-120b`, `qwen3-0.6b`, `qwen3-4b`,
`olmo3-7b-instruct`, `gemma4-12b-it`, `gemma4-31b-it`) declares
`serving: local_vllm`: `submit.sh` submits one self-contained qsub job per model
that starts its vLLM server, waits for health, runs the extraction, and tears the
server down on exit (see `experiments/serving.py`'s `endpoint_for`) -- there is no
separately-submitted "server job" a client job calls into over the network.

`serving: external` covers a commercial API (`gpt-5.6-terra`, `claude-sonnet-5`,
`gemini-3.7-flash`) -- somebody else owns that server's lifecycle, which is a real
distinction from local vLLM, not legacy. Such a model declares no `base_url_env`,
and `endpoint_for()` resolves it to `None`, i.e. the OpenAI SDK's own default
endpoint. It still needs a real API key in `api_key_env`.

## Hardware fields

`hardware.gpu_compute_capability` and `hardware.max_walltime` size the qsub
request `submit.sh` builds per model (`-l gpu_c=...`, `-l h_rt=...`) -- migrated
out of the old hand-written `serve_*.sh`/`extract_*.sh` scripts, where these were
hardcoded per file. `hardware.device: none` means a hosted API with no local
hardware need.
