# experiments

Compares extraction models along two axes: computational time and accuracy.
Accuracy has no gold-labeled dataset to check against, so it's measured by
proxy -- `gpt-oss-120b`'s output is treated as ground truth, and smaller
models (`Qwen3-0.6B`, GLiNER) are fuzzy-matched against it per document, per
field. See `config.py`'s module docstring for the full reproducibility /
serving / reasoning-mode design rationale.

## Layout

```
config.py       MODEL_REGISTRY + ExperimentConfig -- static, committed, "what did we intend to run"
runtime.py       git sha / GPU / package-version capture -> RunManifest, per run
serving.py        LocalVLLMServer + endpoint_for()'s lookup order (file cache -> env var)
serve_model.py      uv run -m experiments.serve_model -- start+register a server, foreground
runner.py             uv run -m experiments.runner   -- extraction + timing
similarity.py           fuzzy per-field comparators (title/authors/dates/agency/document_type)
evaluate.py               uv run -m experiments.evaluate -- score a candidate run vs. ground truth
results.py                  uv run -m experiments.results  -- summary table + plots
```

Outputs (all gitignored, regenerate on demand):
```
runs/<run_id>/                                 one per runner.py invocation
evaluations/<candidate_run_id>__vs__<truth_run_id>/
reports/{summary_table.csv, plots/*.png}
```

## Setup

```bash
uv sync --extra experiments
```

`vllm` is deliberately **not** a dependency of this project (it pins torch/
transformers versions incompatible with `govscape`'s -- confirmed by trying).
`serving.py`'s `LocalVLLMServer` only shells out to a `vllm serve` subprocess,
it never imports `vllm` in-process, so install it separately wherever it
actually runs, e.g. `uv tool install vllm`.

## Serving gpt-oss-120b / Qwen3-0.6B

Both default to `serving="external"` in `config.py` -- served manually on a
remote/persistent GPU box, not launched by `runner.py`. Two ways to point the
runner at them, in the order `endpoint_for()` tries them:

**1. `serve_model.py` (recommended -- no env var to re-set every session).**
On the GPU box (in a `tmux`/`screen` session so it survives you disconnecting):

```bash
uv run -m experiments.serve_model --model-key gpt-oss-120b   # port 8000 by default
uv run -m experiments.serve_model --model-key qwen3-0.6b --port 8001
```

This wraps `vllm serve` (using the `vllm_args` already declared per model in
`config.py`, e.g. gpt-oss-120b's `--reasoning-parser`), streams its normal
startup logs, and once the health check passes, writes the resolved URL to
`experiments/.endpoints.json` (gitignored). `runner.py`/`evaluate.py` then
find it automatically -- no env var needed. Ctrl+C (or `kill`, not `kill -9`)
stops the server and removes its entry.

This only works automatically if whatever runs `experiments.runner` can see
that same file -- i.e. you're also on the GPU box (or SSH'd into it), or it's
on a filesystem shared with wherever the runner runs. A cached entry is only
ever trusted after a live health check, so a stale or crashed server's row is
harmless -- it just falls through to option 2. If your setup is genuinely two
separate machines with no shared filesystem, use option 2 instead (or copy
`experiments/.endpoints.json` over yourself).

**2. Env var fallback**, if you'd rather run `vllm serve` by hand or
`serve_model.py`'s file cache isn't reachable from where the runner runs
(add to a local `.env`, alongside the existing `GOVSCAPE_LLM_*` vars):

```bash
GOVSCAPE_GPT_OSS_120B_BASE_URL=http://<gpu-box>:8000/v1
GOVSCAPE_QWEN3_0_6B_BASE_URL=http://<gpu-box>:8001/v1
GOVSCAPE_LLM_API_KEY=EMPTY   # vLLM ignores it, but the OpenAI SDK requires a non-empty string
```

`--base-url` on `runner.py` overrides both of the above for a single
invocation, e.g. for a quick test against Ollama.

## Usage

```bash
# 0. On the GPU box: start the two LLM servers (see above), leave them running
uv run -m experiments.serve_model --model-key gpt-oss-120b &
uv run -m experiments.serve_model --model-key qwen3-0.6b --port 8001 &

# 1. Run extraction for each model (writes runs/<run_id>/)
uv run -m experiments.runner --model-key gpt-oss-120b --run-name baseline
uv run -m experiments.runner --model-key qwen3-0.6b   --run-name baseline
uv run -m experiments.runner --model-key gliner2-base --run-name baseline

# 2. Score each candidate against the ground-truth run (writes evaluations/<...>/)
uv run -m experiments.evaluate --truth-run <gpt-oss-120b run_id> --candidate-run <qwen3-0.6b run_id>
uv run -m experiments.evaluate --truth-run <gpt-oss-120b run_id> --candidate-run <gliner2-base run_id>

# 3. Compile the summary table + plots (writes reports/)
uv run -m experiments.results
```

`results.py` picks the most recent run per model automatically; pass
`--run <run_id>` (repeatable) to select specific runs explicitly.
