# experiments

Compares extraction models along two axes: computational time and accuracy.
Accuracy has no gold-labeled dataset to check against, so it's measured by
proxy -- the `GROUND_TRUTH_KEY` model's output is treated as ground truth,
and the others are fuzzy-matched against it per document, per field. See
`config.py`'s module docstring for the full reproducibility / serving /
reasoning-mode design rationale.

Ground truth is `gpt-5.6-terra` (hosted OpenAI API) as of 2026-08-03;
`gpt-oss-120b`, previously ground truth, is now scored as a candidate
alongside `Qwen3-0.6B` and GLiNER.

Four more self-served candidates were added on 2026-08-03 -- `qwen3-4b`,
`olmo3-7b-instruct`, `gemma4-12b-it`, `gemma4-31b-it`. They use the same
serve/extract/evaluate path as the existing ones; see
[Serving the self-hosted models](#serving-the-self-hosted-models) for the
per-model ports and the vLLM image they need.

Two caveats specific to a hosted ground truth. Its `wall_seconds` include
network and provider queueing, so it is *not* comparable to the locally
served models on the computational-time axis, even though `results.py`
plots them together. And it rejects an explicit `temperature`, so the
`temperature=0` determinism the self-served models get is unavailable for
it -- ground-truth output is not reproducible even in the best-effort sense
described in `config.py`. Re-run it and you get slightly different truth.

## Layout

```
config.py       MODEL_REGISTRY + ExperimentConfig -- static, committed, "what did we intend to run"
runtime.py       git sha / GPU / package-version capture -> RunManifest, per run
serving.py        LocalVLLMServer + endpoint_for()'s lookup order (file cache -> env var)
serve_model.py      uv run -m experiments.serve_model -- start+register a server, foreground
runner.py             uv run -m experiments.runner   -- extraction + timing
similarity.py           fuzzy per-field comparators + FUSION_COMPARATORS / AGREEMENT_THRESHOLDS
evaluate.py               uv run -m experiments.evaluate -- score a candidate run vs. ground truth
fuse.py                     uv run -m experiments.fuse -- combine the frontier panel's runs into the gold set
results.py                  uv run -m experiments.results  -- summary table + plots
results.ipynb                  interactive counterpart to results.py -- same DataFrame + plots, inline
```

Outputs (`runs/`, `evaluations/`, `reports/` gitignored, regenerate on demand):
```
runs/<run_id>/                                 one per runner.py invocation
evaluations/<candidate_run_id>__vs__<truth_run_id>/
reports/{summary_table.csv, plots/*.png}
data/validation_gold/                          the fused ground-truth set -- COMMITTED (see "Ground-truth dataset" below)
```

## Ground-truth dataset (validation set)

`experiments/README.md`'s comparisons above use `gpt-5.6-terra`'s output as a
*proxy* for truth. Separately, there is a real gold-labeled set for the
held-out **validation** documents, built by 2-of-3 consensus across a
frontier-model panel:

| model_key | model | endpoint |
| --- | --- | --- |
| `gpt-5.6-terra` | `gpt-5.6-terra` | hosted OpenAI (already configured) |
| `claude-sonnet-5` | `claude-sonnet-5` | Anthropic OpenAI-compat (`$GOVSCAPE_ANTHROPIC_BASE_URL`) |
| `gemini-3.7-flash` | `gemini-3.7-flash` | Google Gemini OpenAI-compat (`$GOVSCAPE_GEMINI_BASE_URL`) |

Priority order (left to right) is also the tie-break: on a field with
consensus, the gold value is taken **verbatim from the highest-ranked model
in the agreeing set**. `VALIDATION_PANEL_KEYS` in `config.py` is the single
source of both.

```bash
# 0. Build the validation set (disjoint from data/sample_ocr/)
uv run data/build_validation.py -n 100 --seed 771

# 1. Run each panel model over it. Smoke-test at --limit 3 first.
#    All hosted APIs -> --skip-warmup. Needs real API keys in .env
#    (GOVSCAPE_ANTHROPIC_API_KEY, GOVSCAPE_GEMINI_API_KEY; gpt-5.6-terra uses
#    GOVSCAPE_LLM_API_KEY). Base URLs default in .env.example.
uv run -m experiments.runner --model-key gpt-5.6-terra    --experiment validation --run-name gold --skip-warmup
uv run -m experiments.runner --model-key claude-sonnet-5  --experiment validation --run-name gold --skip-warmup
uv run -m experiments.runner --model-key gemini-3.7-flash --experiment validation --run-name gold --skip-warmup

# 2. Fuse. With no --run it picks the latest run per VALIDATION_PANEL_KEYS.
uv run -m experiments.fuse

# 3. Score any candidate run (over data/validation_ocr) against the gold set:
uv run -m experiments.evaluate --truth-run data/validation_gold \
    --candidate-run <candidate run_id>
```

`data/validation_gold/` is **committed** (the one exception to this repo's
gitignore-because-regenerable policy -- it costs three frontier-model API
passes). It is written run-shaped (`manifest.json` + `metadata/<digest>.json`)
so `evaluate.py`/`results.py` treat it as an ordinary truth run. Alongside:

- `status.json` -- `{digest: {field: {status, n_present, n_agree, source_model}}}` where
  `status` is `consensus` / `consensus_null` / `flagged` / `partial_list`. `n_present < 3`
  (a panel model errored on that doc) or `n_agree < n_present` means the value rests on fewer
  than the full 3 -- filter on those to review borderline consensus.
- `flagged.csv` -- one row per `flagged` / `partial_list` `(digest, field)`: reason
  plus all three models' raw values. **This is the worklist for manual
  extraction** of the pairs the panel couldn't agree on.
- `fusion_report.json` -- per-field resolved/flagged counts, pairwise model
  agreement, `gpt-5.6-terra_outvoted` pairs (where the other two models
  agreed against it), and per-document detail.

Fusion rule, per `(document, field)`:

- **scalar fields** -- agreement = `FUSION_COMPARATORS[field]` score >=
  `AGREEMENT_THRESHOLDS[field]` (stricter than evaluate.py's 0.75; exact for
  `document_type` / `jurisdiction_level` / `report_number`). Largest
  connected component of the agreement graph; size >= 2 -> consensus (value
  from its highest-priority member), else **flagged** (gold value `null`).
  Two models returning null *is* agreement -> `consensus_null`, distinct from
  flagged.
- **list fields** (`authors`, `geographic_coverage`) -- fused element-wise: an
  element enters the gold list when >= 2 models contribute a fuzzily-matching
  element. Field flagged only if *no* element reaches 2-of-3; if some agree
  and some don't, the field keeps the agreed subset and the dropped elements
  are logged (`partial_list`).
- `title` / `issuing_agency` / `performing_organization` / `series` use a
  length-guarded comparator so a truncated value doesn't falsely "agree" with
  the full one (plain `token_set_ratio` is deliberately superset-blind).

Caveat: the panel is capability-asymmetric -- `gpt-5.6-terra` and
`claude-sonnet-5` are both frontier, `gemini-3.7-flash` is a Flash-class model.
If two models share an error they will outvote a correct third -- skim
`gpt-5.6-terra_outvoted` in `fusion_report.json`.

## Setup

```bash
uv sync --extra experiments
```

`vllm` is deliberately **not** a dependency of this project (it pins torch/
transformers versions incompatible with `govscape`'s -- confirmed by trying).
`serving.py`'s `LocalVLLMServer` only shells out to a `vllm serve` subprocess,
it never imports `vllm` in-process, so install it separately wherever it
actually runs, e.g. `uv tool install vllm`.

## Serving the self-hosted models

Six models are self-served: `gpt-oss-120b`, `qwen3-0.6b`, `qwen3-4b`,
`olmo3-7b-instruct`, `gemma4-12b-it`, `gemma4-31b-it`. Each has a matching
pair of SGE scripts in this directory, and each serve script uses a distinct
port so several can share a node:

| model key | HF model | port | `gpu_memory` | scripts |
| --- | --- | --- | --- | --- |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | 8082 | 80G | `{serve,extract}_gpt_oss_120b.sh` |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | 8082 | 24G | `{serve,extract}_qwen_3_0_6b.sh` |
| `qwen3-4b` | `Qwen/Qwen3-4B` | 8083 | 24G | `{serve,extract}_qwen3_4b.sh` |
| `olmo3-7b-instruct` | `allenai/Olmo-3-7B-Instruct` | 8084 | 24G | `{serve,extract}_olmo3_7b_instruct.sh` |
| `gemma4-12b-it` | `google/gemma-4-12B-it` | 8085 | 48G | `{serve,extract}_gemma4_12b_it.sh` |
| `gemma4-31b-it` | `google/gemma-4-31B-it` | 8086 | 80G | `{serve,extract}_gemma4_31b_it.sh` |

The two Gemma entries are the **`-it`** (instruction-tuned) checkpoints, not
the base `google/gemma-4-12B` / `google/gemma-4-31B` repos. That is not a
cosmetic preference: the base checkpoints ship no chat template at all (no
`chat_template.jinja`, no `chat_template` key in `tokenizer_config.json`), and
every request this project makes goes through `/v1/chat/completions`, which
vLLM refuses to serve without one. Both Gemma 4 models are also multimodal
(`image-text-to-text`); this harness only ever sends text, which is a
supported subset, not a workaround.

**Which SIF image**: all four new models need
`vllm-openai_v0.24.0.sif`, which is what every `serve_*.sh` here already
points at. Checked against the containers' actual model registries rather
than assumed -- `vllm-gemma4.sif` is vLLM 0.18.2rc1 and, name
notwithstanding, does *not* register `Gemma4UnifiedForConditionalGeneration`,
the architecture `gemma4-12b-it` uses. It would fail on that model while
succeeding on `gemma4-31b-it`, which is exactly the kind of half-working
setup worth not discovering at 3am.

All six default to `serving="external"` in `config.py` -- served manually on a
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

**Serving from a Singularity/Apptainer image instead of a bare `vllm` on
PATH** (common on HPC clusters): override the base launch command with
`--vllm-command` (or `$GOVSCAPE_VLLM_COMMAND` to set it once and not repeat
it per invocation) -- everything else (health check, endpoint file, teardown)
works unchanged, since Singularity/Apptainer share the host's network
namespace by default (unlike Docker, no `-p` port mapping needed):

```bash
uv run -m experiments.serve_model --model-key gpt-oss-120b \
    --vllm-command "singularity exec --nv /path/to/vllm.sif vllm serve"

# or set it once for the session:
export GOVSCAPE_VLLM_COMMAND="apptainer exec --nv /path/to/vllm.sif vllm serve"
uv run -m experiments.serve_model --model-key gpt-oss-120b
```

`--nv` passes the GPU through to the container. If your `.sif` image's model
cache isn't already visible inside the container (Singularity mounts `$HOME`
by default, so usually `~/.cache/huggingface` just works), add a `--bind
<host-path>:<container-path>` into the `--vllm-command` string.

**2. Env var fallback**, if you'd rather run `vllm serve` by hand or
`serve_model.py`'s file cache isn't reachable from where the runner runs
(add to a local `.env`, alongside the existing `GOVSCAPE_LLM_*` vars):

```bash
GOVSCAPE_GPT_OSS_120B_BASE_URL=http://<gpu-box>:8082/v1
GOVSCAPE_QWEN3_0_6B_BASE_URL=http://<gpu-box>:8082/v1
GOVSCAPE_QWEN3_4B_BASE_URL=http://<gpu-box>:8083/v1
GOVSCAPE_OLMO3_7B_INSTRUCT_BASE_URL=http://<gpu-box>:8084/v1
GOVSCAPE_GEMMA4_12B_IT_BASE_URL=http://<gpu-box>:8085/v1
GOVSCAPE_GEMMA4_31B_IT_BASE_URL=http://<gpu-box>:8086/v1
GOVSCAPE_LLM_API_KEY=EMPTY   # vLLM ignores it, but the OpenAI SDK requires a non-empty string
```

(These are also listed, blank, in the repo-root `.env.example`.)

`--base-url` on `runner.py` overrides both of the above for a single
invocation, e.g. for a quick test against Ollama.

## Usage

```bash
# 0. On the GPU box: start the LLM servers (see above), leave them running.
#    On SGE, submit the paired scripts instead of running these by hand --
#    the extract job must be held until the server job is up, and -hold_jid
#    has to be a qsub argument, not a directive inside the extract script:
#      qsub serve_gemma4_31b_it.sh                             # note the job id
#      qsub -hold_jid <that job id> extract_gemma4_31b_it.sh
#    Submitting both unheld starts extraction before vLLM is listening, and
#    endpoint_for() fails with "no reachable endpoint found".
uv run -m experiments.serve_model --model-key gpt-oss-120b      --port 8082 &
uv run -m experiments.serve_model --model-key qwen3-0.6b        --port 8082 &
uv run -m experiments.serve_model --model-key qwen3-4b          --port 8083 &
uv run -m experiments.serve_model --model-key olmo3-7b-instruct --port 8084 &
uv run -m experiments.serve_model --model-key gemma4-12b-it     --port 8085 &
uv run -m experiments.serve_model --model-key gemma4-31b-it     --port 8086 &

# 1. Run extraction for each model (writes runs/<run_id>/)
#    gpt-5.6-terra is a hosted API: no server to start, but it needs a real
#    GOVSCAPE_LLM_API_KEY in the repo-root .env (runner.py load_dotenv()s it).
#    --skip-warmup: the warmup call amortizes local model load, which a
#    hosted API doesn't have, so it just bills you for docs[0] twice.
uv run -m experiments.runner --model-key gpt-5.6-terra     --run-name baseline --skip-warmup
uv run -m experiments.runner --model-key gpt-oss-120b      --run-name baseline
uv run -m experiments.runner --model-key qwen3-0.6b        --run-name baseline
uv run -m experiments.runner --model-key gliner2-base      --run-name baseline
uv run -m experiments.runner --model-key qwen3-4b          --run-name candidate
uv run -m experiments.runner --model-key olmo3-7b-instruct --run-name candidate
uv run -m experiments.runner --model-key gemma4-12b-it     --run-name candidate
uv run -m experiments.runner --model-key gemma4-31b-it     --run-name candidate

# 1b. If a run dies partway (network drop, walltime kill, Ctrl+C), continue it
#     in place -- same run_id, same directory, no re-extraction of what's done:
uv run -m experiments.runner --model-key gpt-5.6-terra --resume <run_id> --skip-warmup

# 2. Score each candidate against the ground-truth run (writes evaluations/<...>/)
#    -- one invocation per candidate run_id, including the four added 2026-08-03.
uv run -m experiments.evaluate --truth-run <gpt-5.6-terra run_id> --candidate-run <gpt-oss-120b run_id>
uv run -m experiments.evaluate --truth-run <gpt-5.6-terra run_id> --candidate-run <qwen3-0.6b run_id>
uv run -m experiments.evaluate --truth-run <gpt-5.6-terra run_id> --candidate-run <gliner2-base run_id>
uv run -m experiments.evaluate --truth-run <gpt-5.6-terra run_id> --candidate-run <qwen3-4b run_id>
uv run -m experiments.evaluate --truth-run <gpt-5.6-terra run_id> --candidate-run <olmo3-7b-instruct run_id>
uv run -m experiments.evaluate --truth-run <gpt-5.6-terra run_id> --candidate-run <gemma4-12b-it run_id>
uv run -m experiments.evaluate --truth-run <gpt-5.6-terra run_id> --candidate-run <gemma4-31b-it run_id>

# 3. Compile the summary table + plots (writes reports/)
uv run -m experiments.results
```

`results.py` picks the most recent run per model automatically; pass
`--run <run_id>` (repeatable) to select specific runs explicitly.

## Resuming an interrupted run

`runner.py` rewrites `manifest.json` after every document, so a run that dies
partway leaves a valid manifest with `finished_at: null` -- that field is the
marker for "incomplete", and `evaluate.py` refuses such a run rather than
silently scoring its missing documents as extraction failures (override with
`--allow-unfinished`).

`--resume <run_id>` continues that run in place. A document is skipped if it
already has a non-null result; documents whose previous attempt *errored*
(runner writes a literal `null` for those) are retried, so a resume doubles
as a retry pass for transient API failures. `--run-name` and `--resume` are
mutually exclusive. Note the flip side of retrying nulls: a document that
fails *deterministically* (OCR that always breaks the JSON parse, say) is
retried on every resume and the run never reaches 0 errors -- at that point
the document needs looking at, not another resume.

Scope and windowing are read back from the run's own manifest, so you don't
re-pass `--limit`/`--max-pages`/`--max-chars` -- passing one that contradicts
the manifest is an error rather than an override, since blending two
windowings into one run directory produces a reference set no evaluation can
interpret. Resume also refuses if the input directory no longer holds the
same documents the run covered (comparing digests, not just counts -- a
regenerated sample can have an identical document count and share none of
them).

The manifest's top-level `git_sha`/`package_versions` stay those of the
*original* invocation; each resume appends its own to `resume_events`, so
environment drift between the start of a run and its completion stays
visible rather than overwriting the original record.

For interactive exploration (filtering/sorting the summary table, digging into a
single model's raw per-document timings) instead of the one-shot CLI, open
`results.ipynb`. It imports and reuses `results.py`'s functions directly, so its
output never drifts from the CLI's -- run `uv sync --extra experiments` first
(pulls in `ipykernel`).
