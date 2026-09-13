# experiments

Compares extraction models along two axes: computational time and accuracy.
Accuracy has no gold-labeled dataset to check against, so it's measured by
proxy -- the `GROUND_TRUTH_KEY` model's output is treated as ground truth,
and the others are fuzzy-matched against it per document, per field. See
`model-configs/README.md` for the full reproducibility / serving /
reasoning-mode design rationale, and each `model-configs/<key>.yaml`'s own
`notes:` field for that model's specifics.

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

Centralized experiment infrastructure, per-experiment-type orchestrators over
shared building blocks:

```
model-configs/<key>.yaml               one file per model/method -- see model-configs/README.md
dataset-configs/<dataset>.yaml         splits (input_dir) + default windowing for a dataset
experiment-configs/<dataset>/<type>/<id>/<id>.yaml + out/    one committed config per experiment, run output colocated
config.py       loads model-configs/ + dataset-configs/ into HardwareRequirement/LLMModelConfig/GlinerModelConfig/DatasetConfig
utils.py         ExperimentSpec loading + the experiment-level out/run.json, out/metrics.json writers
runtime.py       git sha / GPU / package-version capture -> RunManifest, per model run
serving.py        LocalVLLMServer + endpoint_for()'s lookup order (file cache -> env var)
serve_model.py      uv run -m experiments.serve_model -- start+register a server, foreground
runner.py             run_model() -- one model's extraction + timing over a dataset split; uv run -m experiments.runner is a thin CLI over it
evaluate.py               evaluate_run() -- score a candidate run vs. a truth run; uv run -m experiments.evaluate is a thin CLI over it
run_extraction.py           uv run -m experiments.run_extraction <config.yaml> [--model-key K] [--aggregate] -- "extraction" experiment type
run_ground_truth.py           uv run -m experiments.run_ground_truth <config.yaml> [--model-key K] [--aggregate] [--promote] -- "ground_truth" experiment type
results.py                      uv run -m experiments.results  -- summary table + plots
results.ipynb                      interactive counterpart to results.py -- same DataFrame + plots, inline
```

Fuzzy comparators (`FIELD_COMPARATORS`/`FUSION_COMPARATORS`/`AGREEMENT_THRESHOLDS`/`score_document`)
and the consensus-voting algorithm (`fuse_scalar`/`fuse_list`) live in
`govscape_extract/similarity.py` and `govscape_extract/consensus.py` -- package
capability, not experiment-orchestration glue, colocated with `schema.py`
which they're keyed to.

Outputs: each experiment's `out/` is colocated with its config
(`experiment-configs/.../<id>/out/`, gitignored), containing `run.json` /
`config.snapshot.yaml` / `metrics.json` plus per-model run directories under
`out/runs/<run_name>__<model_key>__<ts>/` (unchanged shape from before this
restructure) and, for `extraction` experiments with a `truth_run`,
`out/evaluations/<candidate>__vs__<truth>/`. A `ground_truth` experiment also
writes `out/fused/` (manifest.json + metadata/ + status.json + flagged.csv +
fusion_report.json) -- promoting a specific one to the committed
`data/validation_gold/` is the separate, explicit `--promote` step.

The legacy flat `runs/`, `evaluations/`, `reports/` directories (pre-dating
this restructure) still hold historical runs and are left as-is.

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
in the agreeing set**. A `ground_truth`-type experiment's `params.model_keys`
(in priority order) is the single source of both -- see
`experiment-configs/govscape/ground_truth/2026-09-12-ground-truth-smoke-01/`
for a worked example (CPU-only smoke, no live panel calls -- copy that
config's `params` and point `split: validation` at a real run to do this for
real).

```bash
# 0. Build the validation set (disjoint from data/sample_ocr/)
uv run data/build_validation.py -n 100 --seed 771

# 1. Run each panel model over it (needs real API keys in .env --
#    GOVSCAPE_ANTHROPIC_API_KEY, GOVSCAPE_GEMINI_API_KEY; gpt-5.6-terra uses
#    GOVSCAPE_LLM_API_KEY). Smoke-test with --limit 3 first.
uv run -m experiments.run_ground_truth <config.yaml> --model-key gpt-5.6-terra
uv run -m experiments.run_ground_truth <config.yaml> --model-key claude-sonnet-5
uv run -m experiments.run_ground_truth <config.yaml> --model-key gemini-3.7-flash

# 2. Fuse the three runs + write the experiment-level out/run.json, out/metrics.json:
uv run -m experiments.run_ground_truth <config.yaml> --aggregate

# 3. Review out/fused/fusion_report.json + flagged.csv, then promote to the
#    committed gold set (never automatic -- a routine or smoke run should
#    never silently overwrite it):
uv run -m experiments.run_ground_truth <config.yaml> --promote

# 4. Score any candidate run (over data/validation_ocr) against the gold set:
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
`olmo3-7b-instruct`, `gemma4-12b-it`, `gemma4-31b-it`. Each declares
`serving: local_vllm` in its `model-configs/<key>.yaml` -- **self-contained**:
`bash experiments/submit.sh <experiment-id> --model-key <key>` submits one
qsub job per model that starts its vLLM server, waits for health, runs the
extraction, and tears the server down on exit (see `serving.py`'s
`endpoint_for`) -- there is no separately-submitted server job a client job
calls into over the network. GPU resource requests (`gpus`, `gpu_memory`,
`gpu_c`, `h_rt`) come from that model's own `hardware` block, not a shared
table -- see `model-configs/README.md`.

The two Gemma entries are the **`-it`** (instruction-tuned) checkpoints, not
the base `google/gemma-4-12B` / `google/gemma-4-31B` repos. That is not a
cosmetic preference: the base checkpoints ship no chat template at all (no
`chat_template.jinja`, no `chat_template` key in `tokenizer_config.json`), and
every request this project makes goes through `/v1/chat/completions`, which
vLLM refuses to serve without one. Both Gemma 4 models are also multimodal
(`image-text-to-text`); this harness only ever sends text, which is a
supported subset, not a workaround.

**Which SIF image**: all four candidates added 2026-08-03 need
`vllm-openai_v0.24.0.sif` -- checked against the containers' actual model
registries rather than assumed. `vllm-gemma4.sif` is vLLM 0.18.2rc1 and, name
notwithstanding, does *not* register `Gemma4UnifiedForConditionalGeneration`,
the architecture `gemma4-12b-it` uses. It would fail on that model while
succeeding on `gemma4-31b-it`, which is exactly the kind of half-working
setup worth not discovering at 3am. `experiments/run_job.sh` (the generic
qsub body `submit.sh` invokes) points at this image and sets
`$GOVSCAPE_VLLM_COMMAND` accordingly -- see its own comments for the module
loads and cache-directory bootstrap, carried over from this cluster's
previously-working (pre-restructure) recipe.

vLLM is deliberately **not** a project dependency (it pins torch/transformers
versions incompatible with `govscape`'s -- confirmed by trying);
`serving.py`'s `LocalVLLMServer` only ever shells out to a `vllm serve`
(or Singularity-wrapped `vllm serve`) subprocess, never imports it in-process.

**Manual/interactive serving** (smoke-testing a model without going through
`submit.sh`/qsub at all) is still available via `serve_model.py`:

```bash
uv run -m experiments.serve_model --model-key qwen3-0.6b --port 8001
# or, from inside a Singularity image:
uv run -m experiments.serve_model --model-key gpt-oss-120b \
    --vllm-command "singularity exec --nv /path/to/vllm.sif vllm serve"
```

This streams `vllm serve`'s normal startup logs and, once healthy, writes the
resolved URL to `experiments/.endpoints.json` (gitignored) -- but only
matters for a model whose `model-configs/<key>.yaml` still says
`serving: external` (a hosted API, or a self-served model you've
deliberately reverted for manual testing); `local_vllm` models never consult
that file, since `endpoint_for()` launches and owns the server itself.
`--base-url` on `runner.py`/`run_extraction.py`/`run_ground_truth.py`
overrides endpoint resolution entirely for a single invocation, e.g. for a
quick test against Ollama.

## Usage

```bash
# 0. Submit self-contained GPU jobs for the local-vllm candidates (each
#    starts its own server, runs, tears down -- see "Serving" above), and
#    run hosted-API / CPU model_keys directly (no GPU needed):
bash experiments/submit.sh <experiment-id>                        # every params.model_keys entry
bash experiments/submit.sh <experiment-id> --model-key qwen3-0.6b   # just one

# 1. Once `qstat` shows the submitted jobs finished, score + write the
#    experiment-level out/run.json and out/metrics.json:
uv run -m experiments.run_extraction <config.yaml> --aggregate

# 1b. If a run dies partway (network drop, walltime kill, Ctrl+C), continue
#     it in place -- same run_id, same directory, no re-extraction of what's
#     done. Find the run_id under out/runs/, then:
uv run -m experiments.runner --model-key <key> --resume <run_id>

# 2. Compile the summary table + plots (writes reports/)
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
