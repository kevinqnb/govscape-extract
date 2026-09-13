#!/bin/bash -l
# Generic SGE job body submitted by experiments/submit.sh. Env vars are set
# via `qsub -v` by submit.sh: REPO_DIR, RUNNER_MODULE, CONFIG_PATH, MODEL_KEY.
#
# Self-contained: MODEL_KEY's model-configs/<key>.yaml declares
# serving: local_vllm, so runner.py starts and tears down its own vLLM
# server inside this same job (see experiments/serving.py's endpoint_for) --
# there is no separately-submitted server job.
#
# The module loads / Singularity bootstrap below is carried over verbatim
# from the pre-restructure serve_*.sh scripts (the proven-working recipe for
# vLLM on this cluster -- vllm is deliberately not a project dependency, see
# pyproject.toml), just run in the same job as the client now instead of a
# separate one. Confirm with a real smoke submission before trusting it, and
# record the outcome in CLAUDE.local.md's "Module loads / job bootstrap"
# section (still a TODO there as of this restructure).

set -euo pipefail

module load python3/3.12.4
module load cuda/12.5

SIF_IMAGE="${VLLM_SIF_DIR:?VLLM_SIF_DIR is not set}/vllm-openai_v0.24.0.sif"
HF_CACHE="${HF_CACHE:?HF_CACHE is not set}"
TMPDIR=/projectnb/mcnet/kevin/tmp
VLLM_CACHE_ROOT=/projectnb/mcnet/kevin/vllm_cache
TRITON_CACHE_DIR=/projectnb/mcnet/kevin/triton_cache
mkdir -p "$TMPDIR" "$HF_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

export GOVSCAPE_VLLM_COMMAND="singularity exec --nv \
    --bind /projectnb/mcnet/kevin:/projectnb/mcnet/kevin \
    --env HF_HOME=$HF_CACHE,TMPDIR=$TMPDIR,VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT,TRITON_CACHE_DIR=$TRITON_CACHE_DIR \
    $SIF_IMAGE vllm serve"

cd "$REPO_DIR"
uv run -m "$RUNNER_MODULE" "$CONFIG_PATH" --model-key "$MODEL_KEY"
