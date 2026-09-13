#!/bin/bash -l
# Submit (or run directly) one experiment's model_keys, self-contained per
# model -- each GPU-requiring model_key's server and client run in one qsub
# job (see experiments/serving.py's endpoint_for / local_vllm), not a
# separately-submitted server job called from another job. Resource requests
# come from that model's own experiments/model-configs/<key>.yaml, not a
# fixed per-experiment-type table -- a gemma4-31b-it job and a qwen3-0.6b job
# have nothing in common.
#
# Usage:
#   bash experiments/submit.sh <experiment-id>                # every params.model_keys entry
#   bash experiments/submit.sh <experiment-id> --model-key K   # just one
#
# A model_key whose hardware.device isn't "cuda" (a hosted API, or GLiNER on
# CPU) runs directly in this shell instead of through qsub -- matching
# CLAUDE.local.md's "API-based work runs on a login node, no qsub" guidance.
#
# Deliberately does NOT aggregate afterward, and does NOT chain a follow-up
# job with -hold_jid: once `qstat` shows the submitted job(s) finished, run
#   uv run -m experiments.run_extraction   <config.yaml> --aggregate
#   uv run -m experiments.run_ground_truth <config.yaml> --aggregate [--promote]
# by hand -- see notes/hub/safety.md on staying in the loop rather than
# auto-chaining cluster jobs.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash experiments/submit.sh <experiment-id> [--model-key <key>]" >&2
    exit 1
fi

EXPERIMENT_ID="$1"
shift

ONLY_MODEL_KEY=""
if [ "${1:-}" = "--model-key" ]; then
    ONLY_MODEL_KEY="${2:?--model-key requires a value}"
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p experiments/out

IFS=$'\t' read -r CONFIG_PATH EXPERIMENT_TYPE ALL_MODEL_KEYS \
    < <(uv run -m experiments.submit_query experiment-info "$EXPERIMENT_ID")

case "$EXPERIMENT_TYPE" in
    extraction)   RUNNER_MODULE=experiments.run_extraction ;;
    ground_truth) RUNNER_MODULE=experiments.run_ground_truth ;;
    *)
        echo "Error: $CONFIG_PATH has params.experiment_type='$EXPERIMENT_TYPE'; expected 'extraction' or 'ground_truth'." >&2
        exit 1
        ;;
esac

if [ -n "$ONLY_MODEL_KEY" ]; then
    MODEL_KEYS="$ONLY_MODEL_KEY"
else
    MODEL_KEYS="$ALL_MODEL_KEYS"
fi

SGE_PROJECT_ARGS=()
if [ -n "${SGE_PROJECT:-}" ]; then
    SGE_PROJECT_ARGS=(-P "$SGE_PROJECT")
fi

SUBMITTED_JOB_IDS=()

for MODEL_KEY in $MODEL_KEYS; do
    IFS=$'\t' read -r DEVICE GPU_COUNT MIN_VRAM_GB GPU_COMPUTE_CAPABILITY MAX_WALLTIME \
        < <(uv run -m experiments.submit_query model-hardware "$MODEL_KEY")

    if [ "$DEVICE" != "cuda" ]; then
        echo "Running $MODEL_KEY directly (hardware.device=$DEVICE, no GPU needed)..."
        uv run -m "$RUNNER_MODULE" "$CONFIG_PATH" --model-key "$MODEL_KEY"
        continue
    fi

    JOB_NAME="$(printf '%s' "job_${EXPERIMENT_ID}_${MODEL_KEY}" | tr -c 'A-Za-z0-9_.-' '_')"
    QSUB_ARGS=(
        -N "$JOB_NAME"
        -l "h_rt=$MAX_WALLTIME"
        -pe omp 16
        -l "gpus=$GPU_COUNT"
        -o "$REPO_DIR/experiments/out/${EXPERIMENT_ID}__${MODEL_KEY}_out.txt"
        -e "$REPO_DIR/experiments/out/${EXPERIMENT_ID}__${MODEL_KEY}_error.txt"
        -m e
        -v "REPO_DIR=$REPO_DIR,RUNNER_MODULE=$RUNNER_MODULE,CONFIG_PATH=$CONFIG_PATH,MODEL_KEY=$MODEL_KEY"
    )
    if [ -n "$MIN_VRAM_GB" ]; then
        QSUB_ARGS+=(-l "gpu_memory=${MIN_VRAM_GB}G")
    fi
    if [ -n "$GPU_COMPUTE_CAPABILITY" ]; then
        QSUB_ARGS+=(-l "gpu_c=$GPU_COMPUTE_CAPABILITY")
    fi

    echo "Submitting GPU job for $MODEL_KEY (gpus=$GPU_COUNT gpu_memory=${MIN_VRAM_GB:-unset}G gpu_c=${GPU_COMPUTE_CAPABILITY:-unset} h_rt=$MAX_WALLTIME)..."
    JOB_OUTPUT="$(qsub "${SGE_PROJECT_ARGS[@]}" "${QSUB_ARGS[@]}" "$REPO_DIR/experiments/run_job.sh")"
    echo "  $JOB_OUTPUT"
    SUBMITTED_JOB_IDS+=("$(echo "$JOB_OUTPUT" | grep -oE '[0-9]+' | head -1)")
done

if [ ${#SUBMITTED_JOB_IDS[@]} -gt 0 ]; then
    echo
    echo "Submitted job(s): ${SUBMITTED_JOB_IDS[*]}"
    echo "Once \`qstat\` shows them finished, aggregate with:"
    echo "  uv run -m $RUNNER_MODULE $CONFIG_PATH --aggregate"
fi
