<!-- Public devlog entry: govscape-extract/devlog/2026-09-13-experiment-infra-01.md.
Written by /devlog as a trimmed version of the private build note at
notes/govscape-extract/builds/2026-09-13-experiment-infra-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-13-experiment-infra-01
kind: build
---

## Session 2026-09-13

### Prompts

1. Proposed a centralized experiment structure ahead of adding more datasets
   and extraction methods: `dataset-configs/`, `model-configs/` (per-model
   settings), `experiment-configs/{dataset}/{type}/{id}/{id}.yaml` + `out/`,
   `run_{type}.py` runners, `submit.sh`, `utils.py`. Requirement: a
   GPU-requiring local model's job must serve and call itself in one
   self-contained submission, never a separately-served job called from
   another. Asked to evaluate the proposal, design a plan together, then
   implement it.
2. Clarifications during planning: retire the old ad hoc job scripts once
   `submit.sh` covers the same jobs, rather than leaving them during a
   transition; keep run output colocated next to its config; move the
   comparator and consensus-voting logic into the main package as real
   capability, not experiment-orchestration glue.
3. Don't call any frontier-model APIs while building/testing the
   ground-truth side.
4. Don't submit a real cluster job without asking first.
5. Confirmed deleting the old ad hoc scripts once `submit.sh` existed.

### Implemented

Replaced the hardcoded model registry with per-model/dataset YAML configs
(`experiments/model-configs/*.yaml`, `experiments/dataset-configs/govscape.yaml`)
loaded and validated by `experiments/config.py`. Added `run_extraction.py` and
`run_ground_truth.py` as per-experiment-type orchestrators over refactored
callable seams in `runner.py`/`evaluate.py`, plus `submit.sh`/`run_job.sh` for
self-contained per-model GPU jobs sized from each model's own config. Moved
the fuzzy comparators and consensus-voting algorithm into the main
`govscape_extract` package; retired the old `experiments/fuse.py` and the 14
ad hoc job scripts. Found and fixed two real bugs along the way (a
`--resume` document-count regression, and local-vLLM jobs never picking up
the cluster's Singularity wrapper) — both caught by real smoke runs and
verified end-to-end without calling any frontier-model API or submitting a
real cluster job. Added a first test suite (23 tests).

### Commits
9352327 Add centralized experiment config/orchestration infrastructure
