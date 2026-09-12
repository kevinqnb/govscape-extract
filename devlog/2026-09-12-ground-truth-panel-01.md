<!-- Public devlog entry: govscape-extract/devlog/2026-09-12-ground-truth-panel-01.md.
Written by /devlog as a trimmed version of the private build note at
notes/govscape-extract/builds/2026-09-12-ground-truth-panel-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-12-ground-truth-panel-01
kind: build
---

## Session 2026-09-12

### Prompts

(Build work happened in an earlier session, ~2026-08-30/31; this entry and the
commit were written up on 2026-09-12.)

1. Build a ground-truth dataset for comparing model extraction accuracy against —
   starting with a validation-set sampling script analogous to `data/build_sample.py`.
2. Use a 3-model panel (claude-haiku-4-5 and gemini-3.7-flash alongside
   gpt-5.6-terra, later swapped to claude-sonnet-5) with a priority-ordered tiebreak:
   on 2-of-3 agreement, take the value verbatim from the highest-ranked model.
3. Implement it, making sure pairs where the panel doesn't agree are clearly tracked
   for manual review later.
4. Debugging round against the real APIs: Anthropic's OpenAI-compat endpoint
   rejecting `response_format={"type":"json_object"}` and an explicit `temperature`;
   Gemini's endpoint rejecting `seed`; a short-by-2 run traced to a token-budget
   cutoff.
5. Once fused, asked for agreement statistics across the panel.

### Implemented

Extracted the sampling core out of `data/build_sample.py` into `data/sampling.py`,
shared by a new `data/build_validation.py` (disjoint validation set via
`--exclude-dir`). Added `claude-sonnet-5` and `gemini-3.7-flash` to
`experiments/config.py`'s `MODEL_REGISTRY`, a `VALIDATION_PANEL_KEYS` priority list,
and a `"validation"` `ExperimentConfig`. Hardened `LLMExtractor`
(`govscape_extract/extractors/llm.py`) with a per-model `response_format` override,
`temperature`/`seed` opt-outs, a one-shot retry on a rejected `response_format`,
refusal/empty-content handling, and lenient JSON parsing (fenced/preamble output).
Added `experiments/fuse.py` plus `FUSION_COMPARATORS` /
`AGREEMENT_THRESHOLDS` in `experiments/similarity.py` to combine the panel's three
runs into a gold set by per-field fuzzy-agreement graph, writing the committed
`data/validation_gold/` (`manifest.json`, `metadata/`, `status.json`,
`flagged.csv`, `fusion_report.json`). Ran the panel over the 100-doc validation set;
79 `(document, field)` pairs didn't reach 2-of-3 agreement and are logged in
`flagged.csv` as a manual-review worklist. No `configs/<id>.yaml` yet — that
retrofit (plus `scripts/run_experiment.py`/`submit.sh`) is deferred to the next
build.

### Commits
3382fca Add frontier-model consensus panel and ground-truth validation set
