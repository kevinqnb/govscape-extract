"""Combine several models' per-field outputs for the same document into one
consensus value -- the voting algorithm behind the ground-truth dataset (see
experiments/run_ground_truth.py and experiments/README.md's "Ground-truth
dataset" section for the run-directory orchestration this plugs into).

For a scalar field: two values "agree" per
govscape_extract.similarity.FUSION_COMPARATORS / AGREEMENT_THRESHOLDS; the
largest mutually-agreeing group wins, tie-broken toward the highest-priority
member of `priority`. A list field (authors, geographic_coverage) is fused
element-wise instead of as a whole list, since a genuine partial mismatch
between two models' lists (a dropped or extra name) should not flag the
entire field the way a scalar mismatch does.

Requires the `experiments` extra -- see govscape_extract/similarity.py.
"""

from __future__ import annotations

import itertools
from typing import Optional

from govscape_extract.similarity import AGREEMENT_THRESHOLDS, FUSION_COMPARATORS, list_element_similarity


def is_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    return bool(str(value).strip())


def _components(keys: list[str], agree) -> list[list[str]]:
    """Connected components of the agreement graph over `keys` (order
    preserved, so each component and the component list stay deterministic)."""
    parent = {k: k for k in keys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in itertools.combinations(keys, 2):
        if agree(a, b):
            parent[find(a)] = find(b)

    comps: dict[str, list[str]] = {}
    for k in keys:
        comps.setdefault(find(k), []).append(k)
    return list(comps.values())


def fuse_scalar(field: str, values: dict[str, object], priority: list[str]) -> dict:
    """values: {model_key: value} for the models that produced *any* metadata
    for this document (value itself may be None). Returns
    {status, value, source_model, agreeing_models}."""
    present = [m for m in priority if m in values]
    comp_fn = FUSION_COMPARATORS[field]
    thr = AGREEMENT_THRESHOLDS[field]

    def agree(a: str, b: str) -> bool:
        return comp_fn(values[a], values[b]) >= thr

    comps = _components(present, agree)
    # largest component wins; ties broken toward the one with the
    # highest-priority member (present is priority-ordered, so comp[0] is that).
    comps.sort(key=lambda c: (-len(c), priority.index(c[0])))
    best = comps[0] if comps else []

    if len(best) >= 2:
        rep = best[0]  # already the highest-priority member
        val = values[rep]
        if is_present(val):
            return {"status": "consensus", "value": val, "source_model": rep, "agreeing_models": best}
        return {"status": "consensus_null", "value": None, "source_model": rep, "agreeing_models": best}

    return {"status": "flagged", "value": None, "source_model": None, "agreeing_models": best}


def fuse_list(field: str, lists: dict[str, object], priority: list[str]) -> dict:
    """Element-wise consensus. Returns
    {status, value, dropped_elements, agreeing_models, source_model}."""
    present = [m for m in priority if m in lists]
    norm = {
        m: [e for e in (lists[m] if isinstance(lists[m], list) else [lists[m]] if lists[m] else []) if is_present(e)]
        for m in present
    }
    nonempty = [m for m in present if norm[m]]

    if len(present) < 2:
        return {"status": "flagged", "value": [], "dropped_elements": [],
                "agreeing_models": [], "source_model": None}
    if not nonempty:
        return {"status": "consensus_null", "value": [], "dropped_elements": [],
                "agreeing_models": present, "source_model": None}

    thr = AGREEMENT_THRESHOLDS[field]
    items = [(m, i, e) for m in present for i, e in enumerate(norm[m])]
    clusters: list[list[tuple[str, int, str]]] = []
    for it in items:
        for cl in clusters:
            if any(list_element_similarity(it[2], other[2]) >= thr for other in cl):
                cl.append(it)
                break
        else:
            clusters.append([it])

    consensus: list[tuple[tuple[str, int, str], set]] = []
    dropped: list[str] = []
    for cl in clusters:
        models_in = {it[0] for it in cl}
        rep = min(cl, key=lambda it: (priority.index(it[0]), it[1]))
        if len(models_in) >= 2:
            consensus.append((rep, models_in))
        else:
            dropped.append(rep[2])

    if not consensus:
        return {"status": "flagged", "value": [], "dropped_elements": dropped,
                "agreeing_models": [], "source_model": None}

    consensus.sort(key=lambda ce: (priority.index(ce[0][0]), ce[0][1]))
    value = [ce[0][2] for ce in consensus]
    agreeing = sorted({m for _, ms in consensus for m in ms}, key=priority.index)
    status = "partial_list" if dropped else "consensus"
    return {"status": status, "value": value, "dropped_elements": dropped,
            "agreeing_models": agreeing, "source_model": None}
