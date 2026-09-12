"""Fuzzy field-level comparators for scoring extracted metadata against a
ground-truth model's output (see experiments/evaluate.py).

There's no gold-labeled dataset for this task, so extracted strings are
compared to another model's output rather than a fixed reference -- they
should never be expected to match exactly (a normalizing LLM vs. GLiNER's
literal document spans, in particular). Each comparator below is chosen for
the specific mismatch shape its field is expected to produce; see each
function's docstring for the rationale.

Null-handling convention applied uniformly by every comparator: both sides
empty -> 1.0 (agreement that the field isn't present); exactly one side
empty -> 0.0; both present -> the comparator's real score.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from dateutil import parser as dateparser
from rapidfuzz import fuzz

from govscape_extract.schema import FIELDS


def _is_empty(s: Optional[str]) -> bool:
    return s is None or not s.strip()


def free_text_similarity(a: Optional[str], b: Optional[str]) -> float:
    """title / issuing_agency / performing_organization / series.

    Uses rapidfuzz.fuzz.token_set_ratio, specifically -- not plain ratio()
    and not partial_ratio(). GLiNER returns a literal document substring;
    the LLM may normalize/reorder/add-or-drop a subtitle (e.g.
    "Dept. of Health and Human Services" vs "HHS -- Department of Health &
    Human Services"). token_set_ratio tokenizes both strings and scores on
    set intersection/union, so it's insensitive to word order and to one
    string being a superset of the other's tokens -- exactly the class of
    harmless mismatch GLiNER-vs-LLM produces. Plain ratio() (whole-string
    edit distance) over-penalizes that reordering/superset case.
    partial_ratio() was rejected because it lets a short wrong string match
    a long correct one perfectly purely by token-subset luck, over-crediting
    garbage extractions.
    """
    if _is_empty(a) and _is_empty(b):
        return 1.0
    if _is_empty(a) or _is_empty(b):
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def exact_match_similarity(a: Optional[str], b: Optional[str]) -> float:
    """Case-insensitive exact match, no partial credit. Used for the two
    kinds of field where near-misses are wrong answers rather than
    harmless variation:

    - Closed-set classification (document_type, jurisdiction_level).
      Fuzzy matching would blur the classification signal itself:
      "technical_report" and "oversight_report" are both individually valid
      labels, not near-misses of one another.
    - Identifiers (report_number). "EPA/600/R-15/047" and "EPA/600/R-15/048"
      are ~97% similar as strings and refer to different documents; token
      overlap would score a wrong number as nearly correct.
    """
    if _is_empty(a) and _is_empty(b):
        return 1.0
    if _is_empty(a) or _is_empty(b):
        return 0.0
    return 1.0 if a.strip().lower() == b.strip().lower() else 0.0


@dataclass
class _ParsedDate:
    year: int
    month: Optional[int]
    day: Optional[int]


def _parse_date(s: str) -> Optional[_ParsedDate]:
    """Parse with two different sentinel defaults; a field only counts as
    "actually present in the string" if both parses agree on it.
    dateutil.parser.parse silently fills any component missing from the
    input text using `default` -- e.g. parse("2009") would otherwise be
    indistinguishable from a full date, spuriously mismatching a fuller date
    that shares the same year but a different (fabricated) month/day.
    """
    try:
        d1 = dateparser.parse(s, default=datetime(1, 1, 1))
        d2 = dateparser.parse(s, default=datetime(2, 2, 2))
    except (ValueError, OverflowError, TypeError):
        return None
    return _ParsedDate(
        year=d1.year,
        month=d1.month if d1.month == d2.month else None,
        day=d1.day if d1.day == d2.day else None,
    )


def date_similarity(a: Optional[str], b: Optional[str]) -> float:
    """publication_date. GLiNER returns the literal written substring; the
    LLM may reformat (e.g. to ISO 8601). Parse both and compare at the
    coarser granularity actually present in both sides (year, then month,
    then day) -- mismatched year is 0.0; if either side lacks month/day
    precision, 1.0 once the coarser fields agree, rather than penalizing
    missing precision as disagreement. Falls back to free_text_similarity if
    either string doesn't parse as a date at all, since a non-date-shaped
    value still deserves credit for textual overlap rather than an
    automatic 0.
    """
    if _is_empty(a) and _is_empty(b):
        return 1.0
    if _is_empty(a) or _is_empty(b):
        return 0.0
    pa, pb = _parse_date(a), _parse_date(b)
    if pa is None or pb is None:
        return free_text_similarity(a, b)
    if pa.year != pb.year:
        return 0.0
    if pa.month is None or pb.month is None:
        return 1.0
    if pa.month != pb.month:
        return 0.0
    if pa.day is None or pb.day is None:
        return 1.0
    return 1.0 if pa.day == pb.day else 0.0


def authors_similarity(
    a: Optional[list[str]], b: Optional[list[str]], match_threshold: float = 0.75
) -> float:
    """authors is an unordered free-text list, not a single string to
    compare directly. Greedy bipartite best-match: for each name in `a`,
    take its highest token_set_ratio unmatched counterpart in `b`; a score
    >= match_threshold counts as a matched pair. From matches:
    precision = matched/len(b), recall = matched/len(a), return their F1 --
    symmetric, penalizing both invented authors (precision) and dropped
    authors (recall), unlike a single intersection/union ratio.

    match_threshold=0.75 is calibrated, not guessed: empirically,
    token_set_ratio("John Smith", "J. Smith") = 77.8 (same person,
    abbreviated -- should match) while token_set_ratio("John Smith", "Jane
    Smith") = 80.0 (different people, shared surname -- should not). These
    two cases straddle any single threshold, so this is a real, accepted
    limitation of using a generic token-overlap metric for personal names
    (a two-token name where one token matches exactly and the other doesn't
    inherently scores "moderately high" regardless of whether that's an
    abbreviation or a different person) rather than a bespoke name-matching
    algorithm. 0.75 favors catching abbreviation variance over rejecting
    shared-surname false positives, since distinct co-authors sharing a
    surname on one document is the rarer case in practice.

    A full Hungarian-algorithm optimal assignment was considered and
    rejected as overkill: author lists here are short (~1-10 names), so
    greedy highest-score-first gives essentially the same result at a
    fraction of the complexity, and the gap between greedy and optimal only
    shows up in adversarial cases not expected from front-matter bylines.
    """
    a = a or []
    b = b or []
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    remaining = list(b)
    matched = 0
    for name_a in a:
        if not remaining:
            break
        scores = [(fuzz.token_set_ratio(name_a, nb) / 100.0, i) for i, nb in enumerate(remaining)]
        best_score, best_i = max(scores)
        if best_score >= match_threshold:
            matched += 1
            remaining.pop(best_i)
    precision = matched / len(b)
    recall = matched / len(a)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def place_list_similarity(a: Optional[list[str]], b: Optional[list[str]]) -> float:
    """geographic_coverage. Same unordered-list F1 as authors_similarity --
    place names have the same mismatch shape as personal names here, one
    side often being a token superset of the other ("Chesapeake Bay" vs
    "Chesapeake Bay watershed"), which token_set_ratio already absorbs. Kept
    as a separate name because --match-threshold is calibrated on personal
    names specifically and shouldn't silently retune place matching too.
    """
    return authors_similarity(a, b)


FIELD_COMPARATORS = {
    "title": free_text_similarity,
    "authors": authors_similarity,
    "publication_date": date_similarity,
    # Verbatim by definition, but a raw string that fails to parse as a date
    # still falls back to token overlap rather than scoring an automatic 0.
    "publication_date_raw": date_similarity,
    "issuing_agency": free_text_similarity,
    "performing_organization": free_text_similarity,
    "document_type": exact_match_similarity,
    "report_number": exact_match_similarity,
    "series": free_text_similarity,
    "jurisdiction_level": exact_match_similarity,
    "geographic_coverage": place_list_similarity,
}

# Guards against schema.py's FIELDS drifting out of sync with this module --
# a newly added field would otherwise be silently skipped by score_document.
assert set(FIELD_COMPARATORS) == {f.name for f in FIELDS}, (
    "FIELD_COMPARATORS is out of sync with govscape_extract.schema.FIELDS"
)


# --- Fusion (experiments/fuse.py) -------------------------------------------
# Combining three models' outputs into one ground-truth value asks a
# *boolean* question -- "are these two the same value?" -- which needs a
# stricter bar than evaluate.py's "does this candidate deserve partial
# credit?". Two knobs, both with drift asserts against FIELDS:
#
#   FUSION_COMPARATORS   -- same shape as FIELD_COMPARATORS, but free-text
#                           fields use a length-guarded comparator.
#   AGREEMENT_THRESHOLDS -- score at/above which fuse.py treats two values as
#                           equal. Higher than evaluate.py's 0.75 for free
#                           text; exact (1.0) for closed sets and identifiers.


def guarded_free_text_similarity(a: Optional[str], b: Optional[str]) -> float:
    """title / issuing_agency / performing_organization / series, for fusion.

    `free_text_similarity` uses `token_set_ratio`, which is *deliberately*
    blind to one string being a token superset of the other (see its
    docstring) -- right for scoring a normalizing LLM against GLiNER's
    literal spans, wrong for deciding that a truncated title "agrees" with
    the full one, or that a bare "EPA" agrees with "EPA, Office of Research
    and Development". Taking the min with `token_sort_ratio` keeps the
    word-order / "&"-vs-"and" insensitivity but drops the score when one
    side carries a whole subtitle or parent-agency the other doesn't, so
    those land in the flagged pile for a human to look at.
    """
    if _is_empty(a) and _is_empty(b):
        return 1.0
    if _is_empty(a) or _is_empty(b):
        return 0.0
    # processor=str.lower: rapidfuzz's fuzz.* apply no processor by default,
    # so without this "Water Quality Assessment" vs "Water quality assessment"
    # scores ~59 on token_sort_ratio (pure casing) and never reaches
    # agreement -- a normalization every model does differently.
    return min(
        fuzz.token_set_ratio(a, b, processor=str.lower),
        fuzz.token_sort_ratio(a, b, processor=str.lower),
    ) / 100.0


def list_element_similarity(a: Optional[str], b: Optional[str]) -> float:
    """One element vs one element, for fuse.py's element-wise consensus on
    `authors` / `geographic_coverage`.

    `authors_similarity` (evaluate.py) uses bare `token_set_ratio` here, which
    is fine for partial-credit scoring but manufactures consensus during
    fusion: `token_set_ratio("Smith", "Jane Smith") == 100` because one token
    set is a subset of the other, so a model that returns a bare surname
    would "agree" with every full name sharing it. `min` with
    `token_sort_ratio` (both case-folded) keeps real variants matching
    ("Jane A. Smith" vs "Jane Smith" ~0.83, "Chesapeake Bay" vs "Chesapeake
    Bay watershed" ~0.74) while dropping the bare-token case ("Smith" vs
    "Jane Smith" ~0.67) below the element threshold (`AGREEMENT_THRESHOLDS`:
    0.75 authors, 0.70 places).
    """
    if _is_empty(a) and _is_empty(b):
        return 1.0
    if _is_empty(a) or _is_empty(b):
        return 0.0
    return min(
        fuzz.token_set_ratio(a, b, processor=str.lower),
        fuzz.token_sort_ratio(a, b, processor=str.lower),
    ) / 100.0


FUSION_COMPARATORS = {
    **FIELD_COMPARATORS,
    "title": guarded_free_text_similarity,
    "issuing_agency": guarded_free_text_similarity,
    "performing_organization": guarded_free_text_similarity,
    "series": guarded_free_text_similarity,
}

# Per-field "same value" thresholds. authors / geographic_coverage are scored
# per element by fuse.py (not as whole lists), so their entry is the
# element-match bar, matching authors_similarity's calibrated 0.75.
AGREEMENT_THRESHOLDS = {
    "title": 0.90,
    "authors": 0.75,
    "publication_date": 0.90,
    "publication_date_raw": 0.90,
    "issuing_agency": 0.85,
    "performing_organization": 0.85,
    "document_type": 1.0,
    "report_number": 1.0,
    "series": 0.88,
    "jurisdiction_level": 1.0,
    # a touch lower than authors: place names legitimately differ by a
    # trailing type word ("Chesapeake Bay" vs "Chesapeake Bay watershed",
    # ~0.74 under the length-guarded metric) far more often than personal
    # names do, while the guarded metric still rejects a bare-token match
    # ("Maryland" vs "Maryland State Highway Administration", ~0.50).
    "geographic_coverage": 0.70,
}

assert set(FUSION_COMPARATORS) == {f.name for f in FIELDS}, (
    "FUSION_COMPARATORS is out of sync with govscape_extract.schema.FIELDS"
)
assert set(AGREEMENT_THRESHOLDS) == {f.name for f in FIELDS}, (
    "AGREEMENT_THRESHOLDS is out of sync with govscape_extract.schema.FIELDS"
)


@dataclass
class DocumentScore:
    digest: str
    field_scores: dict[str, float]
    overall_score: float


def score_document(
    digest: str,
    truth: dict,
    candidate: dict,
    field_comparators: Optional[dict] = None,
) -> DocumentScore:
    """truth/candidate are DocumentMetadata.model_dump()-shaped dicts for
    the same document. overall_score is the equal-weight mean of the
    per-field scores -- equal weighting is the default because nothing in
    the task motivates ranking one field's importance over another for this
    experiment, and it keeps overall_score interpretable as a plain average
    with no extra tunable knob to justify. Per-field weights could be added
    to schema.py's FieldSpec later if a downstream use case argues for it.

    Pass `field_comparators` (e.g. FIELD_COMPARATORS with `authors` rebound
    via functools.partial to a non-default match_threshold) to override the
    default comparators; see evaluate.py's --match-threshold flag.
    """
    field_comparators = field_comparators or FIELD_COMPARATORS
    field_scores = {
        name: comparator(truth.get(name), candidate.get(name))
        for name, comparator in field_comparators.items()
    }
    overall_score = sum(field_scores.values()) / len(field_scores)
    return DocumentScore(digest=digest, field_scores=field_scores, overall_score=overall_score)
