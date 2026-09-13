"""Spot checks on govscape_extract/similarity.py's comparators -- not
exhaustive (the calibration rationale lives in each function's docstring),
just enough to catch a regression in the null-handling convention and each
comparator's basic shape."""

from __future__ import annotations

from govscape_extract.similarity import (
    authors_similarity,
    date_similarity,
    exact_match_similarity,
    free_text_similarity,
)


def test_null_handling_convention_is_uniform():
    for comparator in (free_text_similarity, exact_match_similarity, date_similarity):
        assert comparator(None, None) == 1.0
        assert comparator("", "") == 1.0
        assert comparator("something", None) == 0.0
        assert comparator(None, "something") == 0.0


def test_free_text_similarity_ignores_word_order_and_superset():
    assert free_text_similarity("Department of Health", "Health Department") == 1.0
    assert free_text_similarity("EPA", "EPA Office of Research and Development") == 1.0


def test_exact_match_similarity_rejects_near_miss_identifiers():
    assert exact_match_similarity("EPA/600/R-15/047", "EPA/600/R-15/048") == 0.0
    assert exact_match_similarity("Technical Report", "technical report") == 1.0


def test_date_similarity_matches_at_shared_precision():
    assert date_similarity("2009", "2009-06-15") == 1.0
    assert date_similarity("2009", "2010") == 0.0
    assert date_similarity("June 2009", "2009-07-01") == 0.0


def test_authors_similarity_is_f1_over_matched_pairs():
    assert authors_similarity(["Jane Smith"], ["Jane Smith"]) == 1.0
    assert authors_similarity(["Jane Smith", "John Doe"], ["Jane Smith"]) == 2 / 3
    assert authors_similarity(["Jane Smith"], ["Unrelated Person"]) == 0.0
