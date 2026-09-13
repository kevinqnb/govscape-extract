"""Spot checks on govscape_extract/consensus.py's voting algorithm, moved
here from experiments/fuse.py during the experiment-infrastructure
restructure."""

from __future__ import annotations

from govscape_extract.consensus import fuse_list, fuse_scalar, is_present

PRIORITY = ["a", "b", "c"]


def test_fuse_scalar_two_of_three_consensus_takes_highest_priority_value():
    values = {"a": "Water Quality Report", "b": "Water Quality Report", "c": "Something Else"}
    result = fuse_scalar("title", values, PRIORITY)
    assert result["status"] == "consensus"
    assert result["value"] == "Water Quality Report"
    assert result["source_model"] == "a"
    assert set(result["agreeing_models"]) == {"a", "b"}


def test_fuse_scalar_no_agreement_is_flagged():
    values = {"a": "Report One", "b": "Report Two", "c": "Report Three"}
    result = fuse_scalar("title", values, PRIORITY)
    assert result["status"] == "flagged"
    assert result["value"] is None


def test_fuse_scalar_all_null_is_consensus_null():
    values = {"a": None, "b": None, "c": "something"}
    result = fuse_scalar("title", values, PRIORITY)
    assert result["status"] == "consensus_null"
    assert result["value"] is None


def test_fuse_list_partial_list_keeps_agreed_elements_and_drops_the_rest():
    lists = {
        "a": ["Chesapeake Bay", "Maryland"],
        "b": ["Chesapeake Bay watershed"],
        "c": ["Something Unrelated"],
    }
    result = fuse_list("geographic_coverage", lists, PRIORITY)
    assert result["status"] == "partial_list"
    assert result["value"] == ["Chesapeake Bay"]
    assert "Maryland" in result["dropped_elements"]


def test_fuse_list_needs_at_least_two_models_present():
    result = fuse_list("geographic_coverage", {"a": ["Maryland"]}, PRIORITY)
    assert result["status"] == "flagged"


def test_is_present():
    assert not is_present(None)
    assert not is_present([])
    assert not is_present("   ")
    assert is_present("x")
    assert is_present(["x"])
