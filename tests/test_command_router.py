"""Tests for the natural-language command router."""

from __future__ import annotations

import pytest

from core.command_router import NaturalLanguageRouter, RouteKind
from core.tools import build_sandbox


def _router(tmp_path, **kwargs):
    return NaturalLanguageRouter(build_sandbox(tmp_path, **kwargs))


def test_question_falls_back_to_answering(tmp_path):
    routed = _router(tmp_path).route("what is the relationship between porosity and permeability?")
    assert routed.kind is RouteKind.ANSWER
    assert routed.plan is None


def test_arithmetic_request_becomes_a_calculator_plan(tmp_path):
    routed = _router(tmp_path).route("calculate 2 * (3 + 4)")
    assert routed.kind is RouteKind.TOOL
    assert routed.plan is not None
    assert routed.plan.calls[0].tool == "calculator"
    assert routed.plan.calls[0].arguments["expression"] == "2 * (3 + 4)"


def test_arabic_arithmetic_request(tmp_path):
    routed = _router(tmp_path).route("احسب 5 + 5")
    assert routed.kind is RouteKind.TOOL
    assert routed.plan is not None
    assert routed.plan.calls[0].tool == "calculator"


def test_search_request_becomes_a_search_plan(tmp_path):
    routed = _router(tmp_path).route("search for permeability")
    assert routed.kind is RouteKind.TOOL
    assert routed.plan is not None
    assert routed.plan.calls[0].tool == "search"
    assert routed.plan.calls[0].arguments["query"] == "permeability"


def test_list_request_targets_a_directory(tmp_path):
    routed = _router(tmp_path).route("list the files in notes/")
    assert routed.kind is RouteKind.TOOL
    assert routed.plan is not None
    assert routed.plan.calls[0].tool == "list_dir"
    assert routed.plan.calls[0].arguments["relative"] == "notes/"


def test_write_request_is_ignored_on_a_read_only_sandbox(tmp_path):
    routed = _router(tmp_path, read_only=True).route("write to file notes/a.txt: hello")
    assert routed.kind is RouteKind.ANSWER


def test_write_request_is_routed_on_a_writable_sandbox(tmp_path):
    routed = _router(tmp_path, read_only=False).route("write to file notes/a.txt: hello")
    assert routed.kind is RouteKind.TOOL
    assert routed.plan is not None
    assert routed.plan.calls[0].tool == "write_file"
    assert routed.plan.calls[0].arguments["content"] == "hello"


def test_tools_not_on_the_allow_list_are_never_routed(tmp_path):
    router = _router(tmp_path, allowed_tools=frozenset({"read_file"}))
    assert router.route("calculate 2 + 2").kind is RouteKind.ANSWER
    assert router.route("search for porosity").kind is RouteKind.ANSWER


def test_empty_utterance_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _router(tmp_path).route("   ")


def test_every_routed_tool_is_on_the_allow_list(tmp_path):
    router = _router(tmp_path, read_only=False)
    for utterance in (
        "calculate 1 + 1",
        "search for porosity",
        "list the files in .",
        "read the file notes/a.txt",
        "write to file notes/a.txt: hi",
    ):
        routed = router.route(utterance)
        if routed.plan is not None:
            for call in routed.plan.calls:
                assert call.tool in router.allowed_tools
