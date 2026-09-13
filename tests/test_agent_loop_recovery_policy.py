"""The two per-turn policies the loop counts for: empty recovery and dead calls.

Both are pure decisions with budgets, and the loop owns only the counters. These
tables are what the counters mean, stated before the counting moved next to the
decision: which empty response gets which recovery and in what order, when a
budget is spent, and which repeated tool failure is a stuck loop rather than a
model still adapting.
"""

from __future__ import annotations

import pytest

from opendde_harness.agent.loop.failure_streak import failure_class, is_hard_tool_failure, loop_break_nudge
from opendde_harness.agent.loop.recovery import (
    RecoveryAction,
    RecoveryLimits,
    classify_empty_response,
)
from opendde_harness.providers.base import LLMResponse

LIMITS = RecoveryLimits()


def _response(content: str | None = None, *, reasoning: str | None = None, blocks: list[dict] | None = None):
    return LLMResponse(content=content, reasoning_content=reasoning, thinking_blocks=blocks, finish_reason="stop")


def _classify(response, visible, **state) -> RecoveryAction:
    return classify_empty_response(
        response,
        visible,
        prev_had_tool_calls=state.get("prev_had_tool_calls", False),
        nudges_done=state.get("nudges_done", 0),
        prefill_retries=state.get("prefill_retries", 0),
        empty_retries=state.get("empty_retries", 0),
        limits=state.get("limits", LIMITS),
    )


@pytest.mark.parametrize(
    "case, response, visible, state, expected",
    [
        (
            "text present is a finished turn, whatever else the turn did",
            _response("here is the answer"),
            "here is the answer",
            {"prev_had_tool_calls": True},
            RecoveryAction.COMPLETE,
        ),
        (
            "recovery switched off never recovers",
            _response(""),
            None,
            {"limits": RecoveryLimits(enabled=False)},
            RecoveryAction.COMPLETE,
        ),
        (
            "structured reasoning and no body is a prefill",
            _response("", reasoning="thinking about it"),
            None,
            {},
            RecoveryAction.PREFILL,
        ),
        (
            "an inline think block counts as reasoning too",
            _response("<think>so</think>"),
            None,
            {},
            RecoveryAction.PREFILL,
        ),
        (
            "prefill before nudge: a thinking-only reply after a tool is continued",
            _response("", reasoning="more"),
            None,
            {"prev_had_tool_calls": True},
            RecoveryAction.PREFILL,
        ),
        (
            "nothing at all after a tool ran is a nudge",
            _response(""),
            None,
            {"prev_had_tool_calls": True},
            RecoveryAction.NUDGE,
        ),
        (
            "the nudge fires once per turn; after that it is a plain retry",
            _response(""),
            None,
            {"prev_had_tool_calls": True, "nudges_done": 1},
            RecoveryAction.RETRY,
        ),
        (
            "nothing at all with no tool before it is a plain retry",
            _response(""),
            None,
            {},
            RecoveryAction.RETRY,
        ),
        (
            "a reasoning model whose prefill budget is spent still gets its retries",
            _response("", reasoning="always reasons"),
            None,
            {"prefill_retries": LIMITS.thinking_prefill_max_retries},
            RecoveryAction.RETRY,
        ),
        (
            "every budget spent ends the turn rather than looping",
            _response("", reasoning="always reasons"),
            None,
            {
                "prefill_retries": LIMITS.thinking_prefill_max_retries,
                "empty_retries": LIMITS.empty_content_max_retries,
                "nudges_done": LIMITS.post_tool_empty_max_nudges,
                "prev_had_tool_calls": True,
            },
            RecoveryAction.COMPLETE,
        ),
    ],
)
def test_the_empty_response_table(case, response, visible, state, expected) -> None:
    assert _classify(response, visible, **state) is expected, case


@pytest.mark.parametrize(
    "case, text, counts",
    [
        ("a plain error is deterministic", "Error: file not found", True),
        ("a rate limit clears itself", "Error: 429 rate limit exceeded", False),
        ("so does a timeout", "Error: the request timed out", False),
        ("and a bad gateway", "Error: 503 no healthy upstream", False),
        ("an empty search found nothing and worked", "No matches found.", False),
        ("a non-zero exit code is a failure", "stdout\nExit code: 1", True),
        ("a zero exit code is not", "stdout\nExit code: 0", False),
        ("ordinary output is not a failure", "here are the three files", False),
    ],
)
def test_the_failure_streak_counts_only_deterministic_failures(case, text, counts) -> None:
    """A retry would clear a transient error, so nudging on one is noise."""
    assert is_hard_tool_failure(text) is counts, case


@pytest.mark.parametrize(
    "case, text, expected",
    [
        ("a cut-off payload is its own class", "Error: [truncated] the write was cut", "truncated"),
        (
            "so is an unparseable last call, whose cause is undecided",
            "Error: [incomplete arguments] did not parse",
            "incomplete_arguments",
        ),
        (
            "malformed JSON is not the same as a missing field",
            "Error: [invalid arguments] not valid JSON",
            "invalid_arguments",
        ),
        ("a schema complaint is one class", "Error: invalid parameters for tool", "schema"),
        ("a missing path is another", "Error: file not found", "not_found"),
        ("and a refusal another", "Error: permission denied", "denied"),
        ("anything else is other", "Error: something went sideways", "other"),
    ],
)
def test_two_different_errors_from_one_tool_are_two_classes(case, text, expected) -> None:
    """The streak asks whether the model is repeating one dead call. Counting two
    different errors together fires the nudge at a model that is still adapting."""
    assert failure_class(text) == expected, case


def test_a_repeated_truncation_is_told_to_send_less_not_to_switch_tools() -> None:
    """ "Change approach" is not always somewhere to go: the tool that writes files
    has no counterpart, and the turn's own hint already said to split the call."""
    cut = loop_break_nudge("write", 2, "truncated")
    assert "smaller" in cut and "different tool" not in cut

    undecided = loop_break_nudge("write", 2, "incomplete_arguments")
    assert "smaller payload" in undecided and "schema" in undecided

    ordinary = loop_break_nudge("exec", 3, "not_found")
    assert "change approach" in ordinary and "a different tool" in ordinary
