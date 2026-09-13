"""Fencing untrusted content, and taking the fence off again for a person."""

import json
import time
from pathlib import Path

import pytest

from opendde_harness.security.trust import unwrap_untrusted, wrap_untrusted

VECTORS = json.loads((Path(__file__).parent / "fixtures" / "untrusted_fence_vectors.json").read_text())


@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda case: case["name"])
def test_display_unwrapping_matches_the_shared_vectors(case):
    """The same file drives the TypeScript side (`transcript.test.ts`), so the
    two cannot drift: a session stored by one harness is read by both."""
    assert unwrap_untrusted(case["in"]) == case["out"]


def test_a_fence_round_trips():
    assert unwrap_untrusted(wrap_untrusted("tool output", source="exec")) == "tool output"


def test_every_fence_in_one_message_comes_off():
    text = "\n".join([wrap_untrusted("first", source="exec"), wrap_untrusted("second", source="find")])

    assert unwrap_untrusted(text) == "first\nsecond"


def test_a_nested_fence_comes_off_too():
    inner = wrap_untrusted("the answer", source="subagent")

    assert unwrap_untrusted(wrap_untrusted(inner, source="exec")) == "the answer"


def test_content_that_echoes_a_marker_is_not_a_fence():
    # What the comment always said and the assertion did not: an echoed marker
    # is content. A reply explaining the format writes a short opening line and
    # any closing nonce it likes, and both survive -- the recognizer is the
    # whole sentence this module writes, not the bracket.
    echoed = "[BEGIN UNTRUSTED exec #aaaaaaaa — x]\nbody\n[END UNTRUSTED exec #bbbbbbbb]"

    assert unwrap_untrusted(echoed) == echoed


def test_a_close_that_matches_nothing_is_left_where_it_is():
    opened = wrap_untrusted("body", source="exec").split("\n")[0]
    mismatched = f"{opened}\nbody\n[END UNTRUSTED exec #bbbbbbbb]"

    # The opening was written here, so it goes; the close was not its partner,
    # so it stays rather than being deleted on suspicion.
    assert unwrap_untrusted(mismatched) == "body\n[END UNTRUSTED exec #bbbbbbbb]"


def test_many_unclosed_openings_cost_what_their_length_costs():
    # Scanning for a matching close from every opening made 4,000 of them take
    # seconds of a blocked event loop, on a message well inside the frame
    # limit. One pass over the lines is linear.
    payload = "[BEGIN UNTRUSTED exec #aaaaaaaa — x]\nbody\n" * 4_000

    started = time.perf_counter()
    unwrap_untrusted(payload)
    small = time.perf_counter() - started

    started = time.perf_counter()
    unwrap_untrusted(payload * 2)
    large = time.perf_counter() - started

    # Twice the input, not four times the work. Generous against a loaded host;
    # the quadratic version was ~4x per doubling and seconds in absolute terms.
    assert large < small * 3 + 0.05
    assert large < 1.0


def test_a_real_wrapper_repeated_many_times_is_also_linear():
    one = wrap_untrusted("body", source="exec")
    payload = f"{one}\n" * 4_000

    started = time.perf_counter()

    assert unwrap_untrusted(payload) == "body\n" * 4_000

    assert time.perf_counter() - started < 1.0


def test_half_a_fence_left_by_an_elision_is_removed():
    opened = wrap_untrusted("kept", source="exec").split("\n")[0]

    assert unwrap_untrusted(f"{opened}\nkept") == "kept"


def test_prose_about_the_marker_is_left_alone():
    assert unwrap_untrusted("see the UNTRUSTED fence in trust.py") == "see the UNTRUSTED fence in trust.py"
    assert unwrap_untrusted("plain output") == "plain output"


def test_empty_content_is_never_fenced_and_survives_unwrapping():
    assert wrap_untrusted("   ", source="exec") == "   "
    assert unwrap_untrusted("   ") == "   "
