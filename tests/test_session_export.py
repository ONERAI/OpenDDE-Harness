"""The Markdown transcript, rendered from pi messages.

Nothing else reads a session for a person: the export is where a turn's
thinking, its calls and their results become prose, and it is the one reader
that must show all three.
"""

from __future__ import annotations

from opendde_harness.providers import messages as msg
from opendde_harness.session.export import render_transcript
from opendde_harness.session.manager import Session
from tests import _messages as build


def _session(*messages: dict) -> Session:
    session = Session(key="cli:x")
    for message in messages:
        session.record(message)
    return session


def test_a_turn_renders_its_reasoning_its_call_and_the_result() -> None:
    rendered = render_transcript(
        _session(
            build.user("read a.txt"),
            build.assistant("reading", calls=[("c1", "read", {"path": "a.txt"})], reasoning="the file first"),
            build.tool_result("c1", "read", "the body"),
            build.assistant("it says hello"),
        )
    )

    assert "💭 _thinking_\n> the file first" in rendered
    assert "⏺ **read**" in rendered and '"path": "a.txt"' in rendered
    assert "## 🛠 Tool result: `read`" in rendered and "the body" in rendered
    assert rendered.rstrip().endswith("it says hello")


def test_a_picture_is_named_and_a_marker_renders_as_its_own_text() -> None:
    rendered = render_transcript(
        _session(
            build.user([msg.text_block("look"), msg.image_block("AAA", "image/png")]),
            build.marker({"provider": "p", "model": "m", "items": []}, "[compacted]"),
        )
    )

    assert "look\n[image]" in rendered, "the bytes are not prose; that a picture was sent is"
    assert "AAA" not in rendered
    assert rendered.rstrip().endswith("[compacted]")
