"""The setup wizard's chrome: what each piece of it puts on screen.

The wizard used to draw a box around every screen; these pin the lighter
vocabulary that replaced them, because the thing a reader checks -- does the
step line still carry its progress, does a Chinese label still line its column
up -- is invisible in a diff of markup.
"""

from __future__ import annotations

import pytest

from opendde_harness.cli import _chrome as chrome
from opendde_harness.cli._theme import build_rich_theme


@pytest.fixture
def console():
    from rich.console import Console

    return Console(width=60, theme=build_rich_theme("dark"), record=True, force_terminal=False, legacy_windows=False)


def rendered(console) -> str:
    return console.export_text()


def test_the_lockup_is_the_line_the_tui_opens_with(console):
    """Mark, name, version -- the same three parts as `branding.ts`."""
    from opendde_harness import __logo__, __version__

    chrome.lockup(console)

    text = rendered(console)
    assert f"{__logo__}  OpenDDE Harness v{__version__}" in text
    # At the gutter every prompt sits at, not flush against the edge.
    assert text.splitlines()[1].startswith("  ϒ")


def test_a_step_carries_its_number_title_and_progress(console):
    chrome.step(console, number=2, total=4, title="Long-term memory", word="Step")

    lines = [line for line in rendered(console).splitlines() if line.strip()]

    assert lines[0].startswith("  Step 2/4 · Long-term memory")
    # Two steps reached, two to come, and the mark sits at the far end.
    assert lines[0].rstrip().endswith("● ● ○ ○")
    assert lines[0].rstrip().index("●") > 40
    # A rule under it, the width of the screen inside the gutter.
    assert lines[1] == "  " + "─" * 56


def test_the_steps_ahead_are_numbered_on_one_line(console):
    chrome.steps(console, ["LLM", "Memory", "Protein design", "Web search"])

    assert "① LLM   ② Memory   ③ Protein design   ④ Web search" in rendered(console)


def test_fields_line_up_whatever_the_label_is_made_of(console):
    """A Chinese label is two columns per glyph; a column that counted
    characters put the values of a mixed list in two different places."""
    chrome.fields(console, [("服务商", "custom"), ("Default model", "custom/glm-4.6")])

    from rich.cells import cell_len

    lines = [line for line in rendered(console).splitlines() if line.strip()]
    # Columns, not characters: the label is three glyphs and six columns wide.
    starts = {cell_len(line[: line.index(value)]) for line, value in zip(lines, ["custom", "custom/glm-4.6"])}

    assert len(starts) == 1
    assert lines[0].startswith("  服务商")


def test_every_piece_sits_at_the_same_gutter(console):
    chrome.caption(console, "a caption")
    chrome.hint(console, "a hint")
    chrome.heading(console, "a heading")

    for line in rendered(console).splitlines():
        if line.strip():
            assert line.startswith("  ") and not line.startswith("   ")
