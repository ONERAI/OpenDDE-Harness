"""A row of options, picked with Tab -- and past a handful, a grid of them.

A list of two reads as a list of many: two rows, a pointer, and the shape of
the question hidden in the shape of a menu. A choice between a few things is
one line -- the options side by side, the taken one filled -- which is how the
TUI draws its own scope switch (``Scope: all | scoped``, Tab to flip) and how
a form reads on paper.

Past four the line would wrap on a narrow terminal, so the options go under
the question in aligned columns, four to a line at most and the lines kept
even: nine providers are three lines of three, read at a glance, where a
list of nine is a column to scroll. The keys are the same either way, and
the arrows move across and down the grid the way they move across a row.

Built on prompt_toolkit, which questionary is built on too, so the keys, the
styling and the Ctrl+C behaviour are the ones every other prompt here has.
"""

from __future__ import annotations

import math
from typing import Sequence

#: What the row is drawn with: the taken option, then the ones beside it.
TAKEN = "●"
FREE = "○"

#: The most options one line holds; more than this wraps into a grid.
PER_LINE = 4

#: The gutter every line starts at, the wizard's own.
INDENT = "  "

#: What Escape exits the application with, so a value of ``None`` -- a real
#: option's value, say -- is never mistaken for it.
_BACK = object()


def shape(count: int) -> tuple[int, int]:
    """How many lines and columns ``count`` options take: as few lines as
    :data:`PER_LINE` allows, and the columns spread evenly over them."""
    lines = math.ceil(count / PER_LINE)
    return lines, math.ceil(count / lines)


def row(
    message: str,
    options: Sequence[tuple[str, object]],
    *,
    default: object = None,
    back: object = None,
    scheme: str | None = None,
):
    """Ask ``message`` with the options in a row, or in a grid past a handful.

    ``options`` is ``(label, value)`` in the order they are drawn. ``default``
    names the value the row opens on. Returns ``None`` when the user cancels
    (Ctrl+C), which is what questionary's prompts return and what the wizard
    reads as "stop here". ``back``, when given, is what Escape returns instead
    of cancelling: the screen before this one, for a wizard that has one.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from rich.cells import cell_len

    from opendde_harness.cli._theme import PALETTE, detect_scheme

    if len(options) < 2:
        raise ValueError(f"a row holds 2 options or more, not {len(options)}")

    palette = PALETTE[scheme or detect_scheme()]
    index = next((i for i, (_, value) in enumerate(options) if value == default), 0)
    taken: list[int] = [index]
    lines, columns = shape(len(options))
    # Every cell but the last in its line is padded to one width, so the
    # columns line up down the grid; a lone line keeps the row's own spacing.
    cell = max(cell_len(label) for label, _ in options) + 2 if lines > 1 else 0

    keys_hint = " · ".join(
        [
            "arrows move" if lines > 1 else "tab switch",
            "enter confirm",
            *(["esc back"] if back is not None else []),
            "ctrl+c quit",
        ]
    )

    def fragments():
        line = [("", INDENT), ("class:question", message)]
        if lines > 1:
            line.append(("", "\n" + INDENT))
        else:
            line.append(("", "  "))
        for position, (label, _) in enumerate(options):
            column = position % columns
            if position and column == 0:
                line.append(("", "\n" + INDENT))
            elif position:
                line.append(("", "   " if lines == 1 else " " * (cell - cell_len(options[position - 1][0]))))
            chosen = position == taken[0]
            line.append(("class:mark" if chosen else "class:free", f"{TAKEN if chosen else FREE} "))
            line.append(("class:label" if chosen else "class:free", label))
        return line + [("", "\n"), ("class:hint", f"{INDENT}{keys_hint}")]

    keys = KeyBindings()

    @keys.add("tab")
    @keys.add("right")
    def _next(event):
        taken[0] = (taken[0] + 1) % len(options)

    @keys.add("s-tab")
    @keys.add("left")
    def _previous(event):
        taken[0] = (taken[0] - 1) % len(options)

    @keys.add("down")
    def _down(event):
        # Across the row for a hand that came from a list; down the grid
        # otherwise, staying put where no cell lies beneath.
        if lines == 1:
            taken[0] = (taken[0] + 1) % len(options)
        elif taken[0] + columns < len(options):
            taken[0] += columns

    @keys.add("up")
    def _up(event):
        if lines == 1:
            taken[0] = (taken[0] - 1) % len(options)
        else:
            taken[0] = max(taken[0] - columns, 0)

    @keys.add("enter")
    def _confirm(event):
        event.app.exit(result=options[taken[0]][1])

    @keys.add("c-c")
    def _cancel(event):
        event.app.exit(result=None)

    @keys.add("escape")
    def _back(event):
        event.app.exit(result=_BACK if back is not None else None)

    style = _style(palette)
    application = Application(
        layout=Layout(Window(FormattedTextControl(fragments, focusable=True), height=lines + (2 if lines > 1 else 1))),
        key_bindings=keys,
        style=style,
        # Taken back once it is answered, and what stays is the record below:
        # the row's own keys are an offer, and an offer that has been taken is
        # not worth the lines it sits on.
        erase_when_done=True,
        full_screen=False,
    )
    chosen = application.run()

    if chosen is _BACK:
        return back
    if chosen is not None:
        _record(message, next(label for label, value in options if value == chosen), style)
    return chosen


def _record(message: str, label: str, style) -> None:
    """What the row settled, on one line, where the row was."""
    from prompt_toolkit import print_formatted_text
    from prompt_toolkit.formatted_text import FormattedText

    print_formatted_text(
        FormattedText([("", INDENT), ("class:question", message), ("", "  "), ("class:label", label)]), style=style
    )


def _style(palette: dict[str, str]):
    """The row's four colours, from the palette every other prompt uses."""
    from prompt_toolkit.styles import Style

    from opendde_harness.cli._theme import _prompt_toolkit

    accent = _prompt_toolkit(palette["accent"])
    muted = _prompt_toolkit(palette["muted"])
    disabled = _prompt_toolkit(palette["disabled"])
    return Style(
        [
            ("question", "bold"),
            ("mark", f"fg:{accent} bold"),
            ("label", f"fg:{accent} bold"),
            ("free", f"fg:{muted}"),
            ("hint", f"fg:{disabled}"),
        ]
    )
