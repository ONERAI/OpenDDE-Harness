"""The setup wizard's visual vocabulary: a lockup, a step line, a field list.

One rule, the TUI's own (``ui-tui/src/theme.ts``): violet says identity and
everything else is neutral. So the brand mark, a step's number and a command
name are violet, and every other thing on screen is the terminal's own
foreground or a grey.

A box says "these lines belong together". A wizard that draws one around every
screen says nothing with them and spends four rows and four columns per screen
saying it, so the chrome here is a line of type and the space around it: the
lockup the TUI opens with, a step line with the progress mark at the other end
of it, and a rule under that. What is left of the width is the wizard's.

Everything sits at the same two-column gutter as the prompts questionary draws
(``_theme.QMARK`` is a space, and questionary adds one of its own), so a
heading, a hint and an answer all start in the same column.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rich.console import Console

#: The gutter every line of the wizard starts at.
INDENT = "  "


def lockup(console: "Console") -> None:
    """``ϒ  OpenDDE Harness v0.0.4``, the line the TUI opens with.

    The same three parts in the same three colours as
    ``ui-tui/src/components/branding.ts``: the mark in the accent, the name in
    the terminal's own foreground, the version grey behind it.
    """
    from opendde_harness import __logo__, __version__

    console.print()
    console.print(
        f"{INDENT}[accent][bold]{__logo__}[/bold][/accent]  [heading]OpenDDE Harness[/heading] [muted]v{__version__}[/muted]"
    )


def caption(console: "Console", text: str) -> None:
    """A grey note under the lockup or a heading.

    Indented as a block: a sentence long enough to wrap keeps the gutter on
    its second line, where printing it with the spaces in front of it would
    drop the tail of it against the edge of the terminal.
    """
    _block(console, text, "muted")


def hint(console: "Console", text: str) -> None:
    """The keys a screen answers to, dimmer than the words they label."""
    _block(console, text, "dim")


def _block(console: "Console", text: str, style: str) -> None:
    from rich.padding import Padding
    from rich.text import Text

    console.print(Padding(Text.from_markup(f"[{style}]{text}[/{style}]"), (0, 0, 0, len(INDENT))))


def rule(console: "Console") -> None:
    """A thin line across the width, at the gutter."""
    console.print(f"{INDENT}[border]{'─' * max(1, console.width - len(INDENT) * 2)}[/border]")


def step(console: "Console", *, number: int, total: int, title: str, word: str) -> None:
    """``Step 1/4 · Choose an LLM provider`` with the progress mark at the end.

    The mark is one filled dot per step reached and a hollow one for each still
    to come, which is the whole of the progress reporting: a wizard of four
    screens does not need a bar.
    """
    dots = " ".join("●" if index <= number else "○" for index in range(1, total + 1))
    left = f"{word} {number}/{total} · {title}"
    room = max(1, console.width - len(INDENT) * 2 - _width(left) - _width(dots))

    console.print()
    console.print(
        f"{INDENT}[accent][bold]{word} {number}/{total}[/bold][/accent] [muted]·[/muted] [heading]{title}[/heading]{' ' * room}[accent]{dots[: number * 2 - 1]}[/accent][disabled]{dots[number * 2 - 1 :]}[/disabled]"
    )
    rule(console)
    console.print()


def steps(console: "Console", names: Sequence[str]) -> None:
    """The screens ahead, numbered, on one line."""
    marks = "①②③④⑤⑥⑦⑧⑨"
    parts = [f"[accent]{marks[index]}[/accent] [muted]{name}[/muted]" for index, name in enumerate(names)]

    console.print(f"{INDENT}{'   '.join(parts)}")


def heading(console: "Console", text: str) -> None:
    """A section's name, in the terminal's own foreground."""
    console.print()
    console.print(f"{INDENT}[heading]{text}[/heading]")


def fields(console: "Console", rows: Iterable[tuple[str, str]], *, key_style: str = "muted") -> None:
    """A two-column list: the name of a thing, then what it is set to.

    Borderless and column-aligned rather than tabulated, because a table's
    rules are chrome around six words.
    """
    pairs = [(str(key), str(value)) for key, value in rows]
    width = max((_width(key) for key, _ in pairs), default=0)

    for key, value in pairs:
        console.print(f"{INDENT}[{key_style}]{key}[/{key_style}]{' ' * (width - _width(key) + 2)}{value}")


@contextmanager
def working(console: "Console", label: str, *, note: str | None = None):
    """A phase that takes a while, on one line that refreshes where it stands.

    A wizard that prints a sentence per phase leaves the screen holding a wall
    of things that already finished. This holds one line while the phase runs
    and takes it back when it ends, so what stays on screen is what the phase
    settled -- a :func:`done` line, or the facts it found.

    The line counts the seconds it has been waiting. A spinner alone says a
    program is alive; it does not say whether a phase is taking longer than it
    should, and a minute of it reads as a hang. ``note`` is what the wait is
    for, when the label alone would leave the reader guessing.

    Not around a download: a transfer draws its own progress (``cli/_download``)
    and two live regions on one console cannot both draw.
    """
    import threading
    import time

    def line(seconds: int) -> str:
        elapsed = f" [disabled]{seconds}s[/disabled]" if seconds else ""
        tail = f" [disabled]· {note}[/disabled]" if note else ""
        return f"[muted]{label}[/muted]{elapsed}{tail}"

    status = console.status(line(0), spinner="dots")
    stop = threading.Event()
    started = time.monotonic()

    def tick() -> None:
        while not stop.wait(1.0):
            status.update(line(int(time.monotonic() - started)))

    with status:
        ticker = threading.Thread(target=tick, daemon=True)
        ticker.start()
        try:
            yield status
        finally:
            stop.set()
            ticker.join(timeout=1.0)


def done(console: "Console", text: str) -> None:
    """What a phase settled, one line, with the mark that says it went through."""
    console.print(f"{INDENT}[ok]✓[/ok] {text}")


def _width(text: str) -> int:
    """Columns the text takes: the wide glyphs a Chinese label is made of count
    for two, which is what keeps a mixed-language column aligned."""
    from rich.cells import cell_len

    return cell_len(text)
