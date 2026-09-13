"""Light/dark theme detection and palette for the interactive CLI.

The palette follows the same standard as the terminal UI, whose values live in
the TUI's theme module (``ui-tui/src/theme.ts``). TypeScript and Python cannot
share code, so the values are restated here and must be changed in both places
together.

The standard is one rule: violet says identity, and everything else is neutral.
So the prompt marker, the pointer and the answer keep the brand violet, and the
body text, the greys, the rules and the panel borders take the neutral values
the TUI borrows from the pi-claude-theme reference (see
``LICENSES/README.md``). Body text is the terminal's own foreground rather
than a color of ours, which is what ``INHERIT`` means below; the TUI does the
same, and it is why a heading is bold rather than tinted.

Measured against the ground each palette is drawn on, with body text clearing
WCAG AA 4.5:1: light violet 5.70:1 and grey 5.74:1 on white, dark violet 6.13:1
and grey 5.85:1 on #1E1E1E. Rules, borders and disabled text sit below that
deliberately, which WCAG allows for what is neither body text nor a control.

Detection is kept out of import time and runs on the first themed render (see
``onboard_commands._ThemedConsole``). Only truecolor values are provided;
rich/prompt_toolkit downgrade to 256/16 automatically.
"""

from __future__ import annotations

import os
import sys
from typing import Literal

Scheme = Literal["light", "dark"]

# Prompt chrome, shared by every interactive command so none of them falls back
# to questionary's defaults ("?" marker, ">>" pointer) and reads as a different
# program. A single-space qmark renders as one blank, which -- with
# questionary's own leading space -- puts every prompt line on the same 2-space
# column as our printed help/status lines.
QMARK = " "
POINTER = "❯"

_LUMA_LIGHT_THRESHOLD = 0.6

#: Body text is whatever the terminal already paints with. Each consumer
#: spells that differently, so the palette carries a sentinel and the builders
#: translate it.
INHERIT = "inherit"

PALETTE: dict[Scheme, dict[str, str]] = {
    "dark": {
        "accent": "#A78BFA",
        "text": INHERIT,
        "heading": INHERIT,
        "selected": "#A78BFA",
        "border": "#505050",
        "muted": "#999999",
        "separator": "#505050",
        "disabled": "#666666",
        "error": "#FF6B80",
        "ok": "#4EBA65",
        "warn": "#FFC107",
    },
    "light": {
        "accent": "#7C3AED",
        "text": INHERIT,
        "heading": INHERIT,
        "selected": "#7C3AED",
        "border": "#AFAFAF",
        "muted": "#666666",
        "separator": "#AFAFAF",
        "disabled": "#767676",
        "error": "#AB2B3F",
        "ok": "#2C7A39",
        "warn": "#966C1E",
    },
}


def _rich(color: str) -> str:
    """rich's name for the terminal's own foreground."""
    return "default" if color == INHERIT else color


def _prompt_toolkit(color: str) -> str:
    """prompt_toolkit's name for the same. Its plain ``default`` is not one of
    its color names; ``ansidefault`` is."""
    return "ansidefault" if color == INHERIT else color


_cache: Scheme | None = None


def _parse_hex(value: str) -> tuple[int, int, int] | None:
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    if len(v) != 6:
        return None
    try:
        return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
    except ValueError:
        return None


def _is_light_rgb(rgb: tuple[int, int, int]) -> bool:
    r, g, b = rgb
    luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
    return luma >= _LUMA_LIGHT_THRESHOLD


def _osc_reply_to_rgb(data: str) -> tuple[int, int, int] | None:
    # Terminal answers OSC 11 as e.g. "rgb:ffff/f5f5/eaea" (16-bit per channel)
    # or "#rrggbb"; tolerate a trailing BEL/ST.
    marker = "rgb:"
    idx = data.find(marker)
    if idx != -1:
        parts = data[idx + len(marker) :].strip().strip("\a\033\\").split("/")
        if len(parts) >= 3:
            try:
                return tuple(int(p[:2], 16) for p in parts[:3])  # type: ignore[return-value]
            except ValueError:
                return None
    hidx = data.find("#")
    if hidx != -1:
        return _parse_hex(data[hidx : hidx + 7])
    return None


def _can_probe() -> bool:
    if os.name != "posix":
        return False
    if os.environ.get("CI"):
        return False
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return False
    # GNU screen neither answers OSC 11 nor swallows the query, so the probe
    # bytes echo as visible garbage; tmux defaults to TERM=screen-256color and
    # can't answer a bare query either. Both fall back to COLORFGBG/dark.
    if os.environ.get("TERM", "").startswith("screen"):
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (ValueError, OSError):
        return False


def _probe_osc11() -> tuple[int, int, int] | None:
    try:
        import select
        import termios
        import tty
    except ImportError:
        return None

    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        return None
    try:
        tty.setraw(fd)
        sys.stdout.write("\033]11;?\033\\")
        sys.stdout.flush()
        ready, _, _ = select.select([fd], [], [], 0.1)
        if not ready:
            return None
        raw = os.read(fd, 64).decode("ascii", "replace")
    except (OSError, ValueError):
        return None
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except termios.error:
            pass
    return _osc_reply_to_rgb(raw)


def _detect() -> tuple[Scheme, bool]:
    """Return (scheme, definitive). Non-definitive results (the dark fallback)
    are not cached, so a later call from an interactive path can re-detect."""
    override = os.environ.get("OPENDDE_HARNESS_THEME", "").strip().lower()
    if override in ("light", "dark"):
        return override, True  # type: ignore[return-value]

    hint = os.environ.get("OPENDDE_HARNESS_TERM_BACKGROUND", "")
    rgb = _parse_hex(hint) if hint else None
    if rgb is not None:
        return ("light" if _is_light_rgb(rgb) else "dark"), True

    if _can_probe():
        rgb = _probe_osc11()
        if rgb is not None:
            return ("light" if _is_light_rgb(rgb) else "dark"), True

    fgbg = os.environ.get("COLORFGBG", "")
    if fgbg:
        last = fgbg.split(";")[-1].strip()
        if last in ("7", "15"):
            return "light", True
        if last.isdigit():
            return "dark", True

    return "dark", False


def detect_scheme() -> Scheme:
    global _cache
    if _cache is not None:
        return _cache
    scheme, definitive = _detect()
    if definitive:
        _cache = scheme
    return scheme


def _style_str(color: str, *attrs: str) -> str:
    return " ".join([color, *attrs]) if attrs else color


def build_rich_theme(scheme: Scheme):
    from rich.theme import Theme

    p = PALETTE[scheme]
    return Theme(
        {
            "accent": _rich(p["accent"]),
            "text": _rich(p["text"]),
            "heading": _style_str(_rich(p["heading"]), "bold"),
            "selected": _rich(p["selected"]),
            "border": _rich(p["border"]),
            "muted": _rich(p["muted"]),
            "separator": _rich(p["separator"]),
            "disabled": _rich(p["disabled"]),
            "error": _rich(p["error"]),
            "ok": _rich(p["ok"]),
            "warn": _rich(p["warn"]),
        }
    )


def build_questionary_style(scheme: Scheme):
    from questionary import Style

    p = {key: _prompt_toolkit(value) for key, value in PALETTE[scheme].items()}
    return Style(
        [
            ("qmark", f"fg:{p['accent']} bold"),
            ("question", "bold"),
            ("answer", f"fg:{p['accent']} bold"),
            ("pointer", f"fg:{p['accent']} bold"),
            ("highlighted", f"fg:{p['text']} bold noreverse"),
            ("selected", f"fg:{p['selected']} noreverse"),
            ("separator", f"fg:{p['separator']}"),
            ("instruction", f"fg:{p['muted']} italic"),
            ("disabled", f"fg:{p['disabled']} italic"),
            ("validation-toolbar", f"fg:{p['error']} bold"),
            ("text", f"fg:{p['text']}"),
        ]
    )
