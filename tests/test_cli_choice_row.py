"""A choice between a few things, drawn on one line and switched with Tab --
and past a handful, drawn as a grid the arrows walk.

Driven through prompt_toolkit's own pipe input, so what these exercise is the
widget a user's keystrokes reach, not a stand-in for it.
"""

from __future__ import annotations

import pytest

from opendde_harness.cli import _choice

OPTIONS = [("local", "local"), ("api", "api")]
THREE = [("sign in", "oauth"), ("api key", "key"), ("back", "back")]


def press(keys: str, options=OPTIONS, **kwargs):
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        with create_app_session(input=pipe, output=DummyOutput()):
            return _choice.row("mode", options, scheme="dark", **kwargs)


def test_enter_takes_the_option_the_row_opens_on():
    assert press("\r") == "local"
    assert press("\r", default="api") == "api"
    # A default naming nothing opens on the first option, as a list would.
    assert press("\r", default="nothing") == "local"


def test_tab_walks_the_row_and_comes_back_round():
    assert press("\t\r") == "api"
    assert press("\t\t\r") == "local"
    # Three options: three tabs is all the way round and back to the first.
    assert press("\t\t\t\r", options=THREE) == "oauth"
    assert press("\t\t\r", options=THREE) == "back"


def test_the_arrows_move_it_too_in_both_directions():
    assert press("\x1b[C\r") == "api"  # right
    assert press("\x1b[D\r") == "api"  # left, from the first option, wraps
    assert press("\x1b[B\r") == "api"  # down, for a hand that came from a list


def test_cancelling_answers_nothing_the_way_every_other_prompt_does():
    assert press("\x03") is None  # ctrl+c
    assert press("\x1b") is None  # escape


def test_escape_is_the_way_back_where_the_caller_names_one():
    """A wizard screen with a screen before it hands Escape that screen; Ctrl+C
    still quits, so a value of None is never mistaken for a way back."""
    back = object()
    assert press("\x1b", back=back) is back
    assert press("\x03", back=back) is None
    assert press("\r", back=back) == "local"


NINE = [(f"option {index}", index) for index in range(9)]


def test_past_a_handful_the_options_wrap_into_an_even_grid():
    """Four to a line at most, and the lines kept even: nine are three lines of
    three, not two of four and a stray; five are three and two."""
    assert _choice.shape(2) == (1, 2)
    assert _choice.shape(4) == (1, 4)
    assert _choice.shape(5) == (2, 3)
    assert _choice.shape(8) == (2, 4)
    assert _choice.shape(9) == (3, 3)


def test_the_arrows_walk_the_grid_across_and_down():
    assert press("\x1b[C\r", options=NINE) == 1  # right: the next cell
    assert press("\x1b[C\x1b[C\x1b[C\r", options=NINE) == 3  # and on to the next line
    assert press("\x1b[B\r", options=NINE) == 3  # down: the cell beneath
    assert press("\x1b[B\x1b[B\x1b[B\r", options=NINE) == 6  # and no further than the last line
    assert press("\x1b[B\x1b[A\r", options=NINE) == 0  # up: back to where it was
    assert press("\t\t\t\t\t\t\t\t\t\r", options=NINE) == 0  # tab all the way round


def test_a_row_needs_two_options_to_be_a_choice():
    with pytest.raises(ValueError, match="a row holds"):
        press("\r", options=[("only", 1)])
