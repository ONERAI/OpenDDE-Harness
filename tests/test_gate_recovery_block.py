"""The recovery block, and what the shadow checkpoint thinks the agent edited.

A turn that runs out of its tool-iteration budget is labelled ``interrupted`` on
purpose: the shadow-git commit is tagged, and the next turn on that session gets
a recovery prompt naming the files the last turn changed, so the work can be
picked up. That mechanism is sound. What it is fed is not: the shadow repository
covers the whole workspace, the session journal lives inside the workspace
(``<workspace>/sessions/<channel>/<chat>.jsonl``), and the default exclude list
does not mention it. So the files "the agent modified last turn" are the
harness's own append-only journal and the lock beside it, and the next turn is
told to go and verify them.

These two tests are the gate's evidence for that. The second one states what the
recovery block says now that the shadow repository leaves the harness's own state
alone (``checkpoint.py``'s ``_DEFAULT_EXCLUDES``).
"""

from __future__ import annotations

from pathlib import Path

from tests._gate import (
    MODEL,
    Echo,
    Events,
    declare,
    gate_config,
    loop_for,
    no_injections,
    request,
    requires_service,
    service,  # noqa: F401  -- the fixture the gate shares
    texts,
)

pytestmark = requires_service

KEY = "tui:recovery"


class Scribble(Echo):
    """The scripted model's tool, writing a file of the user's while it answers.

    The recovery block is a list of files the agent changed, so a turn that
    changes one of the user's files is the only turn there is anything to
    recover. The tool is still named ``echo`` because that is the call the faux
    provider scripts.
    """

    def __init__(self, workspace: Path) -> None:
        super().__init__()
        self._workspace = workspace

    async def execute(self, text: str = "", **_kwargs) -> str:
        self.seen.append(text)
        (self._workspace / "notes.txt").write_text(f"{len(self.seen)}\n", encoding="utf-8")
        return f"echoed {text}"


def _config(tmp_path):
    """The default checkpoint policy, which is on for an interactive session."""
    return gate_config(tmp_path, model=MODEL, providers=declare("faux"), defaults={"maxToolIterations": 1})


async def _two_turns(tmp_path, service):  # noqa: F811
    """The first turn exhausts its tool budget, so the second one gets a recovery
    prompt. Returns the second turn's first request."""
    loop = loop_for(service, _config(tmp_path), tools=[Scribble(tmp_path)])
    try:
        await loop.run_turn(request("the first question", KEY), Events(), no_injections, stream=True)
        second = service.streams
        await loop.run_turn(request("the second question", KEY), Events(), no_injections, stream=True)
    finally:
        await loop.close_mcp()
    return service.contexts[second]


async def test_a_turn_that_ran_out_of_iterations_puts_a_recovery_block_in_the_next_one(tmp_path, service):  # noqa: F811
    """The mechanism itself, which is what makes the next test's failure matter."""
    context = await _two_turns(tmp_path, service)

    body = texts(context)
    assert "[Recovery — the previous turn was interrupted before finishing]" in body
    assert "Files modified last turn: notes.txt" in body, "the file the tool actually wrote"
    assert "Verify the current state of these files before continuing." in body


async def test_the_recovery_block_names_no_file_of_the_harnesss_own(tmp_path, service):  # noqa: F811
    """The recovery prompt is for the user's files, not for our bookkeeping.

    Naming the session journal is wrong twice over: the agent did not write it,
    and reading it back is how a turn would be told its own transcript changed
    under it.
    """
    context = await _two_turns(tmp_path, service)

    body = texts(context)
    assert "sessions/" not in body, "the session journal is not something the agent edited"
    assert ".lock" not in body, "and neither is the lock the journal writer holds"
