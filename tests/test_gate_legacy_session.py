"""G8 — a session file written before the records were pi messages still works.

The durable shape changed: a session used to hold OpenAI-style dicts, a verbatim
``pi`` copy beside the assistant's text, ``role: "tool"`` results and a
compaction marker with its boundary. Nobody rewrites those files, so the reader
has to be lossless in both directions -- what the file holds has to come back as
a pi message, travel over the wire as the same object, and still be the bytes on
disk afterwards.

The fixture is a real file in that old shape (``fixtures/session_pre_pi_shape.jsonl``):
all four roles, an assistant turn carrying a signed thinking block, a tool result
naming the call it answers, a multimodal user message, a marker, and lifecycle
records between them.
"""

from __future__ import annotations

import json
from pathlib import Path

from opendde_harness.providers.base import COMPACTION_KEY
from opendde_harness.providers.messages import wire_projection
from opendde_harness.session.manager import SessionManager
from opendde_harness.tui_rpc.methods.session import session_resume
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
)

pytestmark = requires_service

FIXTURE = Path(__file__).parent / "fixtures" / "session_pre_pi_shape.jsonl"
KEY = "tui:legacy"


def _install(workspace: Path) -> Path:
    """The fixture, where a ``tui:legacy`` session would live."""
    manager = SessionManager(workspace)
    path = manager.sessions_dir / "tui" / "legacy.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def _write_home_config(config) -> None:
    """``session.resume`` reads the config for itself; give it this one."""
    path = Path.home() / ".opendde_harness" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.model_dump(mode="json", by_alias=True, exclude_none=True)), encoding="utf-8")


async def test_an_old_session_resumes_over_the_wire_and_replays_verbatim(tmp_path, service):  # noqa: F811
    path = _install(tmp_path)
    before = path.read_text(encoding="utf-8")
    config = gate_config(tmp_path, model=MODEL, providers=declare("faux"), defaults={"maxToolIterations": 1})
    loop = loop_for(service, config, tools=[Echo()])

    stored = loop.sessions.get_or_create(KEY)
    assert [m["role"] for m in stored.messages] == [
        "system",
        "user",
        "assistant",
        "toolResult",
        "assistant",
        "user",
        "assistant",
    ], "every old role read as the shape the harness now writes"
    marker = next(m for m in stored.messages if COMPACTION_KEY in m)
    assert marker[COMPACTION_KEY]["provider"] == "openai-codex", "the boundary survived the read"

    # 1. The TUI can open it: the transcript is the stored messages, and the
    #    lifecycle records the file also carries are not part of it.
    _write_home_config(config)
    resumed = await session_resume({"session_id": KEY}, agent_loop_factory=lambda: loop)
    assert resumed["session_id"] == KEY
    wire = resumed["messages"]
    assert len(wire) == len(stored.messages), "one entry per message; no lifecycle record and no marker note"
    assert "turn.completed" not in json.dumps(wire)
    assert any("Earlier conversation compacted" in (entry.get("text") or "") for entry in wire)
    assert [entry["role"] for entry in wire].count("tool") == 1, "the old tool role travels as the wire's own"

    # 2. And a turn on it sends those records themselves, not a rebuild of them.
    #    Snapshotted first: the session object is the live one the turn appends to.
    legacy = list(stored.messages)
    try:
        await loop.run_turn(request("what now", KEY), Events(), no_injections, stream=True)
    finally:
        await loop.close_mcp()

    sent = service.contexts[0]["messages"]
    # Everything the record holds except the harness's own bookkeeping: the
    # record ids and turn ids ``wire_projection`` drops, and the marker's
    # boundary, which is how this process finds the cut and not something the
    # backend is told.
    expected = [
        {key: value for key, value in wire_projection(m).items() if key != COMPACTION_KEY}
        for m in legacy
        if m["role"] != "system"
    ]
    assert sent[:-1] == expected, "stored record in, the same object out"
    assert sent[-1]["content"].endswith("what now"), "and the new question after them"
    # The stored system record is not a message any more: the rendered prefix is.
    assert "You are terse." not in json.dumps(sent)
    assert service.contexts[0]["systemPrompt"], "the prefix travels in its own field"

    # The signed thinking block is the reason verbatim matters: a re-rendered
    # message loses the signature and the vendor rejects the turn.
    assistant = next(m for m in sent if m.get("model") == "gpt-5")
    assert assistant["content"][0] == {"type": "thinking", "thinking": "signed", "thinkingSignature": "sig-1"}
    assert assistant["responseId"] == "resp_1"

    # 3. The marker names a backend this model is not, so nothing was replayed
    #    and the history it stands for was sent in clear.
    assert service.replays[0] is None, "another backend's marker says nothing here"

    # 4. Reading and appending never rewrote what was already on disk.
    assert path.read_text(encoding="utf-8").startswith(before), "the old records are still the bytes they were"
