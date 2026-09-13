"""G2 — one marker, three models, and what each request carries.

Server-side compaction leaves a marker in the session: the backend's own opaque
summary, and the name of the backend that made it. What the marker means is
entirely about which model the next turn runs on. On the model that made it, the
summary is replayed and the history it swallowed is not sent. On any other
model, the marker names a backend that would not know what to do with it, so the
history is sent whole and nothing is replayed. A ``/model`` switch back makes it
replayable again -- it was never invalidated, only inapplicable.

``faux-codex`` is the offline stand-in for a Codex Responses model, which is the
shape the compaction path gates on; ``PI_COMPACTING`` is widened onto it for the
same reason the package-level suite widens it, and that is the only thing these
tests change about the rule.
"""

from __future__ import annotations

import pytest

from opendde_harness.providers.base import COMPACTION_KEY
from opendde_harness.providers.pi_provider import PI_COMPACTING
from tests._gate import (
    CODEX,
    MODEL,
    Echo,
    Events,
    bind,
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

KEY = "tui:marker"


@pytest.fixture
def compacting(monkeypatch):
    """Let the offline Codex-shaped model compact and replay what it compacted."""
    from opendde_harness.providers import pi_provider

    monkeypatch.setattr(pi_provider, "PI_COMPACTING", PI_COMPACTING | {"faux-codex"})


def _config(tmp_path):
    """A Codex-shaped default, and a second scripted model to switch to.

    The ratio is the smallest the field accepts above zero, so the scripted
    model's own reported prompt size clears it: the trigger reads what the
    backend stated about the call and nothing else.
    """
    providers = {**declare("faux-codex"), **declare("faux")}
    return gate_config(
        tmp_path,
        model=CODEX,
        providers=providers,
        defaults={"maxToolIterations": 1},
        context={"serverCompactRatio": 1e-6},
    )


async def test_a_marker_is_replayed_on_its_own_model_ignored_on_another_and_replayed_again_after_a_switch(
    tmp_path,
    service,  # noqa: F811
    compacting,
):
    config = _config(tmp_path)
    loop = loop_for(service, config, tools=[Echo()])
    events = Events()

    try:
        # 1. A turn over the ratio: the backend is asked for a summary and the
        #    session records it behind every message of the turn it stands for.
        await loop.run_turn(request("the first question", KEY), events, no_injections, stream=True)
        stored = loop.sessions.get_or_create(KEY).messages
        marker = next((m for m in stored if COMPACTION_KEY in m), None)
        assert marker is not None, "the turn ended with a marker in the session"
        assert marker[COMPACTION_KEY]["provider"] == "faux-codex"
        items = marker[COMPACTION_KEY]["items"]
        assert service.compacted, "the backend was asked, rather than a summary being written locally"

        # 2. The next turn on the same model replays it, and sends only what
        #    followed it.
        same = service.streams
        await loop.run_turn(request("the second question", KEY), events, no_injections, stream=True)
        assert service.replays[same] == {"items": items}, "the marker travelled as the replacement history"
        sent = service.contexts[same]
        assert "the first question" not in texts(sent), "what the summary stands for was not sent beside it"
        assert "the second question" in texts(sent)
        # Past our own call: the prompt the service ended up building. The
        # replacement history leads it, and what follows the marker is appended
        # to it rather than the marker being appended to the history.
        payload = (await service.debug())["lastPayload"]["payload"]
        assert payload["input"][: len(items)] == items, "the replacement history is the head of the prompt"
        assert "the second question" in str(payload["input"][len(items) :])

        # 3. A ``/model`` switch to a model whose backend never made it: the
        #    history is sent whole and nothing is replayed.
        loop.set_session_binding(KEY, bind(service, MODEL, config))
        other = service.streams
        await loop.run_turn(request("the third question", KEY), events, no_injections, stream=True)
        assert loop.session_model(KEY) == MODEL
        assert service.replays[other] is None, "the marker is another backend's and says nothing here"
        assert service.models_asked[other] == ("faux", "echo")
        whole = service.contexts[other]
        assert "the first question" in texts(whole), "the swallowed history is sent in clear instead"
        assert "the second question" in texts(whole)

        # 4. And switching back makes it replayable again: it was never
        #    invalidated, only inapplicable.
        loop.set_session_binding(KEY, bind(service, CODEX, config))
        back = service.streams
        await loop.run_turn(request("the fourth question", KEY), events, no_injections, stream=True)
        assert service.replays[back] is not None, "the marker is this backend's again"
        replayed = service.replays[back]["items"]
        assert replayed, "and it carries the opaque items the backend handed back"
        assert "the first question" not in texts(service.contexts[back])
    finally:
        await loop.close_mcp()


async def test_without_the_rule_widened_the_scripted_model_compacts_nothing(tmp_path, service):  # noqa: F811
    """The control, and the reason the fixture exists: which backends compact is
    a rule about real providers, and no faux one is in it."""
    loop = loop_for(service, _config(tmp_path), tools=[Echo()])

    try:
        await loop.run_turn(request("the first question", KEY), Events(), no_injections, stream=True)
    finally:
        await loop.close_mcp()

    stored = loop.sessions.get_or_create(KEY).messages
    assert [m for m in stored if COMPACTION_KEY in m] == [], "no marker, because nothing compacted"
    assert service.compacted == [], "and the backend was never asked"
