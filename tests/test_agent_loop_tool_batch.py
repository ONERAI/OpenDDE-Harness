"""One assistant message's tool calls, run in order, with the stops that bind.

The batch is sequential and every advertised call id gets a result, because an
OpenAI-style backend refuses a history holding a call without one. Two stops are
safety rather than bookkeeping: a terminating decision cancels the siblings behind
it and ends the turn in the loop's own words, and a call the turn was cut in the
middle of must not run at all.
"""

from __future__ import annotations

from opendde_harness.agent.loop import AgentLoop
from opendde_harness.agent.loop.factory import AgentLoopSettings
from opendde_harness.agent.tools.base import Tool, ToolResult
from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import LLMProvider, LLMResponse, StreamDelta
from tests._config import config as build_config
from tests._config import declared

MODEL = "primary/model"
USAGE = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}


async def _noop_token(_text: str) -> None:
    """Wiring a token callback is what puts the loop on the streaming path."""


class Scripted(LLMProvider):
    def __init__(self, script: list[list[StreamDelta]]) -> None:
        super().__init__()
        self.script = [list(deltas) for deltas in script]
        self.calls = 0

    def get_default_model(self) -> str:
        return MODEL

    async def chat(self, messages, tools=None, model=None, **kwargs):
        return LLMResponse(content="ok", finish_reason="stop", usage=dict(USAGE))

    async def chat_stream(self, messages, tools=None, model=None, **kwargs):
        deltas = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        for delta in deltas:
            yield delta


class Recorder(Tool):
    """Counts its runs, so a suppressed sibling is provably suppressed."""

    def __init__(self, name: str, *, aborts: bool = False) -> None:
        self._name = name
        self.runs = 0
        self._aborts = aborts

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "records"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {"text": {"type": "string"}}}

    async def execute(self, text: str = "", **_kwargs):
        self.runs += 1
        if self._aborts:
            return ToolResult(model_text="Error: refused by policy.", abort_action=True)
        return f"ran {self._name}"


def _loop(tmp_path, provider, **kwargs) -> AgentLoop:
    providers = build_config(
        declared("primary", models=[{"id": "model", "contextWindow": 8_000}]), model=MODEL
    ).providers
    return AgentLoop(provider, tmp_path, AgentLoopSettings(model=MODEL, providers=providers), **kwargs)


def _calls(*names: str) -> dict:
    return {
        "tool_calls": [
            {"index": i, "id": f"c{i}", "function": {"name": name, "arguments": '{"text":"x"}'}}
            for i, name in enumerate(names)
        ]
    }


async def test_a_terminating_decision_cancels_the_siblings_behind_it(tmp_path) -> None:
    """One assistant message can ask for ``rm`` and a Python fallback together.

    Once policy terminates the action none of the siblings may run -- and each
    still needs a result under its own id, or the next request is refused for a
    call with no answer. The turn then ends in the loop's own words: another model
    iteration could translate the rejected operation into an equivalent one.
    """
    refusing, fallback = Recorder("danger", aborts=True), Recorder("fallback")
    provider = Scripted(
        [
            [
                StreamDelta(content=None, tool_call_delta=_calls("danger", "fallback")),
                StreamDelta(content=None, finish_reason="tool_calls", usage=dict(USAGE)),
            ]
        ]
    )
    loop = _loop(tmp_path, provider, plugin_tools=[refusing, fallback])
    streamed: list[str] = []

    async def on_token(text: str) -> None:
        streamed.append(text)

    final, messages, _outcome = await loop._run_agent_loop(
        [{"role": "user", "content": "hi"}], session_key="cli:t", on_token_delta=on_token
    )

    assert refusing.runs == 1 and fallback.runs == 0, "the sibling behind the refusal never ran"
    results = [m for m in messages if msg.is_tool_result(m)]
    assert len(results) == 2, "every advertised call id still has a result"
    assert "prior safety decision terminated this action" in str(results[-1])
    assert provider.calls == 1, "and no further iteration is offered the rejected action"
    assert "no alternative method will be attempted" in (final or "")
    assert streamed == [final], "a streaming outlet is told the reply too"


async def test_the_results_come_back_in_the_order_the_calls_were_asked_for(tmp_path) -> None:
    """Sequential, and the history says so: the batch is not a set."""
    first, second = Recorder("alpha"), Recorder("beta")
    provider = Scripted(
        [
            [
                StreamDelta(content=None, tool_call_delta=_calls("alpha", "beta")),
                StreamDelta(content=None, finish_reason="tool_calls", usage=dict(USAGE)),
            ],
            [StreamDelta(content="done"), StreamDelta(content=None, finish_reason="stop", usage=dict(USAGE))],
        ]
    )
    loop = _loop(tmp_path, provider, plugin_tools=[first, second])

    _final, messages, _outcome = await loop._run_agent_loop(
        [{"role": "user", "content": "hi"}], session_key="cli:t", on_token_delta=_noop_token
    )

    bodies = [str(m) for m in messages if msg.is_tool_result(m)]
    assert len(bodies) == 2
    assert "ran alpha" in bodies[0] and "ran beta" in bodies[1]


async def test_a_call_the_turn_was_cut_inside_is_refused_before_it_runs(tmp_path) -> None:
    """A length stop puts the last call in doubt, and the doubt is not dispatched.

    Refusing costs one retry. Running a half-transmitted ``write`` turns an
    append into an overwrite that reports success, so the refusal is the cheap
    side of the trade.
    """
    writer = Recorder("write")
    provider = Scripted(
        [
            [
                StreamDelta(content=None, tool_call_delta=_calls("write")),
                StreamDelta(content=None, finish_reason="length", usage=dict(USAGE)),
            ],
            [StreamDelta(content="done"), StreamDelta(content=None, finish_reason="stop", usage=dict(USAGE))],
        ]
    )
    loop = _loop(tmp_path, provider, plugin_tools=[writer])

    _final, messages, _outcome = await loop._run_agent_loop(
        [{"role": "user", "content": "hi"}], session_key="cli:t", on_token_delta=_noop_token
    )

    assert writer.runs == 0, "the call the reply was cut inside was not run"
    assert "[truncated]" in str([m for m in messages if msg.is_tool_result(m)][0])


async def test_the_same_refusal_applies_to_a_length_stopped_native_message(tmp_path) -> None:
    """The route must not decide whether a cut call is safe to run.

    Same turn, same length stop, same doubt -- stated by the model layer's own
    parsed message rather than assembled from fragments. This is the production
    path for every pi-served model, so a guard only the compatibility path applied
    was a guard that was not there.

    And the doubt covers the whole message, not its tail: generation is sequential,
    so the cut landed inside one of these calls and nothing says which. pi's own
    loop fails every call of a length-stopped message for that reason, and this
    does the same.
    """
    first, second = Recorder("write"), Recorder("bash")
    from opendde_harness.providers.base import ToolCallRequest

    cut = LLMResponse(
        content="",
        tool_calls=[
            ToolCallRequest(id="c0", name="write", arguments={"text": "half"}),
            ToolCallRequest(id="c1", name="exec", arguments={"text": "whole"}),
        ],
        finish_reason="length",
        usage=dict(USAGE),
        truncated=True,
        model=MODEL,
    )
    provider = Scripted(
        [
            [
                StreamDelta(content=None, tool_call_delta=_calls("write", "bash"), final_response=cut),
                StreamDelta(content=None, finish_reason="length", usage=dict(USAGE)),
            ],
            [StreamDelta(content="done"), StreamDelta(content=None, finish_reason="stop", usage=dict(USAGE))],
        ]
    )
    loop = _loop(tmp_path, provider, plugin_tools=[first, second])

    _final, messages, _outcome = await loop._run_agent_loop(
        [{"role": "user", "content": "hi"}], session_key="cli:t", on_token_delta=_noop_token
    )

    assert first.runs == 0, "a cut call is not run, whichever path parsed it"
    assert second.runs == 0, "and neither is the one beside it, which may be the cut one"
    results = [str(m) for m in messages if msg.is_tool_result(m)]
    assert len(results) == 2, "every advertised call id still has a result"
    assert all("[truncated]" in body and "output" in body for body in results)
    assert provider.calls == 2, "the turn carries on so the model can send them again"
