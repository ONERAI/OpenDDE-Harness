"""One model result per call, and one observation of it.

The streamed call is reduced to a single authoritative ``LLMResponse``: the
route's own parsed answer where it produces one, and the fragment accumulation
only where it does not. Everything the turn charges, counts and later compacts is
read from one observation of that result, so a reply the loop wrote itself is not
a call and a retried attempt is not two.
"""

from __future__ import annotations

from typing import Any

import pytest

from opendde_harness.agent.loop import AgentLoop
from opendde_harness.agent.loop.factory import AgentLoopSettings
from opendde_harness.agent.tools.base import Tool, ToolResult
from opendde_harness.providers.base import LLMProvider, LLMResponse, StreamDelta, ToolCallRequest
from opendde_harness.token_wise.base import TokenStrategy, UsageSnapshot
from opendde_harness.token_wise.registry import StrategyRegistry
from tests._config import config as build_config
from tests._config import declared

MODEL = "primary/model"
USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cost": 0.5}


def _providers(window: int = 8_000):
    return build_config(
        declared("primary", models=[{"id": "model", "contextWindow": window}]),
        model=MODEL,
    ).providers


class Scripted(LLMProvider):
    """Streams whatever the script says, one list of deltas per call."""

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


class Answering(LLMProvider):
    """A provider that implements only ``chat``, so the base shim streams it."""

    def __init__(self, response: LLMResponse) -> None:
        super().__init__()
        self.response = response
        self.calls = 0

    def get_default_model(self) -> str:
        return MODEL

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls += 1
        return self.response


class Spy(TokenStrategy):
    """Records one entry per observed call."""

    def __init__(self) -> None:
        self.seen: list[UsageSnapshot] = []

    @property
    def name(self) -> str:
        return "spy"

    async def after_llm_call(self, response: dict[str, Any], usage: UsageSnapshot) -> None:
        self.seen.append(usage)


class Refusing(Tool):
    """A tool whose policy terminates the action: no sibling runs after it, and
    the loop answers in its own words rather than asking the model again."""

    @property
    def name(self) -> str:
        return "danger"

    @property
    def description(self) -> str:
        return "refuses"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **_kwargs):
        return ToolResult(model_text="Error: refused by policy.", abort_action=True)


def _loop(tmp_path, provider, **kwargs) -> AgentLoop:
    return AgentLoop(
        provider,
        tmp_path,
        AgentLoopSettings(model=MODEL, providers=_providers(), **kwargs.pop("settings", {})),
        **kwargs,
    )


async def _noop_token(_text: str) -> None:
    """Wiring a token callback is what puts the loop on the streaming path."""


def _tool_call_delta(name: str, index: int = 0, call_id: str = "c1", arguments: str = "{}") -> dict:
    return {
        "tool_calls": [
            {"index": index, "id": call_id, "function": {"name": name, "arguments": arguments}},
        ]
    }


async def test_a_retry_discards_the_attempts_thinking_tool_calls_and_usage(tmp_path) -> None:
    """The re-run sends the same conversation with no word of what was streamed.

    So nothing the discarded attempt accumulated may survive into the answer --
    not its text, and not the three things that are harder to see: the thinking
    blocks that would go back to Anthropic unpaired, the half-arrived tool call
    that would be dispatched, and the usage the turn would be charged for.
    """
    provider = Scripted(
        [
            [
                StreamDelta(content="first", thinking_blocks=[{"type": "thinking", "thinking": "aa"}]),
                StreamDelta(content=None, tool_call_delta=_tool_call_delta("ghost", call_id="ghost-1")),
                StreamDelta(content=None, usage={"prompt_tokens": 9_999, "completion_tokens": 1}),
                StreamDelta(content=None, retry={"attempt": 2, "total": 3, "reason": "503"}),
                StreamDelta(content="second", thinking_blocks=[{"type": "thinking", "thinking": "bb"}]),
                StreamDelta(content=None, tool_call_delta=_tool_call_delta("real", call_id="real-1")),
                StreamDelta(content=None, finish_reason="tool_calls", usage=dict(USAGE)),
            ]
        ]
    )
    loop = _loop(tmp_path, provider)

    response = await loop._llm_call_stream([{"role": "user", "content": "hi"}], None, MODEL)

    assert response.content == "second"
    assert [block["thinking"] for block in response.thinking_blocks or []] == ["bb"]
    assert [call.name for call in response.tool_calls] == ["real"]
    assert response.usage["prompt_tokens"] == 100, "the void attempt's usage is not billed"


async def test_the_non_streaming_shim_hands_over_the_answer_it_already_has(tmp_path) -> None:
    """A provider with only ``chat`` has the exact result; nothing re-parses it.

    The shim used to replay the answer as loose deltas and let the loop rebuild an
    ``LLMResponse`` from them, which is a second parser working from strictly
    less: fragments carry no repaired-argument flag, no truncation verdict and no
    provider identity. Identity is the assertion, because anything else would
    pass while quietly dropping a field.
    """
    answer = LLMResponse(
        content="the answer",
        tool_calls=[ToolCallRequest(id="c1", name="echo", arguments={"text": "hi"})],
        finish_reason="tool_calls",
        usage=dict(USAGE),
        truncated=True,
        model=MODEL,
    )
    provider = Answering(answer)
    loop = _loop(tmp_path, provider)
    shown: list[str] = []

    async def on_token(text: str) -> None:
        shown.append(text)

    response = await loop._llm_call_stream([{"role": "user", "content": "hi"}], None, MODEL, on_token_delta=on_token)

    assert response is answer, "the route's own result, not a rebuild of it"
    assert response.truncated is True, "which is how a field with no fragment survives"
    assert shown == ["the answer"], "and the text still streams to the outlet"


async def test_a_fragment_only_stream_is_still_reconstructed(tmp_path) -> None:
    """The compatibility path: a route that streams deltas and states no result.

    Kept because test doubles and half-written providers are shaped this way. What
    it must produce is what the accumulation always produced -- merged thinking,
    positional tool calls, the final usage.
    """
    provider = Scripted(
        [
            [
                StreamDelta(content="par", thinking_blocks=[{"type": "thinking", "thinking": "a"}]),
                StreamDelta(content="tial", thinking_blocks=[{"type": "thinking", "signature": "sig"}]),
                StreamDelta(content=None, tool_call_delta=_tool_call_delta("echo", arguments='{"text":')),
                StreamDelta(
                    content=None, tool_call_delta={"tool_calls": [{"index": 0, "function": {"arguments": '"hi"}'}}]}
                ),
                StreamDelta(content=None, finish_reason="tool_calls", usage=dict(USAGE)),
            ]
        ]
    )
    loop = _loop(tmp_path, provider)

    response = await loop._llm_call_stream([{"role": "user", "content": "hi"}], None, MODEL)

    assert response.content == "partial"
    assert response.thinking_blocks == [{"type": "thinking", "thinking": "a", "signature": "sig"}]
    assert [(c.name, c.arguments) for c in response.tool_calls] == [("echo", {"text": "hi"})]
    assert response.usage["total_tokens"] == 120


async def test_each_billed_call_is_observed_once_and_a_synthetic_reply_is_not_a_call(tmp_path) -> None:
    """Two calls, two observations; the reply the loop writes itself adds none.

    The turn's cost is the sum over its calls, so an observation counted twice
    doubles a charge and one counted for a locally written answer invents one.
    """
    spy = Spy()
    provider = Scripted(
        [
            [
                StreamDelta(content=None, tool_call_delta=_tool_call_delta("danger")),
                StreamDelta(content=None, finish_reason="tool_calls", usage=dict(USAGE)),
            ],
            [StreamDelta(content="done"), StreamDelta(content=None, finish_reason="stop", usage=dict(USAGE))],
        ]
    )
    loop = _loop(tmp_path, provider, strategies=StrategyRegistry([spy]), plugin_tools=[Refusing()])

    final, _messages, outcome = await loop._run_agent_loop(
        [{"role": "user", "content": "hi"}], session_key="cli:t", on_token_delta=_noop_token
    )

    assert provider.calls == 1, "the refusal ends the turn instead of asking again"
    assert len(spy.seen) == 1, "one call, one observation"
    assert "no alternative method will be attempted" in (final or ""), "and the loop's own words are the reply"
    assert outcome.usage["cost_usd"] == pytest.approx(0.5)


async def test_a_turn_of_two_calls_is_charged_for_both(tmp_path) -> None:
    """The counterpart: every call the turn really made is in its total."""
    spy = Spy()
    provider = Scripted(
        [
            [
                StreamDelta(content=None, tool_call_delta=_tool_call_delta("echo", arguments='{"text":"x"}')),
                StreamDelta(content=None, finish_reason="tool_calls", usage=dict(USAGE)),
            ],
            [StreamDelta(content="done"), StreamDelta(content=None, finish_reason="stop", usage=dict(USAGE))],
        ]
    )

    class Echo(Tool):
        @property
        def name(self) -> str:
            return "echo"

        @property
        def description(self) -> str:
            return "echo"

        @property
        def parameters(self) -> dict:
            return {"type": "object", "properties": {"text": {"type": "string"}}}

        async def execute(self, text: str = "", **_kwargs) -> str:
            return f"echoed {text}"

    loop = _loop(tmp_path, provider, strategies=StrategyRegistry([spy]), plugin_tools=[Echo()])

    _final, _messages, outcome = await loop._run_agent_loop(
        [{"role": "user", "content": "hi"}], session_key="cli:t", on_token_delta=_noop_token
    )

    assert provider.calls == 2 and len(spy.seen) == 2
    assert outcome.usage["cost_usd"] == pytest.approx(1.0), "summed per call, at each call's own tier"
    assert outcome.usage["completion_tokens"] == 20, "but the token fields describe the last call"


async def test_an_exhausted_turns_reply_keeps_the_native_message_of_the_call_that_wrote_it(tmp_path) -> None:
    """The wrap-up after the budget runs out is a call like any other.

    Its native message is what the next request replays, so it has to reach the
    history with the reply -- and the wrap-up has to be the turn's receipt, or the
    after-turn compaction addresses the call before the last tool ran.
    """
    native = {"role": "assistant", "content": [{"type": "text", "text": "wrapped up"}], "provider": "primary"}
    provider = Scripted(
        [
            [
                StreamDelta(content=None, tool_call_delta=_tool_call_delta("echo", arguments='{"text":"x"}')),
                StreamDelta(content=None, finish_reason="tool_calls", usage=dict(USAGE)),
            ],
            [
                StreamDelta(content="wrapped up"),
                StreamDelta(
                    content=None,
                    finish_reason="stop",
                    usage={"prompt_tokens": 300, "completion_tokens": 7, "total_tokens": 307, "cost": 0.25},
                    final_response=LLMResponse(
                        content="wrapped up",
                        finish_reason="stop",
                        usage={"prompt_tokens": 300, "completion_tokens": 7, "total_tokens": 307, "cost": 0.25},
                        model=MODEL,
                        pi_message=native,
                    ),
                ),
            ],
        ]
    )

    class Echo(Tool):
        @property
        def name(self) -> str:
            return "echo"

        @property
        def description(self) -> str:
            return "echo"

        @property
        def parameters(self) -> dict:
            return {"type": "object", "properties": {"text": {"type": "string"}}}

        async def execute(self, text: str = "", **_kwargs) -> str:
            return f"echoed {text}"

    loop = _loop(tmp_path, provider, plugin_tools=[Echo()], settings={"max_iterations": 1})

    final, messages, outcome = await loop._run_agent_loop(
        [{"role": "user", "content": "hi"}], session_key="cli:t", on_token_delta=_noop_token
    )

    assert outcome.status == "interrupted"
    assert final == "wrapped up"
    # The native message *is* the recorded message: it is what the next request
    # replays, which is how a signature or a native call id survives the turn.
    assert messages[-1] == native, "the wrap-up's own native message rides with its reply"
    assert outcome.final_call is not None
    assert outcome.final_call.prompt_tokens == 300, "the receipt is the wrap-up, not the call before it"
    assert outcome.usage["cost_usd"] == pytest.approx(0.75), "and its price is in the turn's total"
