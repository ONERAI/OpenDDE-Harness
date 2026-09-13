"""How the loop is composed: what gets registered, in what order, and once.

The constructor's collaborators and its tool registry moved to
``agent/loop/factory.py``; these are the properties that move had to keep. Two
of them are load-bearing and neither is visible from a turn: a plugin may
override a built-in tool by name, and the disabled-tools blacklist is applied
after every registration lane so it can strip either group. The third is the
absence of the callback plumbing this pass removed -- stated as a test, because
the whole claim was that nothing consumed it.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from opendde_harness.agent.loop import AgentLoop
from opendde_harness.agent.loop.factory import AgentLoopSettings
from opendde_harness.agent.tools.base import Tool
from opendde_harness.providers.base import LLMProvider, LLMResponse

MODEL = "primary/model"


class Quiet(LLMProvider):
    """A provider nothing in this module calls."""

    async def chat(self, messages, tools=None, model=None, **kwargs):
        return LLMResponse(content="ok", finish_reason="stop")

    def get_default_model(self):
        return MODEL


class Shouty(Tool):
    """A plugin tool that claims a built-in's name."""

    def __init__(self, name: str = "read") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "the plugin's own version"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **_kwargs) -> str:
        return "plugin"


def _loop(tmp_path, **kwargs) -> AgentLoop:
    return AgentLoop(Quiet(), tmp_path, AgentLoopSettings(model=MODEL, **kwargs.pop("settings", {})), **kwargs)


def test_a_plugin_tool_of_the_same_name_replaces_the_builtin(tmp_path) -> None:
    """Registration order is the override rule: plugins go last on purpose.

    A plugin that deliberately contributes ``read`` is replacing the
    built-in, not failing to register. Reversing the order would silently give
    the model the built-in instead, which is a behaviour change nothing would
    report.
    """
    plugin = Shouty()
    loop = _loop(tmp_path, plugin_tools=[plugin])

    assert loop.tools.get("read") is plugin


def test_the_blacklist_strips_a_builtin_and_a_plugin_alike(tmp_path) -> None:
    """Applied after every registration lane, so it can name either group."""
    loop = _loop(
        tmp_path,
        plugin_tools=[Shouty("plugin_only")],
        settings={"disabled_tools": ["plugin_only", "web_search", "never_registered"]},
    )

    assert not loop.tools.has("plugin_only"), "a plugin tool is not exempt"
    assert not loop.tools.has("web_search"), "and neither is a built-in"
    assert loop.tools.has("read"), "everything else is left alone"


def test_the_registry_carries_the_builtin_set(tmp_path) -> None:
    """The list a default install advertises, so a move cannot quietly drop one."""
    loop = _loop(tmp_path)

    for name in ("read", "write", "edit", "ls", "grep", "find", "bash", "spawn", "ask_user"):
        assert loop.tools.has(name), name


def test_tool_search_registers_its_strategy_before_any_other(tmp_path) -> None:
    """``first=True``: the filter runs before a strategy that marks the last
    tool for caching, or the marked tool can be filtered out and the breakpoint
    lost."""
    from opendde_harness.config.schema import ToolSearchConfig

    loop = _loop(tmp_path, settings={"tool_search": ToolSearchConfig(enabled=True)})

    assert loop.tools.has("tool_search") and loop.tools.has("tool_call")
    assert type(loop.strategies.strategies[0]).__name__ == "ToolSearchStrategy"


async def test_two_turns_starting_at_once_connect_the_mcp_servers_once(tmp_path, monkeypatch) -> None:
    """Connecting is once per process, and the blacklist and the counts follow it.

    The counts are what a surface reports as "tools from this server", and they
    are taken after the blacklist rather than before: what a server offered and
    what the agent may call are different numbers whenever the blacklist names
    one of its tools.

    A second caller that arrives while the first is still connecting returns
    without waiting -- current behaviour, stated here so a later change to it is
    a decision rather than a side effect.
    """
    from opendde_harness.agent.tools import mcp as mcp_module

    calls: list[dict] = []

    async def fake_connect(servers, tools, stack, *, executor=None):
        calls.append(dict(servers))
        await asyncio.sleep(0)
        tools.register(Shouty("mcp_demo_search"))
        tools.register(Shouty("mcp_demo_fetch"))
        return {"demo": object()}

    monkeypatch.setattr(mcp_module, "connect_mcp_servers", fake_connect)
    monkeypatch.setattr(
        mcp_module,
        "registered_tool_counts",
        lambda tools, connected: {
            name: sum(1 for t in ("mcp_demo_search", "mcp_demo_fetch") if tools.has(t)) for name in connected
        },
    )
    loop = _loop(
        tmp_path,
        settings={"mcp_servers": {"demo": {"transport": "stdio"}}, "disabled_tools": ["mcp_demo_search"]},
    )

    await asyncio.gather(loop._connect_mcp(), loop._connect_mcp())

    assert len(calls) == 1, "one connect, however many callers asked"
    assert not loop.tools.has("mcp_demo_search"), "the blacklist is re-applied after MCP"
    assert loop.tools.has("mcp_demo_fetch")
    assert loop.mcp_tool_counts == {"demo": 1}, "counted after the blacklist, not before"


@pytest.mark.parametrize(
    "name",
    ["_record_side_call", "is_processing", "_notify_turn_complete", "on_turn_complete", "_processing_lock"],
)
def test_the_retired_callback_plumbing_is_gone(tmp_path, name) -> None:
    """It had no consumer: no caller, and a lock nobody acquired.

    Named one by one rather than checked as a set, so a failure says which piece
    came back.
    """
    loop = _loop(tmp_path)

    assert not hasattr(loop, name), f"{name} was retired; nothing consumed it"


def test_the_context_engine_takes_no_usage_sink(tmp_path) -> None:
    """Its LLM callers -- the curator, the gate, the rewriter -- were deleted, so
    the engine makes no call of its own for a sink to catch."""
    from opendde_harness.context_engine.factory import build_context_engine

    assert "usage_sink" not in inspect.signature(build_context_engine).parameters


def test_the_turn_entry_takes_no_inline_tool_stream_flag(tmp_path) -> None:
    """It was accepted and forwarded by three layers and read by none."""
    from opendde_harness.agent.spine_runner import AgentTurnRunner

    assert "inline_tool_stream" not in inspect.signature(AgentLoop.run_turn).parameters
    assert "inline_tool_stream" not in inspect.signature(AgentTurnRunner.__init__).parameters
