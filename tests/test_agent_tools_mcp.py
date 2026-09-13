"""One MCP server's transport failing must not take the agent turn down with it."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace

import anyio
import mcp
import mcp.client.streamable_http
import pytest

from opendde_harness.agent.tools import mcp as mcp_tools
from opendde_harness.agent.tools.registry import ToolRegistry


def _config(url, transport="streamableHttp"):
    return SimpleNamespace(type=transport, url=url, command="", headers=None, tool_timeout=30)


async def test_a_failing_transport_is_this_servers_error_and_the_next_still_connects(monkeypatch):
    attempted = []

    @asynccontextmanager
    async def fake_streamable_http_client(url, http_client):
        attempted.append(url)
        if url == "https://bad.example/mcp":
            # The SDK's transports run anyio task groups; a failure inside one
            # unwound the enclosing task group and cancelled the turn.
            async with anyio.create_task_group() as group:

                async def fail_transport():
                    await anyio.sleep(0)
                    raise RuntimeError("transport failed")

                group.start_soon(fail_transport)
                yield url, object(), None
        else:
            yield url, object(), None

    class FakeSession:
        def __init__(self, read, write):
            self.read = read

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def initialize(self):
            if self.read == "https://bad.example/mcp":
                await asyncio.Event().wait()

        async def list_tools(self):
            return SimpleNamespace(
                tools=[SimpleNamespace(name="ping", description="", inputSchema={"type": "object", "properties": {}})]
            )

    monkeypatch.setattr(mcp, "ClientSession", FakeSession)
    monkeypatch.setattr(mcp.client.streamable_http, "streamable_http_client", fake_streamable_http_client)

    registry = ToolRegistry()
    async with AsyncExitStack() as stack:
        registered = await mcp_tools.connect_mcp_servers(
            {"bad": _config("https://bad.example/mcp"), "good": _config("https://good.example/mcp")}, registry, stack
        )

    # What the welcome panel reads: the server that connected and how many tools
    # it offered. The one that failed is absent, not present with zero.
    assert registered == {"good": 1}
    assert attempted == ["https://bad.example/mcp", "https://good.example/mcp"]
    assert any(d["function"]["name"].endswith("ping") for d in registry.get_definitions())
    assert asyncio.current_task().cancelling() == 0


async def test_an_unknown_transport_is_skipped_before_any_connection(monkeypatch):
    attempted = False

    @asynccontextmanager
    async def fake_connection(name, cfg, transport_type, executor):
        nonlocal attempted
        attempted = True
        yield None, SimpleNamespace(tools=[])

    monkeypatch.setattr(mcp_tools, "_mcp_server_connection", fake_connection)

    registered = await mcp_tools.connect_mcp_servers(
        {"svc": _config("https://x", transport="carrier-pigeon")}, ToolRegistry(), AsyncExitStack()
    )

    assert attempted is False
    assert registered == {}


async def test_a_transport_failing_after_the_handshake_never_cancels_the_agent_task(monkeypatch):
    @asynccontextmanager
    async def failing_after_handshake(name, cfg, transport_type, executor):
        async with anyio.create_task_group() as group:

            async def drop_later():
                await anyio.sleep(0.05)
                raise RuntimeError("transport dropped")

            group.start_soon(drop_later)
            yield (
                object(),
                SimpleNamespace(
                    tools=[
                        SimpleNamespace(name="ping", description="", inputSchema={"type": "object", "properties": {}})
                    ]
                ),
            )

    monkeypatch.setattr(mcp_tools, "_mcp_transport", failing_after_handshake)

    registry = ToolRegistry()
    async with AsyncExitStack() as stack:
        await mcp_tools.connect_mcp_servers({"svc": _config("https://x")}, registry, stack)
        await asyncio.sleep(0.2)
        assert asyncio.current_task().cancelling() == 0

    assert any(d["function"]["name"].endswith("ping") for d in registry.get_definitions())
    assert asyncio.current_task().cancelling() == 0


async def test_cancelling_a_stuck_handshake_reaps_its_transport_task(monkeypatch):
    @asynccontextmanager
    async def never_ready(name, cfg, transport_type, executor):
        await asyncio.Event().wait()
        yield None, None

    monkeypatch.setattr(mcp_tools, "_mcp_transport", never_ready)

    async def connect():
        async with mcp_tools._mcp_server_connection("stuck", _config("https://x"), "streamableHttp", None):
            pass

    caller = asyncio.create_task(connect())
    await asyncio.sleep(0.05)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    # Reaped before the caller returned, not merely asked to stop.
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "mcp:stuck"]


async def test_a_cancellation_during_close_is_honoured_after_the_reap(monkeypatch):
    closed = asyncio.Event()

    @asynccontextmanager
    async def slow_to_close(name, cfg, transport_type, executor):
        try:
            yield object(), SimpleNamespace(tools=[])
        finally:
            await closed.wait()

    monkeypatch.setattr(mcp_tools, "_mcp_transport", slow_to_close)
    entered = asyncio.Event()
    after_close = []

    async def connect():
        async with mcp_tools._mcp_server_connection("slow", _config("https://x"), "streamableHttp", None):
            entered.set()
            await asyncio.Event().wait()
        after_close.append("ran past the close")

    caller = asyncio.create_task(connect())
    await entered.wait()
    caller.cancel()
    await asyncio.sleep(0.05)
    caller.cancel()  # a second cancellation lands while the close is waiting
    closed.set()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert after_close == []
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "mcp:slow"]


async def test_the_tool_count_is_what_survives_the_disabled_tools_blacklist(monkeypatch):
    """A server offering two tools with one blacklisted contributes one."""

    @asynccontextmanager
    async def fake_connection(name, cfg, transport_type, executor):
        yield (
            None,
            SimpleNamespace(
                tools=[
                    SimpleNamespace(name=tool, description="", inputSchema={"type": "object", "properties": {}})
                    for tool in ("a", "b")
                ]
            ),
        )

    monkeypatch.setattr(mcp_tools, "_mcp_server_connection", fake_connection)

    registry = ToolRegistry()
    offered = await mcp_tools.connect_mcp_servers({"svc": _config("https://x")}, registry, AsyncExitStack())

    # What the server offered, which is what the connector can know.
    assert offered == {"svc": 2}
    assert mcp_tools.registered_tool_counts(registry, offered) == {"svc": 2}

    # The agent applies its blacklist after connecting; the count follows.
    registry.unregister("mcp_svc_b")
    assert mcp_tools.registered_tool_counts(registry, offered) == {"svc": 1}

    # A server whose every tool is disabled still connected, and says zero
    # rather than disappearing into "never connected".
    registry.unregister("mcp_svc_a")
    assert mcp_tools.registered_tool_counts(registry, offered) == {"svc": 0}

    # One server's count never counts another's tools, however similar the name.
    assert mcp_tools.registered_tool_counts(registry, ["svc_extra", "other"]) == {"svc_extra": 0, "other": 0}


async def test_overlapping_server_names_do_not_borrow_each_others_tools(monkeypatch):
    """`a` and `a_b` both prefix-match `mcp_a_b_ping`, and underscores in tool
    names make the ambiguity worse. Ownership is recorded on the wrapper, so
    neither server can claim the other's tool."""
    offered_tools = {"a": ["ping"], "a_b": ["ping", "deep_health_check"]}

    @asynccontextmanager
    async def fake_connection(name, cfg, transport_type, executor):
        yield (
            None,
            SimpleNamespace(
                tools=[
                    SimpleNamespace(name=tool, description="", inputSchema={"type": "object", "properties": {}})
                    for tool in offered_tools[name]
                ]
            ),
        )

    monkeypatch.setattr(mcp_tools, "_mcp_server_connection", fake_connection)

    registry = ToolRegistry()
    servers = {name: _config("https://x") for name in offered_tools}
    offered = await mcp_tools.connect_mcp_servers(servers, registry, AsyncExitStack())

    assert offered == {"a": 1, "a_b": 2}
    # The registered names alone cannot tell these apart: `mcp_a_b_ping` starts
    # with `mcp_a_`, and `mcp_a_b_deep_health_check` starts with `mcp_a_b_`.
    assert set(registry.tool_names) == {"mcp_a_ping", "mcp_a_b_ping", "mcp_a_b_deep_health_check"}
    assert mcp_tools.registered_tool_counts(registry, offered) == {"a": 1, "a_b": 2}

    # The blacklist still applies to the server that actually owns the tool.
    registry.unregister("mcp_a_b_deep_health_check")
    assert mcp_tools.registered_tool_counts(registry, offered) == {"a": 1, "a_b": 1}

    registry.unregister("mcp_a_ping")
    assert mcp_tools.registered_tool_counts(registry, offered) == {"a": 0, "a_b": 1}
