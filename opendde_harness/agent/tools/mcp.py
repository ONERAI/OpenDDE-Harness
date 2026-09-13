"""MCP client: connects to MCP servers and wraps their tools as native OpenDDE Harness tools."""

import asyncio
from collections.abc import Iterable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from opendde_harness.agent.tools import media
from opendde_harness.agent.tools.base import Tool, ToolResult
from opendde_harness.agent.tools.registry import ToolRegistry
from opendde_harness.sandbox import SandboxInitError

if TYPE_CHECKING:
    from opendde_harness.sandbox import SandboxExecutor


@asynccontextmanager
async def _mcp_server_connection(name: str, cfg, transport_type: str, executor: "SandboxExecutor | None"):
    """One server's transport, session and handshake, as a single lifecycle.

    The transport runs in a task of its own. The SDK's transports are anyio
    task groups, and entered in the agent's task they made every later
    transport failure the agent's: a server dropping mid-turn cancelled the
    turn with an ExceptionGroup. Owned by its own task, a failure after the
    handshake ends that task and the server's tools start failing, which the
    wrapper already reports per call. A failure during the handshake is
    raised here, as this server's connection error.
    """
    loop = asyncio.get_running_loop()
    ready: asyncio.Future = loop.create_future()
    closing = asyncio.Event()

    async def run() -> None:
        try:
            async with _mcp_transport(name, cfg, transport_type, executor) as (session, tools):
                ready.set_result((session, tools))
                await closing.wait()
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            raise
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                logger.warning("MCP server '{}': connection ended: {}", name, exc)

    task = asyncio.create_task(run(), name=f"mcp:{name}")

    async def reap() -> None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    try:
        session, tools = await ready
    except BaseException:
        # Never connected (a handshake that hung, or the caller cancelled):
        # nothing to close, and the transport must not outlive the caller.
        await reap()
        raise
    try:
        yield session, tools
    finally:
        closing.set()
        try:
            await asyncio.wait_for(asyncio.shield(task), _MCP_CLOSE_TIMEOUT)
        except asyncio.CancelledError:
            await reap()
            raise
        except BaseException:
            await reap()


#: How long a closing connection may take to unwind its transport.
_MCP_CLOSE_TIMEOUT = 10.0


@asynccontextmanager
async def _mcp_transport(name: str, cfg, transport_type: str, executor: "SandboxExecutor | None"):
    """The transport, session and handshake, entered and left in one task."""
    async with AsyncExitStack() as stack:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client

        if transport_type == "stdio":
            if executor is not None and executor.supports_process_spawning:
                read, write = await executor.start_process(cfg.command, cfg.args, env=cfg.env or None)
            else:
                params = StdioServerParameters(command=cfg.command, args=cfg.args, env=cfg.env or None)
                read, write = await stack.enter_async_context(stdio_client(params))
        elif transport_type == "sse":

            def httpx_client_factory(
                headers: dict[str, str] | None = None,
                timeout: httpx.Timeout | None = None,
                auth: httpx.Auth | None = None,
            ) -> httpx.AsyncClient:
                merged_headers = {**(cfg.headers or {}), **(headers or {})}
                return httpx.AsyncClient(
                    headers=merged_headers or None,
                    follow_redirects=True,
                    timeout=timeout,
                    auth=auth,
                )

            read, write = await stack.enter_async_context(
                sse_client(cfg.url, httpx_client_factory=httpx_client_factory)
            )
        elif transport_type == "streamableHttp":
            http_client = await stack.enter_async_context(
                httpx.AsyncClient(headers=cfg.headers or None, follow_redirects=True, timeout=None)
            )
            read, write, _ = await stack.enter_async_context(streamable_http_client(cfg.url, http_client=http_client))
        else:
            raise ValueError(f"MCP server '{name}': unknown transport type '{transport_type}'")

        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tools = await session.list_tools()
        yield session, tools


class MCPToolWrapper(Tool):
    """Wraps a single MCP server tool as an OpenDDE Harness Tool."""

    # An MCP server declares a name and a schema, not whether calling it changes
    # anything. Treated as if it does: the cost of journaling a read is one
    # line, and the cost of not journaling a write is a job nobody can account
    # for. A wrapper that learns the answer can override this.
    external_effects = True

    def __init__(self, session, server_name: str, tool_def, tool_timeout: int = 30):
        self._session = session
        self._original_name = tool_def.name
        self._server_name = server_name
        self._name = f"mcp_{server_name}_{tool_def.name}"
        self._description = tool_def.description or tool_def.name
        self._parameters = tool_def.inputSchema or {"type": "object", "properties": {}}
        self._tool_timeout = tool_timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def server_name(self) -> str:
        """Which configured MCP server this tool came from.

        Recorded, not inferred. The registered name is
        ``mcp_<server>_<tool>``, and reading ownership back out of that string
        is ambiguous the moment one server's name is another's plus an
        underscore: servers ``a`` and ``a_b`` both prefix-match
        ``mcp_a_b_ping``.
        """
        return self._server_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str | ToolResult:
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(self._original_name, arguments=kwargs),
                timeout=self._tool_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("MCP tool '{}' timed out after {}s", self._name, self._tool_timeout)
            return f"(MCP tool call timed out after {self._tool_timeout}s)"
        except asyncio.CancelledError:
            # MCP SDK's anyio cancel scopes can leak CancelledError on timeout/failure.
            # Re-raise only if our task was externally cancelled (e.g. /stop).
            task = asyncio.current_task()
            if task is not None and task.cancelling() > 0:
                raise
            logger.warning("MCP tool '{}' was cancelled by server/SDK", self._name)
            return "(MCP tool call was cancelled)"
        except Exception as exc:
            logger.exception(
                "MCP tool '{}' failed: {}: {}",
                self._name,
                type(exc).__name__,
                exc,
            )
            return f"(MCP tool call failed: {type(exc).__name__})"

        # str(block) on a pydantic model yields its repr, so an ImageContent used
        # to put its entire base64 payload into the prompt as prose -- the model
        # saw gibberish instead of a picture and nothing errored. Convert per
        # content type instead, and never stringify a payload-bearing block.
        text, blocks = media.blocks_from_mcp_content(result.content)
        if blocks:
            return ToolResult(model_text=text or "(no output)", blocks=blocks)
        return text or "(no output)"


def registered_tool_counts(registry: ToolRegistry, servers: Iterable[str]) -> dict[str, int]:
    """How many tools each named server actually contributes to the registry.

    Counted from the registry rather than from what the server offered, because
    the configured ``disabled_tools`` blacklist runs after the connection and
    unregisters whatever it names. A server whose every tool is disabled stays
    in the mapping with zero: it connected, and that is a different fact from
    never having connected.
    """
    counts = {server: 0 for server in servers}

    for name in registry.tool_names:
        tool = registry.get(name)

        if isinstance(tool, MCPToolWrapper) and tool.server_name in counts:
            counts[tool.server_name] += 1

    return counts


async def connect_mcp_servers(
    mcp_servers: dict,
    registry: ToolRegistry,
    stack: AsyncExitStack,
    executor: "SandboxExecutor | None" = None,
) -> dict[str, int]:
    """Connect to configured MCP servers and register their tools.

    Returns the servers that connected, mapped to how many tools each one
    offered. A server that was skipped or failed to connect is absent from the
    mapping rather than present with zero: those are different facts, and the
    welcome panel reports them differently. The counts are what was offered;
    :func:`registered_tool_counts` is what survived the blacklist, and the
    caller re-counts with it once the blacklist has run.
    """
    registered: dict[str, int] = {}
    for name, cfg in mcp_servers.items():
        # Resolve transport type BEFORE the try/except so the sandbox guard below
        # can raise without being swallowed by the per-server error handler.
        transport_type = cfg.type
        if not transport_type:
            if cfg.command:
                transport_type = "stdio"
            elif cfg.url:
                transport_type = "sse" if cfg.url.rstrip("/").endswith("/sse") else "streamableHttp"
            else:
                logger.warning("MCP server '{}': no command or url configured, skipping", name)
                continue

        # Sandbox guard: fail hard so the agent never starts with a silently broken
        # MCP server. Runs outside try/except — SandboxInitError propagates to
        # _connect_mcp() which surfaces it as a startup error.
        if (
            transport_type == "stdio"
            and executor is not None
            and executor.is_sandboxed
            and not executor.supports_process_spawning
        ):
            raise SandboxInitError(
                f"MCP server '{name}' uses stdio transport, but the active sandbox "
                f"({type(executor).__name__}) does not yet support process spawning. "
                "Either switch to an HTTP/SSE MCP server or set sandbox.backend='none'."
            )

        if transport_type not in ("stdio", "sse", "streamableHttp"):
            logger.warning("MCP server '{}': unknown transport type '{}'", name, transport_type)
            continue

        try:
            # One task per server (see _mcp_server_connection): a failure
            # surfaces here as this server's connection error, the rest still
            # connect, and a later drop never cancels the agent turn.
            session, tools = await stack.enter_async_context(
                _mcp_server_connection(name, cfg, transport_type, executor)
            )
            for tool_def in tools.tools:
                wrapper = MCPToolWrapper(session, name, tool_def, tool_timeout=cfg.tool_timeout)
                registry.register(wrapper)
                logger.debug("MCP: registered tool '{}' from server '{}'", wrapper.name, name)

            registered[name] = len(tools.tools)
            logger.info("MCP server '{}': connected, {} tools registered", name, len(tools.tools))
        except (Exception, BaseExceptionGroup) as e:
            # BaseExceptionGroup is raised by anyio task groups (e.g. streamableHttp cancel
            # scope failures) and is not a subclass of Exception in Python 3.11+.
            logger.error("MCP server '{}': failed to connect: {}", name, e)

    return registered
