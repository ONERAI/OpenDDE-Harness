"""Build an :class:`AgentLoop`, and everything it is composed of, from config.

Every entry point (the TUI RPC server, the protein-design worker) constructs the
loop through :func:`build_agent_loop`, so the config-to-loop mapping exists
once. :class:`AgentLoopSettings` is the flat, config-independent shape the
constructor takes; tests and eval harnesses fill it directly, and
:func:`build_tool_registry` is the loop's tool composition, which is a list of
registrations rather than anything the turn loop needs to read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from opendde_harness.agent.loop.recovery import RecoveryLimits, limits_from_defaults
from opendde_harness.agent.tools.registry import ToolRegistry
from opendde_harness.config.features import ContextConfig, MemoryConfig, RuntimeConfig, SkillForgeConfig
from opendde_harness.config.schema import Config, ExecToolConfig, ProvidersConfig, ToolSearchConfig

if TYPE_CHECKING:
    from pathlib import Path

    from opendde_harness.agent.hook import CompositeHook
    from opendde_harness.agent.loop.main import AgentLoop
    from opendde_harness.agent.subagent import SubagentManager
    from opendde_harness.agent.tools.base import Tool
    from opendde_harness.memory_engine.backend import MemoryBackend
    from opendde_harness.providers.base import LLMProvider
    from opendde_harness.providers.pool import ProviderPool
    from opendde_harness.sandbox import SandboxExecutor
    from opendde_harness.session.manager import SessionManager
    from opendde_harness.token_wise.registry import StrategyRegistry


@dataclass
class AgentLoopSettings:
    """Everything an :class:`AgentLoop` reads from config.

    ``model=None`` means the provider's default. ``providers`` is the whole
    ``providers`` section, carried so a live model switch can find the new
    model's own declared row -- the window and output ceiling are resolved per
    model from that row and then from the provider, never pinned globally.
    ``skill_forge=None`` takes the skill blocklist and the local skill
    directories as their defaults.
    """

    model: str | None = None
    #: The section the boot provider was built from, or None when nothing
    #: routes the model; kept on the default binding so the surfaces that
    #: name a session's provider answer with the route, not a guess.
    provider: str | None = None
    max_iterations: int = 40
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    empty_recovery: RecoveryLimits = field(default_factory=RecoveryLimits)
    max_concurrent_subagents: int = 4
    max_subagent_spawns_per_hour: int = 30
    jina_api_key: str | None = None
    brave_api_key: str | None = None
    web_proxy: str | None = None
    exec_config: ExecToolConfig = field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    disabled_tools: list[str] = field(default_factory=list)
    tool_search: ToolSearchConfig = field(default_factory=ToolSearchConfig)
    skill_forge: SkillForgeConfig | None = None
    context: ContextConfig = field(default_factory=ContextConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    #: ``config.language``. Read here rather than inside prompt rendering, so a
    #: turn's system prompt is rendered from the turn's settings.
    language: str = "en"

    @classmethod
    def from_config(cls, config: Config) -> AgentLoopSettings:
        defaults = config.agents.defaults
        tools = config.tools
        return cls(
            model=defaults.model,
            provider=config.get_provider_name(defaults.model),
            max_iterations=defaults.max_tool_iterations,
            providers=config.providers,
            empty_recovery=limits_from_defaults(defaults),
            max_concurrent_subagents=defaults.max_concurrent_subagents,
            max_subagent_spawns_per_hour=defaults.max_subagent_spawns_per_hour,
            jina_api_key=tools.web.jina_api_key or None,
            brave_api_key=tools.web.brave_api_key or None,
            web_proxy=tools.web.proxy or None,
            exec_config=tools.exec,
            restrict_to_workspace=tools.restrict_to_workspace,
            mcp_servers=tools.mcp_servers,
            disabled_tools=tools.disabled_tools,
            tool_search=tools.tool_search,
            skill_forge=config.skill_forge,
            context=config.context,
            runtime=config.runtime,
            memory=config.memory,
            language=config.language,
        )


def build_agent_loop(
    config: Config,
    *,
    provider: "LLMProvider",
    session_manager: "SessionManager | None" = None,
    backend: "MemoryBackend | None" = None,
    plugin_tools: "list[Tool] | None" = None,
    interactive: bool = True,
    now_fn: Callable[[], datetime] | None = None,
    hooks: "CompositeHook | None" = None,
    strategies: "StrategyRegistry | None" = None,
    provider_pool: "ProviderPool | None" = None,
) -> "AgentLoop":
    """The one place config becomes an :class:`AgentLoop`.

    ``interactive`` is the call site's signal for ``runtime.checkpoint``:
    a one-shot ``-m`` turn has no next turn to recover into, a TUI session
    does. ``provider_pool`` is what lets a session run on a model other than
    ``provider``'s; a one-shot turn has no session to switch and passes none.

    ``backend`` is the memory owner, and this is where the default one is
    decided. A caller that resolved a plugin backend from ``memory.backend``
    passes it; a config that names no backend gets the host's own markdown
    writer, so a fresh install remembers what was said instead of writing
    nothing. Two values leave the loop with no owner at all: ``memory.backend:
    "off"``, which is the user saying so, and a named backend that failed to
    build, because the config declared an owner and quietly substituting a
    different one would start writing files that backend knows nothing about.
    """
    from opendde_harness.agent.loop.main import AgentLoop

    settings = AgentLoopSettings.from_config(config)
    if backend is None and config.memory.backend is None:
        # Unset, not ``"off"``: the sentinel is the one way to have no owner.
        backend = build_host_memory_backend(config, provider=provider, settings=settings, now_fn=now_fn)

    return AgentLoop(
        provider,
        config.workspace_path,
        settings,
        session_manager=session_manager,
        backend=backend,
        plugin_tools=plugin_tools,
        interactive=interactive,
        now_fn=now_fn,
        hooks=hooks,
        strategies=strategies,
        provider_pool=provider_pool,
    )


def build_host_memory_backend(
    config: Config,
    *,
    provider: "LLMProvider",
    settings: AgentLoopSettings,
    now_fn: Callable[[], datetime] | None = None,
) -> "MemoryBackend":
    """The host's own memory writer, bound to the model the host booted on.

    Built here rather than inside the loop because the loop takes its backend as
    a collaborator: one owner, decided by the call site, which is what keeps a
    plugin backend and this one from both writing.

    The binding is the fallback only. Every annotation resolves the running
    turn's binding first (``providers.binding.resolve``), and the drain task a
    turn spawns inherits that turn's context -- so what indexes a conversation is
    the model the conversation ran on, not whichever one the process started
    with.
    """
    from opendde_harness.memory_engine.consolidate.consolidator import MemoryStore
    from opendde_harness.memory_engine.host_backend import HostMarkdownBackend
    from opendde_harness.providers.binding import ModelBinding

    return HostMarkdownBackend(
        MemoryStore(config.workspace_path, now_fn=now_fn),
        binding=ModelBinding(
            provider,
            settings.model or provider.get_default_model(),
            provider_name=settings.provider,
        ),
        enable_foresight=config.memory.foresight,
        now_fn=now_fn,
    )


def apply_disabled_tools(tools: ToolRegistry, disabled: set[str]) -> None:
    """Unregister tools whose names appear in ``tools.disabled_tools``.

    Run after the default registrations and again after MCP connect, so the
    blacklist can cover either group. Silent on misses -- eval configs commonly
    carry an over-broad list that is a no-op for tools this build never
    registered.
    """
    for name in disabled:
        if tools.has(name):
            tools.unregister(name)


def build_tool_registry(
    *,
    workspace: "Path",
    settings: AgentLoopSettings,
    executor: "SandboxExecutor",
    subagents: "SubagentManager",
    skill_registry: Any | None,
    plugin_tools: "list[Tool]",
    strategies: "StrategyRegistry",
) -> tuple[ToolRegistry, Any | None]:
    """The registry a turn calls into, filled in registration order.

    Order is the contract, not an accident. Built-ins first; plugin tools after
    them, so a plugin that deliberately contributes a built-in's name replaces
    it; then the blacklist, which may strip either group. Returns the registry
    and the :class:`ToolSearchController` when progressive disclosure is on --
    the loop holds it because the tools do, not because it reads it.
    """
    from opendde_harness.agent.tools.ask_user import AskUserTool
    from opendde_harness.agent.tools.file_search import FindTool, GrepTool
    from opendde_harness.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
    from opendde_harness.agent.tools.shell import ExecTool
    from opendde_harness.agent.tools.spawn import SpawnTool
    from opendde_harness.agent.tools.web import WebFetchTool, WebSearchTool

    tools = ToolRegistry()
    allowed_dir = workspace if settings.restrict_to_workspace else None
    for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool, GrepTool, FindTool):
        tools.register(cls(workspace=workspace, allowed_dir=allowed_dir))
    exec_config = settings.exec_config
    tools.register(
        ExecTool(
            working_dir=str(workspace),
            timeout=exec_config.timeout,
            restrict_to_workspace=settings.restrict_to_workspace,
            path_append=exec_config.path_append,
            executor=executor,
            extra_deny_patterns=exec_config.extra_deny_patterns,
        )
    )
    tools.register(WebSearchTool(api_key=settings.brave_api_key, proxy=settings.web_proxy))
    tools.register(WebFetchTool(api_key=settings.jina_api_key, proxy=settings.web_proxy))
    tools.register(SpawnTool(manager=subagents))
    # The QuestionBroker is a per-transport singleton, late-bound via
    # set_broker once the transport (TUI RPC server) exists.
    tools.register(AskUserTool())

    # Plugin-contributed tools (e.g. the memory plugin's ``understand_media``).
    for tool in plugin_tools:
        tools.register(tool)

    # ``use_skill`` is the single skill-body loading path: catalog assembly
    # advertises metadata only, the body is loaded on demand.
    if skill_registry is not None:
        from opendde_harness.agent.tools.use_skill import UseSkillTool

        blocklist = list(getattr(settings.skill_forge, "blocklist", None) or [])
        tools.register(UseSkillTool(registry=skill_registry, blocklist=blocklist))

    # Progressive tool disclosure. Registered last so the catalog it searches
    # covers every built-in/plugin tool above; MCP tools join later (registered
    # in ``_connect_mcp``) and the strategy picks them up since it re-reads the
    # registry each turn.
    controller = None
    cfg = settings.tool_search
    if cfg is not None and cfg.enabled:
        from opendde_harness.agent.tools.tool_search import (
            DEFAULT_ALWAYS_VISIBLE,
            ToolCallTool,
            ToolSearchController,
            ToolSearchStrategy,
            ToolSearchTool,
        )

        controller = ToolSearchController(
            tools,
            always_visible=set(DEFAULT_ALWAYS_VISIBLE) | set(cfg.always_visible),
            search_result_limit=cfg.search_result_limit,
        )
        tools.register(ToolSearchTool(controller))
        tools.register(ToolCallTool(controller))
        # ``first=True``: filter the tool list before any strategy that marks
        # the final tool with ``cache_control`` (else the marked tool may be
        # filtered out and the breakpoint lost).
        strategies.register(
            ToolSearchStrategy(controller, compaction_threshold=cfg.compaction_threshold),
            first=True,
        )

    apply_disabled_tools(tools, set(settings.disabled_tools))
    return tools, controller


def checkpoint_active(policy: str, interactive: bool) -> bool:
    """Resolve ``runtime.checkpoint.policy`` against the call site's
    ``interactive`` signal. ``"interactive"`` (the default) skips the snapshot
    for one-shot ``-m`` invocations -- those have no "next turn" to inject
    recovery into, so paying the snapshot cost there is just deadweight.
    ``"always"`` opts in regardless; ``"never"`` opts out regardless."""
    if policy == "never":
        return False
    if policy == "always":
        return True
    return interactive  # policy == "interactive"


def build_checkpoint_service(settings: AgentLoopSettings, workspace: "Path", interactive: bool) -> Any | None:
    """The per-turn shadow-git snapshot, or ``None`` when the gate is closed.

    A bad ``shadow_dir`` disables the safety net rather than failing the loop --
    a config typo should not stop the agent from running.
    """
    from opendde_harness.agent.loop.checkpoint import CheckpointService

    if not checkpoint_active(settings.runtime.checkpoint.policy, interactive):
        return None
    try:
        return CheckpointService(workspace, shadow_dir=settings.runtime.checkpoint.shadow_dir)
    except ValueError as exc:
        # Bad shadow_dir (e.g. ``../escape`` or an absolute path) -> the service
        # refuses to construct. Log and run without the safety net.
        logger.warning("runtime.checkpoint disabled -- {}", exc)
        return None
