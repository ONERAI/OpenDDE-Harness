"""session.* RPC handlers (lifecycle + management).

``session.create`` mints a fresh ``tui:<chat_id>`` key on every call (lazy —
no file is written until the session's first save). ``session.resume`` loads
the stored transcript from disk for a known ``session_id`` and falls back to a
fresh-minted key with empty messages for an unknown or absent id.
``session.close`` flushes any unpersisted messages of the named session.
``session.list`` returns tui-channel sessions sorted by updated_at desc.
``session.delete`` removes a session file and invalidates the cache.
``session.most_recent`` wraps find_most_recent_chat_id("tui").
``session.title`` sets or gets the title field in session metadata (lazy —
title persists on the next save that writes metadata).

Wire shape for session.create/resume: the ``info`` field is the init bundle
consumed by ``ui-tui/src/components/branding.tsx`` (SessionPanel). Requires
``info.skills`` / ``info.tools`` / ``info.model`` — Object.entries(info.skills)
on line 138 will throw if these are missing.

``agent_loop=None`` graceful fallback: empty tools/skills, zero usage,
``lazy=True``. Mirrors ``turn.py``'s factory-exception guard.

Known divergence: ``system.hello`` still advertises ``default_session_key``
``tui:default`` and the ui-tui turn path still hardcodes it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from opendde_harness.cli.update_notice import update_notice
from opendde_harness.config.loader import load_config
from opendde_harness.providers import messages as msg
from opendde_harness.security.trust import unwrap_untrusted
from opendde_harness.session.export import default_export_path, write_transcript
from opendde_harness.session.manager import SessionManager, new_chat_id
from opendde_harness.tui_rpc.errors import TurnInProgressError
from opendde_harness.tui_rpc.methods import instructions
from opendde_harness.tui_rpc.methods import turn as turn_module
from opendde_harness.tui_rpc.methods.system import _opendde_harness_version

if TYPE_CHECKING:
    from opendde_harness.agent.loop.main import AgentLoop
    from opendde_harness.config.schema import Config
    from opendde_harness.session.manager import Session
    from opendde_harness.token_wise.base import UsageSnapshot
    from opendde_harness.token_wise.usage_tracker import CallCounts, SessionUsage
    from opendde_harness.tui_rpc.dispatcher import Dispatcher


AgentLoopFactory = Callable[[], "AgentLoop | None"]


# Cache the package version once at module load — importlib.metadata.version
# walks site-packages dist-info on every call. system._opendde_harness_version()
# already guards PackageNotFoundError for source-checkout environments.
_OPENDDE_HARNESS_VERSION = _opendde_harness_version()

# ``## [0.0.3] - 2026-09-10`` in CHANGELOG.md: the only place this project
# records when a version shipped. Package metadata carries no release date --
# a dist-info directory is dated when it was installed, which is a different
# fact -- so the changelog ships in the wheel beside the package and is read
# from there.
_RELEASE_HEADING_RE = re.compile(r"^##\s*\[(?P<version>[^\]]+)\]\s*-\s*(?P<date>\d{4}-\d{2}-\d{2})\s*$", re.M)


def _changelog_path() -> Path | None:
    """The shipped changelog, or the checkout's own when running from source."""
    package_dir = Path(__file__).resolve().parent.parent.parent
    for candidate in (package_dir / "CHANGELOG.md", package_dir.parent / "CHANGELOG.md"):
        if candidate.is_file():
            return candidate
    return None


def _release_date(version: str) -> str | None:
    """When this version shipped, or None when nothing records it.

    Matched on the exact version first, then on the ``X.Y.Z`` it builds on, so
    a release candidate is dated by the release it is a candidate for rather
    than showing nothing. An unreadable or absent changelog is not an error:
    the panel simply shows the version alone.
    """
    path = _changelog_path()
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    dates = {match["version"].strip().lstrip("vV"): match["date"] for match in _RELEASE_HEADING_RE.finditer(text)}
    if not dates:
        return None
    raw = version.strip().lstrip("vV")
    if raw in dates:
        return dates[raw]
    prefix = re.match(r"\d+\.\d+\.\d+", raw)
    return dates.get(prefix.group(0)) if prefix else None


_OPENDDE_HARNESS_RELEASE_DATE = _release_date(_OPENDDE_HARNESS_VERSION)


def _safe_invoke_factory(
    factory: "AgentLoopFactory | None",
) -> "AgentLoop | None":
    """Invoke ``factory()`` with the same try/except guard ``turn.py`` uses.

    Boot races, transient construction failures, or any other factory-raises
    path must degrade to ``agent_loop=None`` (lazy bundle) rather than crash
    the banner. Mirrors ``turn.py::turn_send`` lines 103-109.
    """
    if factory is None:
        return None
    try:
        return factory()
    except Exception:
        logger.exception("session.*: agent_loop_factory raised")
        return None


def _enumerate_tools(agent_loop: "AgentLoop | None") -> dict[str, list[str]]:
    """Banner ``info.tools`` subfield — single ``"builtin"`` bucket per handoff §3.4."""
    if agent_loop is None:
        return {}
    return {"builtin": sorted(agent_loop.tools.tool_names)}


def _enumerate_skills(agent_loop: "AgentLoop | None") -> dict[str, list[str]]:
    """Banner ``info.skills`` subfield — group by ``source``.

    ``LocalSkillCatalog.list_skills(filter_unavailable=True)`` returns the
    legacy drop-in shape ``list[dict[str, str]]`` (``{name, path, source}``),
    not :class:`SkillMeta` instances.
    """
    if agent_loop is None:
        return {}
    skills = agent_loop.context.skills.list_skills(filter_unavailable=True)
    grouped: dict[str, list[str]] = {}
    for skill in skills:
        grouped.setdefault(skill["source"], []).append(skill["name"])
    return {source: sorted(names) for source, names in grouped.items()}


_MCP_TRANSPORTS = ("stdio", "sse", "streamableHttp")


def _mcp_transport(cfg: Any) -> str | None:
    """The transport ``connect_mcp_servers`` will use for this server.

    The same rules, so the panel never names a transport the connector would
    not use. ``None`` means the connector skips this server entirely, and a
    server nothing will ever connect to is not worth a row.
    """
    declared = getattr(cfg, "type", None)
    if declared:
        return declared if declared in _MCP_TRANSPORTS else None
    if getattr(cfg, "command", None):
        return "stdio"
    url = getattr(cfg, "url", None) or ""
    if url:
        return "sse" if url.rstrip("/").endswith("/sse") else "streamableHttp"
    return None


def _enumerate_mcp_servers(agent_loop: "AgentLoop | None") -> list[dict[str, Any]]:
    """Banner ``info.mcp_servers`` subfield — the configured servers and what
    each has registered.

    MCP connects lazily, on the session's first turn, so a freshly created
    session reports its servers configured and not yet connected. Asking for
    the bundle again (``session.info``) after a turn shows them connected with
    their tool counts.
    """
    if agent_loop is None:
        return []
    counts = getattr(agent_loop, "mcp_tool_counts", {})
    servers = []
    for name, cfg in getattr(agent_loop, "mcp_servers", {}).items():
        transport = _mcp_transport(cfg)
        if transport is None:
            continue
        servers.append(
            {
                "name": name,
                "transport": transport,
                "connected": name in counts,
                "tool_count": counts.get(name, 0),
            }
        )
    return sorted(servers, key=lambda server: server["name"])


def _past_a_replayable_compaction(messages: "list[dict[str, Any]] | None", binding) -> bool:
    """Whether the backend will replace this history with one opaque item.

    A compaction marker stands for everything before it, and the provider that
    wrote it sends it in place of that history on the model that made it. What
    the window will hold on the next call is therefore the marker plus whatever
    followed it -- and the marker's own size is the backend's secret. Summing
    the messages it replaced answers a question nobody asked: 90,000 characters
    of superseded history read as 7% of a window that will never receive them.

    pi treats the same state as unmeasured (``getContextUsage``), and so does
    this: unknown until a call reports on the conversation that exists now.
    Only a marker this session's own binding would replay counts -- another
    provider's marker is inert text, and the history around it really is what
    gets sent.
    """
    if not messages or binding is None:
        return False

    provider = getattr(binding, "provider", None)
    if provider is None:
        return False

    from opendde_harness.providers.base import compaction_boundary

    try:
        return compaction_boundary(list(messages), provider, getattr(binding, "model", None)) is not None
    except Exception:  # pragma: no cover - a provider that cannot answer replays nothing
        logger.debug("session info: could not ask the provider about a compaction marker")
        return False


async def _estimated_context(messages: "list[dict[str, Any]] | None") -> int:
    """How many tokens the stored messages are worth, or 0 if nobody can say.

    Never raises and never fetches. A session has to open on a machine with no
    tokenizer cache and no network -- the first thing a fresh install does --
    so the tokenizer is asked only when it can answer from memory, and the
    character rule the TUI's own live estimate uses answers otherwise. Any
    failure at all leaves the baseline absent: the footer then opens at 0.0%,
    which is what it did before this figure existed, and the first turn
    replaces it with the vendor's own.
    """
    if not messages:
        return 0

    try:
        from opendde_harness.utils.helpers import estimate_prompt_tokens, tokenizer_is_loaded

        if tokenizer_is_loaded():
            return await asyncio.to_thread(estimate_prompt_tokens, list(messages))
        return await asyncio.to_thread(_rough_context, list(messages))
    except Exception:
        logger.debug("session info: could not size the stored history; opening the window figure at zero")
        return 0


def _rough_context(messages: "list[dict[str, Any]]") -> int:
    """Characters over four: the rule every vendor quotes for an estimate, and
    the one ``turn.ts`` counts a live reply with, so the two agree."""
    chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif content is not None:
            chars += len(json.dumps(content, ensure_ascii=False, default=str))
    return chars // 4


async def _occupancy(messages: "list[dict[str, Any]] | None", binding) -> int | None:
    """What the stored conversation puts in front of the next call, or None.

    The one occupancy policy. ``session.info``, ``session.resume`` and
    ``session.undo`` all answer for the same session and must not disagree
    about it: undo used to size the raw history itself, so removing an exchange
    from a session whose window was unmeasured turned a truthful ``?`` into a
    number that counted messages the backend replaces with one opaque marker.

    None means unmeasured, not zero. Only a call that runs after the boundary
    can say what the window holds.
    """
    if _past_a_replayable_compaction(messages, binding):
        return None

    return await _estimated_context(messages)


def _history_usage(
    messages: "list[dict[str, Any]] | None",
    model: str,
    session_key: str,
    *,
    plan_billed: bool,
) -> "tuple[UsageSnapshot, CallCounts] | None":
    """What this session's stored records say its earlier turns moved and cost.

    Every assistant record the model service wrote is a pi message carrying the
    usage of the call that produced it -- the tokens, the cache figures, and
    under ``cost.total`` what pi priced it at from the catalogue that served it.
    Summing them is the only way a process that did not make those calls can
    answer for them, and without it a resumed session opened at zero tokens and
    ``$0.000`` however long its conversation was.

    Priced call by call, because that is how it was stored: each record carries
    its own call's figure, at whatever long-context tier that call reached, so
    the sum is the session's cost rather than a rate applied to a total. pi's
    figure is the model's published price, which is why it lands in
    ``list_cost_usd``; ``estimated_cost_usd`` takes it too unless a plan is
    paying, in which case there is no per-token figure to report.

    A record priced at zero is counted for its tokens and not for its money: pi
    prices a model it carries no rates for at 0, which is "nobody knows" rather
    than "free" (see ``accounting.build_usage_snapshot``). None when no record
    carries a usage figure at all, which is a session written before the model
    service or one with nothing in it -- distinct from a session that really
    moved no tokens.
    """
    from opendde_harness.providers import messages as msg
    from opendde_harness.providers.pi_context import usage_dict
    from opendde_harness.token_wise.base import snapshot_from_usage
    from opendde_harness.token_wise.usage_tracker import CallCounts

    total = snapshot_from_usage({}, model, session_key)
    counts = CallCounts()
    for record in messages or ():
        if not isinstance(record, dict) or record.get("role") != msg.ASSISTANT:
            continue
        usage = usage_dict(record.get("usage"))
        if not usage:
            continue
        call = snapshot_from_usage(usage, model, session_key)
        if not (call.input_tokens or call.output_tokens or call.cache_read_tokens or call.cache_write_tokens):
            continue
        counts.calls += 1
        if call.cache_reported:
            counts.cache_reported += 1
            counts.cache_reported_prompt_tokens += call.input_tokens + call.cache_read_tokens + call.cache_write_tokens
        total.input_tokens += call.input_tokens
        total.output_tokens += call.output_tokens
        total.cache_read_tokens += call.cache_read_tokens
        total.cache_write_tokens += call.cache_write_tokens
        total.cache_reported = total.cache_reported or call.cache_reported
        priced = usage.get("cost")
        if isinstance(priced, (int, float)) and not isinstance(priced, bool) and priced > 0:
            total.list_cost_usd = (total.list_cost_usd or 0.0) + float(priced)
            if not plan_billed:
                total.estimated_cost_usd = (total.estimated_cost_usd or 0.0) + float(priced)
    return (total, counts) if counts.calls else None


def _session_totals(
    agent_loop: "AgentLoop | None",
    session_key: str | None,
    messages: "list[dict[str, Any]] | None",
    model: str,
    *,
    plan_billed: bool,
) -> "SessionUsage | None":
    """The session's spend so far, adopting its stored history the first time.

    The tracker is the one source ``/status`` and the footer both read, so this
    answers from it -- and seeds it from the session file when it has nothing for
    this key, which is every resumed session in a fresh process. Seeded once:
    the tracker refuses a key it already has totals for, because a process that
    has run a turn counted that turn itself and would otherwise count the file's
    copy of it again.
    """
    tracker = getattr(agent_loop, "usage_tracker", None)
    if tracker is None or not session_key:
        return None
    try:
        usage = tracker.session_usage(session_key)
        if usage.counts.calls:
            return usage
        history = _history_usage(messages, model, session_key, plan_billed=plan_billed)
        if history is None:
            return usage
        tracker.seed_session(*history)
        return tracker.session_usage(session_key)
    except Exception:  # pragma: no cover - a tracker that cannot answer reports nothing
        logger.warning("session.info: could not total the usage for {}", session_key)
        return None


async def _prime_window(binding) -> None:
    """Let the model service answer for this model before the banner reports it.

    The window, the output ceiling, what the model can be shown and what it
    costs all come from one row the service reports, and before the first request
    of a process there is no row: ``PiModelProvider.context_window`` is None, so
    a resumed session's footer had a token count and no window to divide it by
    and showed no percentage at all until the first turn landed. One listing
    answers it, and the provider caches the row, so the turn that follows asks
    nothing extra.

    A failure is not a failure of the call. Every fact then reads as unknown,
    which is exactly what it read as before -- no window invented, no price
    reported as zero.
    """
    prime = getattr(getattr(binding, "provider", None), "prime", None)
    if prime is None:
        return
    try:
        await prime()
    except Exception:
        logger.debug("session.info: could not prime the model row; its window stays unknown")


async def _baseline_usage(
    agent_loop: "AgentLoop | None",
    model: str | None,
    messages: "list[dict[str, Any]] | None" = None,
    binding=None,
    *,
    plan_billed: bool = False,
    session_key: str | None = None,
) -> dict[str, Any]:
    """Banner ``info.usage`` subfield — what this session has spent and holds.

    Zero for a session being created: a fresh key carries no prior calls. For a
    resumed one, the session's own totals -- the tracker's if this process has
    run a turn on the key, otherwise its stored records adopted into the tracker
    (:func:`_session_totals`). A resumed session used to open at zero tokens and
    ``$0.000`` with forty calls behind it, and the footer only ever recovered
    the current process's share of the figure. Each turn's ``message.complete``
    carries the same totals from the same tracker afterwards.

    ``context_max`` is the loop's own ladder (``resolve_window``): the model's
    declared row, then what the model layer reports for the model, or 0 when
    neither knows — the UI's empty state, not a borrowed number;
    ``context_source`` says which. The caller primes the row first
    (:func:`_prime_window`), so the middle tier can answer before any call.

    ``plan_billed`` is asked once, by the caller, and handed in: the question
    needs the providers map (a plan is a sign-in on the provider's entry) and
    the banner reports the same answer beside the price, so asking it twice from
    two places is how the two came to disagree. Default metered -- a figure of
    zero reported as free is the failure here, and claiming a plan nobody
    declared would suppress every real figure instead.

    Cost is the exception to reporting a zero: on a subscription there is no
    per-token figure, so the banner says so rather than opening at $0.00.
    ``list_cost_usd`` is the published price of the same spend and is what the
    footer shows, plan or no plan.

    No tier reaches the network except the priming the caller has already done.
    The ladder walk and the record sum are pushed to a thread: this handler runs
    on the event loop (an RPC method), and a resolution that grew a disk read
    would otherwise block every other session in flight.
    """
    from opendde_harness.providers.rates import SOURCE_UNKNOWN

    if agent_loop is not None and model:
        resolved = await asyncio.to_thread(agent_loop.resolve_window, model, binding)
        context_max, context_source = resolved.tokens or 0, resolved.source
    else:
        context_max, context_source = 0, SOURCE_UNKNOWN
    spent = await asyncio.to_thread(
        _session_totals, agent_loop, session_key, messages, model or "", plan_billed=plan_billed
    )
    totals = spent.totals if spent is not None else None
    # What the session already holds. Zero is right for a session with no
    # messages and wrong for a resumed one: its history is in the window from
    # its first call, and a status bar that opens a populated session at 0%
    # says the opposite. Estimated, as pi estimates it when no usage has been
    # reported yet, and replaced by the first real figure the next turn gives.
    # Unless a compaction marker this session's backend replays stands in
    # front of that history, in which case nothing here measures what the next
    # call will send and null says so.
    context_used = await _occupancy(messages, binding)
    return {
        "input": (0 if totals is None else totals.input_tokens + totals.cache_read_tokens + totals.cache_write_tokens),
        "output": 0 if totals is None else totals.output_tokens,
        # None on a plan either way: there the subscription is the price, and a
        # number here -- zero or a sum -- answers a question nobody is billed.
        "cost_usd": None if plan_billed else (0.0 if totals is None else (totals.estimated_cost_usd or 0.0)),
        "list_cost_usd": None if totals is None else totals.list_cost_usd,
        "calls": 0 if spent is None else spent.counts.calls,
        "context_max": context_max,
        "context_source": context_source,
        "context_used": context_used,
        "context_percent": (
            None if context_used is None else round(100 * context_used / context_max) if context_max else 0
        ),
    }


def _session_binding(agent_loop: "AgentLoop | None", session_key: str | None):
    """The model, provider section and provider object this session runs on.

    ``binding_for_session`` is the only thing that answers for a session that
    switched: the loop's own ``provider`` property is the default binding
    outside a running turn, and re-deriving a provider from the model id asks
    the routing rules a question a configured default answers for them. None
    when there is no loop or no session yet, in which case the caller is
    describing the defaults a new session would start on.
    """
    resolve = getattr(agent_loop, "binding_for_session", None)
    if resolve is None or not session_key:
        return None
    try:
        return resolve(session_key)
    except Exception:  # pragma: no cover - a loop that cannot answer is a default
        logger.warning("session.info: could not resolve the binding for {}", session_key)
        return None


def _on_subscription(model: str | None, config: "Config") -> bool:
    """Whether this session's model is billed by plan rather than per token.

    The arrangement is on the provider's own entry: ``login: "oauth"`` is a
    subscription (a ChatGPT account, a coding plan) and a key is metered. Read
    from the config rather than from a table of ours, because what a plan covers
    is the account and only the account's entry says so. The footer marks a plan
    ``(sub)`` because the price beside it describes nothing.

    The model id is the whole question now: it names the provider whose entry
    answers, so this no longer needs the session's binding to tell it which
    provider was resolved -- the binding is built on that same prefix.
    """
    from opendde_harness.providers.rates import is_plan_billed

    return is_plan_billed(model or "", config.providers)


def _auto_compaction(agent_loop: "AgentLoop | None", config: "Config", binding=None) -> bool:
    """Whether anything will compact this session before its window runs out.

    Two mechanisms, either of which is enough:

    * The backend does it. Only the Codex login compacts server-side, and only
      while ``context.server_compact_ratio`` is above zero -- at zero the
      trigger never fires. The session's bound provider answers for itself, and
      a provider that says it compacts nothing is believed: falling back to the
      configured default behind its answer reported compaction for a session
      that had switched away from it. The configured default's own model answers
      only when no binding exists to ask.
    * The context engine does it here. Its history selector fits the session to
      the window on every turn -- it drops the oldest complete exchanges rather
      than summarising them, but the effect the footer reports is the same one:
      the context does not run out. A running loop therefore always answers
      yes, which is why this half asks only whether there is one.

    False with no loop and no server-side provider: nothing is running that
    would compact anything.
    """
    from opendde_harness.cli._helpers import declared_compaction_provider

    if config.context.server_compact_ratio > 0:
        if binding is not None:
            server_side = getattr(binding.provider, "compaction_provider", "")
        else:
            server_side = declared_compaction_provider(config)
        if server_side:
            return True

    return getattr(agent_loop, "context_engine", None) is not None


async def _default_session_info(
    agent_loop: "AgentLoop | None",
    config: "Config",
    session_key: str | None = None,
    messages: "list[dict[str, Any]] | None" = None,
) -> dict[str, Any]:
    """Build the init bundle returned by ``session.create`` / ``session.resume``.

    ``agent_loop=None`` triggers graceful fallback (``tools={}``, ``skills={}``,
    zero usage, ``lazy=True``); version is always real (cached at module load).

    The model reported is the one this session runs on, not the configured
    default: a session that switched has its own, and reporting the default
    would show its user the wrong model. With no ``session_key`` (a session
    being created) the default is the right answer, because that is what a
    new session starts on.
    """
    from opendde_harness.providers import model_id

    model = config.agents.defaults.model
    if session_key and agent_loop is not None:
        model = agent_loop.session_model(session_key)
    # The one object that knows what this session actually runs on, including
    # the provider whose credential was resolved for it.
    binding = _session_binding(agent_loop, session_key)
    # Asked once, for the price the banner opens on and for the `(sub)` marker
    # beside it, which are two readings of one fact.
    plan_billed = _on_subscription(model, config)
    # Before the window is resolved, not after: the middle tier of the ladder is
    # the row the model service reports, and nothing has asked for it yet on a
    # session that is being resumed rather than continued.
    await _prime_window(binding)
    usage = await _baseline_usage(
        agent_loop, model, messages, binding, plan_billed=plan_billed, session_key=session_key
    )
    # What the user declared about this model: the row in
    # `providers.<id>.models`, which is the only thing that knows a fact no
    # catalogue carries.
    row = model_id.row_for(config.providers, model)
    info: dict[str, Any] = {
        "model": model,
        "model_id": model,
        # The provider this session's credential came from. The binding names
        # the one it was built on; without a binding the model id names it, and
        # nothing else does -- there is no configured provider field any more.
        "provider": getattr(binding, "provider_name", None) or model_id.provider_of(model),
        # The level this model thinks at: its own row, else the global default,
        # else the vendor's own (shown as nothing).
        "reasoning_effort": getattr(row, "reasoning_effort", None) or config.agents.defaults.reasoning_effort,
        "context_window": usage["context_max"],
        "lazy": agent_loop is None,
        "skills": _enumerate_skills(agent_loop),
        "tools": _enumerate_tools(agent_loop),
        "usage": usage,
        "version": _OPENDDE_HARNESS_VERSION,
        # When that version shipped, from the changelog; None when the install
        # carries no changelog to read.
        "release_date": _OPENDDE_HARNESS_RELEASE_DATE,
        "cwd": os.getcwd(),
        # The AGENTS.md / ODH.md files this session sends, so the status line
        # and `/memory` can say so without a second round trip. The list only:
        # their contents belong in the prompt, not in a UI payload.
        "project_instructions": instructions.wire_files(instructions.current_files(config, session_key or "")),
        "mcp_servers": _enumerate_mcp_servers(agent_loop),
        # What the footer needs to say `(sub)` and `(auto)` truthfully. Both
        # are facts about the configuration and the objects this session is
        # bound to, so they are answered here rather than guessed in the UI.
        "subscription": plan_billed,
        "auto_compact": _auto_compaction(agent_loop, config, binding),
    }

    # The status bar and the transcript say when PyPI has a newer release,
    # and which command gets it here. The cache is refreshed once a day from
    # the `ddeharness tui` entrypoint (see cli/tui_commands.py); a refresh
    # still in flight from this launch is given a moment, so the first launch
    # after a release is the one that says so.
    notice = update_notice(_OPENDDE_HARNESS_VERSION, wait=2.0)
    if notice is not None:
        info["update_available"], info["update_command"] = notice

    return info


def _get_or_build_manager(config: "Config") -> SessionManager:
    """Return a ``SessionManager`` for the configured workspace.

    Module-level so tests can monkeypatch it to inject a pre-populated manager
    without touching the filesystem (same seam as ``load_config``).
    """
    return SessionManager(config.workspace_path)


def _manager_for(agent_loop: "AgentLoop | None", config: "Config") -> SessionManager:
    """Prefer the loop's shared manager when available; fall back to a fresh one."""
    if agent_loop is not None:
        mgr = getattr(agent_loop, "sessions", None)
        if isinstance(mgr, SessionManager):
            return mgr
    return _get_or_build_manager(config)


def _interrupted_notes(session: "Session | None") -> dict[str, str]:
    """``{turn_id: note}`` for each turn the journal says did not finish.

    A turn with a ``turn.started`` record and no ending is interrupted, and a
    resumed transcript that shows its work without saying so invites the user to
    assume it completed. The note also counts the state-changing tools that were
    started and never reported back: those are the ones whose outcome nobody
    knows, and the harness will not re-run them.
    """
    if session is None:
        return {}
    interrupted = [turn for turn, status in session.turn_status().items() if status == "interrupted"]
    if not interrupted:
        return {}
    uncertain: dict[str, list[str]] = {}
    for record in session.uncertain_tool_calls():
        turn = record.get("turn_id")
        if isinstance(turn, str):
            uncertain.setdefault(turn, []).append(str(record.get("name", "a tool")))
    notes = {}
    for turn in interrupted:
        note = "[this turn was interrupted before finishing]"
        names = uncertain.get(turn)
        if names:
            listed = ", ".join(sorted(set(names)))
            note += f"\n[started and never reported back, so their outcome is unknown: {listed}]"
        notes[turn] = note
    return notes


def _map_to_wire(
    messages: list[dict[str, Any]],
    session_key: str,
    *,
    interrupted: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Map stored session messages to the GatewayTranscriptMessage wire shape.

    The TS side (``gatewayTypes.ts:23``) expects ``{role, text?, context?, name?}``.
    Stored messages carry pi's ``content`` blocks, so the text ones are joined
    and everything else — a picture, the reasoning, the calls a turn asked for —
    is left out: this is a transcript for a person, and the live path shows the
    same thing while the turn runs. All well-formed stored messages are included
    — no consolidation filter; non-dict or roleless entries are skipped with a
    warning so one corrupt line never bricks resume for the whole session.

    A ``toolResult`` record travels as ``role: "tool"`` with its ``toolName`` as
    ``name``. The wire's roles are what the TUI renders by
    (``ui-tui/src/resume.ts``) and there is nothing for it to show differently,
    so the shape it has always had is the shape it keeps.

    ``interrupted`` adds one entry after the last message of each turn the
    journal says did not finish. It uses the shape that already exists rather
    than a new field: an unrecognised ``role`` renders as a system line
    (``ui-tui/src/resume.ts``), which is what this is. Nothing on the wire
    changes shape, and a session stored before the journal existed has no turn
    ids and gets no markers.

    Untrusted fences are removed here, once, for every role. They are the
    harness telling a model which bytes are data (``security/trust.py``); to a
    person reading a transcript they are noise, and resume was showing them
    raw around every tool result. The live path never had them: the UI renders
    the ``result_preview`` the loop emits, which is built before the fence goes
    on. Stripping at this seam keeps the two paths showing the same thing.
    """
    marks = dict(interrupted or {})
    # Where each interrupted turn's last message sits, so the marker lands after
    # the work rather than in the middle of it.
    tail_index: dict[str, int] = {}
    for index, m in enumerate(messages):
        turn = m.get("turn_id") if isinstance(m, dict) else None
        if isinstance(turn, str) and turn in marks:
            tail_index[turn] = index
    last_of = {index: turn for turn, index in tail_index.items()}
    out = []
    for index, m in enumerate(messages):
        if not isinstance(m, dict) or "role" not in m:
            logger.warning("session.resume: skipping malformed stored message in {}", session_key)
            continue
        entry: dict[str, Any] = {"role": "tool" if msg.is_tool_result(m) else m["role"]}
        content = m.get("content", "")
        if isinstance(content, list):
            entry["text"] = " ".join(str(blk.get("text") or "") for blk in content if msg.is_text(blk))
        elif isinstance(content, str):
            entry["text"] = content
        elif content is not None:
            entry["text"] = str(content)
        if isinstance(entry.get("text"), str):
            entry["text"] = unwrap_untrusted(entry["text"])
        if m.get("toolName"):
            entry["name"] = m["toolName"]
        for extra_key in ("context", "name"):
            if extra_key in m:
                entry[extra_key] = m[extra_key]
        out.append(entry)
        turn = last_of.get(index)
        if turn is not None:
            out.append({"role": "system", "text": marks.pop(turn)})
    return out


async def session_create(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """``session.create`` — invoke factory (guarded) and build init bundle.

    Zero-factory invocation (``session_create({})``) is the test/demo path
    and degrades to ``agent_loop=None`` fallback bundle. Production wires
    ``agent_loop_factory`` via :func:`register_session_methods`.

    A fresh ``tui:<chat_id>`` key is minted on every call (lazy — no file
    written until the session's first save). An optional ``title`` param
    is accepted and ignored here; clients set titles via ``session.title``.
    """
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    session_id = f"tui:{new_chat_id()}"
    return {
        "session_id": session_id,
        "info": await _default_session_info(agent_loop, load_config()),
    }


async def session_info(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """``session.info`` — the init bundle again, for a session already open.

    The same facts ``session.create`` and ``session.resume`` return, asked for
    without opening or replacing anything. A live ``/model`` switch changes
    several of them — the model and its window, whether the new provider is
    billed by plan, whether anything will compact the context — and the client
    has nowhere else to learn the new answers.

    An absent ``session_id`` answers for a session about to be created, which
    is what the defaults describe.
    """
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    session_key = params.get("session_id") or None
    messages = None

    if session_key:
        # The same messages resume counts, so the two answers agree about how
        # much of the window this session is already holding.
        try:
            stored = _manager_for(agent_loop, config).peek(session_key)
            messages = stored.messages if stored is not None else None
        except Exception:
            logger.warning("session.info: could not read {} for its context size", session_key)

    return {"info": await _default_session_info(agent_loop, config, session_key, messages)}


async def session_close(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """``session.close`` — flush any unpersisted messages for the given session.

    With per-turn saves the session is normally already fully persisted.
    This handler handles the edge case where a message was added after the
    last save. An absent or unknown ``session_id`` param is silently ignored.
    """
    session_key = params.get("session_id")
    if not session_key:
        return {"ok": True}
    config = load_config()
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    mgr = _manager_for(agent_loop, config)
    try:
        mgr.flush(session_key)
    except Exception:
        logger.warning("session.close: failed to flush {}", session_key)
    return {"ok": True}


async def session_resume(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """``session.resume`` — load stored messages and return the resumed session key.

    Uses manager.peek() (consults cache first, then disk, without caching unknown
    keys). An unknown or absent session_id — or any load failure — falls back to
    a fresh-minted key with empty messages.

    Wire shape: raw session.messages so N stored → N wire (not get_history(),
    which slices and drops leading non-user messages).
    """
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    session_key = params.get("session_id")
    key = session_key if isinstance(session_key, str) else None

    if session_key:
        try:
            mgr = _manager_for(agent_loop, config)
            raw = mgr.peek(session_key)
            if raw is not None:
                # The stored messages first, so the banner's context figure
                # describes what this session is carrying rather than opening
                # a populated session at nothing.
                return {
                    "session_id": session_key,
                    "info": await _default_session_info(agent_loop, config, key, raw.messages),
                    "messages": _map_to_wire(
                        raw.messages,
                        session_key,
                        interrupted=_interrupted_notes(raw),
                    ),
                }
        except Exception:
            logger.exception(
                "session.resume: failed to load {}; falling back to fresh mint",
                session_key,
            )

    # Nothing to resume: a fresh key, and the defaults a new session starts on.
    return {
        "session_id": f"tui:{new_chat_id()}",
        "info": await _default_session_info(agent_loop, config, None),
        "messages": [],
    }


def _session_to_list_item(info: dict[str, Any]) -> dict[str, Any]:
    """Convert a list_sessions entry to the SessionListItem wire shape.

    The TS SessionListItem (gatewayTypes.ts:130) requires:
      id, message_count, preview, started_at (unix timestamp), title.
    started_at maps from created_at ISO string; preview is always empty in
    v0.1 — metadata carries no message content, and the TS picker falls back
    to title or "(untitled)".
    """
    key = info.get("key", "")
    created_at_str = info.get("created_at") or ""
    started_at: float = 0.0
    if created_at_str:
        try:
            started_at = datetime.fromisoformat(created_at_str).timestamp()
        except ValueError:
            pass
    meta = info.get("metadata") or {}
    title = meta.get("title") or ""
    return {
        "id": key,
        "message_count": info.get("message_count", 0),
        "preview": "",
        "source": "tui",
        "started_at": started_at,
        "title": title,
    }


async def session_list(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """``session.list`` — list tui-channel sessions sorted by updated_at desc.

    Filters to channel="tui" (this RPC scopes to the TUI surface). An optional
    positive integer ``limit`` slices after the sort (newest sessions win);
    zero, negative, or non-integer limits are ignored.
    Returns the SessionListResponse shape: {sessions: SessionListItem[]}.
    """
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    mgr = _manager_for(agent_loop, config)
    entries = mgr.list_sessions(channel="tui")
    limit = params.get("limit")
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        entries = entries[:limit]
    return {"sessions": [_session_to_list_item(e) for e in entries]}


async def session_delete(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """``session.delete`` — remove a session file and invalidate its cache entry.

    Returns {deleted: session_id} only when a file was actually removed;
    {deleted: null} otherwise (unknown id, missing param, or removal failure)
    so the UI can tell a typo from a real removal.
    """
    session_key = params.get("session_id", "")
    removed = False
    if session_key:
        agent_loop = _safe_invoke_factory(agent_loop_factory)
        config = load_config()
        mgr = _manager_for(agent_loop, config)
        removed = mgr.delete(session_key)
        # Cleared when the record is gone, not only when this call removed it:
        # a session that switched model before its first save has a binding
        # in memory and no file on disk, and ``delete`` answers False for
        # exactly that case. A record an unlink could not remove keeps its
        # binding, which its metadata still names.
        if agent_loop is not None and (removed or not mgr.exists(session_key)):
            agent_loop.clear_session_binding(session_key)
    return {"deleted": session_key if removed else None}


async def session_most_recent(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """``session.most_recent`` — return the most-recently-updated tui session key.

    Returns the SessionMostRecentResponse shape: {session_id?: string | null, ...}.
    The TS caller (createGatewayEventHandler.ts:242) reads r?.session_id; null
    is the tolerated no-sessions value.
    """
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    mgr = _manager_for(agent_loop, config)
    chat_id = mgr.find_most_recent_chat_id("tui")
    session_id = f"tui:{chat_id}" if chat_id else None
    return {"session_id": session_id}


async def session_title(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """``session.title`` — get or set the title of a session.

    Set path (``title`` param present): the title goes into the session's
    metadata via get_or_create. If the session file already exists on disk,
    it is persisted immediately (metadata-only save) and ``pending`` is
    False; for a never-saved lazy session the title stays in memory
    (``pending`` True — it lands with the session's first save, preserving
    the lazy mint).
    Get path: returns the current title from the cached or disk-loaded
    session.

    Wire shape per SessionTitleResponse (gatewayTypes.ts:154):
      {title?: string, session_key: string, pending: bool}
    """
    session_key = params.get("session_id", "")
    if not session_key:
        return {"title": None, "session_key": "", "pending": False}
    title = params.get("title")
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    mgr = _manager_for(agent_loop, config)

    if title is not None:
        session = mgr.get_or_create(session_key)
        session.set_title(title)
        if mgr.exists(session_key):
            try:
                mgr.save(session)
            except Exception:
                logger.warning("session.title: failed to persist title for {}", session_key)
                return {"title": title, "session_key": session_key, "pending": True}
            return {"title": title, "session_key": session_key, "pending": False}
        return {"title": title, "session_key": session_key, "pending": True}

    raw = mgr.peek(session_key)
    current_title = None
    if raw is not None:
        current_title = (raw.metadata or {}).get("title")
    return {"title": current_title, "session_key": session_key, "pending": False}


async def session_clear(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """``session.clear`` — wipe a session's messages in place, keeping its id.

    Unlike ``session.create`` (which mints a new id), clear preserves the
    session_key so scripts/bookmarks referencing it stay valid. Rejected
    while a turn is in flight (mutating history under a running writer races).
    """
    session_key = params.get("session_id", "")
    if not session_key:
        return {"session_id": "", "cleared": False}
    if turn_module.is_turn_active(session_key):
        raise TurnInProgressError(
            f"session {session_key!r} has an active turn; interrupt it before clearing",
            data={"session_key": session_key},
        )
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    mgr = _manager_for(agent_loop, config)
    session = mgr.get_or_create(session_key)
    session.clear()
    if mgr.exists(session_key):
        try:
            mgr.save(session)
        except Exception:
            logger.warning("session.clear: failed to persist cleared {}", session_key)
    return {"session_id": session_key, "cleared": True}


async def session_undo(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """``session.undo`` — drop the last ``n`` turns (default 1) in place.

    Turn boundaries derive from the role=="user" boundary. Rejected while a
    turn is in flight. ``n`` is reserved for forward-compat; the ui-tui
    ``/undo`` and ``/retry`` commands send no ``n`` (default 1).

    A successful undo also reports ``context_used``: what the conversation
    holds now that the exchange is gone. The client's last figure came from a
    model call that saw the longer conversation, and nothing else would tell it
    the window had emptied until the next turn reported -- so a status bar sat
    at a percentage of messages the session no longer has. Omitted when nothing
    was removed, because then nothing changed.

    Answered by the same policy ``session.info`` answers with, against this
    session's own binding: null where a compaction marker the backend replays
    stands in front of the history, and an offline estimate otherwise. Undoing
    an exchange is not a measurement, and it must not turn an unmeasured window
    into a measured one.
    """
    session_key = params.get("session_id", "")
    if not session_key:
        return {"removed": 0}
    if turn_module.is_turn_active(session_key):
        raise TurnInProgressError(
            f"session {session_key!r} has an active turn; interrupt it before undo",
            data={"session_key": session_key},
        )
    n = params.get("n", 1)
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    mgr = _manager_for(agent_loop, config)
    session = mgr.get_or_create(session_key)
    removed = session.undo_last_turn(n)
    if not removed:
        return {"removed": 0}
    if mgr.exists(session_key):
        try:
            mgr.save(session)
        except Exception:
            logger.warning("session.undo: failed to persist undo for {}", session_key)
    binding = _session_binding(agent_loop, session_key)

    return {"removed": removed, "context_used": await _occupancy(session.messages, binding)}


async def session_branch(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """``session.branch`` — fork the named session into a new diverging child.

    Forks ``session_id`` at its head (full-copy) via ``SessionManager.fork``
    and returns the ``SessionBranchResponse`` shape
    ``{session_id, title, message_count}`` the TUI consumes (it switches ``sid``
    to the returned ``session_id`` and reports ``message_count`` carried). The
    optional ``name`` param becomes the child title when non-empty. An unknown
    or empty (zero-message) source yields ``session_id=None`` so the TUI guard
    treats it as a no-op.
    """
    session_key = params.get("session_id", "")
    if not session_key:
        return {"session_id": None, "title": None}
    name = params.get("name")
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    mgr = _manager_for(agent_loop, config)
    child = mgr.fork(session_key, title=(name or None))
    if child is None:
        return {"session_id": None, "title": None}
    if agent_loop is not None and agent_loop.has_session_binding(session_key):
        # A fork continues its parent's conversation, so it continues on the
        # parent's model; without this it would silently drop to the default.
        agent_loop.set_session_binding(child.key, agent_loop.binding_for_session(session_key))
    return {
        "session_id": child.key,
        "title": child.metadata.get("title"),
        "message_count": len(child.messages),
    }


async def session_export(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """``session.export`` — render a session transcript to a Markdown file.

    Read-only: unlike clear/undo there is no busy-guard. ``session_id`` is
    resolved via the shared cross-channel core; an unresolved value is reported
    as not-found and an ambiguous one returns the candidate keys — neither
    writes a file. On success the rendered Markdown lands at
    ``<workspace>/exports/<sid>.md`` and the absolute path is returned.
    """
    value = params.get("session_id", "")
    if not value:
        return {"exported": False, "path": None, "reason": "not_found"}
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    mgr = _manager_for(agent_loop, config)
    res = mgr.resolve_key(value)
    if res.status == "ambiguous":
        return {
            "exported": False,
            "path": None,
            "reason": "ambiguous",
            "candidates": list(res.candidates),
        }
    session = mgr.peek(res.key) if res.status == "resolved" else None
    if session is None:
        return {"exported": False, "path": None, "reason": "not_found"}
    dest = default_export_path(config.workspace_path, res.key)
    try:
        written = write_transcript(session, dest)
    except OSError:
        logger.warning("session.export: failed to write export for {}", res.key)
        return {"exported": False, "path": None, "reason": "write_failed"}
    return {"exported": True, "path": str(written)}


def register_session_methods(
    dispatcher: "Dispatcher",
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> None:
    """Register the 11 session handlers on a dispatcher.

    Mirrors :func:`opendde_harness.tui_rpc.methods.turn.register_turn_methods` —
    wraps the module-level handlers in single-argument closures that pre-bind
    ``agent_loop_factory``, satisfying the dispatcher's ``params -> dict``
    contract.
    """

    async def _create(params: dict) -> dict:
        return await session_create(params, agent_loop_factory=agent_loop_factory)

    async def _close(params: dict) -> dict:
        return await session_close(params, agent_loop_factory=agent_loop_factory)

    async def _resume(params: dict) -> dict:
        return await session_resume(params, agent_loop_factory=agent_loop_factory)

    async def _list(params: dict) -> dict:
        return await session_list(params, agent_loop_factory=agent_loop_factory)

    async def _delete(params: dict) -> dict:
        return await session_delete(params, agent_loop_factory=agent_loop_factory)

    async def _info(params: dict) -> dict:
        return await session_info(params, agent_loop_factory=agent_loop_factory)

    async def _most_recent(params: dict) -> dict:
        return await session_most_recent(params, agent_loop_factory=agent_loop_factory)

    async def _title(params: dict) -> dict:
        return await session_title(params, agent_loop_factory=agent_loop_factory)

    async def _clear(params: dict) -> dict:
        return await session_clear(params, agent_loop_factory=agent_loop_factory)

    async def _undo(params: dict) -> dict:
        return await session_undo(params, agent_loop_factory=agent_loop_factory)

    async def _branch(params: dict) -> dict:
        return await session_branch(params, agent_loop_factory=agent_loop_factory)

    async def _export(params: dict) -> dict:
        return await session_export(params, agent_loop_factory=agent_loop_factory)

    dispatcher.register("session.create", _create)
    dispatcher.register("session.close", _close)
    dispatcher.register("session.resume", _resume)
    dispatcher.register("session.list", _list)
    dispatcher.register("session.delete", _delete)
    dispatcher.register("session.info", _info)
    dispatcher.register("session.most_recent", _most_recent)
    dispatcher.register("session.title", _title)
    dispatcher.register("session.clear", _clear)
    dispatcher.register("session.undo", _undo)
    dispatcher.register("session.branch", _branch)
    dispatcher.register("session.export", _export)


__all__ = [
    "AgentLoopFactory",
    "session_create",
    "session_close",
    "session_info",
    "session_resume",
    "session_list",
    "session_delete",
    "session_most_recent",
    "session_title",
    "session_clear",
    "session_undo",
    "session_branch",
    "session_export",
    "register_session_methods",
]
