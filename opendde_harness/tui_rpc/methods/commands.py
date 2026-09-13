"""``commands.catalog`` RPC handler — dynamic Typer-reflected slash catalog.

Contract source: ``docs/openspec/changes/harness-command-catalog-dynamic/`` —
proposal.md §2.1 ① + design.md §D1-D2 + specs/tui-ipc.md CAP-CAT-1.

Replaces the v0.0.2 ``_stubs.py`` ``-32012`` placeholder.

Algorithm summary (design.md §D1):

1. Reflect ``opendde_harness.cli.commands.app`` (Typer 0.20+):
   - ``app.registered_commands`` → top-level commands (``CommandInfo``; name
     may be ``None``, fall back to ``callback.__name__`` with ``_`` → ``-``)
   - ``app.registered_groups`` → ``TyperInfo`` entries each with
     ``.name`` (group) + ``.typer_instance.registered_commands`` (children)
   - Groups whose body uses ``@callback(invoke_without_command=True)`` and
     have no subcommands (e.g. ``tui``) are treated as a single group-level
     command tuple ``(group_name,)``.
2. Filter: drop entries matching ``_DISPATCH_BLACKLIST`` (shared with
   ``cli_dispatch.py`` — single source of truth, per design.md §D4.4) plus
   the special-case ``argv == ("agent",)`` (REPL mode).
3. Build response (design.md §D2):
   - ``canon`` dict — ``"/<canonical>" → "/<canonical>"`` (alias = canonical
     1:1 in v0.1; TS-side prefix-1-match in ``createSlashHandler.ts`` handles
     short aliases).
   - ``pairs`` list — same data as ``[(alias, canonical)]`` tuples; TS gates
     on non-empty pairs.
   - ``sub`` dict — ``{group: [subcommand]}``.
   - ``categories`` — ``"(top-level)"`` first then alphabetical groups.
   - ``skill_count`` — SQL count via ``skill_forge.store``; ``0`` + ``warning``
     on failure.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import typer
from loguru import logger

from opendde_harness.tui_rpc.methods._typer_reflect import (
    argument_hint as _argument_hint,
)
from opendde_harness.tui_rpc.methods._typer_reflect import (
    resolve_name as _resolve_name,
)

# Single source of truth for the blacklist — see ``cli_dispatch.py`` header
# for rationale (design.md §D4.4 — one set read by both ``cli.dispatch``
# rejection and ``commands.catalog`` exclusion so the two can never drift).
from opendde_harness.tui_rpc.methods.cli_dispatch import (
    _DISPATCH_BLACKLIST,
    _is_dispatch_compatible,
    dispatch_timeout,
)

if TYPE_CHECKING:
    from opendde_harness.tui_rpc.dispatcher import Dispatcher


# Synthetic category name for top-level commands (no group). Goes first in
# ``categories`` so the hermes UI renders it as the primary cluster.
_TOP_LEVEL_CATEGORY = "(top-level)"

#: Where one sentence ends and the next begins: a full stop, then a space,
#: then a capital. Not every ". ", which cut "(e.g. cli:<id>)" in half.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

#: A completion row is one line on a narrow terminal.
_HELP_MAX = 120


@dataclass(frozen=True)
class _CatalogEntry:
    """Internal reflection record for one CLI command, pre-filter."""

    argv: tuple[str, ...]
    help_text: str
    kind: str  # "top" | "sub" | "group-callback"
    hint: str = ""


def _one_line(raw: str) -> str:
    """One sentence of prose from a help string or a docstring.

    The first paragraph, unwrapped, up to its first full stop: a docstring is
    wrapped for source, and taking its first physical line cut descriptions
    mid-phrase in the popup ("Add a skill to skillForge.blocklist (dropped
    from the injection"). RST's double backticks go: they are markup for
    documentation, and a completion list shows them as the four characters
    they are. Capped, because a popup row is one line.
    """
    paragraph = raw.strip().split("\n\n", 1)[0]
    text = " ".join(paragraph.split()).replace("``", "")

    if not text:
        return ""

    # A sentence ends where the next one begins, which is a capital letter --
    # not at every ". ", or "(e.g. `cli:<id>`)" ended the description.
    sentence = _SENTENCE_SPLIT.split(text, maxsplit=1)[0]

    if len(sentence) <= _HELP_MAX:
        return sentence

    # On a word, with something to say it was cut.
    cut = sentence[:_HELP_MAX].rsplit(" ", 1)[0]

    return f"{cut}…"


def _extract_help(ci: typer.models.CommandInfo) -> str:
    """``help=`` arg → callback docstring → empty string, as one line."""
    return _one_line(ci.help or (ci.callback.__doc__ if ci.callback else None) or "")


def _extract_group_callback_help(sub_app: typer.Typer) -> str:
    """Help text for a group whose body is a single ``@callback`` (no subcommands)."""
    cb = sub_app.registered_callback
    if cb is None or cb.callback is None:
        return ""
    return _one_line(cb.help or (cb.callback.__doc__ or ""))


def _reflect_app(app: typer.Typer) -> list[_CatalogEntry]:
    """Walk a Typer app and return one entry per reflected command (pre-filter).

    Algorithm:
    - Top-level: ``app.registered_commands`` (skip ``hidden`` + unresolvable names)
    - Subgroups: ``app.registered_groups`` (each a ``TyperInfo`` with ``.name`` +
      ``.typer_instance``). For each subgroup:
        - If it has subcommands: emit one entry per subcommand.
        - Else if its body uses ``@callback(invoke_without_command=True)``:
          emit a single ``(group_name,)`` entry (this lets ``tui`` show up
          for blacklist filtering).

    Hidden commands (Typer ``CommandInfo.hidden=True``) are filtered here so
    the catalog mirrors the user-visible ``--help`` surface.
    """
    entries: list[_CatalogEntry] = []

    # Top-level @app.command()
    for ci in app.registered_commands:
        if getattr(ci, "hidden", False):
            continue
        name = _resolve_name(ci)
        if not name:
            continue
        entries.append(_CatalogEntry(argv=(name,), help_text=_extract_help(ci), kind="top", hint=_argument_hint(ci)))

    # Subgroups (app.add_typer(subapp, name=...))
    for ti in app.registered_groups:
        group = ti.name
        sub_app = ti.typer_instance
        if not group or sub_app is None:
            continue
        if not sub_app.registered_commands:
            # Group body itself is the dispatch surface (e.g. ``tui`` uses
            # ``@callback(invoke_without_command=True)``). Emit a single
            # group-level entry so the blacklist filter can hard-reject it.
            if sub_app.info.invoke_without_command:
                entries.append(
                    _CatalogEntry(
                        argv=(group,),
                        help_text=_extract_group_callback_help(sub_app),
                        kind="group-callback",
                    )
                )
            continue
        for ci in sub_app.registered_commands:
            if getattr(ci, "hidden", False):
                continue
            sub_name = _resolve_name(ci)
            if not sub_name:
                continue
            entries.append(
                _CatalogEntry(
                    argv=(group, sub_name),
                    help_text=_extract_help(ci),
                    kind="sub",
                    hint=_argument_hint(ci),
                )
            )

    return entries


def _compute_skill_count() -> tuple[int, str | None]:
    """Return ``(skill_count, warning)`` for the catalog response.

    Phase B-2: reads :class:`SkillRegistry.list_all()` count instead of
    the deleted ``SqliteStore``. The SQLite-backed mass-library mirror
    + agent_cases store have been removed; skill counting now reflects
    the file-based local pool (``<workspace>/skills/`` + builtin) that
    the new :class:`LocalSkillSource` indexes.

    Failure modes (any exception → ``(0, warning_str)``):
    - ``load_config`` raises (missing ``~/.opendde_harness/config.json``, etc.)
    - workspace path unreachable

    Returning ``0`` + warning is the graceful-degrade path: TS-side surfaces
    ``warning`` via the activity strip so the user sees a clear status
    rather than a stalled catalog.
    """
    try:
        from opendde_harness.config.loader import load_config
        from opendde_harness.memory_engine.skill_local.registry import SkillRegistry

        config = load_config()
        workspace = config.workspace_path
        registry = SkillRegistry(workspace=workspace)
        return (len(registry.list_all()), None)
    except Exception as exc:  # noqa: BLE001 — defensive: any failure → fallback
        logger.warning("commands.catalog: skill_count fallback: {!r}", exc)
        return (0, f"skill_count unavailable: {type(exc).__name__}: {exc}")


def _matches_blacklist(argv: tuple[str, ...]) -> bool:
    """True iff ``argv`` is prefixed by any entry in ``_DISPATCH_BLACKLIST``."""
    for prefix in _DISPATCH_BLACKLIST:
        plen = len(prefix)
        if len(argv) >= plen and argv[:plen] == prefix:
            return True
    return False


def _filter_for_catalog(entries: list[_CatalogEntry]) -> list[_CatalogEntry]:
    """Keep exactly what ``cli.dispatch`` will accept.

    One rule, and it is the dispatcher's own: an entry belongs in the catalog
    if ``cli.dispatch`` would run it. That drops the blacklist (interactive
    logins, long-running servers, the recursive ``tui`` and ``onboard``
    wizards) and it drops a bare group head such as ``provider``, which is not
    a command at all -- Typer would print the group's help and the dispatcher
    answers ``not_dispatch_compatible``.

    Measured before this: the catalog offered ten heads, five of which were
    groups. Every one of those five failed the moment it was run, and none of
    the ten carried a description. The subcommands the groups hold are what
    the user wants and what dispatch accepts, so they are what is served.
    """
    return [e for e in entries if _is_dispatch_compatible(list(e.argv))]


def _build_response(entries: list[_CatalogEntry], skill_count: int = 0, warning: str | None = None) -> dict[str, Any]:
    """Project filtered entries into ``CommandsCatalogResponse`` shape.

    - ``pairs`` — one ``(/<head>, /<head>)`` per top-level command or group
      head, in reflection order (alias = canonical 1:1)
    - ``categories`` — ``(top-level)`` first then alphabetical group names
      derived from observed argv prefixes (so a fully-filtered group disappears)

    Heads only: ``slash.exec`` receives the user's ``cmd.slice(1)`` (e.g.
    "skill list") verbatim and cli.dispatch reflection resolves the
    multi-token argv, so the catalog never needs the subcommand surface.
    """
    pairs: list[tuple[str, str]] = []
    seen_groups: set[str] = set()
    has_top_level = False

    def _emit(head: str) -> None:
        canonical = f"/{head}"
        if (canonical, canonical) not in pairs:
            pairs.append((canonical, canonical))

    for entry in entries:
        argv = entry.argv
        if len(argv) == 1:
            if entry.kind == "group-callback":
                # Reflection emitted a bare group head (e.g. a Typer subgroup
                # using ``@callback(invoke_without_command=True)`` with no
                # subcommands). After filtering, the only EC instance
                # (``tui``) is gone, so this branch fires for test fakes.
                # Same treatment as a group with subcommands.
                seen_groups.add(argv[0])
                _emit(argv[0])
            else:
                has_top_level = True
                _emit(argv[0])
        elif len(argv) >= 2:
            seen_groups.add(argv[0])
            _emit(argv[0])

    categories: list[str] = []
    if has_top_level:
        categories.append(_TOP_LEVEL_CATEGORY)
    categories.extend(sorted(seen_groups))

    response: dict[str, Any] = {
        "pairs": pairs,
        # What each of them is, and what it takes. The heads above stay for the
        # client that reads only them; a client that shows a description reads
        # this, and every entry here is one ``cli.dispatch`` accepts.
        "commands": [
            {
                "name": " ".join(entry.argv),
                "description": entry.help_text,
                **({"argument_hint": entry.hint} if entry.hint else {}),
                # Only where it is not the ordinary one: a client reads this to
                # know a command may take a while, and the gateway applies the
                # same figure itself so an old client cannot shorten it.
                **({"timeout_s": seconds} if (seconds := dispatch_timeout(entry.argv)) else {}),
            }
            for entry in entries
        ],
        "categories": categories,
        "skill_count": skill_count,
    }
    if warning is not None:
        response["warning"] = warning
    return response


async def commands_catalog(params: dict[str, Any]) -> dict[str, Any]:
    """Reflect ``opendde_harness.cli.commands.app`` and return a catalog response.

    Lazy-imports the CLI module so an import failure (rare) degrades to an
    empty catalog with a ``warning`` rather than raising — this matches the
    TS-side ``createGatewayEventHandler.ts:198`` graceful-degrade contract.
    """
    try:
        import opendde_harness.cli.commands as ec_commands

        app = ec_commands.app
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.warning("commands.catalog: failed to import ec.cli.commands: {!r}", exc)
        return _build_response(
            entries=[],
            skill_count=0,
            warning=f"cli commands not loadable: {type(exc).__name__}",
        )

    entries = _reflect_app(app)
    filtered = _filter_for_catalog(entries)
    # _compute_skill_count opens a SQLite connection + runs schema init on
    # first call — both blocking I/O. Offload to a worker thread so the
    # JSON-RPC server's event loop keeps servicing other in-flight RPCs
    # (cli.dispatch uses the same pattern via asyncio.to_thread for
    # _invoke_ec_cli — see cli_dispatch.py).
    skill_count, warning = await asyncio.to_thread(_compute_skill_count)
    return _build_response(entries=filtered, skill_count=skill_count, warning=warning)


def register_commands_methods(dispatcher: "Dispatcher") -> None:
    """Register ``commands.catalog`` on a dispatcher instance."""
    dispatcher.register("commands.catalog", commands_catalog)


__all__ = [
    "commands_catalog",
    "register_commands_methods",
]
