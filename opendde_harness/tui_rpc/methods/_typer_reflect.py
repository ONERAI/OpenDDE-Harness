"""Shared Typer-reflection helpers for ``commands.catalog`` + ``cli.dispatch``.

Both ``methods/commands.py`` (catalog handler) and
``methods/cli_dispatch.py`` (dispatch-compat check) need to enumerate the
visible commands registered on a Typer app. Keeping the helpers here
avoids:

1. Drift between the two modules' name-resolution rules (the reviewer
   pre-merge flagged the previous duplication as L1).
2. A circular import — ``commands.py`` imports ``_DISPATCH_BLACKLIST``
   from ``cli_dispatch.py``, so the reverse direction is blocked at
   module load.

Reflection contract (per harness-command-catalog-dynamic design.md §D1):

- ``CommandInfo.name`` is preferred when set; otherwise resolve via
  ``callback.__name__`` with ``_`` → ``-`` to match Typer's CLI surface.
- ``CommandInfo.hidden`` is honored — ``hidden=True`` commands do not
  show up in ``--help`` output and must not show up in the slash
  catalog either.
"""

from __future__ import annotations

import inspect

import typer


def hint_for(app: typer.Typer, argv: "list[str] | tuple[str, ...]") -> str:
    """The argument hint for the command ``argv`` names, or "" for no such command.

    The same walk ``commands.catalog`` does, for the one caller that has an
    argv rather than a reflected entry: the message a usage error turns into.
    """
    if not argv:
        return ""

    head, *rest = argv

    for ci in app.registered_commands:
        if not getattr(ci, "hidden", False) and resolve_name(ci) == head:
            return argument_hint(ci)

    for ti in app.registered_groups:
        if ti.name != head or ti.typer_instance is None or not rest:
            continue
        for ci in ti.typer_instance.registered_commands:
            if not getattr(ci, "hidden", False) and resolve_name(ci) == rest[0]:
                return argument_hint(ci)

    return ""


def argument_hint(ci: typer.models.CommandInfo) -> str:
    """The positional arguments a command takes, as a slash popup shows them.

    ``<file>`` for a positional that is required, ``[file]`` for one that is
    not, in declaration order, and then any option the command cannot run
    without, spelled as it is typed: ``--config <config>``. Empty for a command
    that takes nothing, which is most of them.

    Options that have a default are left out. They are the long tail, a hint is
    what the user has to type next, and a row of a completion popup is one
    line. A required option is not optional whatever its name says: without
    ``--config`` the command answers "Missing option".

    Read from the callback's signature, where Typer leaves its own markers as
    parameter defaults, so it stays true to the command rather than to a
    description somebody wrote once.
    """
    callback = ci.callback

    if callback is None:
        return ""

    positional: list[str] = []
    options: list[str] = []

    for name, parameter in inspect.signature(callback).parameters.items():
        default = parameter.default
        spelled = name.replace("_", "-")
        # Typer spells "no default, so it must be given" as ``...``; anything
        # else, None included, is a value the command can run without.
        required = getattr(default, "default", None) is ...

        if isinstance(default, typer.models.ArgumentInfo):
            positional.append(f"<{spelled}>" if required else f"[{spelled}]")
        elif isinstance(default, typer.models.OptionInfo) and required:
            flag = next((decl for decl in default.param_decls or () if decl.startswith("--")), f"--{spelled}")
            options.append(f"{flag} <{spelled}>")

    return " ".join([*positional, *options])


def resolve_name(ci: typer.models.CommandInfo) -> str | None:
    """Return the canonical (hyphenated) name for a Typer command, or None.

    Typer canonicalises ``@command("foo-bar")`` (explicit) directly. For
    ``@command()`` on a function ``foo_bar`` (implicit), ``ci.name`` is
    ``None`` and the user-facing surface is ``foo-bar`` — we mirror that
    here.
    """
    if ci.name:
        return ci.name
    if not ci.callback:
        return None
    return ci.callback.__name__.replace("_", "-")


def collect_command_names(typer_obj: typer.Typer) -> set[str]:
    """Return all visible (non-hidden) command names registered on a Typer app.

    Works uniformly for the root app and any subgroup — both expose
    ``registered_commands`` of the same shape.
    """
    names: set[str] = set()
    for ci in typer_obj.registered_commands:
        if getattr(ci, "hidden", False):
            continue
        name = resolve_name(ci)
        if name is not None:
            names.add(name)
    return names


__all__ = ["resolve_name", "collect_command_names"]
