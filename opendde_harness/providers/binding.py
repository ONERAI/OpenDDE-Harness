"""The model a piece of work is running on, scoped to that work.

A model id and the credential that serves it are one pair, so they travel
together as a :class:`ModelBinding` rather than as two attributes someone
can update by halves.

The binding is held in a :class:`~contextvars.ContextVar` rather than
passed down, for two reasons. The turn path reads it in three packages --
``opendde_harness.agent`` (the loop and the subagent manager),
``opendde_harness.context_engine`` (the curator and the history trimmer)
and ``opendde_harness.memory_engine`` (the consolidator) -- and threading
a parameter through all of them would
touch far more code than it explains. More importantly,
``asyncio.create_task`` copies the current context, so work detached during
a turn (a subagent, a consolidation task) keeps the binding it was started
under for its whole life, which is exactly the semantics those tasks need:
a subagent spawned before a model switch must not finish on the model
chosen after it.

Nothing here builds providers. :mod:`opendde_harness.providers.pool` does
that, and :class:`~opendde_harness.agent.loop.main.AgentLoop` decides which
binding a turn runs under.

The turn's context window travels beside the binding rather than on it.
It is a fact about the model, but one the loop resolves through its own
ladder (the model's overlay, then what the model layer reports for the model --
see ``AgentLoop.resolve_window``) and re-resolves for whichever model answered.
So the loop enters both at the turn boundary, and the subsystems that size
themselves against the window read :func:`active_window` the way they read
:func:`active_binding`.

Ported from Raven's ``providers/binding.py`` (#284).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opendde_harness.providers.base import LLMProvider


@dataclass(frozen=True)
class ModelBinding:
    """A model id and the provider whose credential serves it.

    ``provider_name`` is the route that was resolved -- the section whose
    credential ``provider`` was built from. Kept on the pair because it cannot
    be reconstructed from the id: a model served through a gateway names
    another vendor, and re-running the routing rules later can answer
    differently once a credential has been added.
    """

    provider: "LLMProvider"
    model: str
    provider_name: str | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("a ModelBinding needs a model id")


_ACTIVE: ContextVar["ModelBinding | None"] = ContextVar("opendde_active_model_binding", default=None)
#: The running turn's window: None is unknown ("do not trim"), which is why
#: ``active_window`` checks ``_ACTIVE`` rather than this value to tell "no
#: turn" from "unknown".
_WINDOW: ContextVar[int | None] = ContextVar("opendde_active_context_window", default=None)


def active_binding() -> "ModelBinding | None":
    """The binding the current turn runs under, or None outside a turn.

    Callers outside a turn (startup, a CLI one-shot, a test) get None and
    should fall back to whatever default they were built with.
    """
    return _ACTIVE.get()


@contextmanager
def use_binding(binding: "ModelBinding", *, window: int | None = None) -> Iterator["ModelBinding"]:
    """Run a block -- and everything it awaits or spawns -- on one binding.

    ``window`` is how many tokens the bound model holds, or None when no table
    lists it; the loop resolves it, this only carries it.
    """
    token = _ACTIVE.set(binding)
    window_token = _WINDOW.set(window)
    try:
        yield binding
    finally:
        _WINDOW.reset(window_token)
        _ACTIVE.reset(token)


def set_active_window(window: int | None) -> None:
    """Correct the running turn's window in place.

    For the loop, once a call has answered: the model that answered is not
    always the one the turn entered under, and its window should size the rest
    of the turn. Outside a turn there is nothing to correct.
    """
    if _ACTIVE.get() is not None:
        _WINDOW.set(window)


def active_window(fallback: int | None) -> int | None:
    """How much the running turn's model can hold, or ``fallback`` outside one.

    The counterpart of :func:`active_binding` for the subsystems that size
    themselves against the window -- the history trimmer, the curator's
    assembler and the consolidator. Each is built once and outlives any number
    of turns, so a window copied at construction answers for the session that
    happened to build it and for no other.
    """
    return _WINDOW.get() if _ACTIVE.get() is not None else fallback


def resolve(pin: "ModelBinding | None", fallback: "ModelBinding") -> "ModelBinding":
    """Which binding should a subsystem use for this call?

    Precedence is the configured rule: a subsystem with a model of its own,
    paired with credentials of its own, uses that; otherwise it follows the
    model of the turn it is running under; outside a turn it uses the fallback
    it was built with.

    ``pin`` is already a pair -- a bare pinned model id never reaches here,
    because a model id without a credential is what mis-pairs one vendor's key
    with another's endpoint.
    """
    if pin is not None:
        return pin
    return _ACTIVE.get() or fallback
