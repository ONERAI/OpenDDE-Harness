"""TokenWise core abstractions — colocated with implementations.

Migrated from ``opendde_harness/core/interfaces.py``. The colocate-with-implementation
rule means strategies live next to the ABC they implement.

Strategies are additive — multiple can be installed. The agent calls each
hook in registration order. A strategy that is not interested in a given
hook inherits the default no-op.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass
class UsageSnapshot:
    """Token usage and cost for a single LLM call.

    Convention: ``input_tokens`` is *fresh* (non-cached) prompt tokens, which
    is what pi reports as ``input`` on every route (it subtracts the cache
    counts a vendor folds into its prompt total before handing the figure
    over), and what ``usage_dict`` passes through as ``prompt_tokens``.
    """

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    # None when the call has no per-token price to state: a plan-billed provider
    # is paid for by subscription, so a number here would be invented. Distinct
    # from 0.0, which means "priced, and it cost nothing".
    estimated_cost_usd: float | None = None
    session_key: str | None = None
    # Whether the provider stated a cache figure for this call at all. A zero
    # with this False is a call nobody counted, not a miss; a relay that
    # forwards no cache fields leaves every call this way.
    cache_reported: bool = False
    # What the call's tokens are worth at the vendor's list price, priced when
    # the call is recorded (pi prices each message as it lands) so a long
    # request pays its own long-context tier. None when no list price is known.
    list_cost_usd: float | None = None


#: The session the current turn belongs to, set by the loop when a turn
#: starts. Read by the tracker for calls made on the turn's behalf outside the
#: loop's own iteration -- a server-side compaction, say -- which see the
#: provider but not the session.
CURRENT_SESSION_KEY: ContextVar[str | None] = ContextVar("opendde_current_session_key", default=None)


def snapshot_from_usage(usage: dict[str, Any] | None, model: str, session_key: str | None) -> UsageSnapshot:
    """One call's usage as a snapshot.

    ``prompt_tokens`` is fresh-only: every usage comes through pi, which
    already takes the cache counts out of a vendor's prompt total, so nothing
    is subtracted here -- a call whose fresh tokens outnumber its cached ones
    (a large tool result on top of a reused prefix) would otherwise be counted
    short by the cached amount. The cache keys are kept whenever the vendor
    stated a figure, zero included, so a key's presence is the report itself.
    """
    usage = usage or {}
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
    cache_write = int(usage.get("cache_creation_input_tokens", 0) or 0)
    return UsageSnapshot(
        model=model,
        input_tokens=int(usage.get("prompt_tokens", 0) or 0),
        output_tokens=int(usage.get("completion_tokens", 0) or 0),
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        session_key=session_key or None,
        cache_reported="cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage,
    )


class TokenStrategy(ABC):
    """Cross-cutting hooks for token and cost optimization.

    Strategies are additive — multiple can be installed. The agent calls each
    hook in registration order. A strategy that is not interested in a given
    hook inherits the default no-op.

    This is a single unified interface rather than four tiny ABCs to keep the
    install point simple. Concrete strategies will typically implement just
    one or two hooks.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (e.g. 'usage_tracker', 'tool_search')."""

    async def before_llm_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, str]:
        """Pre-process the outgoing request. Return (messages, tools, model).

        Used by ToolSearchStrategy (withholds tool schemas). Default: pass
        through.
        """
        return messages, tools, model

    async def after_llm_call(
        self,
        response: dict[str, Any],
        usage: UsageSnapshot,
    ) -> None:
        """Post-call hook. Used by UsageTracker, BudgetAlerter. Default: no-op."""


__all__ = ["TokenStrategy", "UsageSnapshot"]
