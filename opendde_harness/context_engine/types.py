"""Context engine data carriers.

The two value objects one turn's assembly produces and consumes:

- :class:`AssembledContext` — the message list + metadata handed to
  AgentLoop for the LLM call.
- :class:`TokenBudget` — the per-turn budget breakdown, computed from the
  prefix this turn actually renders.

They used to live in ``memory_engine.base`` for historical reasons. They
are the context engine's own contract, so they live beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AssembledContext:
    """Output of :meth:`ContextAssembler.assemble`.

    The agent's LLM call uses exactly these messages. Nothing else from
    session history reaches the model directly.
    """

    messages: list[dict[str, Any]]
    include_indices: list[int] | None = None  # Which session msg indices survived
    metadata: dict[str, Any] = field(default_factory=dict)  # Debug / telemetry


@dataclass(frozen=True)
class TokenBudget:
    """Token budget breakdown for one turn.

    ``context_length`` and ``available_history`` are ``None`` when the model's
    window is unknown -- no table carries it and the user declared none. Every
    consumer checks for that explicitly rather than computing against a stand-in:
    the selector does not trim and :attr:`threshold` reports no trigger. A
    number in their place used to be 65,536 for every such model, which trimmed
    and compacted real windows at a fraction of their size.

    ``reserved_system`` is the cost of the system message the request actually
    carries, not of a second prompt rendered to estimate it.
    """

    context_length: int | None  # Model's context window; None when unknown
    reserved_output: int  # Reserved for completion
    reserved_tools: int  # Tool schemas in prompt
    reserved_system: int  # System prompt, as sent
    available_history: int | None  # What's left for session history; None when unknown

    @property
    def known(self) -> bool:
        return self.available_history is not None


__all__ = [
    "AssembledContext",
    "TokenBudget",
]
