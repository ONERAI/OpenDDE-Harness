"""Turn-scoped authority to ask the person at the keyboard before acting.

One prompt mechanism, shared by every tool that needs it. The shell tool has
always used it for a destructive command; the write and edit tools use it for a
file whose name makes it standing instructions, which is a different kind of
irreversible: nothing is destroyed, but the agent has changed what it will be
told to do on the next turn.

The capability is bound per turn and per origin. A background turn shares this
process with an interactive one and must still fail closed: no responder means
"cannot ask", never "no need to ask".
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Protocol


@dataclass(frozen=True)
class ApprovalDecision:
    """One approval outcome, with the reason the broker recorded for it.

    ``reason`` is the broker's own close reason -- ``allow``, ``deny``,
    ``timeout``, ``cancelled`` or ``error``. A tool result has to tell those
    apart: someone who denied an action knows what they did, and someone whose
    prompt expired while they were away does not.
    """

    approved: bool
    reason: str


class ApprovalResponder(Protocol):
    """Turn-scoped capability that can approve one exact action."""

    async def await_approval(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        tool_call_id: str,
        command: str,
        description: str,
    ) -> ApprovalDecision: ...


@dataclass(frozen=True)
class ApprovalTurn:
    """Approval state isolated by async context for one agent turn.

    ``denied_digests`` suppresses duplicate prompts only within this turn; a
    later user turn receives a fresh decision boundary.
    """

    responder: ApprovalResponder | None = None
    conversation_id: str = ""
    turn_id: str = ""
    tool_call_id: str = ""
    denied_digests: frozenset[str] = frozenset()


#: Reasons :meth:`ApprovalGate.ask` can refuse with, beyond the broker's own
#: ``deny`` / ``timeout`` / ``cancelled`` / ``error``. The wording belongs to
#: each tool: "this command" and "this write" are not the same sentence, and a
#: shared phrasing would have to be vague enough to fit both.
ALREADY_REFUSED = "already"
NOT_INTERACTIVE = "unavailable"


class ApprovalGate:
    """One tool's per-turn approval state and the ask itself.

    Each tool owns a gate rather than sharing a module-level one, so a tool
    constructed for a test carries no state from another test, and so two tools
    cannot consume each other's per-turn refusals.
    """

    def __init__(self, name: str) -> None:
        self._turn: ContextVar[ApprovalTurn] = ContextVar(name, default=ApprovalTurn())

    @property
    def turn(self) -> ApprovalTurn:
        return self._turn.get()

    def start_approval_turn(
        self,
        responder: ApprovalResponder | None,
        *,
        conversation_id: str,
        turn_id: str,
    ) -> None:
        """Bind or revoke interactive approval capability for the current turn."""
        self._turn.set(
            ApprovalTurn(
                responder=responder,
                conversation_id=conversation_id,
                turn_id=turn_id,
            )
        )

    def set_tool_call_id(self, tool_call_id: str) -> None:
        """Attach the provider call ID so approval is auditable end to end."""
        self._turn.set(replace(self._turn.get(), tool_call_id=tool_call_id))

    async def ask(self, *, subject: str, description: str) -> str | None:
        """Ask for one-shot authority, failing closed. None means go ahead.

        ``subject`` is the exact thing being approved and is what the prompt
        shows; a refusal of it is remembered for the rest of the turn so the
        model cannot ask again by retrying.

        Any other return is the reason, for the caller to word: the broker's
        own close reason, or :data:`ALREADY_REFUSED` / :data:`NOT_INTERACTIVE`.
        """
        turn = self._turn.get()
        digest = sha256(subject.encode()).hexdigest()

        if digest in turn.denied_digests:
            return ALREADY_REFUSED

        if turn.responder is None or not turn.conversation_id:
            return NOT_INTERACTIVE

        decision = await turn.responder.await_approval(
            conversation_id=turn.conversation_id,
            turn_id=turn.turn_id,
            tool_call_id=turn.tool_call_id,
            command=subject,
            description=description,
        )

        if decision.approved:
            return None

        self._turn.set(replace(turn, denied_digests=turn.denied_digests | {digest}))

        return decision.reason or "error"


__all__ = [
    "ALREADY_REFUSED",
    "ApprovalDecision",
    "ApprovalGate",
    "ApprovalResponder",
    "ApprovalTurn",
    "NOT_INTERACTIVE",
]
