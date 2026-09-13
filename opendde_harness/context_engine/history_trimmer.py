"""Deterministic history selection — the one policy that decides ``*history``.

The policy, in the order it is applied:

1. **Budget** against what the request actually carries: the rendered system
   prefix, the tool schemas, this turn's user message, and the model's output
   reservation. An unknown window is not a number -- nothing is trimmed.
2. **Marker.** A compaction marker this model's own backend replays is the
   frame, not a candidate: the latest one is kept unconditionally, everything
   it stands for is excluded from selection *and* from costing, and the
   recent-tail rule below applies only after it. A marker the model does not
   replay establishes no boundary at all.
3. **Protected head.** Without an active marker, the first ``protect_first_n``
   user messages are never dropped (the user's own framing of the task, not
   three tool-heavy turns).
4. **Mandatory tail.** The newest completed exchange and the user message that
   encloses it are required.
5. **Fill** the remaining space with the newest complete exchanges, emitted in
   chronological order.
6. **Excerpt before dropping.** When something has to go, the oldest
   tool-result bodies are excerpted first (:mod:`.excerpt`: never a marker,
   never a call's arguments, never in place), because a body the model can ask
   for again is cheaper to give up than a turn of the conversation.
7. **Drop** the oldest unprotected exchanges with what still does not fit.
8. If the mandatory set cannot fit even excerpted, raise
   :class:`ContextBudgetError`. An oversized request is never sent knowingly.

One pass: exchanges and per-message costs are computed once, selection runs
against a running budget, and there is a single exact projection at the end
(plus at most :attr:`HistoryTrimmer._MAX_CORRECTIONS` corrective rounds, for
the gap between summing per-message costs and pricing the whole chain).

``_fit_prompt`` in the loop is deliberately *not* folded in here: tool output
grows after selection, and that growth is a different problem from choosing
which turns to carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

from opendde_harness.context_engine.excerpt import (
    KEEP_RECENT_TOOL_RESULTS,
    excerpt_tool_result,
)
from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import (
    COMPACTION_KEY,
    LLMProvider,
    compaction_boundary,
    orphan_tool_results,
    wire_history,
)
from opendde_harness.providers.binding import ModelBinding, active_window, resolve
from opendde_harness.utils.helpers import estimate_message_tokens, estimate_prompt_tokens_chain


class ContextBudgetError(RuntimeError):
    """The turn cannot be assembled inside this model's window.

    Raised instead of sending a request the window cannot hold: the mandatory
    part of the prompt (system prefix, tool schemas, this turn's user message,
    the protected head and the newest exchange) is over budget even with the
    oldest tool bodies excerpted. Local, before any network call.
    """


@dataclass(frozen=True)
class _Exchange:
    """One indivisible unit of selection.

    A tool exchange is indivisible on the wire: an assistant turn carrying
    ``toolCall`` blocks and the results answering them stand or fall together.
    Both providers refuse half of one -- "tool_use ids were found without
    tool_result blocks" and "messages with role 'tool' must be a response to a
    preceding message with 'tool_calls'".

    ``user_id`` is the message that opened the turn this exchange belongs to;
    selecting the exchange selects it too, so a kept exchange is never left
    without the request it answers.
    """

    ids: tuple[int, ...]
    user_id: int | None
    complete: bool


@dataclass
class SelectionOutcome:
    """What :meth:`HistoryTrimmer.select` decided, and what it cost."""

    history: list[dict[str, Any]]
    included_ids: list[int]
    estimated_tokens: int
    max_prompt_tokens: int | None  # None: the window is unknown, nothing was enforced
    source: str
    excerpted_ids: list[int] = field(default_factory=list)
    dropped_ids: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.max_prompt_tokens is None or self.estimated_tokens <= self.max_prompt_tokens

    @property
    def over_by(self) -> int:
        if self.max_prompt_tokens is None:
            return 0
        return max(0, self.estimated_tokens - self.max_prompt_tokens)


class HistoryTrimmer:
    """Selects and budget-bounds the session history into ``*history``."""

    #: Rounds of corrective removal after the exact projection disagrees with
    #: the sum of per-message costs. Bounded so a disagreement costs a handful
    #: of estimates rather than one per message.
    _MAX_CORRECTIONS = 3
    #: Headroom added to each corrective removal, as a share of the budget, so
    #: a round that removes exactly the overshoot does not come back a token
    #: over on the next projection.
    _CORRECTION_MARGIN = 0.05

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        context_window_tokens: int | None,
        protect_first_n: int = 3,
    ) -> None:
        self._fallback = ModelBinding(provider, model)
        self.get_tool_definitions = get_tool_definitions
        self._fallback_window = context_window_tokens
        self.protect_first_n = protect_first_n

    @property
    def provider(self) -> LLMProvider:
        """The provider of the turn's binding; the build-time one outside a turn."""
        return resolve(None, self._fallback).provider

    @property
    def model(self) -> str:
        """The model of the turn's binding; the build-time one outside a turn."""
        return resolve(None, self._fallback).model

    @property
    def context_window_tokens(self) -> int | None:
        """The running turn's window; the one built with, outside a turn.

        A property because this object outlives any number of turns and two
        sessions can be on models of different sizes at once -- a number copied
        at construction answers for whichever session happened to build it.
        None is unknown, and means "do not trim".
        """
        return active_window(self._fallback_window)

    @context_window_tokens.setter
    def context_window_tokens(self, tokens: int | None) -> None:
        self._fallback_window = tokens

    def set_provider(self, provider: LLMProvider, model: str) -> None:
        """Move the out-of-turn fallback; inside a turn the binding answers."""
        self._fallback = ModelBinding(provider, model)

    # ------------------------------------------------------------------
    # Pure history-shaping helpers (no token estimation / no I/O)
    # ------------------------------------------------------------------

    @staticmethod
    def canonical_ids(messages: list[dict[str, Any]], ids: list[int]) -> list[int]:
        """Close ``ids`` over tool-call / tool-result adjacency.

        Returns the selected indices in order, trimmed so the sequence begins
        at a ``user`` message or at a compaction marker (so history never
        starts mid tool-exchange). A marker opens a history as cleanly as a
        user turn does -- it stands for everything before it -- and skipping
        past one to find a user message deleted the boundary from a selection
        that had kept it. Returns ``[]`` if neither survives.

        Closure is what the front cut may not break: a call before the marker
        whose result comes after it is closed over first and then cut away by
        the alignment, and the result left behind is refused by the provider.
        Past the marker the call is inside the summary, so the orphan goes.
        """
        selected = {mid for mid in ids if isinstance(mid, int) and 0 <= mid < len(messages)}
        tool_parent_by_call, tool_result_by_call = HistoryTrimmer._tool_pairs(messages)

        changed = True
        while changed:
            changed = False
            for call_id, parent_idx in tool_parent_by_call.items():
                result_ids = tool_result_by_call.get(call_id, [])
                if parent_idx in selected:
                    for rid in result_ids:
                        if rid not in selected:
                            selected.add(rid)
                            changed = True
                if any(rid in selected for rid in result_ids) and parent_idx not in selected:
                    selected.add(parent_idx)
                    changed = True

        ordered = sorted(selected)
        for pos, mid in enumerate(ordered):
            if messages[mid].get("role") == "user" or COMPACTION_KEY in messages[mid]:
                aligned = ordered[pos:]
                orphans = orphan_tool_results([messages[mid] for mid in aligned])
                if orphans:
                    logger.warning("dropping {} tool result(s) whose call is before the replay boundary", len(orphans))
                return [mid for at, mid in enumerate(aligned) if at not in orphans]
        return []

    @staticmethod
    def _tool_pairs(messages: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, list[int]]]:
        """Which assistant turn opened each tool call, and which results answer it."""
        parent_by_call: dict[str, int] = {}
        result_by_call: dict[str, list[int]] = {}
        for idx, message in enumerate(messages):
            for call_id in msg.tool_call_ids(message):
                parent_by_call[call_id] = idx
            if msg.is_tool_result(message) and message.get("toolCallId"):
                result_by_call.setdefault(str(message["toolCallId"]), []).append(idx)
        return parent_by_call, result_by_call

    @staticmethod
    def history_from_ids(
        messages: list[dict[str, Any]],
        ids: list[int],
        overrides: dict[int, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Project the selected messages down to provider-safe keys.

        ``overrides`` supplies an excerpted stand-in for an id (see
        :mod:`.excerpt`). The session's own message is never modified: the
        original stays on disk and is what the next turn selects from.
        """
        history: list[dict[str, Any]] = []
        for mid in ids:
            source = (overrides or {}).get(mid, messages[mid])
            clean = msg.wire_projection(source)
            if clean.get("role"):
                history.append(clean)
        return history

    @staticmethod
    def structural_errors(messages: list[dict[str, Any]]) -> list[str]:
        """Tool-call closure validation over a built message list.

        Kept as the oracle the policy is tested against: every selection this
        module produces must come back clean from here.
        """
        errors: list[str] = []
        open_calls: set[str] = set()
        for message in messages:
            open_calls.update(msg.tool_call_ids(message))
            if msg.is_tool_result(message):
                call_id = str(message.get("toolCallId", ""))
                if call_id not in open_calls:
                    errors.append(f"tool result {call_id} has no parent assistant tool call")
                else:
                    open_calls.remove(call_id)
        if open_calls:
            errors.append(f"assistant tool calls missing results: {sorted(open_calls)}")
        return errors

    def validate_candidate(self, messages: list[dict[str, Any]], reserved_output: int) -> dict[str, Any]:
        """Price and structurally check a built message list.

        The test oracle for the policy: an independent answer to "would this
        prompt be accepted", computed from the messages alone rather than from
        the selector's own bookkeeping.
        """
        estimated, source = self._estimate(messages)
        max_prompt = self._max_prompt(reserved_output)
        errors = self.structural_errors(messages)
        return {
            "ok": not errors and (max_prompt is None or estimated <= max_prompt),
            "errors": errors,
            "estimated_tokens": estimated,
            "max_prompt_tokens": max_prompt,
            "over_by": 0 if max_prompt is None else max(0, estimated - max_prompt),
            "source": source,
        }

    # ------------------------------------------------------------------
    # Costing
    # ------------------------------------------------------------------

    def _max_prompt(self, reserved_output: int) -> int | None:
        """The whole prompt's ceiling, or None when there is no usable one.

        No window, no trimming: dropping history to fit a number nobody
        measured is worse than sending it all and letting the provider's
        overflow error say so, which the loop handles by shrinking in place.

        A window the reservation alone fills is the same answer. The
        reservation is the model's whole output ceiling, not a promise about
        the prompt, so a ceiling as large as the window leaves a budget of
        zero -- and refusing every turn on it would be refusing over
        arithmetic rather than over a request the provider will not take.
        """
        window = self.context_window_tokens
        if window is None:
            return None
        room = window - reserved_output
        return room if room > 0 else None

    def _estimate(self, messages: list[dict[str, Any]]) -> tuple[int, str]:
        """What this prompt costs the model the turn is bound to.

        ``wire_history`` first: past a compaction marker the request carries the
        marker and what follows, not the history it stands for, and trimming a
        session down to a budget it is already under drops conversation for
        nothing.
        """
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            wire_history(messages, self.provider, self.model),
            self.get_tool_definitions(),
        )

    def _message_cost(self, message: dict[str, Any]) -> int:
        """One message's share of the prompt, priced the way the request is.

        A marker costs the clear text it carries rather than its placeholder:
        that is what the backend replays in its place, and
        :func:`wire_history` is the one function that knows so.
        """
        if COMPACTION_KEY in message:
            projected = wire_history([message], self.provider, self.model)
            return sum(estimate_message_tokens(m) for m in projected)
        return estimate_message_tokens(message)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        *,
        session_messages: list[dict[str, Any]],
        reserved_output: int,
        build_messages: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    ) -> tuple[list[dict[str, Any]], SelectionOutcome]:
        """Choose this turn's history and return ``(messages, outcome)``.

        ``build_messages`` maps a history list to the full message list
        (system + history + user) — the caller owns prompt composition, this
        module only owns history selection.
        """
        # ── Frame: the marker this model replays, if any ─────────────
        boundary = compaction_boundary(session_messages, self.provider, self.model)
        first = 0 if boundary is None else boundary
        candidates = list(range(first, len(session_messages)))

        # ── Precompute, once: costs and exchanges ────────────────────
        costs = {mid: self._message_cost(session_messages[mid]) for mid in candidates}
        exchanges = self._exchanges(session_messages, candidates, boundary)
        protected = self._protected_user_ids(session_messages, candidates, boundary)

        max_prompt = self._max_prompt(reserved_output)
        if max_prompt is None:
            # Unknown window: ship the candidate range as it stands.
            return self._project(session_messages, candidates, candidates, {}, build_messages, max_prompt=None)

        fixed, _ = self._estimate(build_messages([]))
        room = max_prompt - fixed
        mandatory = self._mandatory_ids(exchanges, protected, boundary)

        selected = self._fill(exchanges, mandatory, costs, room)
        excerpts: dict[int, dict[str, Any]] = {}
        if len(selected) < len(candidates):
            # Before dropping a turn, give up the oldest tool bodies: the model
            # can ask for a file again, it cannot ask for the conversation back.
            excerpts = self._excerpts(session_messages, candidates)
            for mid, excerpted in excerpts.items():
                costs[mid] = self._message_cost(excerpted)
            selected = self._fill(exchanges, mandatory, costs, room)

        mandatory_cost = sum(costs[mid] for mid in mandatory)
        if self._refusable(room, mandatory) and mandatory_cost > room:
            raise ContextBudgetError(self._budget_message(fixed, mandatory_cost, max_prompt, reserved_output))

        messages, outcome = self._project(
            session_messages, candidates, selected, excerpts, build_messages, max_prompt=max_prompt
        )

        # ── One exact projection, then bounded correction ────────────
        # Summing per-message costs is not the same arithmetic as pricing the
        # whole chain (per-message overheads, a provider's own counter). The
        # gap is small and one-sided, so a few rounds of removing the overshoot
        # settle it -- rather than re-estimating once per dropped exchange.
        for _ in range(self._MAX_CORRECTIONS):
            if outcome.ok:
                return messages, outcome
            target = outcome.estimated_tokens + int(max_prompt * self._CORRECTION_MARGIN) - max_prompt
            freed, selected = self._shed(selected, mandatory, costs, target)
            if not freed:
                break
            messages, outcome = self._project(
                session_messages, candidates, selected, excerpts, build_messages, max_prompt=max_prompt
            )
        if not outcome.ok:
            if self._refusable(room, mandatory):
                raise ContextBudgetError(
                    self._budget_message(fixed, outcome.estimated_tokens - fixed, max_prompt, reserved_output)
                )
            logger.warning(
                "the system prompt and tool schemas alone cost {} of the {} tokens {} has for a prompt",
                fixed,
                max_prompt,
                self.model,
            )
        return messages, outcome

    def _exchanges(
        self,
        messages: list[dict[str, Any]],
        candidates: list[int],
        boundary: int | None = None,
    ) -> list[_Exchange]:
        """Partition the candidate range into indivisible units, once.

        One walk: every message belongs to exactly one exchange, tool results
        ride with the assistant turn that called them, and a user message is
        its own unit (it is what encloses the exchanges after it).

        The active marker is not among them. It is the frame, kept whatever the
        budget says, and counting it as an assistant exchange would let it
        answer as "the newest exchange" for a conversation it only stands in
        for. A marker this model does not replay is ordinary content and is
        partitioned like any other assistant message.
        """
        parent_by_call, result_by_call = self._tool_pairs(messages)
        in_range = set(candidates)
        consumed: set[int] = set()
        out: list[_Exchange] = []
        user_id: int | None = None
        for mid in candidates:
            if mid in consumed or mid == boundary:
                continue
            message = messages[mid]
            role = message.get("role")
            if role == "user":
                user_id = mid
                out.append(_Exchange(ids=(mid,), user_id=mid, complete=True))
                consumed.add(mid)
                continue
            call_ids = msg.tool_call_ids(message)
            if role == "assistant" and call_ids:
                results = sorted(rid for cid in call_ids for rid in result_by_call.get(cid, []) if rid in in_range)
                complete = all(any(rid in in_range for rid in result_by_call.get(cid, [])) for cid in call_ids)
                ids = tuple(sorted({mid, *results}))
                out.append(_Exchange(ids=ids, user_id=user_id, complete=complete))
                consumed.update(ids)
                continue
            if msg.is_tool_result(message):
                # A result whose call is outside the range (before the marker,
                # or never recorded) can never be sent: no parent, no exchange.
                parent = parent_by_call.get(str(message.get("toolCallId", "")))
                if parent is None or parent not in in_range:
                    consumed.add(mid)
                    continue
                continue
            out.append(_Exchange(ids=(mid,), user_id=user_id, complete=True))
            consumed.add(mid)
        return out

    def _protected_user_ids(
        self,
        messages: list[dict[str, Any]],
        candidates: list[int],
        boundary: int | None,
    ) -> set[int]:
        """The first ``protect_first_n`` user messages, when no marker is active.

        Past an active marker there is no head to protect: everything before it
        is not sent at all, and the marker itself is what stands for it.
        """
        if boundary is not None or self.protect_first_n <= 0:
            return set()
        users = [mid for mid in candidates if messages[mid].get("role") == "user"]
        return set(users[: self.protect_first_n])

    @staticmethod
    def _mandatory_ids(exchanges: list[_Exchange], protected: set[int], boundary: int | None) -> list[int]:
        """What this turn must carry whatever the budget says.

        The marker (it is the frame), the protected head, and the newest
        completed exchange with the user message that encloses it -- a reply
        with no record of the request it answers is not a conversation.
        """
        mandatory: set[int] = set(protected)
        if boundary is not None:
            mandatory.add(boundary)
        newest = next(
            (ex for ex in reversed(exchanges) if ex.complete and ex.ids != (ex.user_id,)),
            None,
        )
        if newest is not None:
            mandatory.update(newest.ids)
            if newest.user_id is not None:
                mandatory.add(newest.user_id)
        return sorted(mandatory)

    @staticmethod
    def _fill(
        exchanges: list[_Exchange],
        mandatory: list[int],
        costs: dict[int, int],
        room: int,
    ) -> list[int]:
        """Newest-first selection against a running budget; chronological out.

        Incomplete exchanges are never filled in: half a tool exchange is a
        request both vendors refuse. An exchange carries the user message that
        encloses it, so a kept reply is never left without its request.
        """
        selected = set(mandatory)
        remaining = room - sum(costs.get(mid, 0) for mid in selected)
        for ex in reversed(exchanges):
            ids = [*ex.ids, *([ex.user_id] if ex.user_id is not None else [])]
            new_ids = [mid for mid in dict.fromkeys(ids) if mid not in selected]
            if not new_ids or not ex.complete:
                continue
            cost = sum(costs.get(mid, 0) for mid in new_ids)
            if cost > remaining:
                continue
            selected.update(new_ids)
            remaining -= cost
        return sorted(selected)

    @staticmethod
    def _shed(
        selected: list[int],
        mandatory: list[int],
        costs: dict[int, int],
        target: int,
    ) -> tuple[int, list[int]]:
        """Remove the oldest unprotected ids until ``target`` tokens are freed."""
        keep = set(mandatory)
        freed = 0
        out = list(selected)
        for mid in selected:
            if freed >= target:
                break
            if mid in keep:
                continue
            freed += costs.get(mid, 0)
            out.remove(mid)
        return freed, out

    def _excerpts(self, messages: list[dict[str, Any]], candidates: list[int]) -> dict[int, dict[str, Any]]:
        """Excerpted stand-ins for the oldest tool-result bodies in range.

        The newest few results are left whole: they are what the model is
        reasoning from. Markers and call arguments are never touched (see
        :mod:`.excerpt`).
        """
        tool_ids = [mid for mid in candidates if msg.is_tool_result(messages[mid])]
        if len(tool_ids) <= KEEP_RECENT_TOOL_RESULTS:
            return {}
        out: dict[int, dict[str, Any]] = {}
        for mid in tool_ids[: len(tool_ids) - KEEP_RECENT_TOOL_RESULTS]:
            excerpted = excerpt_tool_result(messages[mid])
            if excerpted is not None:
                out[mid] = excerpted
        return out

    def _project(
        self,
        session_messages: list[dict[str, Any]],
        candidates: list[int],
        selected: list[int],
        excerpts: dict[int, dict[str, Any]],
        build_messages: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        *,
        max_prompt: int | None,
    ) -> tuple[list[dict[str, Any]], SelectionOutcome]:
        """Close, project, build and price one candidate selection."""
        canon = self.canonical_ids(session_messages, selected)
        history = self.history_from_ids(session_messages, canon, excerpts)
        messages = build_messages(history)
        estimated, source = self._estimate(messages)
        kept = set(canon)
        dropped = [mid for mid in candidates if mid not in kept]
        used_excerpts = sorted(mid for mid in excerpts if mid in kept)
        warnings: list[str] = []
        if dropped and max_prompt is not None:
            warnings.append(f"dropped {len(dropped)} message(s) to fit the budget")
        if used_excerpts:
            warnings.append(f"excerpted {len(used_excerpts)} older tool result(s)")
        return messages, SelectionOutcome(
            history=history,
            included_ids=canon,
            estimated_tokens=estimated,
            max_prompt_tokens=max_prompt,
            source=source,
            excerpted_ids=used_excerpts,
            dropped_ids=dropped,
            warnings=warnings,
        )

    @staticmethod
    def _refusable(room: int, mandatory: list[int]) -> bool:
        """Is the history what does not fit, or the prompt around it?

        Only the first is selection's to refuse. When the rendered prefix and
        the tool schemas alone are over the window there is no turn to drop
        that would help: the request goes out and the provider's own refusal is
        the truth -- a declared window can simply be wrong, and refusing every
        turn locally would make a wrong number unrecoverable rather than
        visible. The loop's overflow recovery handles it from there.
        """
        return room > 0 and bool(mandatory)

    def _budget_message(self, fixed: int, mandatory_cost: int, max_prompt: int, reserved_output: int) -> str:
        """What to tell somebody whose turn does not fit. No content quoted."""
        window = self.context_window_tokens
        return (
            f"this turn does not fit {self.model}: the system prompt and tool schemas cost {fixed} tokens "
            f"and the messages that cannot be dropped (the protected head and the newest exchange) "
            f"another {mandatory_cost}, against {max_prompt} available "
            f"({window} window less {reserved_output} held back for the reply). "
            "Start a new conversation (/new), switch to a model with a larger window, "
            "or declare a larger window for this model in providers.<provider>.models."
        )


__all__ = ["ContextBudgetError", "HistoryTrimmer", "SelectionOutcome"]
