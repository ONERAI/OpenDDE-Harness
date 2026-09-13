"""The background promise a finished turn leaves behind: index what was said.

Owns the extraction outbox's worker -- when a batch is written, the drain, its
retry budget, the notice a user gets when a batch was abandoned -- and the
best-effort skill-usage feedback beside it. A turn appends one line here and
returns; nothing in this module is on the turn's critical path, and nothing it
holds is lost if the process dies.

A turn is not a unit of memory. Handing each one to the backend as it lands costs
a model call per turn, where the released consolidator spent one on a window's
worth of conversation at a time. So the queue is an accumulator as well as a
handover: entries stay pending, and one session's are written as a single batch
when any of four things is true.

* the queued transcript reaches :data:`BATCH_TOKEN_THRESHOLD` estimated tokens;
* the conversation was closed (``/new``), which is the one moment there is
  certainly nothing more to add to it;
* it has been :data:`BATCH_IDLE_S` since the last turn was queued -- a short
  conversation must not stay unremembered for as long as the user stays away;
* :data:`BATCH_MAX_ENTRIES` turns are queued for it, so the accumulator cannot
  hold the queue up to the bound at which the outbox starts abandoning entries.

Whichever comes first, and a flush writes **every** pending session rather than
only the one that tripped: the acknowledgement is a single prefix cursor, so a
held entry sitting below a written one would be marked written along with it.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from opendde_harness.tracing import semconv, trace
from opendde_harness.utils.helpers import estimate_message_tokens

if TYPE_CHECKING:
    from opendde_harness.memory_engine.backend import MemoryBackend
    from opendde_harness.memory_engine.consolidate.consolidator import MemoryStore
    from opendde_harness.memory_engine.outbox import MemoryOutbox, OutboxEntry

# A turn hands its completed work to the extraction outbox and is done: the answer
# is committed, and indexing it is a separate promise kept in the background. These
# are that background's bounds.
#
# How many times one batch is offered to the backend before it is abandoned.
# A store that fails three times running is failing, not busy, and holding the
# queue behind it would stop every turn after it from being indexed.
STORE_MAX_ATTEMPTS: int = 3
# Pause between attempts at the same batch, multiplied by the attempt number, so a
# wedged service is retried rather than spun on.
STORE_RETRY_BACKOFF_S: float = 1.0
# Teardown's total budget for letting the drain finish what it is holding.
STORE_DRAIN_BUDGET_S: float = 15.0
# What the turn will spend on skill-usage feedback, which is telemetry: it is
# best-effort in both directions, so it gets a budget rather than a turn.
FEEDBACK_BUDGET_S: float = 1.0

#: How much accumulated transcript one write covers, in estimated prompt tokens
#: of the queued messages. The point of the number is the model call behind it:
#: at roughly a window's worth of conversation per call, a backend that
#: summarizes costs what the released consolidator cost.
BATCH_TOKEN_THRESHOLD: int = 8_000

#: How long a session's queued turns may sit before they are written anyway. The
#: token threshold on its own leaves a short conversation unindexed for as long as
#: the user stays away from it, which is most conversations.
BATCH_IDLE_S: float = 600.0

#: Pending entries for one session past which its batch is written whatever it
#: weighs. Half the outbox's capacity: the accumulator holds entries on purpose,
#: and must not be able to hold them up to the bound at which the outbox abandons
#: its oldest.
BATCH_MAX_ENTRIES: int = 32

#: The shortest the drain will wait while holding a batch. A fake clock does not
#: advance on its own, so an idle deadline computed from one can be in the past on
#: every pass; without a floor the drain would spin instead of waiting.
BATCH_POLL_FLOOR_S: float = 1.0


def qualified_ids(ids: list[str] | None, source_prefix: str) -> list[str]:
    """The native ids of a list of qualified ids matching ``<prefix>/<native>``.

    Returns the bare native portion for each match, so the receiving backend does
    not have to re-parse. Non-matching, unprefixed or malformed entries silently
    drop; ``None`` and empty inputs return ``[]``.
    """
    if not ids:
        return []
    needle = f"{source_prefix}/"
    out: list[str] = []
    for qid in ids:
        if not isinstance(qid, str):
            continue
        if qid.startswith(needle):
            native = qid[len(needle) :]
            if native:
                out.append(native)
    return out


@dataclass(frozen=True)
class _Batch:
    """One session's queued turns, as the single slice the backend is offered.

    ``messages`` is the conversation those entries hold, deduplicated: ``/new``
    queues the whole closed session, which overlaps every per-turn entry already
    waiting for it, and a backend handed the overlap would index those turns
    twice.
    """

    session: str
    entries: list["OutboxEntry"]
    messages: list[dict[str, Any]]
    tokens: int
    closing: bool

    @property
    def last(self) -> "OutboxEntry":
        """The newest entry this batch covers, and so what acknowledges it."""
        return self.entries[-1]


def _identity(message: dict[str, Any]) -> Any:
    """What makes two copies of one message the same message.

    The record id when it has one; otherwise the three things that cannot
    coincide for two different messages in one conversation.
    """
    ident = message.get("id")
    if isinstance(ident, str) and ident:
        return ident
    return (
        message.get("role"),
        message.get("timestamp"),
        json.dumps(message.get("content"), sort_keys=True, default=str),
    )


class ExtractionDispatcher:
    """The worker that owes a committed turn its place in the memory index.

    Built with the backend that may be ``None``: a host with no memory owner does
    not extract at all, which is what ``memory.backend: "off"`` buys. The outbox
    file is created on first use for the same reason.
    """

    def __init__(
        self,
        workspace: Path,
        store: "MemoryStore",
        *,
        backend: "MemoryBackend | None",
        now_fn: Callable[[], "datetime"] | None = None,
    ) -> None:
        self._workspace = workspace
        self._store = store
        self._backend = backend
        self._now_fn = now_fn
        self._clock: Callable[[], datetime] = now_fn or datetime.now
        self._outbox: "MemoryOutbox | None" = None
        self._task: asyncio.Task | None = None
        # Set whenever the queue changes, so a drain that is holding a batch
        # re-reads it rather than sleeping out its idle window on a queue that has
        # since crossed the threshold.
        self._wake = asyncio.Event()
        # Write what is held regardless of size: the process is going away, and a
        # batch written now is recallable in the next session rather than the one
        # after.
        self._force = False
        # Said once, on the next turn: some turns were abandoned unindexed.
        # Reported to the user rather than to a shutdown log line nobody reads.
        self._notice_sent = False
        # Feedback sends that outran their budget. Held only so the event loop does
        # not collect a task that is still running.
        self._feedback_inflight: set[asyncio.Task] = set()

    @property
    def backend(self) -> "MemoryBackend | None":
        return self._backend

    @property
    def outbox(self) -> "MemoryOutbox":
        """The extraction outbox, built on first use.

        Lazy because a host with no memory backend never queues anything, and a
        workspace that never queued anything should not carry the file.
        """
        if self._outbox is None:
            from opendde_harness.memory_engine.outbox import MemoryOutbox

            self._outbox = MemoryOutbox(self._workspace, self._store, now_fn=self._now_fn)
        return self._outbox

    @property
    def task(self) -> asyncio.Task | None:
        """The drain, when one is running. Held for teardown and for tests."""
        return self._task

    def queue(
        self,
        session_key: str,
        messages: list[dict],
        *,
        turn_id: str,
        generation: int,
        closing: bool = False,
    ) -> None:
        """Hand a committed turn to the outbox and let the drain pick it up.

        The turn's last durable act, and a cheap one: one line appended, no network,
        no await. ``None`` backend means the host does not extract at all.

        ``closing`` marks the hand-off a closing conversation makes (``/new``).
        It is the one moment there is certainly nothing more to add to that
        session, so it writes the batch instead of accumulating it -- and it is
        recorded on the entry rather than held in memory, because the reset it
        belongs to may be the last thing this process does.

        The queue write is best-effort on top of an already-committed turn: the
        session log has the conversation either way, so a full disk loses the index
        and not the record.
        """
        if self._backend is None or not messages:
            return
        try:
            self.outbox.append(
                turn_id=turn_id,
                session=session_key,
                generation=generation,
                messages=messages,
                closing=closing,
            )
        except OSError:
            logger.exception(
                "could not queue session {} turn {} for extraction; the turn is in the session log",
                session_key,
                turn_id,
            )
            return
        self.kick()

    def kick(self) -> None:
        """Make sure the outbox is being drained, without waiting for it."""
        if self._backend is None:
            return
        if self._task is not None and not self._task.done():
            # A drain that is holding a batch has to look again: the line just
            # appended may be what takes the batch past its threshold.
            self._wake.set()
            return
        try:
            self._task = asyncio.create_task(self._drain())
        except RuntimeError:
            # No running loop (a synchronous caller outside the turn machinery). The
            # entry is on disk; the next turn's kick drains it.
            logger.debug("outbox: no running loop to drain on")

    async def _drain(self) -> None:
        """Hold pending entries until a batch is due, then write every session's.

        Bounded so a failing backend cannot hold the queue: a batch that fails
        ``STORE_MAX_ATTEMPTS`` times is abandoned, counted, and said out loud on
        the next turn -- an unindexed turn is one the user will not be able to
        recall, and silence about it is the part that was wrong.

        A ``store`` that returns ``False`` has not landed and is retried; the
        contract says so, and the old caller discarded the answer.
        """
        outbox = self.outbox
        while True:
            # Cleared before the queue is read, so a kick that arrives after the
            # read is not lost: it sets the event, and the wait below returns at
            # once instead of sitting out the idle window.
            self._wake.clear()
            pending = outbox.pending()
            if not pending:
                self._force = False
                return
            batches = self._batches(pending)
            reason = self._flush_reason(batches)
            if reason is None:
                await self._hold(batches)
                continue
            logger.info("memory: writing {} queued turn(s) -- {}", len(pending), reason)
            for batch in batches:
                await self._write(batch)
            # One prefix cursor, moved once, after every batch this pass holds has
            # been written or abandoned. Acknowledging batch by batch would mark
            # another session's earlier turns written before they were.
            if not outbox.ack(pending[-1]):
                # The store landed and the cursor did not. Stop rather than offer
                # the same batch again: the next run replays it once, which is the
                # trade this design makes.
                return
            if len(outbox.pending()) >= len(pending):
                logger.warning(
                    "memory outbox: the queue is not moving past turn {}; stopping until the next run",
                    pending[-1].turn_id,
                )
                return

    def _batches(self, pending: list["OutboxEntry"]) -> list[_Batch]:
        """Group pending entries by session, oldest session first.

        One batch is one annotation, so the grouping is by conversation: two
        sessions' turns folded into one slice would be summarized as one
        conversation that never happened.
        """
        grouped: dict[str, list["OutboxEntry"]] = {}
        for entry in pending:
            grouped.setdefault(entry.session, []).append(entry)
        batches: list[_Batch] = []
        for session, entries in grouped.items():
            seen: set[Any] = set()
            messages: list[dict[str, Any]] = []
            for entry in entries:
                for message in entry.messages:
                    key = _identity(message)
                    if key in seen:
                        continue
                    seen.add(key)
                    messages.append(message)
            batches.append(
                _Batch(
                    session=session,
                    entries=entries,
                    messages=messages,
                    tokens=sum(estimate_message_tokens(message) for message in messages),
                    closing=any(entry.closing for entry in entries),
                )
            )
        return batches

    def _flush_reason(self, batches: list[_Batch]) -> str | None:
        """Why this pass writes, or None to keep accumulating.

        Read across every pending session: one session coming due writes them all,
        because the acknowledgement cannot say "this one and not that one".
        """
        if self._force:
            return "the session is ending"
        for batch in batches:
            if batch.closing:
                return f"conversation {batch.session} was closed"
            if len(batch.entries) >= BATCH_MAX_ENTRIES:
                return f"{len(batch.entries)} turns are queued for {batch.session}"
            if batch.tokens >= BATCH_TOKEN_THRESHOLD:
                return f"{batch.tokens} tokens of conversation are queued for {batch.session}"
            if self._idle_for(batch) >= BATCH_IDLE_S:
                return f"{batch.session} has had nothing new for {BATCH_IDLE_S:.0f}s"
        return None

    async def _hold(self, batches: list[_Batch]) -> None:
        """Wait for the queue to change, or for the nearest idle deadline."""
        remaining = [BATCH_IDLE_S - self._idle_for(batch) for batch in batches]
        wait = max(BATCH_POLL_FLOOR_S, min(remaining) if remaining else BATCH_IDLE_S)
        try:
            await asyncio.wait_for(self._wake.wait(), wait)
        except asyncio.TimeoutError:
            pass

    def _idle_for(self, batch: _Batch) -> float:
        """Seconds since the newest turn in this batch was queued.

        An entry whose stamp cannot be read counts as fully idle: a malformed
        timestamp must not be a reason to hold a conversation unindexed forever.
        """
        try:
            queued = datetime.fromisoformat(batch.last.queued_at)
        except ValueError:
            return BATCH_IDLE_S
        return max(0.0, (self._clock() - queued).total_seconds())

    async def _write(self, batch: _Batch) -> None:
        """Offer one batch to the backend, retrying a refusal, abandoning a failure."""
        offered = replace(batch.last, messages=batch.messages)
        for attempt in range(1, STORE_MAX_ATTEMPTS + 1):
            if await self._store_entry(offered):
                return
            if attempt < STORE_MAX_ATTEMPTS:
                await asyncio.sleep(STORE_RETRY_BACKOFF_S * attempt)
        logger.warning(
            "backend.store failed {} times for session {} ({} turn(s)); abandoning them",
            STORE_MAX_ATTEMPTS,
            batch.session,
            len(batch.entries),
        )
        self.outbox.deferred += len(batch.entries)

    @trace.instrument("memory.store", extract=semconv.memory_store)
    async def _store_entry(self, entry: "OutboxEntry") -> bool:
        """One ``backend.store`` attempt. Never raises.

        ``entry`` is the batch as one entry: the session and the newest turn id it
        covers, carrying every message it holds. So the span reads as the write it
        is, and the acknowledgement it belongs to is the one on the entry.

        A backend that reports nothing is taken at its word that the write landed:
        the Protocol's return is a ``bool``, and an adapter that returns ``None`` is
        one that does not track landing, not one that failed.
        """
        try:
            landed = await self._backend.store(entry.session, entry.messages)  # type: ignore[union-attr]
        except Exception:
            logger.exception(
                "backend.store failed for session {}; turn data preserved in session log",
                entry.session,
            )
            return False
        return landed is not False

    async def drain(self, timeout: float = STORE_DRAIN_BUDGET_S) -> None:
        """Write what is held before the process goes away.

        The accumulator's last flush: whatever is queued survives the exit -- that
        is what the outbox is for -- so this is about latency, not loss. A batch
        written now is recallable in the next session rather than the one after,
        and a drain still holding past its budget is cancelled rather than left to
        outlive the loop it was started on.
        """
        self._force = True
        self._wake.set()
        self.kick()
        task = self._task
        if task is not None and not task.done():
            await asyncio.wait({task}, timeout=timeout)
            if not task.done():
                task.cancel()
        self._force = False
        if self._outbox is None:
            return
        still_owed = len(self._outbox.pending())
        if still_owed:
            logger.info("{} turn(s) are queued for extraction and will be indexed on the next run", still_owed)
        if self._outbox.deferred:
            logger.warning(
                "{} turn(s) were not indexed: the memory service never accepted them",
                self._outbox.deferred,
            )

    def deferral_notice(self) -> str | None:
        """One line about abandoned extraction, the first time anyone asks.

        Said to the user, once per session. Before this it was a warning in a log at
        shutdown, which is the one moment they are not reading it.
        """
        if self._notice_sent or self._outbox is None or not self._outbox.deferred:
            return None
        self._notice_sent = True
        count = self._outbox.deferred
        turns = "turn" if count == 1 else "turns"
        return (
            f"{count} completed {turns} could not be handed to the memory service and will not be "
            "recalled later. What was said is in the session log."
        )

    @trace.instrument("memory.feedback", extract=semconv.memory_feedback)
    async def send_feedback(self, session_key: str, used_skill_ids: list[str] | None = None) -> None:
        """Forward this turn's skill usage to :meth:`MemoryBackend.feedback`.

        The signal is what the turn actually loaded: the qualified ids the agent
        passed to ``use_skill``. Advertising a skill says nothing about it, so
        nothing is reported for the catalogue.

        Only the ``memory/`` prefix is forwarded -- an on-disk ``local`` skill has no
        feedback channel, and so does an unprefixed id, which predates the
        qualified-id convention and has no safe routing target.

        No-ops without a backend or a qualifying id. Exceptions from
        ``backend.feedback`` are caught and logged: the host must not abort the
        after-turn pipeline because a plugin's handler raised. Feedback is
        best-effort telemetry, not load-bearing state.
        """
        if self._backend is None:
            return
        used_native = qualified_ids(used_skill_ids, "memory")
        if not used_native:
            return
        signals = {"kind": "skill_usage", "session_id": session_key, "used": used_native}

        async def _send() -> None:
            try:
                await self._backend.feedback(signals)  # type: ignore[union-attr]
            except Exception:
                logger.exception("backend.feedback failed for session {}; signals dropped", session_key)

        # Fire and forget, with a budget rather than a deadline: telemetry that is
        # slow is still telemetry, but it does not get to decide when the user can
        # type again. Directly awaited, this was the one after-turn step with no
        # bound on it at all. The reference is held until it finishes, because a
        # pending task nobody refers to can be collected mid-flight.
        task = asyncio.create_task(_send())
        self._feedback_inflight.add(task)
        task.add_done_callback(self._feedback_inflight.discard)
        await asyncio.wait({task}, timeout=FEEDBACK_BUDGET_S)


__all__ = [
    "BATCH_IDLE_S",
    "BATCH_MAX_ENTRIES",
    "BATCH_TOKEN_THRESHOLD",
    "FEEDBACK_BUDGET_S",
    "STORE_DRAIN_BUDGET_S",
    "STORE_MAX_ATTEMPTS",
    "STORE_RETRY_BACKOFF_S",
    "ExtractionDispatcher",
    "qualified_ids",
]
