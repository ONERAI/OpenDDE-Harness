"""The turn journal: what a turn recorded, written while the turn runs.

A turn used to reach disk in one write after it returned. Everything it did
before that -- a submitted remote job, a created artifact, a file it wrote --
was held in a list in memory, and Esc or a lost process took the record of it
with them. A checkpoint of the filesystem cannot reconstruct a job id the
harness was told once.

So the journal writes forward:

* ``turn.started`` and the accepted user message before the first model call;
* ``tool.started`` before a tool that changes state outside this process;
* the assistant message that asked for tools, and each result as it returns;
* the final assistant message and ``turn.completed`` before the runner
  acknowledges the turn.

Cancellation appends ``turn.interrupted`` on the way out. A turn with a start
and neither ending is inferred interrupted on reload, which is the one outcome
no writer can report for itself.

Nothing here invents a tool result. A call the journal has no result for is
recorded as started and nothing more: :func:`drop_unanswered_tool_calls` keeps
it out of the next request, so it is never answered twice and never re-run.

Every write is synchronous. A thread hand-off is the wrong tool on the one path
that matters -- the task is being cancelled, and an ``await`` there is the next
thing to be cancelled. A locked append of a few records costs less than the
tool call that produced them.
"""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger

from opendde_harness.providers import messages as msg
from opendde_harness.session.manager import (
    TOOL_STARTED,
    TURN_COMPLETED,
    TURN_INTERRUPTED,
    TURN_STARTED,
    Session,
    SessionManager,
    new_record_id,
)

#: Set on the live message dict once its sanitized copy is in the journal, so a
#: re-flush of the same tail writes nothing twice. Host-only and never on the
#: wire, like ``_recovery_synthetic``. Empty string for a message the sanitizer
#: refuses (recovery scaffolding), which is journaled as nothing at all.
JOURNAL_KEY = msg.JOURNAL_KEY

#: Sanitizer contract: one live message in, the record to persist out, or None
#: when the message is scaffolding that must never reach disk.
Sanitize = Callable[[dict[str, Any]], "dict[str, Any] | None"]


def strip_inline_images(content: list[Any]) -> list[Any]:
    """Replace inline base64 images with a text placeholder, for persistence.

    Images live for exactly the turn that produced them. Keeping the bytes would
    bloat the session JSONL by megabytes per picture, and every later turn would
    replay them to the model -- paying for an image nobody asked about again.

    A *new* list is returned: the input is the live message the model is still
    working from this turn, and a delivery record only shallow-copies the entry, so
    mutating in place would pull the picture out from under the current request.
    """
    return [msg.text_block("[image]") if msg.is_image(part) else part for part in content]


def sanitized_record(message: dict[str, Any], *, now: Callable[[], Any]) -> dict[str, Any] | None:
    """The durable form of one live message, or ``None`` to record nothing.

    The single rule for what a session file may hold, shared by the turn journal
    and the no-model delivery path so both write the same shape. Three things never
    reach the log: empty-recovery scaffolding, an assistant message that neither
    said nor asked anything, and inline images -- one of those adds megabytes that
    are replayed on every resume and re-fed to the model, and unlike a code bug it
    is not revertible once written. The runtime-context prefix goes too: it is
    rebuilt every turn, and a stored copy is a stale clock in the next prompt.
    """
    from opendde_harness.context_engine.segments import render

    entry = {k: v for k, v in message.items() if k != JOURNAL_KEY}
    role, content = entry.get("role"), entry.get("content")
    if entry.get("_recovery_synthetic"):
        return None  # synthetic recovery nudge — never persist scaffolding
    if role == msg.ASSISTANT and not msg.text_of(entry) and not msg.tool_calls_of(entry):
        return None  # skip empty assistant messages — they poison session context
    if msg.is_tool_result(entry):
        entry["content"] = strip_inline_images(msg.blocks_of(entry))
    if role == msg.USER:
        if isinstance(content, str) and content.startswith(render.RUNTIME_CONTEXT_TAG):
            parts = content.split("\n\n", 1)
            if len(parts) > 1 and parts[1].strip():
                entry["content"] = parts[1]
            else:
                return None
        if isinstance(content, list):
            filtered = [
                block
                for block in strip_inline_images(content)
                if not (msg.is_text(block) and str(block.get("text") or "").startswith(render.RUNTIME_CONTEXT_TAG))
            ]
            if not filtered:
                return None
            entry["content"] = filtered
    entry.setdefault("timestamp", msg.to_ms(now()))
    return entry


def record_delivery(
    session: Session,
    messages: list[dict[str, Any]],
    skip: int,
    *,
    now: Callable[[], Any],
) -> None:
    """Record messages into the session without opening a turn journal.

    The no-model delivery path: a forwarded report is one message and no turn. An
    ordinary turn goes through :class:`TurnJournal`, which records as it goes rather
    than once at the end -- so these are two operations sharing one projection, not
    one operation with a flag.
    """
    for message in messages[skip:]:
        entry = sanitized_record(message, now=now)
        if entry is not None:
            session.record(entry)
    session.updated_at = now()


class TurnJournal:
    """One turn's writes into one session's append-only journal.

    Built per turn, so ``turn_id`` is the turn's identity everywhere downstream
    -- the outbox entry, the resume marker, the recovery prompt.
    """

    def __init__(
        self,
        manager: SessionManager,
        session: Session,
        *,
        sanitize: Sanitize,
        start_index: int,
    ) -> None:
        self.manager = manager
        self.session = session
        self.turn_id = new_record_id()
        self._sanitize = sanitize
        # Where this turn's messages begin in the loop's list. The loop elides
        # and excerpts in place for the wire, which rewrites entries but never
        # adds or drops one, so the offset stays true for the turn's life.
        self._start = start_index
        self._closed = False
        #: The records this turn wrote, in order. What the memory outbox carries
        #: -- the journaled text, not the trimmed copies the last request sent.
        self.written: list[dict[str, Any]] = []

    # ── Lifecycle ──────────────────────────────────────────────────────

    def open(self, messages: list[dict[str, Any]]) -> None:
        """Record ``turn.started`` and the accepted user message.

        Before the first model call, so a turn whose very first call never
        returns still leaves the request that started it.
        """
        self.session.record_lifecycle(
            TURN_STARTED,
            turn_id=self.turn_id,
            generation=self.session.generation,
        )
        self.flush(messages)

    def tool_started(self, call_id: str, name: str, arguments: Any) -> None:
        """Record that a state-changing tool is about to run.

        Written before the call, with the arguments, because the point is the
        case where nothing comes back: the journal then says which tool was
        given what, and the result is unknown rather than absent.
        """
        self.session.record_lifecycle(
            TOOL_STARTED,
            turn_id=self.turn_id,
            call_id=call_id,
            name=name,
            arguments=arguments,
        )
        self._save()

    def close(self, messages: list[dict[str, Any]], *, status: str) -> None:
        """Flush the turn's tail and record how it ended.

        ``status`` is the loop's own: ``"completed"``, ``"interrupted"`` (the
        iteration budget ran out) or ``"error"``. Only a cancellation is
        recorded as ``turn.interrupted``; the other two reached a reply and a
        reload must not offer to resume them.
        """
        if self._closed:
            return
        self.flush(messages)
        self.session.record_lifecycle(
            TURN_COMPLETED,
            turn_id=self.turn_id,
            generation=self.session.generation,
            status=status,
        )
        self._closed = True
        self._save()

    def interrupted(self) -> None:
        """Record ``turn.interrupted``. Called from a cancellation handler.

        Takes no messages: everything the turn completed was flushed as it
        happened, which is the whole point of journaling forward. Failures are
        swallowed -- a cancellation must not be replaced by an OSError.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self.session.record_lifecycle(
                TURN_INTERRUPTED,
                turn_id=self.turn_id,
                generation=self.session.generation,
            )
            self._save()
        except Exception:
            logger.exception("journal: could not record the interruption of turn {}", self.turn_id)

    # ── Messages ───────────────────────────────────────────────────────

    def flush(self, messages: list[dict[str, Any]]) -> None:
        """Persist everything in this turn's tail that is not on record yet.

        Idempotent: a message whose record was written carries its id, so the
        repeated flushes the loop makes after every tool result each write only
        what arrived since. A message the sanitizer refuses is marked too, so it
        is examined once rather than on every flush.
        """
        recorded = 0
        for live in messages[self._start :]:
            if JOURNAL_KEY in live:
                continue
            entry = self._sanitize(live)
            if entry is None:
                live[JOURNAL_KEY] = ""
                continue
            entry["turn_id"] = self.turn_id
            entry["id"] = live[JOURNAL_KEY] = new_record_id()
            self.session.record(entry)
            self.written.append(entry)
            recorded += 1
        if recorded:
            self._save()

    def _save(self) -> None:
        self.manager.save(self.session)


__all__ = ["JOURNAL_KEY", "Sanitize", "TurnJournal"]
