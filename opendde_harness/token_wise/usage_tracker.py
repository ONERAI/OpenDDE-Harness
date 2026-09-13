"""UsageTracker — records token usage and cost for every LLM call.

Accumulates into three tiers:
    - ``per_session[session_key]`` — cumulative usage within one session
    - ``per_day[date]``             — daily roll-up, useful for budgeting
    - ``total``                      — lifetime of this process

Every call is also appended to ``{telemetry_dir}/usage-YYYY-MM-DD.jsonl``
as a single JSON object per line, enabling post-hoc analysis with ``jq``.

The tracker is purely a recorder — it never modifies the outgoing request,
so its ``before_llm_call`` inherits the default no-op pass-through.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from opendde_harness.token_wise.base import TokenStrategy, UsageSnapshot


def _default_telemetry_dir() -> Path:
    return Path.home() / ".opendde_harness" / "telemetry"


@dataclass
class CallCounts:
    """A session's calls, and the subset the provider stated a cache figure for."""

    calls: int = 0
    cache_reported: int = 0
    #: Prompt tokens (fresh + cached + written) across the reported calls only.
    cache_reported_prompt_tokens: int = 0


@dataclass
class SessionUsage:
    """One session's accounting, copied out together so a reader on another
    thread sees one moment rather than three."""

    totals: UsageSnapshot
    by_model: dict[str, UsageSnapshot]
    counts: CallCounts


class UsageTracker(TokenStrategy):
    """Observes every LLM call; persists & rolls up token and cost stats."""

    name = "usage_tracker"

    def __init__(
        self,
        telemetry_dir: Path | None = None,
        flush_every: int = 1,
        persist: bool = True,
    ):
        """Create a tracker.

        Args:
            telemetry_dir: Where to write ``usage-YYYY-MM-DD.jsonl``. Defaults
                to ``~/.opendde_harness/telemetry``.
            flush_every: Buffer N calls before writing to disk. 1 = write every
                call (safest, default). Larger values amortize IO.
            persist: If False, accumulate in memory only (useful for tests).
        """
        self.telemetry_dir = telemetry_dir or _default_telemetry_dir()
        self.flush_every = max(1, flush_every)
        self.persist = persist

        self.per_session: dict[str, UsageSnapshot] = {}
        # The same totals split by the model that produced them: a session
        # that switched models is priced model by model, not at the last one's rate.
        self.per_session_model: dict[str, dict[str, UsageSnapshot]] = {}
        # How many of a session's calls the provider stated a cache figure
        # for, and how much prompt those calls carried: a hit rate is only a
        # rate over the calls that were counted.
        self.per_session_calls: dict[str, CallCounts] = {}
        self.per_day: dict[date, UsageSnapshot] = {}
        self.total: UsageSnapshot = UsageSnapshot(model="__total__")

        self._call_count: int = 0
        self._buffer: list[dict[str, Any]] = []

    # ---- TokenStrategy hook ----

    async def after_llm_call(self, response: dict[str, Any], usage: UsageSnapshot) -> None:
        """Record one LLM call made by the loop."""
        self._record(usage)

    def record_snapshot(self, usage: UsageSnapshot) -> None:
        """Record a call made on the current turn's behalf outside the loop.

        A server-side compaction calls the provider directly, so the loop's
        after-hook never sees it. The loop builds its snapshot the same way it
        builds a turn's own -- same model, same rates -- and hands it here, so
        ``/status`` and the turn footer count the same call at the same figure.
        """
        self._record(usage)

    def _record(self, usage: UsageSnapshot) -> None:
        # ``list_cost_usd`` arrives already priced or not at all. The loop
        # prices every snapshot, at the rates the model layer reports for the
        # model that answered and at the tier the call's own input reaches; a
        # second pass here could only price it from a second table, which is how
        # the footer and ``/status`` came to disagree.
        self._call_count += 1
        self._accumulate(usage)

        if self.persist:
            self._buffer.append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    **asdict(usage),
                }
            )
            if self._call_count % self.flush_every == 0:
                self._flush()

    # ---- Public introspection ----

    def snapshot(self, session_key: str | None = None) -> UsageSnapshot:
        """Return a *copy* of the session accumulator, or the lifetime total."""
        if session_key is not None:
            src = self.per_session.get(session_key) or UsageSnapshot(model="__empty__", session_key=session_key)
        else:
            src = self.total
        return self._copy(src)

    def by_model(self, session_key: str) -> dict[str, UsageSnapshot]:
        """Copies of the session's accumulators, one per model, in first-use order."""
        return {model: self._copy(acc) for model, acc in self.per_session_model.get(session_key, {}).items()}

    def call_counts(self, session_key: str) -> "CallCounts":
        """How many calls the session made, and how many carried a cache figure."""
        counts = self.per_session_calls.get(session_key)
        return CallCounts() if counts is None else replace(counts)

    def seed_session(self, usage: UsageSnapshot, counts: CallCounts) -> bool:
        """Adopt a stored session's own history as the totals this process opens on.

        A resumed session's earlier turns ran in another process, so nothing
        here counted them: ``/status`` opened a conversation with forty calls
        behind it at zero, and the footer beside it said ``$0.000``. The session
        file answers for them -- every stored assistant record carries the usage
        and the price of the call that produced it -- and this is where that
        answer is adopted, once, so the turns that follow add to the real total
        rather than starting a second one.

        Only the per-session tiers. ``per_day`` and ``total`` are what *this*
        process has spent today, and the earlier calls are in the telemetry of
        the day they were made; nothing is written to the log here either, for
        the same reason -- these are not new calls, and counting them as such
        would double every resumed session in the record.

        False, and nothing written, when the session already has totals. A
        process that has run a turn on this key counted that turn through its own
        calls, and adding the file's figures on top would report one spend twice.
        """
        key = usage.session_key or "__no_session__"
        if key in self.per_session:
            return False
        self._accumulate_session(key, usage)
        self.per_session_calls[key] = replace(counts)
        return True

    def session_usage(self, session_key: str) -> SessionUsage:
        """The session's totals, per-model totals and call counts, copied together.

        Called on the thread that also records, so the three agree; a reader
        elsewhere takes this copy rather than the live dictionaries.
        """
        return SessionUsage(self.snapshot(session_key), self.by_model(session_key), self.call_counts(session_key))

    def close(self) -> None:
        """Flush any remaining buffered rows to disk."""
        self._flush()

    # ---- Internals ----

    def _accumulate(self, u: UsageSnapshot) -> None:
        key = u.session_key or "__no_session__"
        self._accumulate_session(key, u)
        counts = self.per_session_calls.setdefault(key, CallCounts())
        counts.calls += 1
        if u.cache_reported:
            counts.cache_reported += 1
            counts.cache_reported_prompt_tokens += u.input_tokens + u.cache_read_tokens + u.cache_write_tokens

        today = date.today()
        day_acc = self.per_day.get(today)
        if day_acc is None:
            day_acc = UsageSnapshot(model="__day__")
            self.per_day[today] = day_acc
        self._add_into(day_acc, u)

        self._add_into(self.total, u)

    def _accumulate_session(self, key: str, u: UsageSnapshot) -> None:
        """Add one call into the session's totals and its own model's, nothing else.

        The two tiers a session's own report reads (``/status``, the footer),
        apart from the day and lifetime roll-ups: a seeded history belongs in
        these and in neither of those.
        """
        session_acc = self.per_session.get(key)
        if session_acc is None:
            session_acc = UsageSnapshot(model=u.model, session_key=key)
            self.per_session[key] = session_acc
        self._add_into(session_acc, u)
        model_acc = self.per_session_model.setdefault(key, {}).get(u.model)
        if model_acc is None:
            model_acc = UsageSnapshot(model=u.model, session_key=key)
            self.per_session_model[key][u.model] = model_acc
        self._add_into(model_acc, u)

    @staticmethod
    def _add_into(acc: UsageSnapshot, add: UsageSnapshot) -> None:
        acc.input_tokens += add.input_tokens
        acc.output_tokens += add.output_tokens
        acc.cache_read_tokens += add.cache_read_tokens
        acc.cache_write_tokens += add.cache_write_tokens
        acc.reasoning_tokens += add.reasoning_tokens
        if add.estimated_cost_usd is not None:
            # A plan-billed call contributes tokens but no money; summing it as
            # zero would read as "these calls were free".
            acc.estimated_cost_usd = (acc.estimated_cost_usd or 0.0) + add.estimated_cost_usd
        if add.list_cost_usd is not None:
            acc.list_cost_usd = (acc.list_cost_usd or 0.0) + add.list_cost_usd

    @staticmethod
    def _copy(src: UsageSnapshot) -> UsageSnapshot:
        return replace(src)

    def _flush(self) -> None:
        if not self._buffer or not self.persist:
            self._buffer.clear()
            return
        try:
            self.telemetry_dir.mkdir(parents=True, exist_ok=True)
            path = self.telemetry_dir / f"usage-{date.today().isoformat()}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                for row in self._buffer:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._buffer.clear()
        except Exception as e:
            logger.warning("UsageTracker flush failed ({}); dropping {} rows", e, len(self._buffer))
            self._buffer.clear()
