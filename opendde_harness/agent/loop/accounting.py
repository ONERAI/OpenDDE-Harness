"""What a turn's model calls cost, observed once per call.

Every billed call of a turn -- an iteration, the wrap-up after the budget runs
out, a server-side compaction -- reaches :class:`TurnAccounting`, which sums the
turn's costs and keeps a receipt of the last call as it was actually made. The
raw vendor usage, the normalized snapshot and the wire projection are three views
of one call, not interchangeable counters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from opendde_harness.config.schema import ProvidersConfig
    from opendde_harness.providers.base import LLMProvider
    from opendde_harness.providers.rates import Resolved
    from opendde_harness.token_wise.base import UsageSnapshot
    from opendde_harness.token_wise.usage_tracker import SessionUsage


@dataclass
class FinalCall:
    """The turn's last billed model call, as it was actually made.

    A routing strategy moves the model and the catalogue away from the session's
    binding, and the exhaustion wrap-up is a call of its own after the loop has
    already stopped. Work that addresses the same backend after the turn --
    server-side compaction -- reads this rather than rebuilding it, because a
    rebuild addresses a different model with a different catalogue against a
    window that is not the one that filled up.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    window: int = 0
    #: Conversation length when the call was made, so the turn's own tail -- its
    #: answer, and on an exhausted turn the tool results before it -- can be told
    #: from what the call had already sent.
    length: int = 0
    #: That tail, once the turn is over. A compaction marker is filed behind
    #: every message of the turn and stands for all of them, so the compaction
    #: input has to carry this too or the next turn replays a summary with a hole
    #: where the turn's own work was.
    trailing: list[dict[str, Any]] = field(default_factory=list)


def reasoning_tokens_of(usage: dict[str, Any]) -> int:
    """The reasoning share of a call's output, where the vendor reports one.

    OpenAI files it under ``completion_tokens_details.reasoning_tokens``, and the
    adapters report every vendor's equivalent in that shape. Zero where nothing
    is reported, which is not the same as no reasoning.
    """
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        return int(details.get("reasoning_tokens", 0) or 0)
    return int(getattr(details, "reasoning_tokens", 0) or 0) if details is not None else 0


def build_usage_snapshot(
    response: Any,
    model: str,
    session_key: str,
    provider: Any = None,
    providers: Any = None,
) -> "UsageSnapshot":
    """Build a UsageSnapshot from an LLMResponse for TokenWise after-hooks.

    Normalizes input_tokens to *fresh* (non-cached) prompt tokens. The
    ``prompt_tokens`` field has two conventions in the wild:
      - Anthropic native: fresh-only (cache_read/write are separate counts)
      - OpenRouter and gateways like it: total (already includes cache_read
        + cache_write)
    We detect by inequality and subtract when needed so downstream code
    (pricing, telemetry) sees a single consistent semantics.

    ``provider`` is the route that answered, asked for the model's published
    rates. Optional because the compaction path hands this method a call it made
    itself; without one the call carries no list price.

    ``providers`` is the configuration's providers map, and the only thing that
    can say whether this model is billed by subscription -- a plan is the entry's
    ``login``, not a fact about the model. Without it every model reads as
    metered, which reports a per-token figure for a call a subscription already
    paid for.
    """
    from opendde_harness.providers.rates import is_plan_billed
    from opendde_harness.token_wise.base import snapshot_from_usage

    snapshot = snapshot_from_usage(response.usage, model, session_key)
    # What the backend itself priced the call at, when it said. The model service
    # reports pi's own figure, computed from the catalogue that actually served
    # the request, and nothing else here prices a call. Two things that figure
    # does not beat:
    #
    #   * a plan. A subscription is the price, and a per-token figure for a turn
    #     nobody is billed per token for is the wrong number whoever computed it,
    #     so ``is_plan_billed`` is asked first;
    #   * its own zero. pi carries no prices for a provider this project declared
    #     (``configure_payload`` sends none), and prices such a call at 0. That is
    #     "nobody knows", not "free", and reporting it as a number would show
    #     `$0.000` for a self-hosted model whose spend is genuinely unknown. Below
    #     zero there is nothing left to ask, so the figure stays None -- which is
    #     what unknown reads as everywhere else here.
    #
    # Left as None for a plan-billed provider either way: the field is optional
    # all the way to the status bar, which renders it only when it is a number.
    reported = (response.usage or {}).get("cost")
    priced = isinstance(reported, (int, float)) and not isinstance(reported, bool) and reported > 0
    snapshot.estimated_cost_usd = float(reported) if priced and not is_plan_billed(model, providers) else None
    # What the call is worth at the vendor's published price, whoever is billing.
    # On a plan there is no per-token figure to report, and a status line that
    # answers `$0.000` reads as free rather than as covered -- so the line shows
    # this and says `(sub)` beside it, which is what pi does. Priced here, next to
    # the figure it stands in for, so nothing downstream has to price it again.
    rates = provider.list_rates() if provider is not None else None
    if rates is not None:
        snapshot.list_cost_usd = rates.cost(
            snapshot.input_tokens,
            snapshot.output_tokens,
            snapshot.cache_read_tokens,
            snapshot.cache_write_tokens,
        )
    return snapshot


def wire_usage(
    response: Any,
    snapshot: Any,
    window: Any,
    *,
    cost_usd: float | None,
    list_cost_usd: float | None,
) -> dict[str, Any]:
    """The ``usage`` a turn puts on the wire, for one call and the turn's costs.

    Token and context figures describe this call, because that is what the status
    bar means by "the last call". The two costs are the turn's running totals,
    summed per call: a long call was priced at its own long-context tier when it
    landed, which no rate applied to a session total reproduces.
    """
    reported_usage = response.usage or {}
    prompt_tokens = int(reported_usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(reported_usage.get("completion_tokens", 0) or 0)
    # The share of this call's prompt the backend served from its cache, on the
    # snapshot's fresh-plus-cached total so the figure means the same whichever
    # convention the vendor reports prompt_tokens in. None when nothing was sent
    # or the vendor stated no figure: an unreported call is not a miss.
    cached = snapshot.cache_read_tokens
    prompt_total = snapshot.input_tokens + cached + snapshot.cache_write_tokens
    reported = snapshot.cache_reported and prompt_total > 0
    context_max = (window.tokens or 0) if window is not None else 0
    context_used = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(reported_usage.get("total_tokens", 0) or 0),
        "cache_hit_percent": round(100 * cached / prompt_total) if reported else None,
        "cost_usd": cost_usd,
        "list_cost_usd": list_cost_usd,
        "context_max": context_max,
        "context_source": window.source if window is not None else "",
        "context_used": context_used,
        "context_percent": round(100 * context_used / context_max) if context_max else 0,
    }


def session_wire_usage(session: "SessionUsage | None") -> dict[str, Any]:
    """What the session has spent, to travel beside what the turn has spent.

    Two questions, and one number used to answer both. :func:`wire_usage`
    describes one turn -- the tokens and context of its last call, the cost of
    all of them -- and a footer that shows that as "what this session is
    costing" answers a thirty-call turn with the turn's figure while ``/status``
    answers with the session's. The fields here are the session's, taken from
    the tracker that ``/status`` also reads, so the two render the same dollar
    figure for the same session or the test that compares them fails.

    ``session_input_tokens`` counts cached prompt tokens in, which is what both
    the footer's ``↑`` and the ``/status`` token line mean by input. The two
    costs are None where nothing states one: a plan reports no vendor price, and
    a model in no price table has no list price -- neither is zero.

    Empty for a caller with no tracker to ask. The turn's own fields still
    travel, so a surface that shows only those is unaffected.
    """
    if session is None:
        return {}
    totals = session.totals
    return {
        "session_input_tokens": totals.input_tokens + totals.cache_read_tokens + totals.cache_write_tokens,
        "session_output_tokens": totals.output_tokens,
        "session_cost_usd": totals.estimated_cost_usd,
        "session_list_cost_usd": totals.list_cost_usd,
        "session_calls": session.counts.calls,
    }


class TurnAccounting:
    """One turn's billed calls: the sums, the wire view and the last receipt.

    Per turn, not per loop: the loop is a long-lived singleton shared across
    sessions, so a counter on it would carry one turn's spend into the next.
    Every call the turn makes is observed here exactly once -- iterations, the
    exhaustion wrap-up and the after-turn compaction alike -- which is what keeps
    the turn footer and ``/status`` saying the same thing about the same call.
    """

    def __init__(
        self,
        *,
        session_key: str,
        providers: "ProvidersConfig",
        provider: Callable[[], "LLMProvider"],
        resolve_window: Callable[[str], "Resolved"],
        note_window: Callable[[str, "Resolved"], None],
    ) -> None:
        self._session_key = session_key
        self._providers = providers
        self._provider = provider
        self._resolve_window = resolve_window
        self._note_window = note_window
        #: Vendor output for the whole turn, summed per call. A tool-using turn
        #: makes several billed calls, and reporting the last one's figure as the
        #: turn's is what this replaces.
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        #: None until some call states a figure: a plan reports no vendor price at
        #: all, and zero would read as free. Summed per call rather than priced
        #: once over the totals, because each call was billed at the tier its own
        #: prompt fell in.
        self.cost_usd: float | None = None
        self.list_cost_usd: float | None = None
        #: The last billed call's wire projection, which is what a status bar
        #: means by "the last call".
        self.usage: dict[str, Any] = {}
        #: That call as it was made. Compaction is triggered by its numbers, so it
        #: has to be sent to its model with its catalogue and its message list --
        #: not the session's binding and the full tool registry.
        self.last_call: FinalCall | None = None
        #: How many calls reported usage, so a caller can assert that a synthetic
        #: reply added none.
        self.billed_calls = 0
        #: The normalized usage of the last billed call, already priced. Kept so a
        #: call made inside the turn but outside the iteration's after-hook -- the
        #: exhaustion wrap-up -- can be handed to the session tracker without
        #: being priced a second time from a second table. None until a call is
        #: billed.
        self.last_snapshot: "UsageSnapshot | None" = None

    def snapshot(self, response: Any, model: str) -> "UsageSnapshot":
        """This call's normalized usage, priced at this call's own tier.

        Built for every call, billed or not: the strategies' after-hook observes
        each response, and a call the vendor reported nothing for is still a call.
        """
        return build_usage_snapshot(response, model, self._session_key, self._provider(), self._providers)

    def record(
        self,
        response: Any,
        snapshot: "UsageSnapshot",
        *,
        model: str,
        sent: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        history_len: int = 0,
    ) -> bool:
        """Add one call to the turn, and make it the receipt. False if unbilled.

        A call whose usage the vendor never stated adds nothing and replaces
        nothing: the turn's figures still describe the last call that *was*
        billed, and a synthetic reply the loop wrote itself is not a call at all.
        """
        if not response.usage:
            return False
        self.billed_calls += 1
        self.completion_tokens += int(response.usage.get("completion_tokens", 0) or 0)
        self.reasoning_tokens += reasoning_tokens_of(response.usage)
        if snapshot.estimated_cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + snapshot.estimated_cost_usd
        if snapshot.list_cost_usd is not None:
            self.list_cost_usd = (self.list_cost_usd or 0.0) + snapshot.list_cost_usd
        # The call's own model, which is not always the turn's (a strategy rewrote
        # it, or the chain fell back). Unknown to the catalogue, 0 tells the UI to
        # show its empty state rather than a number that is not this model's.
        window = self._resolve_window(model)
        self._note_window(model, window)
        # Snapshotted, not referenced: the loop appends this call's own answer to
        # the very list it sent, and compaction has to know the difference.
        self.last_call = FinalCall(
            messages=list(sent or []),
            tools=list(tools or []),
            model=model,
            prompt_tokens=int(response.usage.get("prompt_tokens", 0) or 0),
            window=window.tokens or 0,
            length=history_len,
        )
        self.last_snapshot = snapshot
        self.usage = wire_usage(response, snapshot, window, cost_usd=self.cost_usd, list_cost_usd=self.list_cost_usd)
        return True


def charge(usage: dict[str, Any], snapshot: "UsageSnapshot") -> None:
    """Charge a call made after the turn's own figures were built.

    The server-side compaction: a billed call on the full prompt, made because of
    this turn and before its figures reach the wire. It adds cost without
    replacing the answer's token and context fields -- those describe the prompt
    the compaction just replaced. The same two sums as a turn's own call, spelled
    once, because the turn's accumulator is gone by the time this runs.
    """
    if snapshot.estimated_cost_usd is not None:
        usage["cost_usd"] = (usage.get("cost_usd") or 0.0) + snapshot.estimated_cost_usd
    if snapshot.list_cost_usd is not None:
        usage["list_cost_usd"] = (usage.get("list_cost_usd") or 0.0) + snapshot.list_cost_usd


__all__ = [
    "FinalCall",
    "TurnAccounting",
    "build_usage_snapshot",
    "charge",
    "reasoning_tokens_of",
    "session_wire_usage",
    "wire_usage",
]
