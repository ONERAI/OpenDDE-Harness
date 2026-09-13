"""Spine wiring for the TUI RPC turn path: the runner (a TuiTurnRunner driving
the agent loop's native run_turn with stream=True), the outlet that maps each
spine event to its wire event (token.delta / thinking.delta / tool.*), and the
sink that fires ``message.complete`` / ``error`` after the render barrier.

The TUI runs turns through spine (submit -> lane -> run_turn -> hub -> outlet).
All of token/reasoning/tool/Text flow through the hub to the TuiOutlet, so they
share one per-outlet FIFO. spine never imports tui_rpc; tui_rpc imports spine.

Why ``message.complete`` is fired from the sink (not from a stream-close): it is
an unconditional per-turn signal — the front-end clears its turn slot on it, so a
turn that streams nothing (empty reply, tool-only) must still emit it or the UI
wedges. The sink awaits the hub's per-conversation delivery fence first, so it
lands after the turn's last ``token.delta``; an empty turn never built a queue,
so the barrier returns at once. The fence is bounded and reports whether the
output actually rendered, so a wedged or failing outlet cannot leave the turn
without a ``message.complete``. This is the REPL's ``result() -> wait_idle``
render barrier moved into the sink.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from opendde_harness.agent.loop.accounting import session_wire_usage
from opendde_harness.agent.spine_runner import AgentTurnRunner
from opendde_harness.agent.tools.approval import ApprovalResponder
from opendde_harness.spine import (
    Deliverable,
    EpisodeStart,
    Notice,
    NoticeKind,
    Origin,
    OriginPools,
    Reasoning,
    Scheduler,
    Text,
    ToolEvent,
    ToolPhase,
    TurnEnded,
    TurnFailed,
    TurnOutcome,
    TurnRequest,
    TurnRetry,
    TurnStarted,
    TurnUsage,
)
from opendde_harness.spine.delivery import Capabilities, DeliveryHub, DeliveryReport
from opendde_harness.spine.events import TurnEvent
from opendde_harness.spine.runner import Drain, Emit
from opendde_harness.tui_rpc.subscriptions import SubscriptionEmitter

_TURN_FAILED_CODE = -32099


def _conversation_id(req: TurnRequest) -> str:
    return req.conversation or f"{req.source.channel}:{req.source.chat_id}"


class TuiTurnRunner(AgentTurnRunner):
    """Runs a TUI turn through the agent loop's native run_turn (stream=True), so
    token/reasoning/tool/Text all flow through the hub to the TuiOutlet (one
    per-outlet FIFO — no dual path). Two TUI-specific bits the generic runner
    does not carry:

    - it keeps ``outcome.usage_detail`` so the sink can attach the full usage
      (cost / context, richer than the three-field TurnOutcome.usage) to
      ``message.complete``, and adds the session's running totals from the same
      tracker ``/status`` reads, so the footer's dollar figure and ``/status``'s
      are one number rather than two that disagree;
    - it fires the synthetic tool.complete when the turn replied via the
      message tool (the loop's general tool path skips the message tool), so the
      UI records that the agent acted.
    """

    def __init__(
        self,
        agent_loop: Any,
        emitter: SubscriptionEmitter,
        usages: dict[str, dict[str, Any]],
        turn_ids: dict[str, str],
        approval_responder: ApprovalResponder | None = None,
    ) -> None:
        super().__init__(agent_loop, stream=True)
        self._emitter = emitter
        self._usages = usages
        self._turn_ids = turn_ids
        self._approval_responder = approval_responder

    async def run(self, req: TurnRequest, emit: Emit, drain: Drain) -> TurnOutcome:
        cid = _conversation_id(req)
        tools = getattr(self._loop, "tools", None)
        # Approval capability is rebound for every turn. Only USER origin
        # receives the TUI responder; background origins share this process
        # but must still fail closed as non-interactive.
        # The IDs bind any response to this exact conversation and turn.
        #
        # Every tool that can ask, not the shell alone: writing AGENTS.md or
        # ODH.md is a privileged act too, and a gate nobody binds is a gate
        # that refuses every time. Duck-typed, because the alternative is a
        # base class that most tools would implement emptily.
        for tool in _approvable(tools):
            tool.start_approval_turn(
                self._approval_responder if req.origin is Origin.USER else None,
                conversation_id=cid,
                turn_id=self._turn_ids.get(cid, ""),
            )
        outcome = await self._loop.run_turn(req, emit, drain, stream=True)

        # The turn's own figures, plus the session's. Read here rather than in
        # the sink because this is where the loop is: the last call and any
        # compaction made on the turn's behalf are recorded by the time run_turn
        # returns, so the totals are complete, and the sink only forwards them.
        usage = dict(outcome.usage_detail)
        usage.update(session_wire_usage(_session_usage(self._loop, cid)))
        self._usages[cid] = usage
        return outcome


def _session_usage(agent_loop: Any, session_key: str) -> Any:
    """This session's totals from the loop's usage tracker, or None without one.

    ``session_key`` is the conversation id, which is the key the loop records
    every call of the turn under (``AgentLoop.run_turn`` sets the same string on
    ``CURRENT_SESSION_KEY``); asking under any other spelling would answer for a
    session with no calls in it.
    """
    tracker = getattr(agent_loop, "usage_tracker", None)
    if tracker is None:
        return None
    try:
        return tracker.session_usage(session_key)
    except Exception:  # pragma: no cover - a tracker that cannot answer reports nothing
        logger.debug("tui usage: could not read session totals for {!r}", session_key)
        return None


def _approvable(tools: Any) -> list[Any]:
    """Every registered tool that can be handed an approval transport.

    Asks the registry by name and fetches each one, which is all this module
    has ever assumed of it.
    """
    if tools is None:
        return []

    found = []

    for name in getattr(tools, "tool_names", []):
        tool = tools.get(name)

        if callable(getattr(tool, "start_approval_turn", None)):
            found.append(tool)

    return found


#: The Notice kinds the TUI is shown. PROGRESS and TOOL_HINT are the turn
#: narrating itself, which the transcript already does; these two are facts
#: about the turn the user would otherwise never learn.
_NOTICE_ON_THE_WIRE = frozenset({NoticeKind.MODEL_FALLBACK, NoticeKind.DELIVERY_FAILED})


class TuiOutlet:
    """The TUI's send surface. Maps each spine event to its wire event on the
    conversation's subscription: streamed token content via ``send_stream_chunk``
    (-> token.delta), and the discrete deliverables via ``deliver`` (Reasoning ->
    thinking.delta, ToolEvent -> tool.start / tool.complete, a non-streamed Text
    -> a token.delta, TurnRetry -> turn.retry, TurnUsage -> turn.usage). The turn's completion (``message.complete``) and failure
    (``error``) are emitted by the sink after the render barrier. A Notice rides
    turn.notice, but only for the kinds that are the user's business rather than
    the turn's chatter: see ``_NOTICE_ON_THE_WIRE``. MediaOut is eaten — the wire
    protocol has no event for it and the TUI shows no tool media today (a known
    gap, deferred)."""

    def __init__(self, channel: str, emitter: SubscriptionEmitter) -> None:
        self.name = channel
        self.capabilities = Capabilities(streaming=True)
        self._emitter = emitter

    async def deliver(self, out: Deliverable) -> None:
        cid = out.conversation_id
        if isinstance(out, Reasoning):
            if out.content:
                await self._emitter.emit(cid, {"type": "thinking.delta", "payload": {"text": out.content}})
        elif isinstance(out, ToolEvent):
            if out.phase is ToolPhase.START:
                await self._emitter.emit(
                    cid,
                    {
                        "type": "tool.start",
                        "payload": {
                            "tool_call_id": out.tool_call_id,
                            "name": out.name,
                            "arguments": out.arguments or {},
                            "display": out.display,
                        },
                    },
                )
            else:
                await self._emitter.emit(
                    cid,
                    {
                        "type": "tool.complete",
                        "payload": {
                            "tool_call_id": out.tool_call_id,
                            "result_preview": out.result_preview,
                            "truncated": out.truncated,
                        },
                    },
                )
        elif isinstance(out, Text):
            # A non-streamed reply (clarification / hook short-circuit / empty
            # fallback) rides one token.delta into the same buffer the streamed
            # reply uses, so message.complete finalizes it like any other text.
            if out.content:
                await self._emitter.emit(cid, {"type": "token.delta", "payload": {"text": out.content}})
        elif isinstance(out, EpisodeStart):
            # Boundary marker; the TUI buckets this model call's reasoning +
            # text + tools into one collapsible episode.
            await self._emitter.emit(cid, {"type": "episode.start", "payload": {"index": out.index}})
        elif isinstance(out, TurnUsage):
            await self._emitter.emit(
                cid,
                {
                    "type": "turn.usage",
                    "payload": {
                        "completion_tokens": out.completion_tokens,
                        "reasoning_tokens": out.reasoning_tokens,
                        "calls": out.calls,
                    },
                },
            )
        elif isinstance(out, Notice):
            if out.kind in _NOTICE_ON_THE_WIRE and out.detail:
                await self._emitter.emit(
                    cid,
                    {"type": "turn.notice", "payload": {"kind": str(out.kind), "text": out.detail}},
                )
        elif isinstance(out, TurnRetry):
            await self._emitter.emit(
                cid,
                {
                    "type": "turn.retry",
                    "payload": {
                        "attempt": out.attempt,
                        "total": out.total,
                        "reason": out.reason,
                        "discard": out.discard,
                    },
                },
            )
        # MediaOut, and every other Notice kind: eaten (no wire event today).

    async def send_stream_chunk(self, chat_id: str, stream_id: str, delta: str, *, done: bool = False) -> None:
        if done:
            # The front-end has no stream-done event; the turn is finalized by
            # message.complete (emitted by the sink). done=True only lets the hub
            # close its stream state.
            return
        if not delta:
            return
        await self._emitter.emit(stream_id, {"type": "token.delta", "payload": {"text": delta}})

    async def emit_complete(self, conversation_id: str, turn_id: str | None, usage: dict[str, Any]) -> None:
        await self._emitter.emit(
            conversation_id,
            {"type": "message.complete", "payload": {"turn_id": turn_id, "usage": usage}},
        )

    async def emit_notice(self, conversation_id: str, kind: NoticeKind, text: str) -> None:
        """A fact about the turn the transcript cannot show, sent straight to the
        subscription rather than through the hub — the hub's queue for this
        conversation is what a delivery failure is about."""
        await self._emitter.emit(
            conversation_id,
            {"type": "turn.notice", "payload": {"kind": str(kind), "text": text}},
        )

    async def emit_error(self, conversation_id: str, code: int, message: str, reason: str, detail: str = "") -> None:
        payload: dict[str, Any] = {"code": code, "message": message, "reason": reason}
        if detail:
            payload["detail"] = detail
        await self._emitter.emit(conversation_id, {"type": "error", "payload": payload})


def _make_tui_sink(
    hub: DeliveryHub,
    outlet: TuiOutlet,
    channel: str,
    turn_ids: dict[str, str],
    usages: dict[str, dict[str, Any]],
    on_turn_end: Callable[[str], None] | None,
) -> Callable[[TurnEvent], Awaitable[None]]:
    """Adapt the hub into the scheduler's EventSink for the TUI. Deliverables
    route through the hub; a turn's end fires message.complete / error after the
    render barrier (so they land after the last token.delta). ``on_turn_end`` is
    called at each turn exit (before message.complete) so turn.send's active-turn
    slot is cleared before the front-end is told it may submit the next turn.
    This sink is build_tui's alone — the CLI keeps its own lifecycle-dropping
    sink."""

    async def _finish(conversation_id: str) -> DeliveryReport:
        # close_stream clears the hub's per-stream state (so the next turn on this
        # conversation reopens cleanly); the fence then blocks until every queued
        # token.delta of *this* conversation has been delivered or explicitly
        # failed — an empty turn never built a queue, so it returns at once. The
        # fence is bounded and per-conversation: neither a wedged outlet nor
        # another chat's backlog can hold message.complete.
        await hub.close_stream(conversation_id)
        return await hub.fence(channel, conversation_id)

    async def _report_lost_output(conversation_id: str, report: DeliveryReport) -> None:
        """Tell the user their output did not all reach the screen. Best-effort:
        message.complete matters more than this notice, so a failure here is
        logged, never raised."""
        logger.warning(
            "tui delivery incomplete: conversation={!r} failed={} timed_out={}",
            conversation_id,
            report.failed,
            report.timed_out,
        )
        detail = "Some output could not be shown" + (" (delivery timed out)" if report.timed_out else "")
        try:
            await outlet.emit_notice(conversation_id, NoticeKind.DELIVERY_FAILED, detail)
        except Exception:
            logger.exception("tui delivery notice failed: conversation={!r}", conversation_id)

    def _drop(conversation_id: str) -> None:
        turn_ids.pop(conversation_id, None)
        usages.pop(conversation_id, None)
        if on_turn_end is not None:
            on_turn_end(conversation_id)

    async def sink(event: TurnEvent) -> None:
        if isinstance(event, TurnEnded):
            report = await _finish(event.conversation_id)
            if not report.ok:
                await _report_lost_output(event.conversation_id, report)
            turn_id = turn_ids.get(event.conversation_id)
            usage = usages.get(event.conversation_id) or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            _drop(event.conversation_id)
            await outlet.emit_complete(event.conversation_id, turn_id, usage)
            return
        if isinstance(event, TurnFailed):
            await _finish(event.conversation_id)  # a failed turn's own error is the report
            _drop(event.conversation_id)
            # A cancelled turn's error is emitted by turn.cancel, not here, to
            # avoid a double error event.
            if not event.cancelled:
                await outlet.emit_error(
                    event.conversation_id,
                    _TURN_FAILED_CODE,
                    "turn_failed",
                    "internal",
                    event.error or "",
                )
            return
        if isinstance(event, TurnStarted):
            # message.start is emitted by turn.send (it owns the turn_id).
            return
        await hub.dispatch(event)

    return sink


def build_tui(
    agent_loop: Any,
    emitter: SubscriptionEmitter,
    *,
    channel: str = "tui",
    on_turn_end: Callable[[str], None] | None = None,
    approval_responder: ApprovalResponder | None = None,
    user_pool: int = 1,
    system_pool: int = 1,
) -> tuple[Scheduler, DeliveryHub, dict[str, str], Callable[[], Awaitable[None]]]:
    """Wire the spine pieces a TUI turn flows through: a hub with the channel's
    TuiOutlet, and a Scheduler whose runner streams the agent loop and whose sink
    fires message.complete / error after the render barrier. Returns those plus
    the ``turn_ids`` map (turn.send binds conversation_id -> turn_id so the sink
    can attach it to message.complete) and a ``teardown`` the caller awaits on
    exit (stop the scheduler, then close the hub's workers). ``on_turn_end`` lets
    turn.send drop its active-turn slot at each turn exit.

    ``approval_responder`` is an interactive capability, not a process-wide
    permission. The runner binds it only to USER-origin turns and explicitly
    revokes it for background origins."""
    hub = DeliveryHub()
    outlet = TuiOutlet(channel, emitter)
    hub.register(outlet)
    turn_ids: dict[str, str] = {}
    usages: dict[str, dict[str, Any]] = {}
    scheduler = Scheduler(
        TuiTurnRunner(
            agent_loop,
            emitter,
            usages,
            turn_ids,
            approval_responder=approval_responder,
        ),
        OriginPools(user=user_pool, system=system_pool),
        _make_tui_sink(hub, outlet, channel, turn_ids, usages, on_turn_end),
    )

    async def teardown() -> None:
        await scheduler.shutdown(grace=0.0)
        await hub.aclose()

    return scheduler, hub, turn_ids, teardown
