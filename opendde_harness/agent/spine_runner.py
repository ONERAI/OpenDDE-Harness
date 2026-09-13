"""The agent/spine boundary: a ``TurnRunner`` that holds a loop, and the adapter
that turns the loop's per-turn callbacks into spine events on one ``emit``.

It lives on the agent side because it holds the loop; spine never imports the
agent. ``stream`` is the canon Q2-D assembly switch: a streaming outlet (TUI)
passes True so the reply streams as StreamDelta and dissolves, a non-streaming
outlet (REPL) passes False so the reply is one Text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opendde_harness.spine.events import (
    EpisodeStart,
    MediaOut,
    Notice,
    NoticeKind,
    Reasoning,
    StreamDelta,
    Text,
    ToolEvent,
    ToolPhase,
    TurnRetry,
    TurnUsage,
)
from opendde_harness.spine.message import Media

if TYPE_CHECKING:
    from opendde_harness.agent.loop import AgentLoop
    from opendde_harness.spine.runner import Drain, Emit, TurnOutcome
    from opendde_harness.spine.turn import TurnRequest


class TurnEvents:
    """One turn's agent callbacks, fanned onto the spine's single ``emit``.

    The loop takes a callback per kind of output and knows nothing about events;
    the spine takes events in one ordered stream and knows nothing about the loop.
    This is the whole translation, and it keeps one piece of state: whether
    anything streamed, which is what decides if the reply also goes out as a
    closing ``Text``. Emitting both would show a TUI the answer twice.
    """

    def __init__(self, emit: "Emit") -> None:
        self._emit = emit
        #: True once a token reached the outlet. Reset by a discarding retry,
        #: because the attempt that streamed has been declared void.
        self.streamed = False

    async def token(self, text: str) -> None:
        if not text:
            return
        self.streamed = True
        await self._emit(StreamDelta(delta=text))

    async def reasoning(self, text: str) -> None:
        if text:
            await self._emit(Reasoning(content=text))

    async def episode(self, index: int) -> None:
        await self._emit(EpisodeStart(index=index))

    async def retry(self, attempt: int, total: int, reason: str, discard: bool) -> None:
        """A new attempt at the same call. ``discard`` voids what was shown."""
        if discard:
            self.streamed = False
        await self._emit(TurnRetry(attempt=attempt, total=total, reason=reason, discard=discard))

    async def usage(self, completion_tokens: int, reasoning_tokens: int, calls: int) -> None:
        await self._emit(TurnUsage(completion_tokens=completion_tokens, reasoning_tokens=reasoning_tokens, calls=calls))

    async def tool(self, phase: str, info: dict[str, Any]) -> None:
        if phase == "start":
            await self._emit(
                ToolEvent(
                    phase=ToolPhase.START,
                    tool_call_id=info["tool_call_id"],
                    name=info["name"],
                    arguments=info["arguments"],
                    display=info.get("display"),
                )
            )
        else:
            await self._emit(
                ToolEvent(
                    phase=ToolPhase.COMPLETE,
                    tool_call_id=info["tool_call_id"],
                    result_preview=info["result_preview"],
                    truncated=info["truncated"],
                )
            )

    async def progress(self, text: str, tool_hint: bool = False) -> None:
        """Progress and tool hints stay distinct so an outlet can gate each on its
        own config flag (send_progress vs send_tool_hints). Outlets that render
        neither eat both kinds anyway."""
        if text:
            await self._emit(Notice(kind=NoticeKind.TOOL_HINT if tool_hint else NoticeKind.PROGRESS, detail=text))

    async def notice(self, kind: NoticeKind, detail: str) -> None:
        await self._emit(Notice(kind=kind, detail=detail))

    async def media(self, paths: list[str]) -> None:
        """Independent of the stream, and emitted before the reply."""
        await self._emit(
            MediaOut(media=tuple(Media(path=p, mime="application/octet-stream", kind="file") for p in paths))
        )

    async def text(self, content: str) -> None:
        """The whole reply, for an outlet that streamed nothing."""
        await self._emit(Text(content=content))


class AgentTurnRunner:
    def __init__(self, agent_loop: AgentLoop, *, stream: bool) -> None:
        self._loop = agent_loop
        self._stream = stream

    async def run(self, req: TurnRequest, emit: Emit, drain: Drain) -> TurnOutcome:
        return await self._loop.run_turn(req, emit, drain, stream=self._stream)


__all__ = ["AgentTurnRunner", "TurnEvents"]
