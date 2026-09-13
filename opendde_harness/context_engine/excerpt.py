"""The deterministic excerpt policy for tool-result bodies.

One rule, used in the two places a prompt is cut down:

- at turn start, by :class:`~opendde_harness.context_engine.history_trimmer.HistoryTrimmer`,
  before it drops whole exchanges (a body the model can ask for again is
  cheaper to give up than a turn of the conversation);
- mid-turn, by ``agent.loop.context_policy.emergency_shrink``, when tool output grown since
  selection pushes the request over the window.

What it may touch is narrow on purpose:

- only ``toolResult`` bodies. A compaction marker is a boundary, not text,
  and an assistant turn's ``toolCall`` arguments are what the results are
  answers *to* -- excerpting either breaks the exchange rather than shrinking
  it;
- never in place. The caller's message dicts (the session log) are left
  whole, so the original is still on disk and still what the next turn's
  selection sees; the excerpt exists only in the request;
- every key but ``content`` survives, so the result's ``toolCallId`` /
  ``toolName`` (the reference back to the call, and to whatever artifact the
  call produced) still reaches the model. A picture in a result goes with the
  text: its bytes are the densest thing in the window, and the placeholder says
  the body was elided.
"""

from __future__ import annotations

from typing import Any

from opendde_harness.providers import messages as msg

#: What the model is shown in place of an elided body. Short and identical
#: everywhere, so a second pass recognises its own work and does not re-elide
#: (or re-count) what it already replaced.
TOOL_BODY_PLACEHOLDER = "[earlier tool output elided to fit the context window]"

#: Tool results kept whole by the list-level policy: the ones the model is
#: currently reasoning about. Older bodies are the bulk of context growth.
KEEP_RECENT_TOOL_RESULTS = 3


def excerpt_tool_result(message: dict[str, Any]) -> dict[str, Any] | None:
    """A copy of one tool result with its body replaced, or ``None``.

    ``None`` means there is nothing to excerpt here: the message is not a tool
    result, carries no body, or has already been excerpted.
    """
    if not msg.is_tool_result(message):
        return None
    blocks = msg.blocks_of(message)
    if not blocks or blocks == [msg.text_block(TOOL_BODY_PLACEHOLDER)]:
        return None
    return msg.with_blocks(message, [msg.text_block(TOOL_BODY_PLACEHOLDER)])


def excerpt_older_tool_results(
    messages: list[dict[str, Any]],
    keep_recent: int = KEEP_RECENT_TOOL_RESULTS,
) -> tuple[list[dict[str, Any]], int]:
    """Excerpt every tool-result body but the newest ``keep_recent``.

    Returns ``(messages, count)``; ``count == 0`` means there was nothing
    worth eliding and the list is the one that came in.
    """
    tool_idxs = [i for i, m in enumerate(messages) if msg.is_tool_result(m)]
    if len(tool_idxs) <= keep_recent:
        return messages, 0
    targets = set(tool_idxs[: len(tool_idxs) - keep_recent] if keep_recent else tool_idxs)
    out: list[dict[str, Any]] = []
    count = 0
    for idx, message in enumerate(messages):
        excerpted = excerpt_tool_result(message) if idx in targets else None
        if excerpted is None:
            out.append(message)
            continue
        out.append(excerpted)
        count += 1
    return (out, count) if count else (messages, 0)


__all__ = [
    "KEEP_RECENT_TOOL_RESULTS",
    "TOOL_BODY_PLACEHOLDER",
    "excerpt_older_tool_results",
    "excerpt_tool_result",
]
