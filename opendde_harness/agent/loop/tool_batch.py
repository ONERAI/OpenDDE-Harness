"""One assistant message's tool calls, run in order, and what a result may be.

Owns the sequential batch and the result policy around it: the per-result size
cap, where a picture goes for a model that cannot see one, and the safety stop
that cancels the siblings behind a terminating decision. The registry still
validates and dispatches each call; the durable boundaries are the caller's, which
is why the journal is passed in rather than reached for.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from opendde_harness.agent.loop.failure_streak import failure_class, is_hard_tool_failure
from opendde_harness.providers import messages as msg

if TYPE_CHECKING:
    from opendde_harness.agent.context import ContextBuilder
    from opendde_harness.agent.tools.registry import ToolRegistry
    from opendde_harness.providers.base import ToolCallRequest
    from opendde_harness.session.journal import TurnJournal

#: Longest tool result the model is shown, and therefore the longest one a session
#: record holds. Applied where the result enters the turn, so the prompt and the
#: record agree; a tool that has more to say is told how much was cut and asked
#: for a narrower slice. Measured before the cap: three status dumps of 250KB each
#: in one iteration made a 360k-token request.
TOOL_RESULT_MAX_CHARS = 16_000
#: How much of a result the transcript row is sent: what ctrl+o expands to.
TOOL_PREVIEW_MAX_CHARS = 1_024

#: What the model is told when a sibling call was cancelled. Every advertised call
#: id needs a result: OpenAI-style providers reject a history holding an assistant
#: tool call without its matching tool result.
SKIPPED_AFTER_ABORT = "Error: Tool call was not executed because a prior safety decision terminated this action."


def truncated_refusal(hint: str | None = None) -> str:
    """What a call of a reply that stopped at the output limit is told.

    Every call of that message, not only the last one: generation is sequential, so
    the cut landed inside one of them, and nothing the upstream said names which.
    Running the others on the assumption that only the tail was affected is how a
    ``write`` whose content stopped mid-file replaces the whole file with the part
    that arrived and reports success. Refusing all of them costs one retry.

    Keeps the ``[truncated]`` marker: it is the failure class the loop-break nudge
    keys on, and a payload that keeps outrunning the limit needs to be told to send
    less rather than to switch tools. ``hint`` is the tool's own advice about how.
    """
    body = (
        "Error: [truncated] The reply that asked for this call stopped at the output "
        "limit, so any of its calls may have been cut off part-way through. None of "
        "them were run. Send the call again."
    )
    return f"{body} {hint}" if hint else body


def image_placeholder_text(
    blocks: list[dict[str, Any]],
    *,
    describe_tool: str | None = None,
) -> str:
    """Text standing in for images a model with no vision will not receive.

    Keeps the tool's own text (it already names the file and its geometry) and
    appends a line per dropped image so the model knows a picture exists and where
    it came from, rather than silently seeing nothing.

    Nothing follows this message, so the note must not say a picture does: it would
    leave the model waiting for one that never arrives. It points at the tool that
    can read the file instead. Where a model *can* see, the blocks ride along in
    the result and the model layer decides where they go on the wire -- pi's Chat
    Completions adapter moves them into a following user message itself, which is
    why the loop no longer says so here.

    ``describe_tool`` names the reading tool, or is ``None`` when the caller has
    none to offer: it is contributed by the long-term memory plugin and absent on a
    default install, and naming a tool the model was never given is an instruction
    it cannot follow. The note then says only that a picture exists.
    """
    texts = [str(b.get("text") or "") for b in blocks if msg.is_text(b)]
    images = sum(1 for b in blocks if msg.is_image(b))
    body = "\n".join(t for t in texts if t)
    if images:
        noun = "image" if images == 1 else "images"
        hint = f"; use the {describe_tool} tool to read the file" if describe_tool else ""
        body += f"\n[{images} {noun} not shown — you cannot see images directly{hint}]"
    return body.strip()


def cap_tool_result(text: str) -> str:
    """Cut a tool result to ``TOOL_RESULT_MAX_CHARS``, saying what was cut.

    The head is kept: for a listing, a log or a JSON document that is where the
    shape is, and the model can ask for the rest by offset or filter.
    """
    limit = TOOL_RESULT_MAX_CHARS
    if len(text) <= limit:
        return text
    return (
        text[:limit] + f"\n\n[tool result truncated: the first {limit:,} of {len(text):,} characters are shown; "
        "ask for a narrower slice (an offset, a filter, a single item) to see the rest]"
    )


def cap_tool_blocks(blocks: list[dict]) -> list[dict]:
    """A multimodal result's text under the same cap as a plain one.

    The blocks replace the text on an image-capable route, so a cap on the text
    alone left the long text inside them uncut. Pictures pass; text parts over the
    cap collapse into one cut part, and the notice says so.
    """
    texts = [b["text"] for b in blocks if b.get("type") == "text" and isinstance(b.get("text"), str)]
    joined = "\n".join(texts)
    if len(joined) <= TOOL_RESULT_MAX_CHARS:
        return blocks
    return [{"type": "text", "text": cap_tool_result(joined)}, *(b for b in blocks if b.get("type") != "text")]


def tool_hint(tool_calls: list) -> str:
    """Format tool calls as a concise hint, e.g. 'web_search("query")'."""

    def _fmt(tc):
        args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
        val = next(iter(args.values()), None) if isinstance(args, dict) else None
        if not isinstance(val, str):
            return tc.name
        return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'

    return ", ".join(_fmt(tc) for tc in tool_calls)


@dataclass
class ToolBatchResult:
    """What one batch did, beyond the messages it appended.

    ``aborted`` is the safety stop: a terminating decision ran, its siblings were
    cancelled and the turn may not offer the model another iteration. ``failure``
    names the last result's deterministic failure, for the streak that decides
    whether the model is repeating a dead call. ``refused`` means nothing ran at
    all: the reply was cut off at the output limit, so none of its calls could be
    trusted.
    """

    messages: list[dict[str, Any]]
    used_skill_ids: list[str] = field(default_factory=list)
    aborted: bool = False
    refused: bool = False
    failure: tuple[str, str] | None = None


class ToolBatch:
    """Runs the calls of one assistant message, in the order they were asked for.

    Sequential on purpose: "one terminating action stops all remaining siblings" is
    only expressible in an order, and it is the reason this is not a gather.
    """

    def __init__(
        self,
        *,
        tools: "ToolRegistry",
        context: "ContextBuilder",
        route_images: Callable[[str, list[dict] | None, str], tuple[str, list[dict] | None]],
        on_tool_event: Callable[[str, dict], Awaitable[None]] | None = None,
        journal: "TurnJournal | None" = None,
    ) -> None:
        self._tools = tools
        self._context = context
        self._route_images = route_images
        self._on_tool_event = on_tool_event
        self._journal = journal

    async def run(
        self,
        calls: list["ToolCallRequest"],
        messages: list[dict[str, Any]],
        *,
        model: str,
        truncated: bool = False,
    ) -> ToolBatchResult:
        """Execute ``calls`` against ``messages``, appending a result for each.

        ``truncated`` is the whole story about a reply that hit its output ceiling:
        one field, read once, whichever route parsed the message. When it is set no
        call is dispatched -- see :func:`truncated_refusal` -- and the turn carries
        on with a refusal per call, which is what a model needs to send them again.
        """
        out = ToolBatchResult(messages=messages)
        if truncated and calls:
            return self._refuse_all(calls, out)
        for index, call in enumerate(calls):
            logger.info("Tool call: {}({})", call.name, json.dumps(call.arguments, ensure_ascii=False)[:200])
            # Looked up once, outside the event branch: three things below ask the
            # tool about itself, and resolving it only when an outlet is listening
            # left the other two reading a name that existed only on the streaming
            # path.
            tool = self._tools.get(call.name)
            await self._emit(
                "start",
                {
                    "tool_call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                    # Tool-authored call label; None -> UI derives one.
                    "display": tool.display_call(call.arguments) if tool else None,
                },
            )
            # Whichever tool is about to run, if it can ask for approval, the
            # prompt it raises names this call. Not the shell alone: a write to
            # AGENTS.md asks too, and an approval with no call id behind it is not
            # auditable.
            if callable(getattr(tool, "set_tool_call_id", None)):
                tool.set_tool_call_id(call.id)
            # A tool that changes something outside this process is recorded before
            # it is called. If nothing comes back, the journal says which tool was
            # given what and that its outcome is unknown; a filesystem checkpoint
            # cannot reconstruct a submitted job or the id it was told once.
            if self._journal is not None and getattr(tool, "external_effects", False):
                self._journal.tool_started(call.id, call.name, call.arguments)
            started = time.monotonic()
            result = await self._tools.execute(call.name, call.arguments, run_meta=call.run_meta)
            duration_ms = int((time.monotonic() - started) * 1000)
            # The registry already unwrapped any ToolResult: ``result`` is the
            # model-facing text, with the optional display string riding along on it
            # (ToolOutput). The model always gets the model text; the UI preview
            # prefers the display string.
            model_text = cap_tool_result(str(result))
            self._note_skill(call, model_text, out.used_skill_ids)
            display_src = getattr(result, "display_text", None) or model_text
            # The log stays one line; the UI event keeps newlines so a tool that
            # reports several items (e.g. ask_user's question -> answer pairs)
            # renders one row each.
            preview = display_src[:TOOL_PREVIEW_MAX_CHARS]
            logger.info(
                "Tool result: {} duration={}ms result={}",
                call.name,
                duration_ms,
                preview.replace("\n", " ")[:200],
            )
            # ``truncated`` says the preview does not hold the whole result: what
            # ctrl+o expands to is its head. Whether the model's own copy was cut
            # is the cap's business, and is said inside the result itself.
            await self._emit(
                "complete",
                {
                    "tool_call_id": call.id,
                    "result_preview": preview,
                    "truncated": len(display_src) > TOOL_PREVIEW_MAX_CHARS,
                },
            )
            model_text, blocks = self._route_images(model_text, getattr(result, "blocks", None), model)
            if blocks:
                out.messages = self._context.add_tool_result(
                    out.messages, call.id, call.name, model_text, cap_tool_blocks(blocks)
                )
            else:
                # Keep the long-standing 4-arg call for text results so no existing
                # caller or test double sees a signature change.
                out.messages = self._context.add_tool_result(out.messages, call.id, call.name, model_text)
            # The completed exchange, as soon as the tool returned.
            if self._journal is not None:
                self._journal.flush(out.messages)
            if getattr(result, "abort_action", False):
                out.aborted = True
                for skipped in calls[index + 1 :]:
                    out.messages = self._context.add_tool_result(
                        out.messages, skipped.id, skipped.name, SKIPPED_AFTER_ABORT
                    )
                return out
            # Consecutive same-tool deterministic failures count; transient ones do
            # not -- a retry would clear those.
            out.failure = (call.name, failure_class(model_text)) if is_hard_tool_failure(model_text) else None
        return out

    def _refuse_all(self, calls: list["ToolCallRequest"], out: ToolBatchResult) -> ToolBatchResult:
        """Answer every call of a cut-off reply without running any of it.

        No start event either: nothing was dispatched, so an outlet has no call to
        show as running. The results are journaled like any others, because the
        assistant message that asked for them is already on disk and a call with no
        result is dropped from the next request rather than answered twice.
        """
        logger.warning(
            "the reply was cut at the output limit; refusing its {} tool call(s) unrun",
            len(calls),
        )
        text = ""
        for call in calls:
            tool = self._tools.get(call.name)
            text = truncated_refusal(getattr(tool, "truncation_hint", None))
            out.messages = self._context.add_tool_result(out.messages, call.id, call.name, text)
        if self._journal is not None:
            self._journal.flush(out.messages)
        out.refused = True
        # Counted like any other deterministic failure: a payload that keeps
        # outrunning the limit is exactly what the loop-break nudge is for.
        out.failure = (calls[-1].name, failure_class(text)) if is_hard_tool_failure(text) else None
        return out

    async def _emit(self, phase: str, info: dict[str, Any]) -> None:
        if self._on_tool_event is not None:
            await self._on_tool_event(phase, info)

    @staticmethod
    def _note_skill(call: "ToolCallRequest", model_text: str, used: list[str]) -> None:
        """Record a skill the agent actually loaded, which is the feedback signal.

        Advertising a skill says nothing about it, so only ``use_skill`` counts, and
        only when it did not come back as an error.
        """
        if call.name != "use_skill" or model_text.lstrip().lower().startswith("error:"):
            return
        skill_id = call.arguments.get("skill_id")
        if isinstance(skill_id, str) and "/" in skill_id and skill_id not in used:
            used.append(skill_id)


__all__ = [
    "SKIPPED_AFTER_ABORT",
    "TOOL_PREVIEW_MAX_CHARS",
    "TOOL_RESULT_MAX_CHARS",
    "ToolBatch",
    "ToolBatchResult",
    "cap_tool_blocks",
    "cap_tool_result",
    "image_placeholder_text",
    "tool_hint",
    "truncated_refusal",
]
