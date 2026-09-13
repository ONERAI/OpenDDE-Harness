"""One streamed model call, relayed and reduced to one authoritative result.

The route's own parsed answer wins outright; the fragment accumulation below it
is the compatibility path for a provider that streams deltas without one -- the
base class's non-streaming shim and the test doubles shaped like it. Nothing here
decides whether a failed call is repeated: that budget is the model service's.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import aclosing
from typing import Any, Awaitable, Callable

from loguru import logger

from opendde_harness.providers.base import (
    ErrorClassification,
    LLMProvider,
    LLMResponse,
    RunMeta,
    ToolCallRequest,
    format_llm_error,
)
from opendde_harness.providers.truncation import flag_truncation


async def llm_call_stream(
    provider: LLMProvider,
    messages: list[dict],
    tools: list[dict] | None,
    model: str | None,
    *,
    on_token_delta: Callable[[str], Awaitable[None]] | None = None,
    on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    on_tool_event: Callable[[str, dict], Awaitable[None]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    on_retry: Callable[[int, int, str, bool], Awaitable[None]] | None = None,
) -> LLMResponse:
    """One streamed call, accumulated into an ``LLMResponse``.

    Each non-empty content chunk fires the callback; tool_call fragments are
    merged positionally; the final response object is shape-compatible with what
    ``chat()`` would have returned.

    One call, however many attempts it takes. A failure the model layer calls
    transient is run again inside the model service, by pi's own retry loop, under
    the budget the request carried (``agents.defaults.llm_retries``); what arrives
    here is a ``retry`` delta before each new attempt. Everything the discarded
    attempt accumulated is dropped at that point and ``on_retry(attempt, total,
    reason, discard)`` says so -- ``discard`` is whether any of it reached a
    callback, which is what tells an outlet showing live text to start over.

    A failure on this side of the pipe is nobody's to repeat: the model layer's
    two stream deadlines and a broken generator are past the point the service
    could run the call again, and text already shown cannot be unshown. Such a
    turn keeps what it has and says why it stopped.
    """
    content_buf: list[str] = []
    reasoning_buf: list[str] = []
    thinking_blocks: list[dict] = []
    tool_call_slots: list[dict[str, Any]] = []
    final_usage: dict[str, Any] | None = None
    had_error = False
    error_content: str | None = None
    error_classification: ErrorClassification | None = None
    upstream_finish_reason: str | None = None
    delivered = False
    # The adapter's own parsed answer, when the route produces one. It wins
    # outright: the accumulation below is a second parser working from strictly
    # less -- fragments carry no repaired-argument flag, no exact item ids, no
    # final usage and no provider identity -- and that is how a streamed result
    # and a non-streamed one came to disagree about the same bytes.
    authoritative: LLMResponse | None = None

    def _error(content: str | None, classification: ErrorClassification | None) -> LLMResponse:
        return LLMResponse(
            content=content,
            finish_reason="error",
            error_classification=classification or ErrorClassification("unknown"),
            usage=final_usage or {},
        )

    # aclosing() guarantees the async generator (and its underlying stream) is
    # closed when an error unwinds the loop, so a stalled stream terminates with a
    # structured error instead of hanging or leaking the connection.
    try:
        async with aclosing(
            provider.chat_stream(messages=messages, tools=tools, model=model, tool_choice=tool_choice)
        ) as stream:
            async for delta in stream:
                retry = getattr(delta, "retry", None)
                if retry is not None:
                    # The attempt just ended is void: the re-run sends the same
                    # conversation without a word of what it streamed, so nothing
                    # it accumulated may survive into the answer.
                    logger.warning(
                        "LLM call retried ({}/{}) model={}, delivered={}: {}",
                        retry["attempt"],
                        retry["total"],
                        model,
                        delivered,
                        retry["reason"],
                    )
                    if on_retry is not None:
                        await on_retry(int(retry["attempt"]), int(retry["total"]), str(retry["reason"]), delivered)
                    content_buf.clear()
                    reasoning_buf.clear()
                    thinking_blocks.clear()
                    tool_call_slots.clear()
                    final_usage = None
                    had_error = False
                    error_content = None
                    error_classification = None
                    upstream_finish_reason = None
                    authoritative = None
                    delivered = False
                    continue
                if getattr(delta, "final_response", None) is not None:
                    authoritative = delta.final_response
                if delta.finish_reason == "error":
                    # A non-streaming provider's chat() error, replayed by the
                    # base class as its single terminal delta. Its content is the
                    # error text, not a token to render or accumulate -- surface it
                    # via error_classification instead of the normal success
                    # collation below.
                    had_error = True
                    error_content = delta.content
                    error_classification = delta.error_classification
                    if delta.usage is not None:
                        final_usage = delta.usage
                    continue
                if delta.finish_reason:
                    upstream_finish_reason = delta.finish_reason
                merge_thinking_blocks(thinking_blocks, getattr(delta, "thinking_blocks", None))
                reasoning_delta = getattr(delta, "reasoning_content", None)
                if reasoning_delta:
                    reasoning_buf.append(reasoning_delta)
                    if on_reasoning_delta is not None:
                        delivered = True
                        await on_reasoning_delta(reasoning_delta)
                if delta.content:
                    content_buf.append(delta.content)
                    if on_token_delta is not None:
                        delivered = True
                        await on_token_delta(delta.content)
                if delta.builtin_tool_event and on_tool_event is not None:
                    delivered = True
                    event = delta.builtin_tool_event
                    await on_tool_event(str(event["phase"]), event)
                if delta.tool_call_delta:
                    merge_tool_call_fragments(tool_call_slots, delta.tool_call_delta)
                if delta.usage is not None:
                    final_usage = delta.usage
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        classification = provider.classify_error(exc)
        head = getattr(provider, "provider_name", None) or None
        summary = format_llm_error(exc, classification, provider=head)
        if not delivered:
            return _error(summary, classification)
        # Part of the reply was shown. Nothing sends the call again, so this is
        # the turn's text: say why it stopped instead of ending the turn as if it
        # were complete.
        partial = "".join(content_buf)
        logger.warning(
            "LLM stream interrupted after {} chars [{}]: {}",
            len(partial),
            getattr(classification, "category", "unknown"),
            summary[:200],
        )
        return _error(f'{partial}\n\n[Reply interrupted: {summary}. Send "continue" to resume.]', classification)

    if authoritative is not None:
        # The route stated the result. Two fields are normalized rather than
        # trusted: the model, which a route may leave to the caller's request, and
        # a failure's classification, which every consumer reads as present.
        authoritative.model = authoritative.model or model
        if authoritative.finish_reason == "error" and authoritative.error_classification is None:
            authoritative.error_classification = ErrorClassification("unknown")
        return authoritative

    if had_error:
        return _error(error_content, error_classification)

    tool_calls = finalize_tool_calls(tool_call_slots)

    # No ceiling passed: the provider resolves its own after the loop has handed
    # over the request, so the number this turn carried is not knowable here.
    # Nothing is compared against it, so nothing is missing.
    sent_max_tokens, truncated = flag_truncation(
        finish_reason=upstream_finish_reason,
        usage=final_usage,
        tool_calls=tool_calls,
    )

    finish_reason = upstream_finish_reason or ("tool_calls" if tool_calls else "stop")

    return LLMResponse(
        content="".join(content_buf),
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=final_usage or {},
        reasoning_content="".join(reasoning_buf) or None,
        thinking_blocks=thinking_blocks or None,
        truncated=truncated,
        max_tokens=sent_max_tokens,
        model=model,
    )


def merge_thinking_blocks(blocks: list[dict], incoming: list[dict] | None) -> None:
    """Accumulate Anthropic thinking blocks arriving one delta at a time.

    The text arrives in fragments and the signature closes them, in a delta of its
    own carrying empty text, so the fragments fold into one block and the
    signature lands on it. Forwarding the raw per-delta list would send Anthropic
    dozens of unsigned scraps. Redacted blocks are opaque and stay separate
    entries, exactly as they arrived.
    """
    if not incoming:
        return
    for block in incoming:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "redacted_thinking":
            blocks.append(dict(block))
            continue
        target = next((item for item in blocks if item.get("type") != "redacted_thinking"), None)
        if target is None:
            target = {"type": "thinking"}
            blocks.append(target)
        for key, value in block.items():
            if key == "thinking" and isinstance(value, str):
                target["thinking"] = f"{target.get('thinking', '')}{value}"
            elif value not in (None, ""):
                target[key] = value


def merge_tool_call_fragments(slots: list[dict[str, Any]], delta: dict[str, Any]) -> None:
    """Merge a single chat_stream tool_call_delta into accumulator slots.

    Each slot follows the shape ``{id, function: {name, arguments_buf: [str]}}``.
    Per OpenAI Chat Completions chunk semantics: each tool call fragment carries an
    ``index`` field; ``id`` / ``function.name`` typically appear in the first
    fragment for that index, ``function.arguments`` is a JSON string streamed in
    pieces.

    Respects the ``index`` field so parallel multi-tool streams do not collapse
    into ``slots[0]``. Fragments without an ``index`` default to 0 (single-tool
    case, backward-compatible).
    """
    incoming = delta.get("tool_calls") or []
    if not incoming:
        return
    for tc in incoming:
        idx = int(tc.get("index", 0) or 0)
        while len(slots) <= idx:
            slots.append(
                {"id": None, "function": {"name": None, "arguments_buf": []}, "fields": None, "fn_fields": None}
            )
        slot = slots[idx]
        if tc.get("id") and not slot["id"]:
            slot["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name") and not slot["function"]["name"]:
            slot["function"]["name"] = fn["name"]
        if fn.get("arguments"):
            slot["function"]["arguments_buf"].append(fn["arguments"])
        # A provider that signs its tool calls (Gemini) sends the signature in
        # these fields, and the next request has to carry them back. Dropped here,
        # the streamed call went back unsigned while the non-streamed one did not,
        # which is the kind of difference nothing downstream can see.
        if tc.get("provider_specific_fields") and not slot.get("fields"):
            slot["fields"] = tc["provider_specific_fields"]
        if fn.get("provider_specific_fields") and not slot.get("fn_fields"):
            slot["fn_fields"] = fn["provider_specific_fields"]


def finalize_tool_calls(slots: list[dict[str, Any]]) -> list[ToolCallRequest]:
    """Convert accumulator slots into a final ToolCallRequest list.

    A slot whose arguments do not parse is flagged on that call, not on the turn:
    an unparseable blob is evidence about one call, and reducing it to "something
    in this turn failed" would leave the loop guessing which. An incomplete JSON
    blob is also the one piece of evidence that needs no cooperation from the
    backend, which matters where a backend reports a clean stop on a cut-off
    reply.
    """
    result: list[ToolCallRequest] = []
    for slot in slots:
        name = slot["function"]["name"]
        if not name:
            continue
        args_text = "".join(slot["function"]["arguments_buf"])
        repaired = False
        try:
            args = json.loads(args_text) if args_text else {}
        except json.JSONDecodeError:
            args = {"_raw_arguments": args_text}
            repaired = True
        if not isinstance(args, dict):
            args = {"_raw_arguments": args_text}
            repaired = True
        result.append(
            ToolCallRequest(
                id=slot["id"] or "",
                name=name,
                arguments=args,
                provider_specific_fields=slot.get("fields"),
                function_provider_specific_fields=slot.get("fn_fields"),
                run_meta=RunMeta(arguments_repaired=True) if repaired else None,
            )
        )
    return result


__all__ = [
    "finalize_tool_calls",
    "llm_call_stream",
    "merge_thinking_blocks",
    "merge_tool_call_fragments",
]
