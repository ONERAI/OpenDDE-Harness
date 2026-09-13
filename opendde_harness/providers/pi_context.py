"""The loop's messages as pi-ai's ``Context`` -- and its events back.

The agent loop, the session log and the context engine all speak pi's own
``Message`` (:mod:`opendde_harness.providers.messages`), so there is no
translation left to do here: the messages pass through, the system prefix folds
into pi's single ``systemPrompt``, and the tool schemas become pi ``Tool``s.
What remains is the other direction -- pi's ``AssistantMessageEvent`` JSON as
:class:`~opendde_harness.providers.base.StreamDelta`, plus the one event that is
the service's own: ``retry``, written when pi's retry loop is about to run a
failed call again.

This used to be a two-way conversion: the loop spoke OpenAI-style dicts and
every turn was rebuilt into pi's shape on the way out, with a verbatim copy of
pi's own message carried in the history so the rebuild could be skipped for
turns pi had written. One shape in the store retired all of that.

Pure functions, plain dicts in and out, no I/O.
"""

from __future__ import annotations

import json
from typing import Any

from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import (
    COMPACTION_KEY,
    ErrorClassification,
    LLMResponse,
    StreamDelta,
    ToolCallRequest,
)

#: pi's terminal reason -> the loop's ``finish_reason``. ``deferred`` is
#: unreachable: nothing in the service's ``StreamParams`` asks for one.
_FINISH_REASONS = {"stop": "stop", "toolUse": "tool_calls", "length": "length"}

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``{"type":"function","function":{...}}`` schemas as pi ``Tool``s."""
    out: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") or tool
        name = str(function.get("name") or "")
        if not name:
            continue
        parameters = function.get("parameters")
        out.append(
            {
                "name": name,
                "description": str(function.get("description") or ""),
                "parameters": parameters if isinstance(parameters, dict) else dict(_EMPTY_SCHEMA),
            }
        )
    return out


def _assistant(message: dict[str, Any], *, provider: str, model: str) -> dict[str, Any]:
    """One stored assistant record as a message pi will accept.

    A message the model service answered carries pi's own ``api`` and is sent
    exactly as it came back, which is what keeps thinking signatures and native
    tool-call ids alive across turns.

    Without one, the record is something this harness wrote and the envelope is
    completed here. Two records reach this: a compaction marker on a model that
    does *not* replay it -- ordinary content by then -- and a turn a provider
    that is not pi produced, which is a test double. The declared api is
    :data:`~opendde_harness.providers.messages.REPLAY_API`, so pi renders the
    turn for the target wire rather than trusting fields that are not there.

    Either way the boundary itself stops here: ``compaction`` is how the session
    log and the provider's own split recognise a marker, and the request carries
    the summary or the text, never the key.
    """
    entry = (
        msg.wire_projection(message)
        if "api" in message
        else msg.assistant_message(msg.blocks_of(message), provider=provider, model=model)
    )
    entry.pop(COMPACTION_KEY, None)
    return entry


def _user(message: dict[str, Any], pending: list[str]) -> dict[str, Any]:
    """One user message, with any mid-history instructions prefixed onto it."""
    entry = msg.wire_projection(message)
    if not pending:
        return entry
    prefix = "\n\n".join(pending)
    content = entry.get("content")
    if isinstance(content, list):
        entry["content"] = [msg.text_block(prefix), *msg.blocks_of(message)]
    else:
        entry["content"] = f"{prefix}\n\n{content}" if content else prefix
    return entry


def to_context(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    provider: str,
    model: str,
) -> dict[str, Any]:
    """The loop's messages and tools as a pi ``Context``.

    A ``system`` (or ``developer``) message that leads the history joins the
    ``systemPrompt``. One that appears later is prefixed onto the next user
    message instead of being hoisted: pi has a single ``systemPrompt``, emitted
    before everything, so hoisting would move a mid-history instruction ahead of
    the content it was written to follow and would change the cached prefix on
    every turn it survives. With no user message after it there is nowhere to
    put it, and it joins the ``systemPrompt`` after all. No producer in the tree
    emits one today; every caller builds the system message first.
    """
    system: list[str] = []
    pending: list[str] = []
    out: list[dict[str, Any]] = []
    started = False

    for message in messages:
        role = message.get("role")
        if role in msg.SYSTEM_ROLES:
            text = msg.text_of(message)
            if text:
                (pending if started else system).append(text)
            continue
        started = True
        if role == msg.ASSISTANT:
            out.append(_assistant(message, provider=provider, model=model))
        elif role == msg.TOOL_RESULT:
            out.append(msg.wire_projection(message))
        else:
            out.append(_user(message, pending))
            pending = []

    system.extend(pending)
    context: dict[str, Any] = {"messages": out}
    if system:
        context["systemPrompt"] = "\n\n".join(system)
    if tools:
        context["tools"] = _tools(tools)
    return context


def usage_dict(usage: Any) -> dict[str, Any]:
    """pi ``Usage`` as the loop's OpenAI-style usage dict.

    ``input`` is fresh-only on every pi route -- pi subtracts the cache counts
    a vendor folds into its prompt total -- and is passed through as
    ``prompt_tokens`` with that meaning. The cache keys appear only when pi
    reported them, so an absent key still means "nobody counted" rather than
    zero.
    """
    if not isinstance(usage, dict):
        return {}
    prompt = int(usage.get("input") or 0)
    completion = int(usage.get("output") or 0)
    out: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(usage.get("totalTokens") or prompt + completion),
    }
    if "cacheRead" in usage:
        out["cache_read_input_tokens"] = int(usage["cacheRead"] or 0)
    if "cacheWrite" in usage:
        out["cache_creation_input_tokens"] = int(usage["cacheWrite"] or 0)
    if usage.get("reasoning") is not None:
        out["completion_tokens_details"] = {"reasoning_tokens": int(usage["reasoning"] or 0)}
    cost = usage.get("cost")
    if isinstance(cost, dict) and cost.get("total") is not None:
        out["cost"] = float(cost["total"])
    return out


def _classification(event: dict[str, Any]) -> ErrorClassification:
    """The verdict on a pi ``error`` event, which is entirely the service's.

    Three fields, each pi's own function run against the failed message:
    ``retryable`` is ``isRetryableAssistantError``, ``overflow`` is
    ``isContextOverflow`` against the window the service reports for the model,
    and ``code`` is pi's ``ModelsErrorCode`` when a ``ModelsError`` is what
    failed -- ``auth`` and ``oauth`` for a credential that is missing or was
    rejected, which no retry fixes -- or the service's own code for a failure
    it produced itself rather than read, which today is ``timeout``: a gap
    budget that expired with nothing arriving.

    Nothing here reads the message text. The tables that used to are pi's now,
    which is what makes a new vendor wording a version bump rather than an edit.
    ``category`` is a label for the log; the two flags are what the loop acts on.
    """
    if event.get("reason") == "aborted":
        return ErrorClassification("aborted")
    code = str(event.get("code") or "")
    if code in ("auth", "oauth"):
        return ErrorClassification("auth")
    return ErrorClassification(
        category=code or "server",
        retryable=bool(event.get("retryable", False)),
        should_compress=bool(event.get("overflow", False)),
    )


#: How much of a failure's wording rides on the one line a retry is announced
#: with. A provider's error can be a whole JSON body; the line says which
#: attempt is running and why, not the whole of it.
_RETRY_REASON_CHARS = 80


def _retry_reason(raw: Any) -> str:
    """The failed attempt's wording, as one short line."""
    text = " ".join(str(raw or "").split())
    return text[: _RETRY_REASON_CHARS - 1] + "\u2026" if len(text) > _RETRY_REASON_CHARS else text


def event_to_delta(event: dict[str, Any]) -> StreamDelta | None:
    """One pi ``AssistantMessageEvent`` as a loop delta, or None to ignore it.

    Tool calls arrive whole on ``toolcall_end``, shaped as the OpenAI chunk
    fragment ``model_call.merge_tool_call_fragments`` expects; the ``contentIndex`` is the
    slot index, so parallel calls stay apart. The ``done`` delta carries the
    whole answer as its ``final_response``, which the loop takes as
    authoritative, so a streamed turn keeps pi's message for replay and agrees
    with the non-streamed one about the same bytes.
    """
    kind = event.get("type")
    if kind == "text_delta":
        return StreamDelta(content=str(event.get("delta") or ""))
    if kind == "thinking_delta":
        return StreamDelta(content=None, reasoning_content=str(event.get("delta") or ""))
    if kind == "toolcall_end":
        call = event.get("toolCall") or {}
        arguments = call.get("arguments")
        return StreamDelta(
            content=None,
            tool_call_delta={
                "tool_calls": [
                    {
                        "index": int(event.get("contentIndex") or 0),
                        "id": str(call.get("id") or ""),
                        "type": "function",
                        "function": {
                            "name": str(call.get("name") or ""),
                            "arguments": json.dumps(
                                arguments if isinstance(arguments, dict) else {}, ensure_ascii=False
                            ),
                        },
                    }
                ]
            },
        )
    if kind == "done":
        message = event.get("message") or {}
        return StreamDelta(
            content=None,
            finish_reason=_FINISH_REASONS.get(str(event.get("reason")), "stop"),
            usage=usage_dict(message.get("usage")) or None,
            final_response=response_from_done(message),
        )
    if kind == "retry":
        # pi counts the retries: `attempt` is the retry about to run, 1-indexed,
        # out of `maxAttempts` retries. The loop counts calls, so both gain the
        # initial one, which is neither a retry nor free.
        return StreamDelta(
            content=None,
            retry={
                "attempt": int(event.get("attempt") or 0) + 1,
                "total": int(event.get("maxAttempts") or 0) + 1,
                "reason": _retry_reason(event.get("errorMessage")),
            },
        )
    if kind == "error":
        error = event.get("error") or {}
        text = str(error.get("errorMessage") or "") or f"the model service ended the stream ({event.get('reason')})"
        return StreamDelta(
            content=text,
            finish_reason="error",
            error_classification=_classification(event),
            usage=usage_dict(error.get("usage")) or None,
        )
    return None


def response_from_done(message: dict[str, Any]) -> LLMResponse:
    """A final pi ``AssistantMessage`` as one ``LLMResponse``.

    The message itself rides along under ``pi_message``, whole: it *is* the
    record the session stores for this turn, and it replays through
    :func:`to_context` untouched, which is how thinking signatures and native
    tool-call ids carry forward. A failed turn carries none -- pi drops an
    errored assistant message from the history it sends.

    ``usage`` is part of that, and stripping it was a crash. pi reads
    ``usage.totalTokens`` off every message it replays
    (``utils/estimate.js:4``) without checking the field is there, so a stored
    turn without one failed the next request outright -- "Cannot read properties
    of undefined". pi's own app stores the usage with the message for the same
    reason. It is also what makes the record say what the turn cost.
    """
    text: list[str] = []
    thinking: list[str] = []
    calls: list[ToolCallRequest] = []
    for block in message.get("content") or ():
        kind = block.get("type")
        if kind == "text":
            text.append(str(block.get("text") or ""))
        elif kind == "thinking":
            thinking.append(str(block.get("thinking") or ""))
        elif kind == "toolCall":
            arguments = block.get("arguments")
            calls.append(
                ToolCallRequest(
                    id=str(block.get("id") or ""),
                    name=str(block.get("name") or ""),
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )
    stop = str(message.get("stopReason") or "stop")
    failed = stop in ("error", "aborted")
    content = str(message.get("errorMessage") or "") if failed else "".join(text)
    return LLMResponse(
        content=content or "".join(text),
        tool_calls=calls,
        finish_reason="error" if failed else _FINISH_REASONS.get(stop, "tool_calls" if calls else "stop"),
        usage=usage_dict(message.get("usage")),
        reasoning_content="".join(thinking) or None,
        error_classification=_classification({"reason": stop}) if failed else None,
        truncated=stop == "length",
        model=str(message.get("responseModel") or message.get("model") or "") or None,
        pi_message=None if failed else dict(message),
    )
