"""pi's ``Message``, as the harness stores and passes it.

One representation, everywhere: the session file, the journal, the extraction
outbox, the context engine and the agent loop all hold pi-ai's own message
shape, and the model layer forwards it. Three roles:

* ``{"role": "user", "content": str | [text|image], "timestamp": ms}``
* pi's ``AssistantMessage`` exactly as the model service returned it -- its
  ``api`` / ``provider`` / ``model`` triple, its ``usage``, its ``stopReason``
  and its ``content`` of ``text`` / ``thinking`` / ``toolCall`` blocks;
* ``{"role": "toolResult", "toolCallId", "toolName", "content": [text|image],
  "isError", "timestamp": ms}``.

There used to be two: an OpenAI-style dict the loop built, plus a verbatim copy
of pi's own message under a ``pi`` key so replaying a turn kept its thinking
signatures and native tool-call ids. Every reader had to know which of the two
it was looking at, and the projection between them was a translation layer with
its own bugs. Now the thing that is stored is the thing that is sent.

Two shapes here are *not* pi messages, and both are deliberate:

* a system message (``{"role": "system", "content": str}``) is the prompt
  prefix the assembler builds; pi has a single ``systemPrompt``, so
  :func:`~opendde_harness.providers.pi_context.to_context` folds it in rather
  than sending it as a message;
* a compaction marker is an assistant-role record carrying ``compaction``. It
  is a boundary, not an answer -- the provider replays what it stands for and
  never sends the record itself -- so it carries one text block and no
  envelope. ``to_context`` completes the envelope on the one path where such a
  record does reach a request: a marker the model being asked does not replay,
  which is ordinary content.

Every function is pure, takes and returns plain dicts, and imports nothing from
this package.
"""

from __future__ import annotations

import base64
import binascii
import time
from typing import Any, Literal, TypedDict

# ── Roles ──────────────────────────────────────────────────────────────

USER = "user"
ASSISTANT = "assistant"
TOOL_RESULT = "toolResult"
SYSTEM = "system"

#: Roles whose text joins pi's single ``systemPrompt`` instead of travelling as
#: a message. ``developer`` is the OpenAI spelling of the same thing.
SYSTEM_ROLES = frozenset({SYSTEM, "developer"})

# ── Content block types ────────────────────────────────────────────────

TEXT = "text"
IMAGE = "image"
THINKING = "thinking"
TOOL_CALL = "toolCall"

#: pi replays an assistant message as "same model" only when its ``api``,
#: ``provider`` and ``model`` all match the model now being asked
#: (``transform-messages.ts:94-98``). A message pi did not write -- one this
#: harness synthesised, or a record from a session written before the model
#: service existed -- carries this instead, which declares it cross-model. The
#: consequence is deliberate: pi re-renders replayed thinking as plain text and
#: renormalises tool-call ids for the target API rather than trusting
#: signatures the record does not carry.
REPLAY_API = "opendde-harness-replay"

#: The journal's stamp on a live message whose record is already on disk. Named
#: here because :data:`HOST_KEYS` is what keeps it off the wire;
#: :data:`opendde_harness.session.journal.JOURNAL_KEY` is what it means.
JOURNAL_KEY = "_journal_id"

#: Keys the harness keeps on a message for its own bookkeeping and never sends:
#: the record id, the turn it belongs to, the journal stamp and the
#: empty-recovery scaffolding flag.
HOST_KEYS = frozenset({"id", "turn_id", JOURNAL_KEY, "_recovery_synthetic"})


def now_ms() -> int:
    """The wall clock in pi's unit: milliseconds since the epoch."""
    return int(time.time() * 1000)


def to_ms(moment: Any) -> int:
    """A ``datetime`` as pi's millisecond stamp."""
    return int(moment.timestamp() * 1000)


# ── Blocks ─────────────────────────────────────────────────────────────


def text_block(text: str) -> dict[str, Any]:
    """One pi ``TextContent``."""
    return {"type": TEXT, "text": text}


def image_block(data: str, mime: str) -> dict[str, Any]:
    """One pi ``ImageContent``: base64 bytes and their media type.

    pi carries bytes, never a URL -- there is no remote-reference form to send
    and nothing here fetches one.
    """
    return {"type": IMAGE, "data": data, "mimeType": mime or "image/png"}


def tool_call_block(call_id: str, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    """One pi ``ToolCall`` block: arguments as an object, never a JSON string."""
    return {
        "type": TOOL_CALL,
        "id": call_id,
        "name": name,
        "arguments": arguments if isinstance(arguments, dict) else {},
    }


def thinking_block(thinking: str, *, signature: str | None = None, redacted: bool = False) -> dict[str, Any]:
    """One pi ``ThinkingContent``, signed when the history kept a signature."""
    block: dict[str, Any] = {"type": THINKING, "thinking": thinking}
    if signature:
        block["thinkingSignature"] = signature
    if redacted:
        block["redacted"] = True
    return block


class TextPart(TypedDict):
    """A text content block."""

    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    """An image content block: base64 bytes and their media type."""

    type: Literal["image"]
    data: str
    mimeType: str


#: What a tool hands back as content. Deliberately not used to type what this
#: project *reads*: pi's own content carries blocks this union does not model (a
#: thinking block, a tool call, a field a later pi version adds), and the
#: pass-through code that forwards them unchanged would otherwise become a type
#: error for doing the right thing. So read-side helpers take ``Any`` and check
#: shape at runtime.
ContentPart = TextPart | ImagePart


def is_text(block: Any) -> bool:
    """True for a pi text block."""
    return isinstance(block, dict) and block.get("type") == TEXT


def is_image(block: Any) -> bool:
    """True for a pi image block. Every pi image carries its bytes inline."""
    return isinstance(block, dict) and block.get("type") == IMAGE


def zero_usage() -> dict[str, Any]:
    """The ``Usage`` a synthesised assistant message carries.

    Zeroes rather than nothing: pi's type requires the field, and no adapter
    reads the usage of a message it is replaying.
    """
    return {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0},
    }


# ── Constructors ───────────────────────────────────────────────────────


def user_message(content: str | list[dict[str, Any]], *, timestamp: int | None = None) -> dict[str, Any]:
    """One pi ``UserMessage``.

    A plain string stays one: that is the shape every vendor expects for text,
    and wrapping it in a block list would change the request for no reason.
    """
    return {"role": USER, "content": content, "timestamp": now_ms() if timestamp is None else timestamp}


def assistant_message(
    content: str | list[dict[str, Any]] | None = None,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
    thinking_blocks: list[dict[str, Any]] | None = None,
    provider: str = "",
    model: str = "",
    timestamp: int | None = None,
) -> dict[str, Any]:
    """An assistant turn this harness wrote, as a pi ``AssistantMessage``.

    For everything the model service did not answer: a recovery prefill, the
    reply a terminated action gets, a forwarded report, a subagent's turn, and
    a record read back from a session written before the service existed. The
    service's own message is stored verbatim and never comes through here.

    Block order is pi's: thinking first, then text, then the calls. The
    envelope declares :data:`REPLAY_API`, so pi treats the turn as another
    model's and re-renders it for whatever API the next request uses.
    ``stopReason`` is ``toolUse`` or ``stop`` and never ``error`` / ``aborted``:
    pi drops such a message from the history entirely
    (``transform-messages.ts:195``).
    """
    blocks: list[dict[str, Any]] = list(_thinking_blocks(reasoning_content, thinking_blocks))
    if isinstance(content, str):
        if content:
            blocks.append(text_block(content))
    elif isinstance(content, list):
        blocks.extend(block for block in content if isinstance(block, dict))
    blocks.extend(call for call in (tool_calls or ()) if isinstance(call, dict))
    return {
        "role": ASSISTANT,
        "content": blocks,
        "api": REPLAY_API,
        "provider": provider,
        "model": model,
        "stopReason": "toolUse" if any(b.get("type") == TOOL_CALL for b in blocks) else "stop",
        "usage": zero_usage(),
        "timestamp": now_ms() if timestamp is None else timestamp,
    }


def _thinking_blocks(
    reasoning_content: str | None,
    thinking_blocks: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Signed blocks when a caller kept them, else the reasoning text."""
    out: list[dict[str, Any]] = []
    for block in thinking_blocks or ():
        if not isinstance(block, dict):
            continue
        if block.get("type") == "redacted_thinking":
            out.append(thinking_block("", signature=str(block.get("data") or ""), redacted=True))
            continue
        signature = block.get("signature") or block.get("thinkingSignature")
        out.append(thinking_block(str(block.get("thinking") or ""), signature=str(signature) if signature else None))
    if out:
        return out
    return [thinking_block(reasoning_content)] if reasoning_content else []


def tool_result_message(
    call_id: str,
    tool_name: str,
    content: str | list[dict[str, Any]],
    *,
    is_error: bool = False,
    details: Any = None,
    timestamp: int | None = None,
) -> dict[str, Any]:
    """One pi ``ToolResultMessage``.

    ``toolName`` is how pi pairs a result with the call it answers, alongside
    ``toolCallId``; both are required and neither is derivable downstream.
    """
    message: dict[str, Any] = {
        "role": TOOL_RESULT,
        "toolCallId": call_id,
        "toolName": tool_name,
        "content": [text_block(content)] if isinstance(content, str) else list(content),
        "isError": is_error,
        "timestamp": now_ms() if timestamp is None else timestamp,
    }
    if details is not None:
        message["details"] = details
    return message


def system_message(text: str) -> dict[str, Any]:
    """The prompt prefix. Folded into pi's ``systemPrompt``, never sent as one."""
    return {"role": SYSTEM, "content": text}


# ── Readers ────────────────────────────────────────────────────────────


def is_assistant(message: dict[str, Any]) -> bool:
    return message.get("role") == ASSISTANT


def is_tool_result(message: dict[str, Any]) -> bool:
    return message.get("role") == TOOL_RESULT


def blocks_of(message: dict[str, Any]) -> list[dict[str, Any]]:
    """A message's content as blocks; a plain string becomes one text block."""
    content = message.get("content")
    if isinstance(content, str):
        return [text_block(content)] if content else []
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def text_of(message: dict[str, Any]) -> str:
    """The message's own words: its text blocks, joined. No thinking, no calls."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "".join(str(block.get("text") or "") for block in blocks_of(message) if block.get("type") == TEXT)


def thinking_of(message: dict[str, Any]) -> str:
    """The reasoning the message carries, as text."""
    return "".join(str(block.get("thinking") or "") for block in blocks_of(message) if block.get("type") == THINKING)


def tool_calls_of(message: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``toolCall`` blocks of an assistant message, in order."""
    if not is_assistant(message):
        return []
    return [block for block in blocks_of(message) if block.get("type") == TOOL_CALL]


def tool_call_ids(message: dict[str, Any]) -> list[str]:
    """The ids of the calls an assistant message asked for."""
    return [str(block.get("id")) for block in tool_calls_of(message) if block.get("id")]


def has_images(message: dict[str, Any]) -> bool:
    return any(is_image(block) for block in blocks_of(message))


#: How much of an image is decoded to read its dimensions. Every format this
#: project inlines states its size in the first few hundred bytes; decoding the
#: whole of a 4MB picture on every budget probe would cost far more than the
#: estimate is worth.
IMAGE_HEADER_BYTES = 4096


def image_payload(block: dict[str, Any]) -> bytes | None:
    """An image block's decoded head, or None when it does not decode."""
    try:
        return base64.b64decode(str(block.get("data") or "")[:IMAGE_HEADER_BYTES], validate=False)
    except (binascii.Error, ValueError):
        return None


# ── Rewriting ──────────────────────────────────────────────────────────


def with_blocks(message: dict[str, Any], blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """A copy of ``message`` carrying ``blocks`` as its content.

    A copy, always: the caller's dict is the live message a request may still
    be built from, and the projections that shrink a prompt must not reach back
    into the session log.
    """
    return {**message, "content": blocks}


def with_text(message: dict[str, Any], text: str) -> dict[str, Any]:
    """A copy of ``message`` whose words are ``text``, thinking and calls kept.

    For the one case where the harness knows better than the model what the
    model said: a reply carrying ``<think>`` debris the wire was never meant to
    show. The reasoning and the tool calls are untouched; the text blocks
    collapse into one, which costs any ``textSignature`` pi had put on them --
    the signature describes bytes that are no longer what is stored.
    """
    kept = [block for block in blocks_of(message) if block.get("type") != TEXT]
    thinking = [block for block in kept if block.get("type") == THINKING]
    calls = [block for block in kept if block.get("type") == TOOL_CALL]
    return with_blocks(message, [*thinking, *([text_block(text)] if text else []), *calls])


def with_appended_text(message: dict[str, Any], text: str) -> dict[str, Any]:
    """A copy of ``message`` with ``text`` added after its own words."""
    if isinstance(message.get("content"), str):
        return {**message, "content": f"{message['content']}{text}"}
    return with_blocks(message, [*blocks_of(message), text_block(text)])


def wire_projection(message: dict[str, Any]) -> dict[str, Any]:
    """``message`` without the harness's own bookkeeping.

    A deny-list, not an allow-list: pi's ``AssistantMessage`` carries a dozen
    optional fields (``responseId``, ``providerThinkingLevel``,
    ``rawStopReason``, ``deferred``, ``diagnostics``, ...) and replaying a turn
    means replaying all of them. Only :data:`HOST_KEYS` is ours to drop.
    """
    return {key: value for key, value in message.items() if key not in HOST_KEYS}


__all__ = [
    "ASSISTANT",
    "ContentPart",
    "HOST_KEYS",
    "IMAGE",
    "JOURNAL_KEY",
    "ImagePart",
    "REPLAY_API",
    "SYSTEM",
    "SYSTEM_ROLES",
    "TEXT",
    "THINKING",
    "TOOL_CALL",
    "TOOL_RESULT",
    "TextPart",
    "USER",
    "assistant_message",
    "blocks_of",
    "has_images",
    "image_block",
    "image_payload",
    "is_assistant",
    "is_image",
    "is_text",
    "is_tool_result",
    "now_ms",
    "system_message",
    "text_block",
    "text_of",
    "thinking_block",
    "thinking_of",
    "to_ms",
    "tool_call_block",
    "tool_call_ids",
    "tool_calls_of",
    "tool_result_message",
    "user_message",
    "wire_projection",
    "with_appended_text",
    "with_blocks",
    "with_text",
]
