"""Reading a session file written before the records were pi messages.

Sessions used to be logged as OpenAI-style dicts: roles ``system`` / ``user`` /
``assistant`` / ``tool``, ``tool_calls`` whose ``function.arguments`` is a JSON
string, ``tool_call_id`` on a result, content either a string or ``text`` /
``image_url`` parts -- and, on an assistant turn the model service answered, a
verbatim copy of pi's own message under ``pi``. The store holds pi's shape now
(:mod:`opendde_harness.providers.messages`), so a file written by an older
build is converted as it is read.

Converted, never rewritten. Adding a field is not a reason to rewrite a
conversation, and an append-only log has no place to put one anyway: the
conversion lives in memory, the file keeps the bytes it was written with, and
:meth:`SessionManager.save` still appends only past what is on disk.

What each record becomes:

* an assistant dict carrying ``pi`` **is** that pi message -- the service wrote
  it, so its thinking signatures, native tool-call ids and ``api`` /
  ``provider`` / ``model`` triple survive exactly as they did when the copy was
  the thing being replayed. It gains a zero ``usage`` if it has none: the copy
  was stored without one, and pi reads ``usage.totalTokens`` off every message
  it replays without checking the field is there;
* an assistant dict without one is synthesised, with the
  :data:`~opendde_harness.providers.messages.REPLAY_API` marker that declares
  it cross-model: nothing in the record says which wire wrote it, and pi
  re-renders such a turn rather than trusting signatures it does not carry;
* a ``tool`` dict becomes a ``toolResult``, with ``toolName`` taken from the
  assistant call it answers (the record's own ``name`` is the fallback);
* a compaction marker passes through -- it was already an assistant record
  carrying ``compaction``, and only its content becomes a text block;
* a ``system`` record passes through unchanged. It is not a pi message and was
  never meant to be: ``to_context`` folds it into pi's single ``systemPrompt``,
  which is what happened to it before as well.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import COMPACTION_KEY

#: Keys the old shape put beside ``content`` that the new one expresses inside
#: it. Any one of them names a record written by the older build.
_RETIRED = ("pi", "tool_calls", "reasoning_content", "thinking_blocks")


def _blocks(content: Any) -> list[dict[str, Any]]:
    """Old content as pi blocks. An image that is not inline bytes is dropped.

    pi carries bytes only, so a remote ``image_url`` has nothing to become --
    and fetching one while loading a session file is not this function's job.
    Sessions store a placeholder in place of image bytes anyway, so a record
    carrying real ones is from a build that persisted them.
    """
    if isinstance(content, str):
        return [msg.text_block(content)] if content else []
    if not isinstance(content, list):
        return [] if content is None else [msg.text_block(json.dumps(content, ensure_ascii=False))]
    out: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == msg.TEXT:
            out.append(msg.text_block(str(part.get("text") or "")))
        elif kind == msg.IMAGE:
            out.append(part)
        elif kind == "image_url":
            url = part.get("image_url")
            raw = url.get("url") if isinstance(url, dict) else url
            if isinstance(raw, str) and raw.startswith("data:"):
                header, _, payload = raw.partition(",")
                if payload:
                    out.append(msg.image_block(payload, header[5:].split(";")[0]))
                    continue
            logger.debug("session: dropped a stored image that is not inline bytes")
    return out


def _arguments(raw: Any) -> dict[str, Any]:
    """An old tool call's arguments as an object. Anything else becomes ``{}``."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _merge(entry: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """``entry`` plus the record's own bookkeeping, which a conversion keeps.

    The ids and the turn are the journal's; ``compaction`` is the boundary a
    marker record is. The stored timestamp wins over the one the conversion
    minted: it is when the message was recorded, which is the fact the file
    holds, and no adapter reads a replayed message's timestamp at all.
    """
    out = dict(entry)
    for key in ("id", "turn_id", "timestamp"):
        if key in record:
            out[key] = record[key]
    if COMPACTION_KEY in record:
        out[COMPACTION_KEY] = record[COMPACTION_KEY]
    return out


def _has_openai_image(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(part, dict) and part.get("type") == "image_url" for part in content
    )


def _needs_conversion(record: dict[str, Any]) -> bool:
    """Whether this record is in the old shape.

    A pi message and a compaction marker both carry block-list content and
    neither carries the retired keys; everything else was written by a build
    that logged OpenAI dicts.
    """
    role = record.get("role")
    if role == "tool":
        return True
    if role == msg.ASSISTANT:
        if any(key in record for key in _RETIRED):
            return True
        return not isinstance(record.get("content"), list) or _has_openai_image(record["content"])
    if role == msg.USER:
        content = record.get("content")
        if isinstance(content, str):
            return False
        return not isinstance(content, list) or _has_openai_image(content)
    return False


def _assistant(record: dict[str, Any]) -> dict[str, Any]:
    """One old assistant record as a pi message."""
    stored = record.get("pi")
    if isinstance(stored, dict):
        # The zero shape rather than a guess: what the turn cost was not stored
        # beside the copy, and pi needs the field present, not truthful -- no
        # adapter bills a message it is replaying.
        return _merge({"usage": msg.zero_usage(), **stored}, record)
    calls = [
        msg.tool_call_block(
            str(call.get("id") or ""),
            str((call.get("function") or {}).get("name") or call.get("name") or ""),
            _arguments((call.get("function") or {}).get("arguments", call.get("arguments"))),
        )
        for call in record.get("tool_calls") or ()
        if isinstance(call, dict)
    ]
    return _merge(
        msg.assistant_message(
            _blocks(record.get("content")),
            tool_calls=calls,
            reasoning_content=record.get("reasoning_content"),
            thinking_blocks=record.get("thinking_blocks"),
        ),
        record,
    )


def read_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``records`` as pi messages, converting whatever was written in the old shape.

    One pass, because a ``tool`` record's ``toolName`` comes from the assistant
    call it answers, which is the record before it.
    """
    out: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    converted = 0
    for record in records:
        for call in msg.tool_calls_of(record):
            names[str(call.get("id"))] = str(call.get("name") or "")
        for call in record.get("tool_calls") or ():
            if isinstance(call, dict) and call.get("id"):
                names[str(call["id"])] = str((call.get("function") or {}).get("name") or call.get("name") or "")
        if not _needs_conversion(record):
            out.append(record)
            continue
        converted += 1
        role = record.get("role")
        if role == msg.ASSISTANT:
            out.append(_assistant(record))
        elif role == "tool":
            call_id = str(record.get("tool_call_id") or "")
            out.append(
                _merge(
                    msg.tool_result_message(
                        call_id,
                        names.get(call_id) or str(record.get("name") or ""),
                        _blocks(record.get("content")),
                        is_error=bool(record.get("is_error")),
                    ),
                    record,
                )
            )
        else:
            out.append({**record, "content": _blocks(record.get("content"))})
    if converted:
        logger.debug("session: read {} record(s) written in the pre-pi message shape", converted)
    return out


__all__ = ["read_records"]
