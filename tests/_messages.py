"""Terse pi-message builders for the tests.

One place, so a fixture says what the turn *was* rather than spelling out pi's
envelope. Thin wrappers over
:mod:`opendde_harness.providers.messages` -- the production constructors -- so a
test can never drift into a shape the harness does not write.
"""

from __future__ import annotations

from typing import Any

from opendde_harness.providers import messages as msg

user = msg.user_message
system = msg.system_message


def assistant(
    text: str = "",
    *,
    calls: list[tuple[str, str, dict[str, Any]]] | None = None,
    reasoning: str | None = None,
    thinking: list[dict[str, Any]] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """An assistant turn the harness wrote: text, and calls as ``(id, name, args)``."""
    return {
        **msg.assistant_message(
            text,
            tool_calls=[msg.tool_call_block(cid, name, args) for cid, name, args in calls or ()],
            reasoning_content=reasoning,
            thinking_blocks=thinking,
        ),
        **fields,
    }


def pi_assistant(
    text: str = "",
    *,
    calls: list[tuple[str, str, dict[str, Any]]] | None = None,
    thinking: list[dict[str, Any]] | None = None,
    api: str = "openai-completions",
    provider: str = "openai",
    model: str = "gpt-5",
    **fields: Any,
) -> dict[str, Any]:
    """An assistant turn as the model service answers it, envelope and all.

    The difference from :func:`assistant` is the ``api`` / ``provider`` /
    ``model`` triple: pi replays such a message as its own, signatures included,
    where a synthesised one is declared cross-model.
    """
    content: list[dict[str, Any]] = list(thinking or ())
    if text:
        content.append(msg.text_block(text))
    content.extend(msg.tool_call_block(cid, name, args) for cid, name, args in calls or ())
    return {
        "role": msg.ASSISTANT,
        "content": content,
        "api": api,
        "provider": provider,
        "model": model,
        "stopReason": "toolUse" if calls else "stop",
        "usage": msg.zero_usage(),
        "timestamp": msg.now_ms(),
        **fields,
    }


def tool_result(call_id: str, name: str, body: str | list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    """One tool result, body as text or as blocks."""
    return {**msg.tool_result_message(call_id, name, body), **fields}


def marker(compaction: dict[str, Any], text: str = "[compacted]") -> dict[str, Any]:
    """A compaction marker record: one text block and the boundary beside it."""
    return {"role": msg.ASSISTANT, "content": [msg.text_block(text)], "compaction": compaction}


def text_of(message: dict[str, Any]) -> str:
    return msg.text_of(message)


__all__ = ["assistant", "marker", "pi_assistant", "system", "text_of", "tool_result", "user"]
