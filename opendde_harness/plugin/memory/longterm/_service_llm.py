"""The memory library's LLM client, served by this project's own model layer.

The library builds its language model through one seam -- the client
builder ``_library.patch_llm_client`` replaces -- and talks to it through
everalgo's small chat protocol (``LLMClient.chat``). This is that client. Every
call goes to the conversation's default model through the model service
(pi-ai), the same way a turn does: a Codex sign-in serves memory as it serves
the conversation, and memory needs no model, key or endpoint of its own. The
``[llm]`` and ``[multimodal]`` sections of the library's toml hold placeholders
so the library builds a client at all; nothing in them is read here.

Which model: ``agents.defaults.model`` as the config stands at the moment of
the call, so a default changed in the picker reaches the next extraction
without a restart. The library's own ``model`` argument is ignored on purpose:
it is the placeholder from the toml, or the ``[decider]`` fallback to it.

Structured output is a forced tool call. The library asks for a pydantic
class (``response_format``) and reads ``parsed`` back; pi has no structured
outputs wire that every provider speaks, and a tool whose parameters are the
class's schema is what every provider does speak. A model that answers in
prose instead is given one more chance: JSON in the text, validated the same
way. Images travel as pi image blocks; a data URI of any other type -- a PDF,
an audio clip -- is refused rather than sent as text pretending to be the
file, because the model service carries bytes for images only.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from typing import Any, Callable

from everalgo.llm.config import LLMConfig
from everalgo.llm.errors import LLMError
from everalgo.llm.types import ChatMessage, ChatResponse, Usage
from pydantic import BaseModel, ValidationError

from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import LLMProvider, LLMResponse

logger = logging.getLogger("opendde_harness.plugin.memory.longterm")

#: The name of the one tool a structured answer is asked through.
EMIT_TOOL = "emit"

ProviderFactory = Callable[[], tuple[LLMProvider, str]]


def _default_provider() -> tuple[LLMProvider, str]:
    """The provider for the configured default model, and that model's id."""
    from opendde_harness.cli._helpers import make_provider
    from opendde_harness.config.loader import load_config

    config = load_config()
    model = str(config.agents.defaults.model or "")
    if not model:
        raise LLMError("no default model is configured, so memory has no model to run on: run `ddeharness onboard`")
    return make_provider(config), model


def _data_uri(url: str) -> tuple[str, str] | None:
    """``(mime, base64 payload)`` of a data URI, or None for any other URL."""
    if not url.startswith("data:"):
        return None
    header, sep, payload = url[5:].partition(",")
    if not sep or ";base64" not in header:
        return None
    mime = header.split(";", 1)[0].strip()
    try:
        base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None
    return mime, payload


def _blocks(content: Any) -> str | list[dict[str, Any]]:
    """everalgo content parts as pi content blocks."""
    if isinstance(content, str):
        return content
    blocks: list[dict[str, Any]] = []
    for part in content or ():
        dump = getattr(part, "model_dump", None)
        item = dump(exclude_none=True) if callable(dump) else part
        if not isinstance(item, dict):
            blocks.append(msg.text_block(str(item)))
            continue
        if item.get("type") == "text":
            blocks.append(msg.text_block(str(item.get("text", ""))))
            continue
        if item.get("type") == "image_url":
            image = item.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            parsed = _data_uri(str(url or ""))
            if parsed is None:
                raise LLMError("the model service carries image bytes, not a URL to fetch: pass the image inline")
            mime, payload = parsed
            if not mime.startswith("image/"):
                raise LLMError(f"the conversation model is reached through pi, which carries images only, not {mime}")
            blocks.append(msg.image_block(payload, mime))
            continue
        blocks.append(msg.text_block(json.dumps(item, ensure_ascii=False, default=str)))
    return blocks


def _pi_message(message: ChatMessage) -> dict[str, Any]:
    content = _blocks(message.content)
    if message.role == "system":
        return msg.system_message(content if isinstance(content, str) else "\n".join(_texts(content)))
    if message.role == "assistant":
        return msg.assistant_message(content)
    return msg.user_message(content)


def _texts(blocks: list[dict[str, Any]]) -> list[str]:
    return [str(block.get("text", "")) for block in blocks if block.get("type") == msg.TEXT]


def _emit_tool(schema: type[BaseModel]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": EMIT_TOOL,
                "description": f"Answer with one {schema.__name__}. Call this exactly once; do not answer in prose.",
                "parameters": schema.model_json_schema(),
            },
        }
    ]


def _structured(response: LLMResponse, schema: type[BaseModel]) -> BaseModel:
    """The answer as the class the library asked for, from the call or the text."""
    candidate: Any = None
    if response.has_tool_calls:
        candidate = response.tool_calls[0].arguments
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except json.JSONDecodeError:
                candidate = None
    if candidate is None and response.content:
        text = response.content.strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1] if "\n" in text else ""
        try:
            candidate = json.loads(text)
        except json.JSONDecodeError:
            candidate = None
    if not isinstance(candidate, dict):
        raise LLMError(f"the model did not answer with a {schema.__name__}")
    try:
        return schema.model_validate(candidate)
    except ValidationError as exc:
        raise LLMError(f"the model's {schema.__name__} does not validate: {exc}") from exc


class ModelServiceLLMClient:
    """everalgo's ``LLMClient``, over the conversation's own model."""

    def __init__(self, config: LLMConfig, *, provider_factory: ProviderFactory | None = None) -> None:
        # Nothing of the library's config is used: see the module docstring.
        self._factory = provider_factory or _default_provider
        self._provider: LLMProvider | None = None
        self._model = ""

    def _resolve(self) -> tuple[LLMProvider, str]:
        provider, model = self._factory()
        if self._provider is None or model != self._model:
            self._provider, self._model = provider, model
        return self._provider, self._model

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: type[BaseModel] | None = None,
        **extra: Any,
    ) -> ChatResponse:
        del model, temperature, extra  # the placeholder model; generation settings are the model row's
        provider, model_id = self._resolve()
        structured = isinstance(response_format, type) and issubclass(response_format, BaseModel)
        kwargs: dict[str, Any] = {"messages": [_pi_message(m) for m in messages], "model": model_id}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if structured:
            kwargs["tools"] = _emit_tool(response_format)
            kwargs["tool_choice"] = "required"

        try:
            response = await provider.chat(**kwargs)
        except Exception as exc:  # noqa: BLE001 - the library's contract is one error type
            raise LLMError(str(exc)) from exc
        if response.finish_reason == "error":
            raise LLMError(response.content or "the model service reported an error")

        parsed = _structured(response, response_format) if structured else None
        content = response.content or ""
        if parsed is not None:
            content = parsed.model_dump_json()
        usage_dict = response.usage or {}
        usage = Usage(
            prompt_tokens=_count(usage_dict.get("prompt_tokens")),
            completion_tokens=_count(usage_dict.get("completion_tokens")),
        )
        finish: Any = "length" if response.truncated or response.finish_reason == "length" else "stop"
        return ChatResponse(
            content=content, model=response.model or model_id, usage=usage, finish_reason=finish, parsed=parsed
        )


def _count(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = ["EMIT_TOOL", "ModelServiceLLMClient"]
