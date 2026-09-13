"""The memory library's LLM client, served by the model layer.

Everything the library asks its client for is answered by the conversation's
default model through the model service: no key, no endpoint and no model of
memory's own. The provider is faked here -- what is under test is the
translation between everalgo's chat protocol and this project's provider
contract, and that the model followed is the config's default at call time.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest
from everalgo.llm.config import LLMConfig
from everalgo.llm.errors import LLMError
from everalgo.llm.types import ChatMessage, ImageUrlInner, ImageUrlPart, TextPart
from pydantic import BaseModel

from opendde_harness.plugin.memory.longterm._service_llm import EMIT_TOOL, ModelServiceLLMClient
from opendde_harness.providers.base import LLMResponse, ToolCallRequest

PLACEHOLDER = LLMConfig(model="conversation-model", api_key="served-by-the-model-service", base_url="http://x")


class Episode(BaseModel):
    summary: str
    tags: list[str]


class _Provider:
    """Records the one request it is given and answers a scripted response."""

    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        return self.response


def _client(response: LLMResponse, model: str = "openai-codex/gpt-5.6-luna") -> tuple[ModelServiceLLMClient, _Provider]:
    provider = _Provider(response)
    return ModelServiceLLMClient(PLACEHOLDER, provider_factory=lambda: (provider, model)), provider


def _tool_answer(arguments: Any) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id="c1", name=EMIT_TOOL, arguments=arguments)],
        usage={"prompt_tokens": 12, "completion_tokens": 3},
        model="gpt-5.6-luna",
    )


async def test_a_structured_answer_is_asked_for_as_a_forced_tool_call_and_read_back_parsed():
    client, provider = _client(_tool_answer({"summary": "PR #12 merged", "tags": ["pr"]}))

    reply = await client.chat(
        [ChatMessage(role="system", content="annotate"), ChatMessage(role="user", content="the chunk")],
        model="conversation-model",
        response_format=Episode,
    )

    request = provider.calls[0]
    assert request["tool_choice"] == "required"
    tool = request["tools"][0]["function"]
    assert tool["name"] == EMIT_TOOL
    assert tool["parameters"]["properties"].keys() == {"summary", "tags"}
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert reply.parsed == Episode(summary="PR #12 merged", tags=["pr"])
    assert json.loads(reply.content) == {"summary": "PR #12 merged", "tags": ["pr"]}
    assert reply.usage is not None and (reply.usage.prompt_tokens, reply.usage.completion_tokens) == (12, 3)
    assert reply.finish_reason == "stop"
    assert reply.model == "gpt-5.6-luna"


async def test_a_model_that_answered_in_prose_is_read_for_the_json_it_wrote():
    """The model service has no wire spelling for a required tool call on every
    provider, so a model can answer in text; JSON there is still an answer."""
    client, _ = _client(LLMResponse(content='```json\n{"summary": "x", "tags": []}\n```'))

    reply = await client.chat([ChatMessage(role="user", content="go")], response_format=Episode)

    assert reply.parsed == Episode(summary="x", tags=[])


async def test_an_answer_that_is_not_the_class_asked_for_is_an_llm_error():
    client, _ = _client(LLMResponse(content="I would rather not."))

    with pytest.raises(LLMError, match="did not answer with a Episode"):
        await client.chat([ChatMessage(role="user", content="go")], response_format=Episode)

    client, _ = _client(_tool_answer({"summary": 3}))
    with pytest.raises(LLMError, match="does not validate"):
        await client.chat([ChatMessage(role="user", content="go")], response_format=Episode)


async def test_a_plain_answer_travels_as_text_without_a_tool():
    client, provider = _client(LLMResponse(content="two lines", truncated=True))

    reply = await client.chat([ChatMessage(role="user", content="summarise")], temperature=0.2, max_tokens=50)

    assert "tools" not in provider.calls[0] and "tool_choice" not in provider.calls[0]
    assert provider.calls[0]["max_tokens"] == 50
    assert reply.content == "two lines" and reply.parsed is None
    assert reply.finish_reason == "length"


async def test_the_model_is_the_configs_default_at_call_time_never_the_librarys_placeholder():
    """The toml's model is a placeholder and the decider's fallback is that
    placeholder too; the one that answers is whatever the config names now."""
    provider = _Provider(LLMResponse(content="ok"))
    models = iter(["openai-codex/gpt-5.6-luna", "anthropic/claude-opus-5"])
    client = ModelServiceLLMClient(PLACEHOLDER, provider_factory=lambda: (provider, next(models)))

    await client.chat([ChatMessage(role="user", content="a")], model="conversation-model")
    await client.chat([ChatMessage(role="user", content="b")], model="conversation-model")

    assert [call["model"] for call in provider.calls] == ["openai-codex/gpt-5.6-luna", "anthropic/claude-opus-5"]


async def test_an_inline_image_travels_as_image_bytes_and_anything_else_is_refused():
    payload = base64.b64encode(b"\x89PNG").decode()
    client, provider = _client(LLMResponse(content="a picture"))
    parts = [
        TextPart(text="what is this"),
        ImageUrlPart(image_url=ImageUrlInner(url=f"data:image/png;base64,{payload}")),
    ]

    await client.chat([ChatMessage(role="user", content=parts)])

    blocks = provider.calls[0]["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "what is this"}
    assert blocks[1] == {"type": "image", "data": payload, "mimeType": "image/png"}

    pdf = [ImageUrlPart(image_url=ImageUrlInner(url=f"data:application/pdf;base64,{payload}"))]
    with pytest.raises(LLMError, match="images only, not application/pdf"):
        await client.chat([ChatMessage(role="user", content=pdf)])
    with pytest.raises(LLMError, match="not a URL to fetch"):
        await client.chat(
            [ChatMessage(role="user", content=[ImageUrlPart(image_url=ImageUrlInner(url="https://x/y.png"))])]
        )


async def test_a_failed_model_call_is_the_librarys_one_error_type():
    client, _ = _client(LLMResponse(content="rate limited", finish_reason="error"))

    with pytest.raises(LLMError, match="rate limited"):
        await client.chat([ChatMessage(role="user", content="a")])


async def test_no_default_model_is_reported_rather_than_guessed(monkeypatch, tmp_path):
    from opendde_harness.config import loader
    from opendde_harness.plugin.memory.longterm import _service_llm
    from tests._config import write_config

    path = write_config(tmp_path / "config.json", {}, model="")
    monkeypatch.setattr(loader, "_current_config_path", path)
    loader._cache.clear()

    with pytest.raises(LLMError, match="no default model is configured"):
        _service_llm._default_provider()


def test_the_runner_points_the_library_at_this_client_and_at_the_given_config(monkeypatch, tmp_path):
    from opendde_harness.config import loader
    from opendde_harness.plugin.memory.longterm import _library, _server_runner

    seen: dict[str, Any] = {}
    monkeypatch.setattr(_library, "patch_llm_client", lambda factory: seen.setdefault("factory", factory))
    monkeypatch.setattr(_library, "start_server", lambda root: seen.setdefault("root", root))
    monkeypatch.setattr(_server_runner, "patch_llm_client", _library.patch_llm_client)
    monkeypatch.setattr(_server_runner, "start_server", _library.start_server)
    monkeypatch.setattr(loader, "_current_config_path", None)

    config = tmp_path / "config.json"
    _server_runner.main([*_library.SERVER_CMDLINE.split(), "--root", str(tmp_path / "memory"), "--config", str(config)])

    assert seen["factory"] is ModelServiceLLMClient
    assert seen["root"] == (tmp_path / "memory").resolve()
    assert loader.get_config_path() == config.resolve()

    with pytest.raises(SystemExit, match="--config PATH is required"):
        _server_runner.main([*_library.SERVER_CMDLINE.split(), "--root", str(tmp_path)])
