"""The context the model service is sent, and the events it sends back.

The messages pass through: the store holds pi's shape, so what is tested here
is the folding of the system prefix, the tool schemas, and that nothing of the
harness's own bookkeeping rides along. The mapping that used to live here --
OpenAI dicts in, pi types out -- is gone with the shape it converted; what is
left of it reads a session file written before the change
(``session/legacy.py``, ``tests/test_session_legacy.py``).

The mapping tests are pure. The round-trips at the end need the built bundle
(``cd ui-tui && npm run build``) and skip without it; neither reaches a vendor
-- the first talks to the faux provider, the last is refused for want of a key
before any request is sent.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from opendde_harness.providers import messages as msg
from opendde_harness.providers.model_service import ModelService
from opendde_harness.providers.pi_context import event_to_delta, response_from_done, to_context
from tests._messages import assistant, marker, pi_assistant, system, tool_result, user

BUNDLE = Path(__file__).resolve().parent.parent / "ui-tui" / "dist" / "model-service.js"
NODE = shutil.which("node")

PIXEL = "iVBORw0KGgoAAAANSUhEUg=="
DATA_URL = f"data:image/png;base64,{PIXEL}"

#: Every vendor key pi-ai would find in the environment. Cleared before the
#: no-key round-trip so the child cannot reach a real endpoint.
VENDOR_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_OAUTH_TOKEN",
    "AZURE_OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "OPENROUTER_API_KEY",
    "XAI_API_KEY",
)


def convert(messages, tools=None, *, provider="openai", model="gpt-4o-mini"):
    return to_context(messages, tools, provider=provider, model=model)


# -- Context: system ---------------------------------------------------------


def test_leading_system_messages_join_into_one_system_prompt():
    context = convert([system("one"), {"role": "developer", "content": "two"}, user("hi")])
    assert context["systemPrompt"] == "one\n\ntwo"
    assert [m["role"] for m in context["messages"]] == ["user"]


def test_no_system_message_leaves_the_key_out():
    assert "systemPrompt" not in convert([user("hi")])


def test_a_late_system_message_is_prefixed_onto_the_next_user_message():
    context = convert([system("rules"), user("first"), system("the repo moved"), user("second")])
    assert context["systemPrompt"] == "rules"
    assert [m["content"] for m in context["messages"]] == ["first", "the repo moved\n\nsecond"]


def test_a_late_system_message_prefixes_a_block_list_too():
    context = convert([user("first"), system("note"), user([msg.text_block("second")])])
    assert context["messages"][1]["content"] == [
        {"type": "text", "text": "note"},
        {"type": "text", "text": "second"},
    ]


def test_a_trailing_system_message_with_no_user_after_it_joins_the_system_prompt():
    context = convert([system("rules"), user("hi"), system("and one more thing")])
    assert context["systemPrompt"] == "rules\n\nand one more thing"
    assert [m["role"] for m in context["messages"]] == ["user"]


# -- Context: what passes through -------------------------------------------


def test_a_plain_user_string_stays_a_string():
    message = convert([user("hello")])["messages"][0]
    assert message["role"] == "user"
    assert message["content"] == "hello"
    assert isinstance(message["timestamp"], int)


def test_a_user_message_carrying_a_picture_passes_through_whole():
    blocks = [msg.text_block("look"), msg.image_block(PIXEL, "image/png")]
    assert convert([user(blocks)])["messages"][0]["content"] == blocks


def test_the_harness_own_bookkeeping_never_reaches_the_wire():
    context = convert(
        [
            {**user("hi"), "id": "r1", "turn_id": "t1", "_journal_id": "j1"},
            {**pi_assistant("there"), "id": "r2", "turn_id": "t1"},
            {**tool_result("c1", "read", "body"), "id": "r3", "turn_id": "t1", "_recovery_synthetic": True},
        ]
    )
    for message in context["messages"]:
        assert not {"id", "turn_id", "_journal_id", "_recovery_synthetic"} & set(message)


PI_ASSISTANT = pi_assistant(
    "reading",
    thinking=[{"type": "thinking", "thinking": "signed reasoning", "thinkingSignature": "sig-1"}],
    calls=[("fc_68a|b" * 9, "read", {"path": "a.txt"})],
    api="openai-responses",
    responseId="resp_1",
    providerThinkingLevel="medium",
    timestamp=1,
)
PI_ASSISTANT["content"][2]["thoughtSignature"] = "tsig"
PI_ASSISTANT["usage"] = {"input": 5, "output": 7, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 12, "cost": {}}


def test_the_service_own_message_is_replayed_exactly_as_pi_produced_it():
    assert convert([PI_ASSISTANT])["messages"][0] == PI_ASSISTANT


def test_an_assistant_record_the_harness_wrote_is_declared_cross_model():
    """A compaction marker on a model that does not replay it: ordinary content.

    The envelope is completed here because the record has none -- it is a
    boundary in the session log, not an answer -- and the ``compaction`` key
    itself goes no further than that log.
    """
    message = convert([marker({"provider": "openai", "model": "other", "items": []}, "[compacted]")])["messages"][0]
    assert message["api"] == "opendde-harness-replay"
    assert message["provider"] == "openai"
    assert message["model"] == "gpt-4o-mini"
    assert message["stopReason"] == "stop"
    assert message["content"] == [msg.text_block("[compacted]")]
    assert message["usage"]["totalTokens"] == 0
    assert "compaction" not in message


def test_a_tool_result_passes_through_with_its_pairing():
    context = convert([pi_assistant(calls=[("call_1", "read", {})]), tool_result("call_1", "read", "file body")])
    assert context["messages"][1] == {
        "role": "toolResult",
        "toolCallId": "call_1",
        "toolName": "read",
        "content": [msg.text_block("file body")],
        "isError": False,
        "timestamp": context["messages"][1]["timestamp"],
    }


# -- Context: tools ----------------------------------------------------------


def test_a_tool_schema_passes_through_whole():
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "enum": ["a", "b"]}},
        "required": ["path"],
    }
    context = convert(
        [user("hi")],
        [{"type": "function", "function": {"name": "read", "description": "read a file", "parameters": schema}}],
    )
    assert context["tools"] == [{"name": "read", "description": "read a file", "parameters": schema}]


def test_no_tools_leaves_the_key_out():
    assert "tools" not in convert([user("hi")], [])


# -- Events ------------------------------------------------------------------


def test_a_text_delta_is_content():
    delta = event_to_delta({"type": "text_delta", "contentIndex": 0, "delta": "hel"})
    assert delta.content == "hel"
    assert delta.finish_reason is None


def test_a_thinking_delta_is_reasoning():
    delta = event_to_delta({"type": "thinking_delta", "contentIndex": 0, "delta": "hmm"})
    assert delta.content is None
    assert delta.reasoning_content == "hmm"


def test_a_finished_tool_call_is_one_openai_style_fragment():
    delta = event_to_delta(
        {
            "type": "toolcall_end",
            "contentIndex": 2,
            "toolCall": {"type": "toolCall", "id": "call_9", "name": "read", "arguments": {"path": "a.txt"}},
        }
    )
    assert delta.tool_call_delta == {
        "tool_calls": [
            {
                "index": 2,
                "id": "call_9",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path": "a.txt"}'},
            }
        ]
    }


@pytest.mark.parametrize(
    ("reason", "finish"),
    [("stop", "stop"), ("toolUse", "tool_calls"), ("length", "length"), ("deferred", "stop")],
)
def test_done_maps_every_stop_reason(reason, finish):
    delta = event_to_delta({"type": "done", "reason": reason, "message": {"role": "assistant", "content": []}})
    assert delta.finish_reason == finish


def test_done_maps_usage_and_the_cost_pi_computed():
    delta = event_to_delta(
        {
            "type": "done",
            "reason": "stop",
            "message": {
                "role": "assistant",
                "content": [],
                "usage": {
                    "input": 100,
                    "output": 20,
                    "cacheRead": 60,
                    "cacheWrite": 5,
                    "reasoning": 8,
                    "totalTokens": 185,
                    "cost": {"input": 0.1, "output": 0.2, "cacheRead": 0.01, "cacheWrite": 0.02, "total": 0.33},
                },
            },
        }
    )
    assert delta.usage == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 185,
        "cache_read_input_tokens": 60,
        "cache_creation_input_tokens": 5,
        "completion_tokens_details": {"reasoning_tokens": 8},
        "cost": 0.33,
    }


def test_usage_without_cache_figures_keeps_the_keys_out():
    delta = event_to_delta(
        {
            "type": "done",
            "reason": "stop",
            "message": {"usage": {"input": 10, "output": 2, "totalTokens": 12, "cost": {"total": 0.0}}},
        }
    )
    assert "cache_read_input_tokens" not in delta.usage
    assert "cache_creation_input_tokens" not in delta.usage
    assert delta.usage["cost"] == 0.0


def test_an_error_event_is_terminal_and_carries_the_services_own_verdict():
    """Every flag is pi's, read in the service and forwarded; none is guessed here."""
    delta = event_to_delta(
        {
            "type": "error",
            "reason": "error",
            "retryable": True,
            "overflow": False,
            "error": {"stopReason": "error", "errorMessage": "429 rate limit exceeded"},
        }
    )
    assert delta.finish_reason == "error"
    assert delta.content == "429 rate limit exceeded"
    assert delta.error_classification.category == "server"
    assert delta.error_classification.retryable is True
    assert delta.error_classification.should_compress is False


def test_an_error_without_a_retryable_flag_is_not_retryable():
    delta = event_to_delta(
        {"type": "error", "reason": "error", "error": {"errorMessage": "401 invalid api key"}},
    )
    assert delta.error_classification.retryable is False
    assert delta.error_classification.should_compress is False


@pytest.mark.parametrize("code", ["auth", "oauth"])
def test_a_credential_code_is_auth_and_never_retryable(code):
    """pi raises these before a request is sent, so nothing to repeat."""
    delta = event_to_delta(
        {
            "type": "error",
            "reason": "error",
            "code": code,
            "retryable": True,
            "error": {"errorMessage": "Provider is not configured: openai"},
        }
    )
    assert delta.error_classification.category == "auth"
    assert delta.error_classification.retryable is False


def test_any_other_code_labels_the_failure_and_keeps_the_services_retry_verdict():
    delta = event_to_delta(
        {
            "type": "error",
            "reason": "error",
            "code": "stream",
            "retryable": True,
            "error": {"errorMessage": "503 service unavailable"},
        }
    )
    assert delta.error_classification.category == "stream"
    assert delta.error_classification.retryable is True


def test_the_overflow_flag_is_what_asks_the_loop_to_compress():
    """The wording is pi's business. Here it is a boolean, and nothing reads the text."""
    delta = event_to_delta(
        {
            "type": "error",
            "reason": "error",
            "overflow": True,
            "error": {"errorMessage": "This model's maximum context length is 128000 tokens"},
        }
    )
    assert delta.error_classification.should_compress is True

    # The same wording without the flag asks for nothing: a service that did not
    # say so did not see an overflow, and this end does not second-guess it.
    unflagged = event_to_delta(
        {
            "type": "error",
            "reason": "error",
            "error": {"errorMessage": "This model's maximum context length is 128000 tokens"},
        }
    )
    assert unflagged.error_classification.should_compress is False


def test_an_abort_is_its_own_category_and_never_retryable():
    delta = event_to_delta(
        {"type": "error", "reason": "aborted", "retryable": True, "error": {"errorMessage": "aborted"}}
    )
    assert delta.error_classification.category == "aborted"
    assert delta.error_classification.retryable is False


def test_a_retry_event_is_the_attempt_about_to_run_and_voids_what_came_before():
    """pi counts retries and the loop counts calls, so both numbers gain one."""
    delta = event_to_delta(
        {
            "type": "retry",
            "attempt": 1,
            "maxAttempts": 3,
            "delayMs": 2000,
            "errorMessage": "503 service unavailable",
        }
    )
    assert delta.retry == {"attempt": 2, "total": 4, "reason": "503 service unavailable"}
    assert delta.finish_reason is None, "a retry does not end the stream"
    assert delta.content is None


def test_a_retry_reason_is_one_short_line():
    """A provider's failure can be a whole JSON body; the announcement is a line."""
    delta = event_to_delta(
        {"type": "retry", "attempt": 1, "maxAttempts": 1, "errorMessage": "line one\nline two " + "x" * 200}
    )
    assert "\n" not in delta.retry["reason"]
    assert len(delta.retry["reason"]) <= 80
    assert delta.retry["reason"].endswith("\u2026")


@pytest.mark.parametrize(
    "kind", ["start", "text_start", "text_end", "thinking_start", "thinking_end", "toolcall_start", "toolcall_delta"]
)
def test_events_the_loop_does_not_need_are_ignored(kind):
    assert event_to_delta({"type": kind, "contentIndex": 0, "delta": "x"}) is None


# -- The non-streaming path --------------------------------------------------


def test_a_done_message_becomes_one_response():
    response = response_from_done(
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "because"},
                {"type": "text", "text": "here you go"},
                {"type": "toolCall", "id": "call_1", "name": "read", "arguments": {"path": "a.txt"}},
            ],
            "model": "gpt-4o-mini",
            "responseModel": "gpt-4o-mini-2024-07-18",
            "stopReason": "toolUse",
            "usage": {"input": 5, "output": 7, "totalTokens": 12},
        }
    )
    assert response.content == "here you go"
    assert response.reasoning_content == "because"
    assert response.finish_reason == "tool_calls"
    assert response.model == "gpt-4o-mini-2024-07-18"
    assert response.usage["prompt_tokens"] == 5
    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [("call_1", "read", {"path": "a.txt"})]


def test_a_done_response_keeps_pis_whole_message_for_replay_usage_included():
    """The usage travels with the message it belongs to.

    pi reads ``usage.totalTokens`` off every message it replays and does not
    check the field is there, so a stored turn without one failed the next
    request outright. It is also what makes the record say what the turn cost.
    """
    response = response_from_done(PI_ASSISTANT)
    assert response.pi_message == PI_ASSISTANT
    assert response.pi_message["usage"]["totalTokens"] == 12
    assert response.usage["prompt_tokens"] == 5
    # What it keeps is exactly what to_context replays.
    assert convert([response.pi_message])["messages"][0] == PI_ASSISTANT


def test_a_failed_turn_keeps_no_message_because_pi_would_drop_it():
    response = response_from_done({"content": [], "stopReason": "error", "errorMessage": "boom"})
    assert response.pi_message is None


def test_the_done_delta_carries_the_whole_answer_as_the_authoritative_response():
    delta = event_to_delta({"type": "done", "reason": "toolUse", "message": PI_ASSISTANT})
    assert delta.final_response.finish_reason == "tool_calls"
    assert delta.final_response.pi_message == PI_ASSISTANT


def test_a_length_stop_is_reported_as_truncated():
    response = response_from_done({"content": [{"type": "text", "text": "cut"}], "stopReason": "length"})
    assert response.finish_reason == "length"
    assert response.truncated is True


def test_an_errored_message_carries_its_message_and_a_classification():
    response = response_from_done({"content": [], "stopReason": "error", "errorMessage": "401 invalid api key"})
    assert response.finish_reason == "error"
    assert response.content == "401 invalid api key"
    # A `done` event never carries a failure -- pi ends such a turn with an
    # `error` event -- so there are no service flags to read here, and the
    # verdict says only that the turn failed.
    assert response.error_classification.category == "server"
    assert response.error_classification.retryable is False


# -- Round trips through the real service ------------------------------------

pytestmark_service = pytest.mark.skipif(
    not BUNDLE.exists() or NODE is None,
    reason="ui-tui/dist/model-service.js is not built (cd ui-tui && npm run build) or node is missing",
)

FULL_HISTORY = [
    system("You are terse."),
    user("read a.txt"),
    assistant("reading", calls=[("call_1", "read", {"path": "a.txt"})]),
    tool_result("call_1", "read", "the file body"),
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    }
]


@pytestmark_service
async def test_pi_accepts_a_translated_context_end_to_end():
    context = convert(FULL_HISTORY, TOOLS, provider="faux", model="echo")
    service = ModelService(node=NODE, bundle=BUNDLE, env={"OPENDDE_MODEL_SERVICE_FAUX": "1"})
    await service.start()
    try:
        events = [e async for e in service.stream("faux", "echo", context)]
    finally:
        await service.close()

    assert events[-1]["type"] == "done"
    assert events[-1]["reason"] == "toolUse"
    deltas = [d for d in (event_to_delta(e) for e in events) if d is not None]
    assert "".join(d.content for d in deltas if d.content) != ""
    assert deltas[-1].finish_reason == "tool_calls"
    response = deltas[-1].final_response
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "echo"
    assert response.pi_message["api"] == events[-1]["message"]["api"]


@pytestmark_service
async def test_pi_accepts_its_own_message_replayed_verbatim():
    """The turn pi produced, stored and sent straight back, is accepted."""
    service = ModelService(node=NODE, bundle=BUNDLE, env={"OPENDDE_MODEL_SERVICE_FAUX": "1"})
    await service.start()
    try:
        first = [e async for e in service.stream("faux", "echo", convert(FULL_HISTORY, TOOLS))]
        stored = response_from_done(first[-1]["message"]).pi_message
        call = next(b for b in stored["content"] if b["type"] == "toolCall")
        history = [*FULL_HISTORY, stored, tool_result(call["id"], call["name"], "echoed")]
        context = convert(history, TOOLS)
        assert context["messages"][-2] == stored
        second = [e async for e in service.stream("faux", "echo", context)]
    finally:
        await service.close()

    assert second[-1]["type"] == "done"
    assert second[-1]["reason"] == "toolUse"


@pytestmark_service
async def test_a_keyless_builtin_model_fails_on_auth_not_on_message_shape(monkeypatch):
    for key in VENDOR_KEYS:
        monkeypatch.delenv(key, raising=False)
    context = convert(FULL_HISTORY, TOOLS, provider="openai", model="gpt-4o-mini")
    service = ModelService(node=NODE, bundle=BUNDLE, env={})
    await service.start()
    try:
        events = [e async for e in service.stream("openai", "gpt-4o-mini", context)]
    finally:
        await service.close()

    assert [e["type"] for e in events] == ["error"]
    delta = event_to_delta(events[0])
    assert delta.finish_reason == "error"
    text = delta.content.lower()
    # pi checks credentials before it converts the history, so a missing key is
    # the whole failure: nothing in the message names a message or a block.
    assert delta.error_classification.category == "auth", f"not an auth failure: {delta.content}"
    shape_words = ("messages[", "invalid_request_error", "tool_call_id", "content block", "unexpected role")
    assert not any(word in text for word in shape_words), (
        f"pi rejected the message shapes, not the missing key: {delta.content}"
    )
    assert json.dumps(context)  # the context this test sent is JSON, as the wire requires
