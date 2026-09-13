"""JSON-driven LLMProvider used by the real AgentLoop in the PTY tests."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from opendde_harness.providers.base import (
    ErrorClassification,
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    StreamDelta,
    ToolCallRequest,
)
from opendde_harness.token_wise.base import CURRENT_SESSION_KEY


def user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            return "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


class ScriptedProvider(LLMProvider):
    """Rules match user text; response steps advance through real tool calls.

    A gate is a file the driver releases after observing the UI. Waiting is
    cancellable, so slow/queue tests do not depend on model/network timing.
    Unknown prompts are errors, never plausible-looking fallback answers.
    """

    provider_name = "scripted"

    def __init__(self, config, journal):
        super().__init__()
        self.model = config.agents.defaults.model
        self.model_overlays = config.providers.model_overlays()
        self.generation = GenerationSettings(timeout=30, first_token_timeout=10, idle_timeout=30)
        self.rules = json.loads(Path(os.environ["OPENDDE_HARNESS_E2E_SCRIPT"]).read_text())["rules"]
        self.journal = journal
        self.sequence = 0

    def get_default_model(self):
        return self.model

    async def chat_stream(self, messages, tools=None, model=None, **kwargs):
        prompt = user_text(messages)
        rule = next((r for r in self.rules if r["match"] in prompt), None)
        if rule is None:
            self.journal("script_mismatch", reason="no matching prompt rule")
            raise RuntimeError("ScriptedProvider received an unexpected prompt")
        session = CURRENT_SESSION_KEY.get() or "unscoped"
        name = rule["match"]
        # A turn can be cancelled while a tool runs, after this generator has
        # already returned. Derive the step from the real current-turn history
        # instead of retaining a cursor that would survive that cancellation.
        user_index = max(i for i, m in enumerate(messages) if m.get("role") == "user")
        index = sum(bool(m.get("tool_calls")) for m in messages[user_index + 1 :] if m.get("role") == "assistant")
        responses = rule["responses"]
        if index >= len(responses):
            self.journal("script_mismatch", reason="response script exhausted", rule=name)
            raise RuntimeError("ScriptedProvider response script exhausted")
        response = responses[index]
        self.sequence += 1
        call = self.sequence
        self.journal("provider_start", rule=name, step=index, session=session, call=call)
        outcome = "complete"
        try:
            expected = response.get("expect_tool_result")
            if expected:
                results = [m.get("content", "") for m in messages if m.get("role") == "tool"]
                if not any(expected in str(value) for value in results):
                    self.journal("script_mismatch", reason="real tool result missing", rule=name)
                    raise RuntimeError("AgentLoop did not return the expected real tool result")
                self.journal("tool_result_verified", rule=name, expected=expected)

            for chunk in response.get("chunks", []):
                if chunk.get("delay_ms"):
                    await asyncio.sleep(chunk["delay_ms"] / 1000)
                yield StreamDelta(content=chunk.get("text"), reasoning_content=chunk.get("thinking"))

            if gate := response.get("gate"):
                self.journal("provider_waiting", rule=name, gate=gate)
                async with asyncio.timeout(60):
                    while not Path(gate).exists():
                        await asyncio.sleep(0.025)
                for text in response.get("after_gate", []):
                    yield StreamDelta(content=text)

            calls = response.get("tool_calls", [])
            if calls:
                yield StreamDelta(
                    content=None,
                    tool_call_delta={
                        "tool_calls": [
                            {
                                "index": i,
                                "id": f"e2e-{call}-{i}",
                                "function": {"name": t["name"], "arguments": json.dumps(t["arguments"])},
                            }
                            for i, t in enumerate(calls)
                        ]
                    },
                )
            if error := response.get("error"):
                outcome = "error"
                yield StreamDelta(
                    content=error,
                    finish_reason="error",
                    error_classification=ErrorClassification("scripted_error", retryable=False),
                )
            else:
                yield StreamDelta(
                    content=None,
                    finish_reason="tool_calls" if calls else "stop",
                    usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                )
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            self.journal("provider_end", rule=name, step=index, session=session, call=call, outcome=outcome)

    async def chat(self, messages, tools=None, model=None, **kwargs):
        """Non-streamed consumers use the same script, with no network path."""
        text, thinking, calls = [], [], []
        terminal: dict[str, Any] = {}
        async for delta in self.chat_stream(messages, tools, model, **kwargs):
            if delta.content:
                text.append(delta.content)
            if delta.reasoning_content:
                thinking.append(delta.reasoning_content)
            if delta.tool_call_delta:
                for call in delta.tool_call_delta["tool_calls"]:
                    calls.append(
                        ToolCallRequest(
                            id=call["id"],
                            name=call["function"]["name"],
                            arguments=json.loads(call["function"]["arguments"]),
                        )
                    )
            if delta.finish_reason:
                terminal = {
                    "finish_reason": delta.finish_reason,
                    "usage": delta.usage or {},
                    "error_classification": delta.error_classification,
                }
        return LLMResponse(
            content="".join(text),
            reasoning_content="".join(thinking) or None,
            tool_calls=calls,
            model=model or self.model,
            **terminal,
        )
