"""A failed model call is run again, bounded and in the user's sight.

The budget is spent inside the model service, by pi-ai's own retry loop: the
request carries ``retry: {"maxRetries": n}`` from ``agents.defaults.llm_retries``
and pi decides which failures are worth repeating and waits between attempts.
What reaches the loop is one ``retry`` event per new attempt, which voids
whatever the discarded one streamed and is announced through ``TurnRetry``. A
failure pi calls deterministic is the turn's answer at once, whatever the budget.

The ``faux-flaky`` model in the service scripts the failures: its prompt says
how many attempts fail and whether pi reads the wording as transient.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from opendde_harness.agent.loop import AgentLoop
from opendde_harness.agent.loop.factory import AgentLoopSettings
from opendde_harness.providers.base import GenerationSettings
from opendde_harness.providers.model_service import ModelService
from opendde_harness.providers.pi_provider import PiModelProvider

BUNDLE = Path(__file__).resolve().parent.parent / "ui-tui" / "dist" / "model-service.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    not BUNDLE.exists() or NODE is None,
    reason="ui-tui/dist/model-service.js is not built (cd ui-tui && npm run build) or node is missing",
)

#: Fails to order, then echoes. See ``fauxStep`` in the service's ``main.ts``.
FLAKY = "faux-flaky/echo"


class Counting(ModelService):
    """The service, plus how many stream requests it was sent."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.streams = 0

    async def stream_with_id(self, provider, model, context, options=None, *, replay=None, retry=None, timeouts=None):
        self.streams += 1
        return await super().stream_with_id(
            provider, model, context, options, replay=replay, retry=retry, timeouts=timeouts
        )


@pytest.fixture
async def service():
    svc = Counting(node=NODE, bundle=BUNDLE, env={"OPENDDE_MODEL_SERVICE_FAUX": "1"})
    await svc.start()
    try:
        yield svc
    finally:
        await svc.close()


def _loop(tmp_path, svc: ModelService, retries: int = 3) -> AgentLoop:
    built = PiModelProvider(svc, "faux-flaky", "echo")
    # The backoff pi waits is 2s, doubled per attempt; a test that spent it
    # would be testing the clock. The gap budgets are what bound the wait here:
    # nothing arrives between the failed attempt and the next one, so a short
    # idle budget would end the stream instead of letting it retry -- hence a
    # generous one, and a scripted failure that is instant.
    built.generation = GenerationSettings(retries=retries, first_token_timeout=30.0, idle_timeout=30.0)
    return AgentLoop(built, tmp_path, AgentLoopSettings(model=FLAKY))


async def _call(loop: AgentLoop, prompt: str):
    shown: list[str] = []
    retries: list[tuple[int, int, str, bool]] = []

    async def on_token(text: str) -> None:
        shown.append(text)

    async def on_retry(attempt: int, total: int, reason: str, discard: bool) -> None:
        retries.append((attempt, total, reason, discard))

    response = await loop._llm_call_stream(
        [{"role": "user", "content": prompt}],
        None,
        FLAKY,
        on_token_delta=on_token,
        on_retry=on_retry,
    )
    return response, shown, retries


async def test_a_failed_attempt_is_announced_and_the_next_one_answers(tmp_path, service):
    """One request, two attempts. The first one's failure never ends the turn."""
    loop = _loop(tmp_path, service, retries=3)

    response, shown, retries = await _call(loop, "fail:1 please")

    assert service.streams == 1, "the attempts are the service's, not two requests"
    assert retries == [(2, 4, "503 service unavailable", False)], "announced, with pi's own wording"
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "echo"
    assert "".join(shown).startswith("Echo: fail:1 please.")


async def test_the_second_attempts_answer_is_the_only_one_kept(tmp_path, service):
    """Two failures, then an echo. Nothing the discarded attempts left survives."""
    loop = _loop(tmp_path, service, retries=3)

    response, _shown, retries = await _call(loop, "fail:2 twice")

    assert [(a, t) for a, t, _r, _d in retries] == [(2, 4), (3, 4)]
    assert response.finish_reason == "tool_calls"
    assert response.content.count("Echo: fail:2 twice.") == 6, "one attempt's text, not three"


async def test_a_budget_of_zero_is_one_attempt(tmp_path, service):
    """The same scripted failure; only the budget differs."""
    loop = _loop(tmp_path, service, retries=0)

    response, _shown, retries = await _call(loop, "fail:1 please")

    assert retries == []
    assert response.finish_reason == "error"
    assert "503 service unavailable" in (response.content or "")
    assert response.error_classification.retryable is True, "transient, but nothing was left to spend"


async def test_a_deterministic_failure_is_the_answer_at_once(tmp_path, service):
    """What separates this from the retried case is pi's verdict, not the budget."""
    loop = _loop(tmp_path, service, retries=3)

    response, _shown, retries = await _call(loop, "fatal, do not repeat this")

    assert retries == []
    assert response.finish_reason == "error"
    assert response.error_classification.retryable is False
    assert "malformed" in (response.content or "")


async def test_text_already_shown_is_marked_void_when_the_attempt_had_delivered(tmp_path, service):
    """``discard`` is what tells an outlet to start its live text over.

    The scripted failure arrives before any token, so the flag is False above.
    Here the first attempt streams and the loop is told the re-run replaces it.
    """
    loop = _loop(tmp_path, service, retries=1)
    delivered: list[bool] = []

    async def on_token(_text: str) -> None:
        pass

    async def on_retry(_attempt: int, _total: int, _reason: str, discard: bool) -> None:
        delivered.append(discard)

    # A retry the loop sees after it has delivered: the delta carrying the
    # retry is fed in after a token one, which is the ordering a stream that
    # broke mid-answer produces.
    from opendde_harness.providers.base import StreamDelta

    async def broken(**_kwargs):
        yield StreamDelta(content="half an answer")
        yield StreamDelta(content=None, retry={"attempt": 1, "total": 2, "reason": "503"})
        yield StreamDelta(content="the whole answer")
        yield StreamDelta(content=None, finish_reason="stop")

    loop.provider.chat_stream = broken  # type: ignore[method-assign]
    response = await loop._llm_call_stream(
        [{"role": "user", "content": "hi"}], None, FLAKY, on_token_delta=on_token, on_retry=on_retry
    )

    assert delivered == [True]
    assert response.content == "the whole answer", "the discarded attempt's text is not in the answer"
