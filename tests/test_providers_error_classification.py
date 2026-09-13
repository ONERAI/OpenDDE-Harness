"""How a failure on *this* side of the pipe is bucketed.

A model's failure is read by pi, in the model service: whether it is transient,
whether the context window refused it, and pi's own error code all travel on the
error event, and ``providers/pi_context`` turns them into the verdict the loop
acts on (tests/test_pi_context.py). The tables of vendor wordings and status
codes that used to live here are pi's now.

What is left is the short list of failures that happen beside the model: the
child process gone, and the two stream deadlines ``pi_provider`` enforces because
pi has none. Both are transient; everything else is this process's own mistake.
"""

import asyncio

import pytest

from opendde_harness.providers.base import LLMProvider
from opendde_harness.providers.model_service import ModelServiceError


@pytest.mark.parametrize(
    "exc",
    [
        # What `asyncio.wait_for` raises when a gap budget expires. Its class
        # name is "timeouterror" and its str() is empty, so only the isinstance
        # check answers for it.
        asyncio.TimeoutError(),
        TimeoutError("the model service went silent before its first event"),
        # The child process exited, or closed its input under a send.
        ModelServiceError("gone", "the model service is not running"),
    ],
)
def test_a_failure_beside_the_model_is_a_retryable_network_error(exc):
    verdict = LLMProvider.classify_error(exc)

    assert (verdict.category, verdict.retryable) == ("network", True)


def test_a_refused_request_is_labelled_by_its_code_and_not_repeated():
    """The service's own refusals: a model it does not serve, a malformed call."""
    for code in ("model_not_found", "no_max_tokens", "invalid_params"):
        verdict = LLMProvider.classify_error(ModelServiceError(code, "no"))
        assert (verdict.category, verdict.retryable) == (code, False)


def test_anything_else_is_unknown_and_not_repeated():
    """A bug in this process. Sending the request again would reproduce it."""
    verdict = LLMProvider.classify_error(KeyError("tool_calls"))

    assert (verdict.category, verdict.retryable) == ("unknown", False)


def test_nothing_reads_the_message_text_any_more():
    """The wordings a vendor invents are pi's business, not this method's."""
    verdict = LLMProvider.classify_error(RuntimeError("429 rate limit exceeded, please retry"))

    assert verdict.category == "unknown", "a status code in a string is not a classification here"
