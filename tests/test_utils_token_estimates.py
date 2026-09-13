"""Token estimates are additive and cached by content, so a turn costs the new text."""

from opendde_harness.providers import messages as msg
from opendde_harness.utils import helpers
from opendde_harness.utils.helpers import (
    count_text_tokens,
    estimate_message_tokens,
    estimate_prompt_tokens,
    take_tokens,
    truncate_to_tokens,
)
from tests import _messages as build


def test_a_repeated_estimate_never_encodes_twice(monkeypatch):
    encodes = []
    real = helpers.tiktoken.get_encoding

    class _Counting:
        def __init__(self, enc):
            self._enc = enc

        def encode(self, text, **kwargs):
            encodes.append(len(text))
            return self._enc.encode(text, **kwargs)

    monkeypatch.setattr(helpers.tiktoken, "get_encoding", lambda name: _Counting(real(name)))
    helpers._TOKEN_COUNT_CACHE.clear()
    message = build.user("the same words, counted once " * 50)

    first = estimate_message_tokens(message)
    second = estimate_message_tokens(dict(message))

    assert first == second > 0
    assert len(encodes) == 1


def test_a_truncation_is_a_prefix_of_the_source_at_every_boundary():
    """The cut lands on a token boundary, which is not a character boundary;
    decoding across one returns a replacement character instead of a prefix."""
    for source in ("\U0001f60axyz", "\u4f60\u597d\u4e16\u754c", "\u6f22\u5b57", "\U0001f44d\U0001f3fdtest"):
        for limit in range(1, 5):
            kept = truncate_to_tokens(source, limit)
            assert source.startswith(kept), (source, limit, kept)
            assert count_text_tokens(kept) <= limit


def test_a_truncation_holds_its_budget_for_text_the_tokenizer_refuses():
    """``<|endoftext|>`` in user text used to raise inside the encoder, and both
    the count and the cut fell back to characters -- which is not a token
    budget for anything dense."""
    dense = "\u4f60\u597d\u4e16\u754c " * 500 + "<|endoftext|>"

    kept = truncate_to_tokens(dense, 100)

    assert count_text_tokens(kept) <= 100
    assert dense.startswith(kept)


def test_a_tokenizer_outage_falls_back_to_a_budget_nothing_can_exceed(monkeypatch):
    """The early "it already fits" check used to read the approximate count --
    tiktoken's when tiktoken works, characters over four when it does not -- and
    handed 80,000 dense characters back whole against a 20,000-token budget.
    One byte is at most one token, so bytes are the conservative answer."""
    dense = "\u4f60\u597d\u4e16\u754c " * 4_000

    def broken(payload):
        raise RuntimeError("no tokenizer here")

    monkeypatch.setattr(helpers, "_encode", broken)
    helpers._TOKEN_COUNT_CACHE.clear()
    kept = truncate_to_tokens(dense, 1_000)
    monkeypatch.undo()
    helpers._TOKEN_COUNT_CACHE.clear()

    assert dense.startswith(kept)
    assert count_text_tokens(kept) <= 1_000, "measured with the real counter once it is back"


def test_a_cut_reports_what_it_actually_cost(monkeypatch):
    """A caller spending one shared allowance cannot work the cost out
    afterwards: counting the returned prefix asks the estimator that answers
    characters-over-four when the tokenizer is away, which is not what the cut
    was measured in."""
    chunk = "\u4f60\u597d\u4e16\u754c " * 200

    text, cost = take_tokens(chunk, 100)
    assert cost == count_text_tokens(text) <= 100, "an exact count while the tokenizer answers"

    def broken(payload):
        raise RuntimeError("no tokenizer here")

    monkeypatch.setattr(helpers, "_encode", broken)
    helpers._TOKEN_COUNT_CACHE.clear()
    text, cost = take_tokens(chunk, 100)
    monkeypatch.undo()
    helpers._TOKEN_COUNT_CACHE.clear()

    assert cost == len(text.encode("utf-8")) == 100, "bytes, because bytes are what it was cut by"
    assert count_text_tokens(text) <= cost, "and no token count can exceed them"


def test_the_prompt_estimate_is_the_sum_of_its_messages_and_tools():
    messages = [build.system("be brief"), build.user("hello there")]
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]

    assert estimate_prompt_tokens(messages, tools) == sum(
        estimate_message_tokens(m) for m in messages
    ) + count_text_tokens(helpers.json.dumps(tools, ensure_ascii=False))
    assert estimate_prompt_tokens([], None) == 0


def test_an_edited_message_is_a_miss_not_a_stale_hit():
    message = build.tool_result("c1", "read", "x" * 4000)
    before = estimate_message_tokens(message)
    message["content"] = [msg.text_block("[elided]")]

    assert estimate_message_tokens(message) < before


def test_the_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(helpers, "_TOKEN_COUNT_CACHE_MAX", 8)
    helpers._TOKEN_COUNT_CACHE.clear()
    for i in range(20):
        count_text_tokens(f"payload number {i}")

    assert len(helpers._TOKEN_COUNT_CACHE) == 8


def test_counting_from_many_threads_at_once_is_safe():
    import threading

    from opendde_harness.utils import helpers

    helpers._TOKEN_COUNT_CACHE.clear()
    errors: list[BaseException] = []
    texts = [f"text {i} " * 20 for i in range(helpers._TOKEN_COUNT_CACHE_MAX + 200)]

    def work(offset: int):
        try:
            for i in range(len(texts)):
                helpers.count_text_tokens(texts[(i + offset) % len(texts)])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(k * 700,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(helpers._TOKEN_COUNT_CACHE) <= helpers._TOKEN_COUNT_CACHE_MAX


def test_replayed_reasoning_counts_as_its_text_not_its_encrypted_blob():
    """Measured on the Codex login: counting the base64 blob read a 31k-token
    prompt as 161k, and the fitter elided the history on that number."""
    thinking = "Plan: read the config, then validate it."
    blob = "A" * 100_000
    plain = build.assistant("done", reasoning=thinking)
    signed = build.assistant("done", thinking=[{"type": "thinking", "thinking": thinking, "signature": blob}])
    redacted = build.assistant("done", thinking=[{"type": "redacted_thinking", "data": blob}])

    assert estimate_message_tokens(signed) == estimate_message_tokens(plain)
    assert estimate_message_tokens(signed) < count_text_tokens(blob) // 100
    assert estimate_message_tokens(redacted) < count_text_tokens(blob) // 100


def test_a_tool_call_costs_its_name_and_its_arguments():
    """What the request spells out. Counting neither read a turn as one token."""
    call = build.assistant(calls=[("c1", "read", {"path": "src/main.py", "offset": 200})])

    assert estimate_message_tokens(call) >= count_text_tokens("read") + count_text_tokens(
        '{"path": "src/main.py", "offset": 200}'
    )


def test_an_image_is_priced_by_its_patches_not_by_its_base64():
    """A 1000x1000 JPEG is ~1.3k image tokens and ~460k base64 characters;
    charging the transport encoding starved the history budget."""
    import base64

    png = base64.b64encode(
        bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 8 + (280).to_bytes(4, "big") + (280).to_bytes(4, "big")
    ).decode()
    picture = build.user([msg.text_block("look"), msg.image_block(png, "image/png")])

    assert estimate_message_tokens(picture) == count_text_tokens("look") + 100


def test_a_vocabulary_that_is_not_loaded_yet_is_reported_as_absent(monkeypatch):
    """tiktoken fetches its vocabulary over the network the first time it is
    asked. Callers that must answer offline ask this first, so the answer has
    to be about what is in this process, not about what could be fetched."""
    from opendde_harness.utils.helpers import tokenizer_is_loaded

    count_text_tokens("load the vocabulary if it is not already here")
    assert tokenizer_is_loaded() is True

    monkeypatch.setattr(helpers.tiktoken.registry, "ENCODINGS", {})
    assert tokenizer_is_loaded() is False

    # A tiktoken that keeps its encodings somewhere else costs an estimate its
    # precision, never its correctness.
    monkeypatch.delattr(helpers.tiktoken, "registry")
    assert tokenizer_is_loaded() is False
