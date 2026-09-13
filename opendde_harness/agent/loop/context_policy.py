"""What a request is allowed to be: the reply reservation and the mid-turn fit.

Turn-start history selection belongs to the context engine; this is the second
horizon, after the strategies have chosen the actual messages, tools and model.
Every function here takes the call's own model and limits explicitly, because the
call is not always addressed to the model the turn entered under.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import send_max_tokens, wire_history
from opendde_harness.providers.model_id import row_for
from opendde_harness.utils.helpers import estimate_prompt_tokens

if TYPE_CHECKING:
    from opendde_harness.config.schema import ProvidersConfig
    from opendde_harness.providers.base import LLMProvider

#: Most recent tool results kept intact when emergency-shrinking; older ones are
#: elided (their bodies are the bulk of mid-turn context growth).
KEEP_RECENT_TOOL_RESULTS = 3
#: Image-bearing messages kept intact when emergency-shrinking. Tighter than the
#: tool-result count because one image can cost 1568 tokens: the picture the
#: model is currently reasoning about is worth keeping, older ones are the
#: cheapest thing to give up.
KEEP_RECENT_IMAGES = 1


def reserved_output(window: int | None, model: str, *, provider: "LLMProvider", providers: "ProvidersConfig") -> int:
    """Tokens to hold back for the reply: the model's own output ceiling, never
    more than the window.

    Zero when nothing knows the ceiling -- no declared row names one and the
    model layer has not reported one yet. Reserving a number nobody measured is
    the failure this whole ladder exists to avoid, and the request names no
    ceiling either, so there is nothing to hold back for.
    """
    ceiling = send_max_tokens(
        getattr(provider, "generation", None),
        # The qualified id, the same one the request's own bound resolves under:
        # a row is declared beside its provider, so a bare id names the model
        # without saying whose row to read -- and reading another provider's row
        # answers for the wrong model or for none.
        model,
        overlay=row_for(providers, model),
        provider=provider,
    )
    if ceiling is None:
        return 0
    # The whole ceiling, not a share of it. Requests no longer name a ceiling, so
    # the one that applies is the model's own -- whatever the vendor fills in.
    # Reserving less than that hands out a prompt the reply cannot coexist with:
    # measured on this repo's default model, a share leaves the prompt 150000 of
    # a 200000 window against a reply allowed 64000, and the sum is refused at
    # request time. ``emergency_shrink`` only elides tool bodies, so a history
    # grown on conversation gets no retry from that refusal.
    return ceiling if window is None else min(ceiling, window)


def fit_prompt(
    messages: list[dict],
    tools: list[dict] | None,
    model: str,
    *,
    provider: "LLMProvider",
    limit: int | None,
) -> list[dict]:
    """Elide older tool bodies before a call the window cannot hold.

    Mid-turn growth is tool output, which the trimmer never sees: it bounds
    history at turn start. A run of large results used to reach the request uncut
    and either overflow -- caught by the retry in the loop -- or, on a backend
    that accepts more than the table says, go through at a long-context price.
    The estimate is the same count the trimmer uses; an unknown budget elides
    nothing, and so does a prompt whose last three results alone are over, which
    the per-result cap keeps small.

    ``limit`` is this call's budget: the window of the model the call goes to,
    less that model's reply reservation. ``None`` means the window is unknown.
    """
    if limit is None:
        return messages
    # Past a compaction marker this model's backend replays, the request carries
    # the marker and what follows it; the history it stands for is not sent and
    # must not be counted, or every turn after a compaction elides tool bodies to
    # fit a prompt it is nowhere near.
    estimate = estimate_prompt_tokens(wire_history(messages, provider, model), tools)
    if estimate <= limit:
        return messages
    shrunk, elided = emergency_shrink(messages)
    if not elided:
        return messages
    logger.warning(
        "prompt estimate of {} tokens exceeds the {}-token budget of {}; elided {} older tool result(s)",
        estimate,
        limit,
        model,
        elided,
    )
    return shrunk


def emergency_shrink(messages: list[dict]) -> tuple[list[dict], int]:
    """Elide the bodies of older tool-result messages to fit a tighter window.

    Mid-turn context overflow is almost always accumulated tool output, so
    replacing the content of all but the most recent few ``role="tool"`` messages
    with a short placeholder frees the most tokens while keeping system / user /
    assistant reasoning intact. Deterministic, no extra LLM call. Returns
    ``(new_messages, num_elided)``; ``num_elided == 0`` means there was nothing
    worth eliding (the caller should not bother retrying).

    The tool-body rule is :mod:`opendde_harness.context_engine.excerpt`, the same
    one turn-start selection applies before it drops a whole exchange.
    """
    from opendde_harness.context_engine.excerpt import excerpt_older_tool_results

    messages, elided = elide_older_images(messages)
    shrunk, excerpted = excerpt_older_tool_results(messages, KEEP_RECENT_TOOL_RESULTS)
    return shrunk, elided + excerpted


def elide_older_images(messages: list[dict]) -> tuple[list[dict], int]:
    """Drop inline images from all but the most recent image-bearing message.

    Run before the tool-text pass because an image is by far the densest thing in
    the window -- one costs up to 1568 tokens, which is more than most tool
    outputs -- so dropping a stale picture buys more room than eliding several
    text results, and costs less of what the model still needs.

    Not restricted to a tool result: an endpoint that cannot carry an image in
    one has the model layer move the picture into a following ``user`` message,
    and that message would otherwise be untouchable here.
    """
    bearing = [i for i, m in enumerate(messages) if msg.has_images(m)]
    if len(bearing) <= KEEP_RECENT_IMAGES:
        return messages, 0

    target = set(bearing[:-KEEP_RECENT_IMAGES] if KEEP_RECENT_IMAGES else bearing)
    out: list[dict[str, Any]] = []
    elided = 0
    for i, m in enumerate(messages):
        if i not in target:
            out.append(m)
            continue
        # A copy: the caller's messages may still be referenced elsewhere.
        out.append(
            msg.with_blocks(
                m,
                [
                    msg.text_block("[image elided to fit the context window]") if msg.is_image(block) else block
                    for block in msg.blocks_of(m)
                ],
            )
        )
        elided += 1
    return out, elided


__all__ = [
    "KEEP_RECENT_IMAGES",
    "KEEP_RECENT_TOOL_RESULTS",
    "elide_older_images",
    "emergency_shrink",
    "fit_prompt",
    "reserved_output",
]
