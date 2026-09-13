"""Empty-response recovery for the agent loop.

Pure decision logic, no I/O — the loop owns the side effects (appending
messages, incrementing counters, ``continue``). Kept separate from the loop so
the branching can be unit-tested in isolation, mirroring the repo's other small
policy units (nudge_policy, decision_router).

A turn that ends with no visible text would otherwise surface a canned
"no response to give" dud — a zero-score turn on weaker models. This recovers
the turn before giving up, in three bounded modes:

  PREFILL  thinking-only — the model emitted only reasoning (a structured field
           or an inline <think> block) and no body. Re-feed its own reasoning so
           it continues into the answer.
  NUDGE    post-tool empty — the model ran a tool then returned nothing. Inject a
           short user nudge so it processes the tool result.
  RETRY    plain empty — re-request as-is.

This is distinct from Sentinel's NudgeInjector / NudgePolicy, which inject
*proactive suggestions* onto an outbound reply; this module instead recovers an
empty turn before it is ever sent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Awaitable, Callable

from loguru import logger

from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import LLMResponse

if TYPE_CHECKING:
    from opendde_harness.agent.loop.accounting import TurnAccounting

# In-content thinking markers. Some models (Ollama, certain Qwen gateways) put
# the reasoning in ``content`` as <think>…</think> rather than in the structured
# reasoning_content field, so a content scan is required — checking only the
# structured fields would miss them.
# Matches an opening or a closing tag, with or without a vendor namespace
# prefix (``<mm:think>``). A turn cut off inside an inlined block arrives as
# a lone closing tag, so an opener-only pattern reads it as ordinary prose.
_THINK_TAG_RE = re.compile(r"</?(?:[a-z][\w.-]{0,15}:)?(?:think|thinking|reasoning)>", re.IGNORECASE)
_NS = r"(?:[a-z][\w.-]{0,15}:)?"
_THINK_BLOCK_RE = re.compile(
    rf"<{_NS}(think|thinking|reasoning)>[\s\S]*?</{_NS}\1>",
    re.IGNORECASE,
)

POST_TOOL_NUDGE = (
    "You executed tool calls but returned an empty response. Use the tool "
    "results above to continue the task, or give your final answer now."
)


class RecoveryAction(Enum):
    """What the loop should do about an empty assistant response."""

    COMPLETE = auto()  # visible text present, or budgets spent → finish the turn
    PREFILL = auto()  # thinking-only → re-feed reasoning, re-request
    NUDGE = auto()  # post-tool empty → inject (empty) + user nudge, re-request
    RETRY = auto()  # plain empty → re-request as-is


@dataclass(frozen=True)
class RecoveryLimits:
    """Per-turn retry budgets."""

    enabled: bool = True
    post_tool_empty_max_nudges: int = 1
    thinking_prefill_max_retries: int = 2
    empty_content_max_retries: int = 3


def limits_from_defaults(defaults: object) -> RecoveryLimits:
    """Build limits from an ``agents.defaults`` config object (duck-typed).

    Centralizes the config→RecoveryLimits mapping so the several AgentLoop
    construction sites don't each repeat the field plumbing.
    """
    return RecoveryLimits(
        enabled=getattr(defaults, "empty_recovery_enabled", True),
        post_tool_empty_max_nudges=getattr(defaults, "post_tool_empty_max_nudges", 1),
        thinking_prefill_max_retries=getattr(defaults, "thinking_prefill_max_retries", 2),
        empty_content_max_retries=getattr(defaults, "empty_content_max_retries", 3),
    )


def strip_think_blocks(text: str) -> str:
    """Remove paired think blocks. Debris is left in place for the check below.

    Matched by the same shape as ``_THINK_TAG_RE``: any of the three tag names,
    with or without a vendor namespace prefix. A single spelling here would let
    a complete ``<mm:think>...</mm:think>`` block through to the user in full,
    which is the one outcome stripping exists to prevent.

    The backreference keeps the two ends the same tag -- ``<think>x</thinking>``
    is a malformed pair, and deleting everything between two unrelated tags
    would take real content with it. The prefixes are not tied to each other:
    a backend that stamps one end and not the other still wrote one block.
    """
    return _THINK_BLOCK_RE.sub("", text).strip()


def is_only_think_debris(text: str) -> bool:
    """True when the text is nothing but think tags once they are removed.

    The shape a turn cut off inside an inlined reasoning block arrives in: a
    lone closing tag with no opener, which pairs with nothing and so survives
    ``strip_think_blocks`` as an eleven-character string that reads like a real
    answer. Asked of the residue rather than of vendor spellings -- which
    prefix a backend picked is not knowable in advance.
    """
    return bool(text) and not _THINK_TAG_RE.sub("", text).strip()


def strip_think(text: str | None) -> str | None:
    """The user-facing text of a reply, or ``None`` when there is none left.

    Paired blocks are removed, and what is left is then checked for being nothing
    but tag debris: when a backend inlines its reasoning and the turn is cut off
    inside it, content arrives as a lone closing tag with no opener to pair
    against, so the substitution finds nothing and an eleven-character string reads
    as a real answer. Recovery is then skipped and the tag is what the user sees.
    """
    if not text:
        return None
    cleaned = strip_think_blocks(text)
    if is_only_think_debris(cleaned):
        return None
    return cleaned or None


def has_inline_thinking(content: str | None) -> bool:
    """True when raw content carries a <think>/<thinking>/<reasoning> marker."""
    return bool(content) and bool(_THINK_TAG_RE.search(content))


def has_thinking(response: LLMResponse) -> bool:
    """True when the response produced reasoning in any form (structured or inline)."""
    return bool(response.reasoning_content or response.thinking_blocks or has_inline_thinking(response.content))


def classify_empty_response(
    response: LLMResponse,
    visible: str | None,
    *,
    prev_had_tool_calls: bool,
    nudges_done: int,
    prefill_retries: int,
    empty_retries: int,
    limits: RecoveryLimits,
) -> RecoveryAction:
    """Decide how to handle a no-tool-call assistant response.

    ``visible`` is ``response.content`` after stripping <think> blocks — i.e. the
    user-facing text. Non-empty ``visible`` (or recovery disabled) means the turn
    is done.

    Ordering puts PREFILL before NUDGE so a thinking-only response is continued
    via prefill rather than spending the post-tool nudge on it; the
    ``not thinking`` guard on NUDGE keeps them mutually exclusive.
    """
    if visible or not limits.enabled:
        return RecoveryAction.COMPLETE

    thinking = has_thinking(response)

    # thinking-only prefill — the model reasoned but produced no body.
    if thinking and prefill_retries < limits.thinking_prefill_max_retries:
        return RecoveryAction.PREFILL

    # post-tool empty nudge — exclude thinking-only (handled above).
    if prev_had_tool_calls and not thinking and nudges_done < limits.post_tool_empty_max_nudges:
        return RecoveryAction.NUDGE

    # Fallback plain retry. The ``prefill_exhausted`` clause is load-bearing:
    # some models (e.g. mimo-v2-pro via OpenRouter) always populate a reasoning
    # field, so gating retry on ``not thinking`` alone would permanently block
    # retries for every reasoning model once prefill is spent.
    prefill_exhausted = thinking and prefill_retries >= limits.thinking_prefill_max_retries
    if empty_retries < limits.empty_content_max_retries and (not thinking or prefill_exhausted):
        return RecoveryAction.RETRY

    return RecoveryAction.COMPLETE


@dataclass
class RecoveryState:
    """One turn's empty-response budgets, and the decision they gate.

    Per turn, not per loop: the AgentLoop is a long-lived singleton shared across
    sessions, so counters on it would leak one turn's spent budget into the next.
    ``after_tool_calls`` is the other half of the decision -- a post-tool empty and
    a first-message empty are different failures with different recoveries.
    """

    limits: RecoveryLimits = field(default_factory=RecoveryLimits)
    after_tool_calls: bool = False
    nudges: int = 0
    prefills: int = 0
    retries: int = 0

    def decide(self, response: LLMResponse, visible: str | None) -> RecoveryAction:
        """Classify this response and spend the budget the answer uses.

        The counter moves here rather than at the call site, so "which budget did
        that decision spend" has one answer. COMPLETE spends nothing.
        """
        action = classify_empty_response(
            response,
            visible,
            prev_had_tool_calls=self.after_tool_calls,
            nudges_done=self.nudges,
            prefill_retries=self.prefills,
            empty_retries=self.retries,
            limits=self.limits,
        )
        if action is RecoveryAction.PREFILL:
            self.prefills += 1
        elif action is RecoveryAction.NUDGE:
            self.nudges += 1
        elif action is RecoveryAction.RETRY:
            self.retries += 1
        return action


# Asks the model for a best-effort wrap-up after the iteration budget is spent.
# Tools are withheld on this call, so the prompt must not invite another tool use
# or a question — there is no further turn to answer it.
MAX_ITER_SYNTHESIS_PROMPT = (
    "You've used up the tool-calling budget for this turn, so no tools are "
    "available now. Using only what you've already gathered, give your best "
    "final answer: summarize what you accomplished, deliver any partial "
    "results, and briefly note what's left undone. Do not ask questions — "
    "there is no further turn to answer them. Reply in the same language as "
    "the user's request (this instruction is in English, but it is not the "
    "conversation language)."
)

# Returned only if the synthesis call itself fails — never leave the turn silent.
MAX_ITER_STATIC_FALLBACK = (
    "I reached the maximum number of tool call iterations ({n}) without "
    "completing the task. You can try breaking the task into smaller steps."
)


async def synthesize_on_exhaustion(
    messages: list[dict],
    model: str | None,
    *,
    stream: Callable[..., Awaitable[LLMResponse]],
    plain: Callable[..., Awaitable[LLMResponse]],
    tool_defs: list[dict] | None,
    accounting: "TurnAccounting",
    max_iterations: int,
    default_model: str,
    on_token_delta: Callable[[str], Awaitable[None]] | None = None,
    on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    on_retry: Callable[[int, int, str, bool], Awaitable[None]] | None = None,
) -> tuple[str, LLMResponse | None]:
    """One tools-disabled call to wrap up after the iteration budget runs out.

    Instead of returning a canned apology, ask the model to summarize what it
    accomplished and deliver its best partial answer. Tools are declared and then
    refused (``tool_choice="none"``) so it cannot start another tool call -- or an
    ``ask_user`` -- at the cliff edge: withholding them entirely is what "no more
    tool calls" should mean, but a budget can only be exhausted by calling tools,
    so this history always carries tool_use and tool_result blocks, and Anthropic
    rejects a request holding those with no tools defined.

    When the turn caller wired streaming callbacks, this reply must stream too --
    otherwise it never reaches a streaming outlet: the run_turn boundary only emits
    a closing ``Text`` when nothing streamed, so a non-streamed wrap-up after an
    already-streamed turn gets dropped. The static fallback is pushed through the
    same callback for the same reason.

    The call is billed like any other and observed through the same ``accounting``:
    it is the turn's last call, so it becomes the receipt that after-turn work
    addressing the same backend reads, and its price reaches the turn's total. Left
    out, the one call a user actually read was the one the turn never mentioned.

    Returns the reply and the response that carried it, or ``None`` when the call
    failed and the static fallback is what the user reads.
    """
    synth_messages = [*messages, msg.user_message(MAX_ITER_SYNTHESIS_PROMPT)]
    try:
        if on_token_delta is not None or on_reasoning_delta is not None:
            response = await stream(
                messages=synth_messages,
                tools=tool_defs,
                model=model,
                on_token_delta=on_token_delta,
                on_reasoning_delta=on_reasoning_delta,
                tool_choice="none",
                on_retry=on_retry,
            )
        else:
            response = await plain(
                messages=synth_messages,
                tools=tool_defs,
                tool_choice="none",
                model=model,
            )
        text = strip_think(response.content)
        # The wrap-up's own admission rule: a reported prompt is the evidence that
        # the call was made and billed. Without it the turn keeps the receipt of
        # the last call that was, rather than replacing it with a call nothing was
        # charged for.
        if int((response.usage or {}).get("prompt_tokens") or 0) > 0:
            answered = response.model or model or default_model
            # ``messages``, not ``synth_messages``: the wrap-up instruction is local
            # scaffolding the session never records, and a marker standing for this
            # span must stand for the conversation, not for an order to stop working.
            accounting.record(
                response,
                accounting.snapshot(response, answered),
                model=answered,
                sent=messages,
                tools=tool_defs,
                history_len=len(messages),
            )
        if response.finish_reason != "error" and text:
            return text, response
        logger.warning(
            "Max-iter synthesis returned no usable content (finish_reason={})",
            response.finish_reason,
        )
    except Exception as exc:
        logger.warning("Max-iter synthesis call failed: {}", exc)
    fallback = MAX_ITER_STATIC_FALLBACK.format(n=max_iterations)
    if on_token_delta is not None:
        await on_token_delta(fallback)
    return fallback, None
