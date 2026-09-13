"""Base LLM provider interface."""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from opendde_harness.providers import messages as pi
from opendde_harness.tracing import semconv, trace

if TYPE_CHECKING:
    from opendde_harness.config.schema import ModelEntry
    from opendde_harness.providers.rates import ListRates


@dataclass(frozen=True)
class ErrorClassification:
    """Structured verdict on a failed LLM call.

    Describes what went wrong; it authorises nothing by itself:
      - ``retryable``       → the failure is the transient kind. Whether the
        request is sent again is the model service's decision, under the
        budget the request carried (``GenerationSettings.retries``)
      - ``should_compress`` → context-window overflow; shrink then retry
    ``category`` is for logging/telemetry only.

    A model's failure is read by pi, in the model service, and travels on the
    error event; a failure on this side of the pipe is read by
    :meth:`LLMProvider.classify_error`.
    """

    category: str
    retryable: bool = False
    should_compress: bool = False


_EXC_NAME_PREFIX_RE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception))\s*:\s*")
_JSON_MESSAGE_RE = re.compile(r'"message"\s*:\s*"([^"]*)"')


def _strip_json_error_body(text: str) -> str:
    """Replace a raw JSON error body with its human-readable message.

    Rewrites only when a message is actually extracted, and only within the
    parsed object's own boundary -- trailing text after the JSON survives, and
    a body yielding no message leaves the text unchanged rather than truncated.
    """
    idx = text.find("{")
    if idx == -1:
        return text
    candidate = text[idx:]
    if not candidate.startswith(('{"', "{'")):
        return text
    try:
        obj, end = json.JSONDecoder().raw_decode(candidate)
    except ValueError:
        # Malformed JSON has no knowable boundary; treat the rest of the text
        # as the body, which is the shape a swallowed error carries.
        obj, end = None, len(candidate)
    message = ""
    if isinstance(obj, dict):
        err = obj.get("error")
        found = err.get("message") if isinstance(err, dict) else None
        found = found or obj.get("message")
        if isinstance(found, str):
            message = found.strip()
    if not message:
        m = _JSON_MESSAGE_RE.search(candidate[:end])
        message = m.group(1).strip() if m else ""
    if not message:
        return text
    head = text[:idx].rstrip()
    tail = candidate[end:].strip()
    return " ".join(part for part in (head, message, tail) if part)


def format_llm_error(
    exc: BaseException,
    classification: ErrorClassification,
    provider: str | None = None,
) -> str:
    """Build the canonical content for a failed LLM call.

    Shape: ``Error calling LLM (<category>[@<provider>]): <detail>``. The head
    names the category and the provider so a rendering surface can show a
    diagnosis and a fix hint instead of the raw exception; the detail drops
    duplicated exception-name prefixes and raw JSON error bodies.
    """
    detail = str(exc).strip()
    names: list[str] = []
    while True:
        m = _EXC_NAME_PREFIX_RE.match(detail)
        if not m:
            break
        name = m.group(1).rsplit(".", 1)[-1]
        if name not in names:
            names.append(name)
        detail = detail[m.end() :]
    detail = _strip_json_error_body(detail).strip()
    prefix = "".join(f"{n}: " for n in names)
    detail = f"{prefix}{detail}".strip().rstrip(":-").strip() or type(exc).__name__
    head = f"{classification.category}@{provider}" if provider else classification.category
    return f"Error calling LLM ({head}): {detail}"


@dataclass(frozen=True)
class TruncationInfo:
    """Where the output stopped, on a call that never finished arriving."""

    at_tokens: int | None = None

    def as_error(self, tool_name: str) -> str:
        """What is known, and what is only inferred, kept apart.

        Known: the turn stopped at the output limit, because the upstream said
        so. Inferred: that this call was the cut one, which follows from
        generation being sequential but not from anything the upstream said --
        a turn can finish a call and then hit the limit in the prose after it.
        Saying "this call was cut" as a fact sends a model to split up a call
        that was whole.

        The refusal is not conditional on that inference. A call that may be
        incomplete is not dispatched either way; being wrong costs one retry.

        What to do about it is the tool's, via ``Tool.truncation_hint``.
        """
        at = f" at the {self.at_tokens}-token output limit" if self.at_tokens else " at the output limit"
        return (
            f"Error: [truncated] This turn stopped{at}, and this call was the last thing "
            f"being written, so it may have been cut short. It was not run. Send it again."
        )


@dataclass(frozen=True)
class RunMeta:
    """What happened around a call, as opposed to what the call asks for.

    Kept apart from ``arguments`` because the two travel differently: anything
    inside that dict reaches the assistant message the history keeps, and the
    loop records that before the registry sees the call -- so a flag stored
    there is already fixed into the conversation by the time anyone strips it,
    and the model reads back a field it never wrote.

    An empty instance means "nothing worth noting", which is the normal turn.

    The two fields sit at different layers on purpose. ``arguments_repaired``
    is an observation the provider can make -- this call's JSON had to be
    repaired to parse. ``truncation`` is the loop's conclusion drawn from it
    plus the response-level signals. Keeping the observation separate is what
    lets a single decision point serve both response paths: the streaming one
    assembles its tool calls in the loop, where a provider has nothing to
    attach a conclusion to.

    No ``__bool__``: it would have to pick one field to mean "non-empty", and
    every later field would silently fall outside it.
    """

    truncation: TruncationInfo | None = None
    arguments_repaired: bool = False
    #: This call was the last one of its turn. Only ever set alongside
    #: ``arguments_repaired``, because that is the only place it means
    #: anything: generation is sequential, so nothing arrives after a cut, and
    #: a repair with no calls after it is the shape a cut leaves. Recorded as
    #: the position it is rather than as a verdict -- whether the turn ran out
    #: of room is a reading of these two facts, and it belongs in the sentence
    #: the model reads, not in the record.
    last_of_turn: bool = False


@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]
    provider_specific_fields: dict[str, Any] | None = None
    function_provider_specific_fields: dict[str, Any] | None = None
    run_meta: RunMeta | None = None

    def to_pi_tool_call(self) -> dict[str, Any]:
        """This call as the pi ``toolCall`` block an assistant message carries.

        Only for a turn the model service did not answer: its own message
        already holds the blocks, signatures included, and is stored as it came
        back. The two ``provider_specific_fields`` slots are not carried -- they
        are a Chat Completions chunk's, read while a stream is being assembled,
        and pi has no message field they correspond to.
        """
        return pi.tool_call_block(self.id, self.name, self.arguments)


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    # Nested details ride in here too (``completion_tokens_details``), so the
    # values are not all integers -- the annotation says so rather than
    # leaving every reader to discover it.
    usage: dict[str, Any] = field(default_factory=dict)
    reasoning_content: str | None = None  # Kimi, DeepSeek-R1 etc.
    thinking_blocks: list[dict] | None = None  # Anthropic extended thinking
    # Set when finish_reason == "error". Providers that have the live exception
    # attach a precise classification here; otherwise the retry layer fills it
    # in from the error string.
    error_classification: "ErrorClassification | None" = None
    # Generation stopped at the output ceiling rather than because the model
    # was done. Distinct from finish_reason: upstream does not always say so
    # (some backends report "stop" on a truncated response), and a tool call
    # whose arguments were cut mid-JSON is truncated no matter what the
    # backend claims.
    truncated: bool = False
    # The ceiling that produced it, for the message shown to the model.
    max_tokens: int | None = None
    # The model that answered, as the backend reported it, so usage is priced
    # under the id that served rather than the spelling the caller used.
    model: str | None = None
    # Who actually answered and under what identity: the exact model ref, the
    # protocol, the provider instance, the backend's own response id and its
    # native status. ``model`` above stays the caller's spelling because usage
    # and session records are filed under it; this is where the precise
    # identity lives, so the two never have to be the same string.
    #
    # An ``AdapterResult.meta``; typed loosely here so ``providers.base`` does
    # not import the adapter layer that imports it.
    result_meta: Any = None
    # The model layer's own final message, kept whole so the next request can
    # replay it verbatim instead of a re-rendered copy. pi-ai's
    # ``AssistantMessage`` (minus its usage, which nothing reads back) is the
    # only producer today; see ``providers/pi_context.py``. None on every other
    # route, and on a failed call -- an errored turn is not replayed.
    pi_message: dict[str, Any] | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


@dataclass
class StreamDelta:
    """Single normalized delta from a streaming LLM response.

    Producers (provider.chat_stream) yield one of these per non-empty chunk.
    Consumers (AgentLoop on_token_delta path, TUI SubscriptionEmitter) read
    `.content` for incremental token text; `tool_call_delta` / `usage` are
    optional carriers for in-stream tool deltas and final usage snapshots.

    `finish_reason` / `error_classification` are only ever set on the
    terminal delta of a stream (mirroring `LLMResponse`); mid-stream deltas
    leave both as ``None``.
    """

    content: str | None
    tool_call_delta: dict[str, Any] | None = None
    builtin_tool_event: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    reasoning_content: str | None = None  # Kimi, DeepSeek-R1, qwen, o-series thinking stream
    # Anthropic extended thinking. Carried separately from reasoning_content
    # because the blocks are signed: Anthropic requires the originals back,
    # unmodified, on the next request of a tool-use turn, and a re-rendered
    # copy of the text is not the same object.
    thinking_blocks: list[dict] | None = None
    finish_reason: str | None = None
    error_classification: ErrorClassification | None = None
    # Set only on the delta that announces a retry, and on no other:
    # ``{"attempt", "total", "reason"}``, where ``attempt`` is the one about to
    # run out of ``total``. Not terminal -- the same stream carries the next
    # attempt's deltas after it -- but everything streamed before it is void.
    retry: dict[str, Any] | None = None
    # The whole answer, on the terminal delta of an adapter that produces one.
    #
    # A stream used to be re-parsed into a response by whoever consumed it,
    # which is a second parser working from strictly less information: the
    # fragments carry no repaired-argument flag, no exact item ids, no final
    # usage and no provider identity, so the streamed result and the
    # non-streamed result could disagree about the same bytes. When this is
    # set it is authoritative and the accumulation is only a preview.
    #
    # ``LLMResponse``; untyped to keep the forward reference out of the
    # dataclass, since ``LLMResponse`` is defined above in the same module.
    final_response: Any = None


def declared_tokens(row: "ModelEntry | None", field: str) -> int | None:
    """A positive token count the user wrote for this model, or None.

    The first tier of both ladders -- the window (``context_window``) and the
    output ceiling (``max_tokens``) -- because somebody describing their own
    deployment is the authority on it, and for a self-hosted model the only
    source there is. One reader so the two ladders cannot disagree about what
    counts as a declaration.
    """
    value = getattr(row, field, None) if row is not None else None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return int(value)
    return None


def send_max_tokens(
    generation: Any,
    model: str | None,
    *,
    pinned: int | None = None,
    overlay: "ModelEntry | None" = None,
    provider: Any = None,
) -> int | None:
    """The output ceiling a request will actually carry.

    One function for both the request body and the agent loop's ceiling check.
    Computed separately the two would drift the moment either side grew a
    bound, and the check would stop firing without ever failing -- which is
    the exact shape of the defect this branch exists to remove.

    A pin is a call site asking for a deliberately short answer, so it wins --
    but never above what the model accepts. Every pin in the tree today is far
    below any real ceiling, which is exactly why the bound has to be written
    down: the first pin that is not would be a rejected request, and nothing
    about the call site would say why.

    ``pinned`` is the per-call argument (``judge`` asks for 64, the curator for
    2048); the settings object carries the per-provider one. The argument has
    to arrive here rather than bypass the function, or the two ways of asking
    for a short answer are bounded by different rules and only one of them is
    the number truncation is judged against.

    ``overlay`` is the model's own declared row, which beats every table.

    The model's own ceiling is the answer, unbounded by anything else. How much
    of the window a turn holds back for its reply is the budget's business, and
    it reserves exactly this number: requests no longer name a ceiling, so the
    one that applies is the model's own, and the prompt has to fit beside it.

    Two tiers and then the pin. The user's own declaration for this model, then
    what ``provider`` says its model accepts (the model service's row); a pin
    is bounded by whichever answered, and stands alone when neither did.
    ``None`` comes back when nothing knows and nobody pinned: there is no
    ceiling to check and the request names none, which is pi sending the
    model's own.
    """
    ceiling = declared_tokens(overlay, "max_tokens")
    if ceiling is None:
        reported = getattr(provider, "max_output_tokens", None)
        reported = reported() if callable(reported) else None
        if isinstance(reported, int) and not isinstance(reported, bool) and reported > 0:
            ceiling = reported
    pin = pinned if pinned is not None else getattr(generation, "max_tokens", None)
    if pin:
        return min(int(pin), ceiling) if ceiling else int(pin)
    return ceiling


@dataclass(frozen=True)
class GenerationSettings:
    """Default generation parameters for LLM calls.

    Stored on the provider so every call site inherits the same defaults
    without having to pass max_tokens / reasoning_effort through every layer.
    Individual call sites can still override by passing explicit keyword
    arguments to chat() / chat_with_retry().

    No temperature. It is per model or nothing: a default here rode on every
    request, including to models that reject the parameter outright, and a
    number that suits one model is the wrong number for the next. Where a
    temperature belongs is the model's own row (``providers.<id>.models[]``),
    which the request path reads, and no call site can name one.
    """

    #: ``None`` means "no opinion" -- the ceiling is resolved from the model's
    #: own metadata at request time. A number pins it, which is what an
    #: explicit ``chat(max_tokens=...)`` at a call site wants.
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    #: Every call is a stream, and a stream is bounded per gap rather than in
    #: total: ``first_token_timeout`` covers the silence before its first event
    #: (a model with hidden reasoning thinks for minutes before it),
    #: ``idle_timeout`` every silence after. A stream that keeps delivering has
    #: no total cap; a long reply is not a fault.
    first_token_timeout: float = 300.0
    idle_timeout: float = 120.0
    #: How many times a failed model call may be run again. Sent with the
    #: request as its retry budget and spent inside the model service, by pi's
    #: own ``retryAssistantCall``: pi decides which failures are worth
    #: repeating and waits between attempts, and each retry it schedules is
    #: announced to the loop so what a discarded attempt streamed is replaced
    #: rather than appended to. Zero unless configuration says otherwise;
    #: ``agents.defaults.llmRetries`` is what a built provider carries.
    retries: int = 0


#: The key a session message carries when it stands for a server-side
#: compaction: ``{"provider", "model", "items"}``. It lives here rather than
#: with the one provider that writes it because everything between the session
#: log and the request has to carry it through -- the history projections, the
#: coalescer, the trimmer -- and none of those may import a vendor module.
COMPACTION_KEY = "compaction"


def replays_marker(marker: dict[str, Any], provider_name: str, model: str | None) -> bool:
    """The one rule: a marker replaces the history before it only in a request
    to the same provider, on the same model, that made it."""
    return bool(provider_name) and marker.get("provider") == provider_name and marker.get("model") == model


def compaction_boundary(
    messages: list[dict[str, Any]], provider: "LLMProvider | None", model: str | None
) -> int | None:
    """Index of the marker a request to ``model`` replays, or None for none.

    None rather than zero, because zero is a real answer: the capped session
    slice and the curator projection both start their result *at* the marker,
    and a sentinel that collided with that index left the boundary unprotected,
    unreinserted and uncosted in exactly the lists those two produce.

    The latest marker wins: a second compaction's summary already covers the
    first, so the request starts at the newer one.
    """
    for idx in range(len(messages) - 1, -1, -1):
        marker = messages[idx].get(COMPACTION_KEY)
        if isinstance(marker, dict) and provider is not None and provider.replays_compaction(marker, model):
            return idx
    return None


def orphan_tool_results(messages: list[dict[str, Any]]) -> set[int]:
    """Positions of tool results in ``messages`` whose call is not in it.

    A history is only ever cut at the front -- by a cap, or by aligning to a
    compaction marker -- and a result whose call the cut removed is an orphan
    both vendors refuse outright ("messages with role 'tool' must be a response
    to a preceding message with 'tool_calls'"). Aligning to a user message
    could never produce one; aligning to a marker can, because a call may sit
    before the marker and its result after. The call is inside the summary by
    then and cannot be brought back, so the result goes with it.

    A cut is all this answers for. A marker left *inside* the list orphans a
    result the same way, but only on the model that replays it, and only the
    provider's own conversion knows which model that is.
    """
    opened: set[str] = set()
    orphans: set[int] = set()
    for idx, message in enumerate(messages):
        opened.update(pi.tool_call_ids(message))
        if pi.is_tool_result(message) and str(message.get("toolCallId", "")) not in opened:
            orphans.add(idx)
    return orphans


def _marker_cost(message: dict[str, Any]) -> dict[str, Any]:
    """A marker rendered as the ordinary message its request costs.

    ``items`` is the user's own recent words, verbatim and in clear, beside one
    opaque summary; the request sends all of it and the marker's own placeholder
    text none of it. So the clear text is what the estimators must price -- up
    to 20k tokens of it, counted as nothing before this -- while the opaque item
    is left uncounted for the reason replayed reasoning is: it is a blob
    standing for a conversation nobody local can price, and counting its bytes
    read a 31k-token prompt as 161k.
    """
    marker = message.get(COMPACTION_KEY) or {}
    texts: list[str] = []
    for item in marker.get("items") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "input_text":
                texts.append(str(part.get("text") or ""))
    return pi.user_message("\n".join(texts))


def wire_history(
    messages: list[dict[str, Any]], provider: "LLMProvider | None", model: str | None
) -> list[dict[str, Any]]:
    """``messages`` as a request to ``model`` will carry them, priced as ordinary ones.

    A compaction marker stands for every message before it on the model whose
    backend made it, so only the instructions, the marker's own clear content
    and what follows it reach that wire; on every other model the marker says
    nothing and the local history is what is sent. Budgeting has to count the
    same thing the request carries, or a session the backend has already
    compacted keeps being trimmed against a size it no longer has.

    The one instruction kept is the last system message anywhere in the list,
    because the converter has a single ``instructions`` field and the last
    writer wins it. Every other system message is dropped, wherever it sits --
    one after the marker was priced in full and sent not at all.

    This is a costing projection, not a request: the marker becomes the text it
    stands in for. What actually goes on the wire is the provider's own
    conversion of the untouched list.
    """
    boundary = compaction_boundary(messages, provider, model)
    if boundary is None:
        return messages
    tail = [
        _marker_cost(m) if idx == boundary else m
        for idx, m in enumerate(messages[boundary:], boundary)
        if m.get("role") != "system"
    ]
    instructions = next((m for m in reversed(messages) if m.get("role") == "system"), None)
    return [instructions, *tail] if instructions is not None else tail


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implementations should handle the specifics of each provider's API
    while maintaining a consistent interface.
    """

    #: Whether the backend compacts a conversation itself (see
    #: ``OpenAICodexProvider.compact``); the loop asks it instead of trimming
    #: locally when a session's prompt nears the window.
    supports_compaction = False
    #: The name this provider writes into the markers it makes, and the only
    #: one it answers for. Declared rather than overridden so ``replays_marker``
    #: stays the single rule every provider is read by.
    compaction_provider = ""
    _SENTINEL = object()

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base
        self.generation: GenerationSettings = GenerationSettings()

    def replays_compaction(self, marker: dict[str, Any], model: str | None) -> bool:
        """Will a request to ``model`` send ``marker`` in place of the history
        before it? Only the provider that wrote the marker can, and only on the
        model that made it; a provider that writes none answers no to every
        marker."""
        return replays_marker(marker, self.compaction_provider, model or self.get_default_model())

    def effort_for(self, model: str | None) -> str | None:
        """The thinking level a call with no pin uses for this model.

        The model's own declared row first, then ``generation.reasoning_effort``;
        None leaves the vendor's default. Read through ``getattr`` because
        wrappers that never run this ``__init__`` carry the providers map as a
        property.
        """
        from opendde_harness.providers import model_id

        row = model_id.row_for(getattr(self, "providers", None), model or "")
        declared = getattr(row, "reasoning_effort", None)
        if declared:
            return declared
        generation = getattr(self, "generation", None)
        return generation.reasoning_effort if generation is not None else None

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            model: Model identifier (provider-specific).
            max_tokens: Maximum tokens in response.
            tool_choice: Tool selection strategy ("auto", "required", or specific tool dict).

        Returns:
            LLMResponse with content and/or tool calls.
        """
        pass

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """Non-streaming fallback: emit the full ``chat()`` response as a single
        terminal delta.

        The TUI agent loop drives turns via ``chat_stream``; a provider without
        a real streaming implementation would otherwise ``AttributeError``
        there. This default makes any provider that implements ``chat`` usable
        in the streaming path — without token-level streaming.
        ``PiModelProvider`` overrides it with true streaming, so this is
        reached only by a test double or a provider still being written.

        Generation defaults resolve from ``self.generation`` the same way
        ``chat_with_retry`` does: literal defaults here would shadow the user's
        configuration, since the agent loop calls this with messages/tools/model
        only.

        ``generation`` is read defensively -- a subclass that never runs this
        ``__init__`` (thin adapters, test doubles) reaches this method with the
        attribute missing, and a crash there would be worse than the settings
        it is meant to restore.
        """
        gen = getattr(self, "generation", None) or GenerationSettings()
        if max_tokens is self._SENTINEL:
            max_tokens = gen.max_tokens
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.effort_for(model)
        response = await self.chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        tool_call_delta: dict[str, Any] | None = None
        if response.tool_calls:
            tool_call_delta = {
                "tool_calls": [
                    {
                        "index": i,
                        "id": tc.id,
                        # A provider that signs its tool calls puts the signature
                        # here; replaying a non-streaming answer as one delta used
                        # to drop it, so the same call came back signed or unsigned
                        # depending on which path served it.
                        "provider_specific_fields": tc.provider_specific_fields,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            "provider_specific_fields": tc.function_provider_specific_fields,
                        },
                    }
                    for i, tc in enumerate(response.tool_calls)
                ]
            }
        yield StreamDelta(
            content=response.content,
            tool_call_delta=tool_call_delta,
            usage=response.usage or None,
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
            finish_reason=response.finish_reason,
            error_classification=response.error_classification,
            # The answer itself, beside the deltas that render it. The fields
            # above are what a token stream can carry; this call already has the
            # whole result, and rebuilding it from fragments drops what no
            # fragment spells -- the truncation verdict, a repaired-argument
            # flag, the native message. The reader takes this and relays the
            # rest.
            final_response=response,
        )

    @staticmethod
    def classify_error(exc: BaseException) -> ErrorClassification:
        """The bucket a failure on *this* side of the pipe falls in.

        Not the model's failure. That one is read by pi, in the model service,
        and its verdict travels on the error event -- retryable, context
        overflow, and pi's own error code -- so the tables of vendor wordings
        and status codes that used to live here are pi's, and a change to them
        arrives with a version bump rather than an edit.

        What is left is what fails before or beside the model: the service
        gone, and the two stream deadlines ``pi_provider`` enforces because pi
        has none. Both are transient. Anything else is something this process
        did wrong, and sending it again would do it again.
        """
        from opendde_harness.providers.model_service import ModelServiceError

        if isinstance(exc, TimeoutError):
            return ErrorClassification("network", retryable=True)
        if isinstance(exc, ModelServiceError):
            # "gone" is the child process having exited or closed its input,
            # which the next request restarts; every other code is a refusal.
            if exc.code == "gone":
                return ErrorClassification("network", retryable=True)
            return ErrorClassification(exc.code)
        return ErrorClassification("unknown")

    # -- what this provider knows about its own model -----------------------
    #
    # Four facts the loop decides with: how much context there is, how much
    # reply to hold back for, whether a picture can be shown, and what a call
    # is worth. A provider that knows them answers; the default is None, which
    # every caller reads as "nobody knows" -- no trimming against an invented
    # window, no ceiling to check, no denial of an image, no price. They are
    # asked rather than looked up because the model layer is the only thing that
    # has the figures for the model it is about to call.

    def context_window(self) -> int | None:
        """How many tokens this provider's model accepts, or None if unknown."""
        return None

    def max_output_tokens(self) -> int | None:
        """The model's own output ceiling, or None -- meaning no ceiling to check."""
        return None

    def input_modalities(self) -> tuple[str, ...] | None:
        """What the model accepts (``("text", "image")``), or None if unknown.

        ``None`` is not a denial. A picture withheld from a model that could
        have seen it fails silently; a picture sent to one that cannot is
        refused, loudly. Callers take the optimistic reading of None.
        """
        return None

    def list_rates(self) -> "ListRates | None":
        """The model's published price per token, or None where there is none."""
        return None

    @trace.instrument("llm.call", extract=semconv.llm_call)
    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """One call to ``chat``, with the caller's omissions settled and a
        failure returned rather than raised.

        The name is what the call sites say. What it used to do -- a ladder of
        billed attempts and then every configured fallback model -- is gone.
        Whether a request the server never accepted is sent again is the
        transport's business, under the policy the binding declares
        (``GenerationSettings.retries``); re-running a call whose stream was
        accepted belongs to the agent loop, and there is no other model to try.

        Parameters default to ``self.generation`` when not explicitly passed,
        so callers need not thread max_tokens / reasoning_effort through every
        layer. There is no temperature to pass: a model that wants one declares
        it in its own row, and that row is the only thing the request path reads
        it from.
        """
        if max_tokens is self._SENTINEL:
            max_tokens = self.generation.max_tokens
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.effort_for(model)

        # A pin is per call but a ceiling is per model. Left as ``None`` when
        # nobody pinned, which is the provider's cue to resolve the model's own.
        if max_tokens is None:
            sent = None
        else:
            from opendde_harness.providers import model_id

            # The qualified id: a row is declared under its provider, and a
            # declared ceiling has to bound a pin the same way it bounds the
            # budget's reservation.
            row = model_id.row_for(getattr(self, "providers", None), model or "")
            sent = send_max_tokens(self.generation, model or "", pinned=max_tokens, overlay=row, provider=self)

        try:
            response = await self.chat(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=sent,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            classification = self.classify_error(exc)
            return LLMResponse(
                content=format_llm_error(exc, classification, provider=getattr(self, "provider_name", None)),
                finish_reason="error",
                error_classification=classification,
            )

        if response.finish_reason == "error":
            # The model layer classifies its own failures; nothing here can
            # improve on a verdict it did not make, so an unclassified one
            # stays unclassified rather than being guessed at from its text.
            response.error_classification = response.error_classification or ErrorClassification("unknown")
            return response

        # Judged here, not by the caller: this runs inside the ``llm.call``
        # span, and ``trace.instrument`` extracts its attributes in a
        # ``finally`` that closes the span before the caller sees the result.
        from opendde_harness.providers.truncation import flag_truncation

        response.max_tokens, response.truncated = flag_truncation(
            sent=sent,
            finish_reason=response.finish_reason,
            usage=response.usage,
            tool_calls=response.tool_calls,
        )
        response.model = response.model or model or self.get_default_model()
        return response

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass
