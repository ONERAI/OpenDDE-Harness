"""The agent loop's provider, answered by pi-ai in the model service.

The loop knows one contract -- :class:`~opendde_harness.providers.base.LLMProvider`
-- and this is that contract served by ``@earendil-works/pi-ai`` running in a
Node child. Nothing here speaks HTTP: the messages become a pi ``Context``
(``providers/pi_context.py``), the service streams pi's own events back, and
each one becomes a :class:`StreamDelta`.

Retry is not this class's either. The budget travels with the request --
``retry: {"maxRetries": n}`` from ``GenerationSettings.retries`` -- and is spent
inside the service by pi's own ``retryAssistantCall``, which decides what is
worth repeating and waits between attempts. What arrives here is a ``retry``
delta before each new attempt, so the loop can void what the discarded one
streamed; the attempts themselves are never visible as separate requests.

A backend that compacts a conversation itself is served here too, on the routes
that have a wire item for one (:data:`PI_COMPACTING`): ``compact`` asks for the
summary and writes the marker the session records, and every later request to
the same model replays that marker in place of the history it stands for,
sending only what came after it (``_split_at_marker``).

The two deadlines travel with the request rather than being enforced here: a
stream is bounded per gap, ``first_token_timeout`` covering the silence before
its first event and ``idle_timeout`` every silence after, and the service ends
an attempt that goes quiet past its budget. That is where they have to be --
the service is what can abort the request it is billing, and what can run the
call again, and a stalled stream is the commonest failure a retry exists for.

This is the only model layer. ``cli._helpers.make_provider`` builds one for
every model and every provider, with nothing to choose between.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from loguru import logger

from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import (
    COMPACTION_KEY,
    LLMProvider,
    LLMResponse,
    StreamDelta,
    compaction_boundary,
    format_llm_error,
    orphan_tool_results,
    replays_marker,
)
from opendde_harness.providers.model_service import ModelService, ModelServiceError
from opendde_harness.providers.pi_context import event_to_delta, to_context, usage_dict

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opendde_harness.config.schema import Config, ModelEntry, ProvidersConfig
    from opendde_harness.providers.rates import ListRates

#: pi's ``ThinkingLevel``. Our ``ReasoningEffort`` vocabulary is the same set
#: plus ``off``, which pi spells by leaving the option out: every adapter reads
#: a missing ``reasoning`` as "no thinking" (``anthropic-messages.js:661``, and
#: the same shape in the Google and Responses adapters). pi clamps a level the
#: model does not serve to one it does, so nothing here has to.
_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high", "xhigh", "max"})

#: pi's ``ToolChoice``. Our ``required`` and the per-tool dict have no pi
#: spelling, so they are left out rather than approximated by ``auto``.
_TOOL_CHOICES = frozenset({"auto", "none"})

#: The pi provider ids whose backend compacts a conversation itself. pi's own
#: extension gates the feature on the same two (``supportsRemoteCompactionModel``):
#: the direct OpenAI Responses route and the Codex login. Every other provider
#: is asked nothing and the loop's local trimming is what shrinks a session.
PI_COMPACTING = frozenset({"openai", "openai-codex"})

#: What a compaction marker says where a message's text would be. The loop
#: shows it in place of the conversation it stands for; the meaning is entirely
#: in the ``compaction`` key beside it. The record is an assistant message with
#: this one text block and no pi envelope: it is a boundary, and the request
#: that replays it sends the summary rather than the record
#: (:meth:`PiModelProvider._split_at_marker`).
_MARKER_TEXT = "[Earlier conversation compacted by the model's own backend; replayed on this model.]"


#: pi's own spelling of the four rates it publishes, in the order
#: :class:`~opendde_harness.providers.rates.ListRates` takes them.
_COST_FIELDS = ("input", "output", "cacheRead", "cacheWrite")


def _per_token(cost: "Mapping[str, Any]") -> tuple[float, float, float, float] | None:
    """pi's four per-million rates as per-token floats, or None where there are none.

    All four at zero is pi carrying no price for this model rather than a model
    that is free: it prices a provider this project declared at zero, because
    ``configure`` sends no rates for one. Answered as None so the figure reads
    as unknown, which is what it is.
    """
    rates = [
        float(value) / 1e6
        if isinstance(value := cost.get(field), (int, float)) and not isinstance(value, bool)
        else 0.0
        for field in _COST_FIELDS
    ]
    if not any(rates):
        return None
    return rates[0], rates[1], rates[2], rates[3]


def _tiers(cost: "Mapping[str, Any]") -> dict[int, tuple[float, float, float, float]]:
    """pi's long-context tiers, keyed by the input count each applies above."""
    out: dict[int, tuple[float, float, float, float]] = {}
    for tier in cost.get("tiers") or ():
        if not isinstance(tier, dict):
            continue
        above = tier.get("inputTokensAbove")
        rates = _per_token(tier)
        if isinstance(above, int) and not isinstance(above, bool) and above > 0 and rates is not None:
            out[above] = rates
    return out


def _timeouts(generation: Any) -> dict[str, int]:
    """The two gap budgets in the milliseconds the service takes.

    Only the positive ones: a budget of zero or less is nobody's deadline, and
    sending it would end every attempt before its first event.
    """
    out: dict[str, int] = {}
    for key, field in (("firstTokenMs", "first_token_timeout"), ("idleMs", "idle_timeout")):
        value = getattr(generation, field, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            out[key] = int(value * 1000)
    return out


class PiModelProvider(LLMProvider):
    """One ``(provider, model)`` pair on a shared model service.

    ``service`` is either the service itself or something that returns one --
    the process-wide starter in ``providers/pi_service.py``, which cannot run
    at construction time because building a provider is synchronous and
    starting the child is not.
    """

    def __init__(
        self,
        service: "ModelService | Callable[[], Awaitable[ModelService]]",
        provider_id: str,
        model_id: str,
        *,
        stored_model: str | None = None,
        providers: "ProvidersConfig | None" = None,
    ) -> None:
        super().__init__()
        self._service = service
        self.provider_id = provider_id
        self.model_id = model_id
        #: What ``format_llm_error`` names and what the loop logs.
        self.provider_name = provider_id
        #: The qualified id this project files the model under, which is
        #: ``<pi provider id>/<model id>`` -- the same spelling the config
        #: declares the model's row under, and the one every caller sees.
        self.stored_model = stored_model or f"{provider_id}/{model_id}"
        #: The whole providers map, so a live model switch can find the new
        #: model's declared row without the loop holding the config.
        self.providers: "ProvidersConfig | None" = providers
        #: The service's own row for this model -- window, output ceiling, input
        #: modalities, price -- read once, on the first request. Every fact the
        #: loop used to take from a vendored table comes from here.
        self._row: dict[str, Any] | None = None
        self._row_asked = False
        #: The cache key of a provider serving no session (the design worker's):
        #: constant across its own calls, which is all a prefix cache needs.
        self._own_session = uuid.uuid4().hex

    # -- the loop's contract -------------------------------------------------

    @property
    def supports_compaction(self) -> bool:  # type: ignore[override]
        """Whether this backend compacts a conversation itself.

        Only where the wire has an item to carry one: the Responses routes
        (:data:`PI_COMPACTING`). The loop asks this after every turn and
        compacts through the backend instead of trimming locally when it is
        true, so answering yes for a route that cannot is a billed request that
        fails every turn.
        """
        return self.provider_id in PI_COMPACTING

    @property
    def compaction_provider(self) -> str:  # type: ignore[override]
        """The name written into the markers this provider makes, and the only
        one it answers for.

        pi's id, not our section's: the two differ by a character for the login
        that matters most (``openai_codex`` is pi's ``openai-codex``), and a
        marker is only ever read back through this same property.

        Empty where the backend compacts nothing, which is the base class's
        spelling for "writes no marker and replays none" -- and what the TUI
        reads to answer whether anything at all will compact this session
        (``tui_rpc.methods.session._auto_compaction``). A name there would have
        it promise compaction on every route pi serves.
        """
        return self.provider_id if self.supports_compaction else ""

    def replays_compaction(self, marker: dict[str, Any], model: str | None) -> bool:
        return replays_marker(marker, self.compaction_provider, self.resolved_model(model))

    def resolved_model(self, model: str | None) -> str:
        """Every spelling this route answers to, as the one it records.

        The loop names a model
        in the stored ``provider/model`` form, the catalogue in pi's own id, and
        a finished response in the bare one; a marker is written under one of
        them and asked about under another, and comparing them literally has the
        budgeter decide a marker will not be replayed while the request replays
        it -- budgeting one history and sending a different one.
        """
        if not model or model in self._known_spellings():
            return self.stored_model
        return model

    def _known_spellings(self) -> frozenset[str]:
        return frozenset(
            value for value in (self.stored_model, f"{self.provider_id}/{self.model_id}", self.model_id) if value
        )

    def get_default_model(self) -> str:
        return f"{self.provider_id}/{self.model_id}"

    def context_window(self) -> int | None:
        """The window the service reports for this model, once it has said.

        ``None`` until the first request has been made: the number comes from
        the service's ``models`` listing, and there is no service to ask before
        one is started. Unknown reads as unknown -- the loop does not trim,
        gate or compact against a number nothing measured.
        """
        return self._fact("contextWindow")

    def max_output_tokens(self) -> int | None:
        """The output ceiling the service reports, or None while nobody knows.

        The same row, and the same "unknown until the first request" caveat.
        ``None`` is not a licence to invent one: a request that names no ceiling
        gets the model's own, which is exactly what pi already sends.
        """
        return self._fact("maxTokens")

    def input_modalities(self) -> tuple[str, ...] | None:
        """What this model accepts, as pi states it: ``("text", "image")``.

        ``None`` while the row has not been read, which callers must read as
        "no answer" rather than as a denial -- a picture withheld from a model
        that could see it fails silently, and the request being refused does not.
        """
        kinds = (self._row or {}).get("input")
        if not isinstance(kinds, list):
            return None
        return tuple(kind for kind in kinds if isinstance(kind, str)) or None

    def list_rates(self) -> "ListRates | None":
        """This model's published price per token, from the service's row.

        pi states its rates per million tokens; :class:`ListRates` is per token,
        which is what the loop's own arithmetic consumes. The long-context
        ``tiers`` travel with them, so a call is priced at the tier its own
        input reaches -- pi's ``calculateCost`` rule.

        ``None`` where there is no price rather than a price of zero. pi carries
        no rates for a provider this project declared and prices such a call at
        0; reporting that as a number reads as free, which is the one thing an
        unmeasured spend is not.
        """
        from opendde_harness.providers.rates import ListRates, ListTier

        cost = (self._row or {}).get("cost")
        if not isinstance(cost, dict):
            return None
        rates = _per_token(cost)
        if rates is None:
            return None
        tiers = tuple(ListTier(*tier, input_tokens_above=above) for above, tier in sorted(_tiers(cost).items()))
        return ListRates(*rates, tiers=tiers)

    def _fact(self, field: str) -> int | None:
        """One positive integer from the service's row, or None."""
        value = (self._row or {}).get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = LLMProvider._SENTINEL,
        reasoning_effort: object = LLMProvider._SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """One request, collected from the stream it is served by.

        The ``done`` event's message is the answer, whole -- the same object
        the streamed path calls authoritative -- so the two agree about the
        same bytes by construction rather than by two parsers agreeing.
        """
        final: LLMResponse | None = None
        failure: StreamDelta | None = None
        stream = self.chat_stream(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        async with aclosing(stream) as deltas:
            async for delta in deltas:
                if delta.final_response is not None:
                    final = delta.final_response
                elif delta.finish_reason == "error":
                    failure = delta
        if final is not None:
            return final
        if failure is not None:
            return LLMResponse(
                content=failure.content,
                finish_reason="error",
                error_classification=failure.error_classification,
                usage=failure.usage or {},
            )
        # No terminal event at all. The service writes one even when the
        # adapter does not, so this is the pipe having gone away underneath.
        gone = self._failed(ModelServiceError("gone", "the stream ended with no result"))
        return LLMResponse(content=gone.content, finish_reason="error", error_classification=gone.error_classification)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = LLMProvider._SENTINEL,
        reasoning_effort: object = LLMProvider._SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """pi's events for one request, as the loop's deltas.

        Every omitted parameter resolves from ``self.generation`` and the
        model's overlay, the same way every other provider settles them, so a
        loop that passes messages and tools only still sends the user's
        configuration.
        """
        generation = self.generation
        stored = model or self.stored_model
        if max_tokens is self._SENTINEL:
            max_tokens = generation.max_tokens
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.effort_for(stored)
        options = self._options(
            stored,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        replay, sendable = self._split_at_marker(messages, stored)
        context = to_context(sendable, tools, provider=self.provider_id, model=self.model_id)
        # The two budgets this request is bounded by, both spent inside the
        # service: the retry ladder, and the per-gap deadlines that make a
        # stalled stream one of the failures it covers. Sent on every request
        # rather than enforced here, because an attempt this end gave up on is
        # one the retry loop never sees.
        retry = {"maxRetries": max(0, int(generation.retries))}
        timeouts = _timeouts(generation)

        try:
            service = await self._resolve_service()
            events = service.stream(
                self.provider_id, self.model_id, context, options, replay=replay, retry=retry, timeouts=timeouts
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a failure to open is this call's answer, not a crash
            yield self._failed(exc)
            return

        # A plain read, with no deadline of its own. A silence is the service's
        # to end -- it is the one that can abort the request it is billing, and
        # the one whose retry budget then covers it.
        async with aclosing(events):
            try:
                async for event in events:
                    delta = event_to_delta(event)
                    if delta is not None:
                        yield delta
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the failure is the turn's answer, not a crash
                yield self._failed(exc)

    # -- compaction ----------------------------------------------------------

    async def compact(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Ask the backend to compact this history. Raises unless it did.

        The same request a turn makes, with the backend asked for a summary
        instead of an answer: the same split, the same context, the same tool
        catalogue, so what is compacted is the conversation that was actually
        being sent. The answer is one marker standing for all of it, and the
        usage of the call that made it.

        Nothing here returns a marker it is not sure of. The caller keeps its
        local history on a failure, which is only safe while a compaction that
        did not happen cannot be mistaken for one that did.
        """
        if not self.supports_compaction:
            raise ModelServiceError("unsupported_model", f"{self.provider_id} does not compact conversations")
        stored = model or self.stored_model
        replay, sendable = self._split_at_marker(messages, stored)
        context = to_context(sendable, tools, provider=self.provider_id, model=self.model_id)
        level = reasoning_effort or self.effort_for(stored)
        service = await self._resolve_service()
        result = await service.compact(
            self.provider_id,
            self.model_id,
            context,
            replay=replay,
            session_id=self._cache_session_id(),
            reasoning=level if level in _THINKING_LEVELS else None,
        )
        items = [item for item in result.get("items") or () if isinstance(item, dict)]
        if not items:
            raise ModelServiceError("compaction_failed", "the service compacted the conversation into no items")
        marker = {
            "role": "assistant",
            "content": [msg.text_block(_MARKER_TEXT)],
            COMPACTION_KEY: {
                "provider": self.compaction_provider,
                # Not the caller's spelling: this route serves one model
                # whatever it is asked for, and a vendor may answer under a
                # dated build id nothing will ever name again. A marker keyed to
                # one of those is a compaction billed every turn and replayed
                # never, so what is recorded is the one spelling every accepted
                # one resolves to.
                "model": self.stored_model,
                "items": [dict(item) for item in items],
            },
        }
        return marker, usage_dict(result.get("usage"))

    def _split_at_marker(
        self, messages: list[dict[str, Any]], model: str | None
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """``(what to replay, what to convert)`` for one request to ``model``.

        A marker this model replays stands for every message before it, so the
        request carries the marker's own items and then the history after it.
        The instructions are the exception: this project builds them at the head
        of every history, which is before any marker, and they are what pi's
        ``systemPrompt`` is made of -- cut with the rest, the request would go
        out with no system prompt at all.

        A tool result whose call sits before the boundary goes with it. The call
        is inside the opaque summary from here on, and a result without its call
        is refused outright ("must be a response to a preceding message with
        tool_calls"); ``orphan_tool_results`` answers for exactly this cut.
        """
        boundary = compaction_boundary(messages, self, model)
        if boundary is None:
            return None, messages
        marker = messages[boundary].get(COMPACTION_KEY) or {}
        after = messages[boundary + 1 :]
        orphans = orphan_tool_results(after)
        if orphans:
            logger.warning(
                "compaction replaced the calls {} answered; their results are in the summary",
                sorted(str(after[idx].get("toolCallId") or "") for idx in orphans),
            )
        kept = [message for idx, message in enumerate(after) if idx not in orphans]
        instructions = [m for m in messages[:boundary] if m.get("role") in msg.SYSTEM_ROLES]
        return {"items": [dict(item) for item in marker.get("items") or () if isinstance(item, dict)]}, [
            *instructions,
            *kept,
        ]

    # -- request options -----------------------------------------------------

    def _overlay(self, model: str) -> "ModelEntry | None":
        from opendde_harness.providers import model_id

        return model_id.row_for(self.providers, model)

    def _options(
        self,
        model: str,
        *,
        max_tokens: object,
        reasoning_effort: object,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        """The ``SimpleStreamOptions`` subset the service accepts.

        Only what pi models. The retry budget and the two gap deadlines ride
        beside the options rather than in them -- pi's own timeout option is a
        whole-call one, and these are per silence -- and an ``off`` thinking
        level is spelled by leaving ``reasoning`` out.
        """
        overlay = self._overlay(model)
        options: dict[str, Any] = {}

        # A temperature is sent only because the model's own row declares one.
        # No caller can name one: there is no default to inherit -- pi's own app
        # sends none unless the user set one for the model -- and a model that
        # does not take the parameter refuses the whole request over it
        # ("Unsupported parameter: temperature" from the Codex wire).
        temperature = getattr(overlay, "temperature", None)
        if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
            options["temperature"] = float(temperature)

        level = str(reasoning_effort) if reasoning_effort else ""
        if level in _THINKING_LEVELS:
            options["reasoning"] = level

        cap = self._output_cap(max_tokens, overlay)
        if cap is not None:
            options["maxTokens"] = cap

        if isinstance(tool_choice, str) and tool_choice in _TOOL_CHOICES:
            options["toolChoice"] = tool_choice
        elif tool_choice is not None:
            logger.debug("model service: no pi spelling for tool_choice {!r}; omitted", tool_choice)

        options["sessionId"] = self._cache_session_id()

        return options

    def _cache_session_id(self) -> str:
        """The key pi files this conversation's prompt cache under.

        The session the loop is serving, so consecutive calls of one
        conversation share the prefix the backend already holds: Codex sends
        it as ``prompt_cache_key`` and as its session header, and without one
        every request opens a fresh cache under a fresh id and reads nothing
        from the last. A provider serving no session keys on itself.
        """
        from opendde_harness.token_wise.base import CURRENT_SESSION_KEY

        return CURRENT_SESSION_KEY.get() or self._own_session

    def _output_cap(self, pinned: object, overlay: "ModelEntry | None") -> int | None:
        """The ceiling this request carries, or None to leave it to the model.

        A model's own maximum is not a pin and is not sent: pi already knows it
        and floors nothing. What is a pin is the caller's number or a
        ``maxTokens`` the user wrote for this model, the smaller of the
        two when there are both. A nonpositive result is dropped rather than
        forwarded: pi would send it as the request's ceiling.
        """
        declared = getattr(overlay, "max_tokens", None)
        if isinstance(pinned, bool) or not isinstance(pinned, int):
            pinned = declared
        elif declared and pinned > 0:
            pinned = min(pinned, int(declared))
        if not isinstance(pinned, int) or isinstance(pinned, bool) or pinned <= 0:
            return None
        return pinned

    # -- plumbing ------------------------------------------------------------

    async def prime(self) -> None:
        """Read this model's row now, rather than at the first request.

        Every fact about the model comes from that row -- the window a footer
        divides an occupancy by, the output ceiling a budget reserves for, what
        the model can be shown, what it costs -- and before the first request of
        a process there is no row at all, so each of those reads as unknown. A
        resumed session's status line therefore had a token count and no window
        to state it against. One listing settles all of them, and the row is kept,
        so the turn that follows asks for nothing extra.

        Safe to call more than once and from anywhere: the listing is read once
        per provider, and a failure to read it is left as the unknown it already
        was (``_prime_row``).
        """
        await self._resolve_service()

    async def _resolve_service(self) -> ModelService:
        """The service, and this model's row with it.

        The row is primed here rather than at the stream, so every path that
        reaches the service -- a turn, a compaction -- leaves the model's facts
        readable afterwards by the synchronous callers that need them (the
        budget's reservation, the usage report's price).
        """
        service = self._service
        if not isinstance(service, ModelService):
            service = await service()
        await self._prime_row(service)
        return service

    async def _prime_row(self, service: ModelService) -> None:
        """Ask the service once for this model's row, and keep it.

        One round trip for every fact about the model: the window to trim
        against, the output ceiling to reserve for, what it can be shown, and
        what it costs. The listing is what the service was configured with, so
        asking again would answer the same thing.

        A failure to read it is not a reason to fail the request. Every fact
        then reads as unknown, which is the answer the callers are built for --
        no trimming against an invented window, no price reported as zero.
        """
        if self._row_asked:
            return
        self._row_asked = True
        try:
            rows = await service.models()
        except Exception as exc:  # noqa: BLE001 - a missing listing is not a failed request
            logger.debug("model service: could not read the model listing ({})", exc)
            return
        for row in rows:
            if row.get("provider") == self.provider_id and row.get("id") == self.model_id:
                self._row = dict(row)
                return

    def _failed(self, exc: BaseException) -> StreamDelta:
        """A terminal delta for a failure that never became a pi event."""
        classification = self.classify_error(exc)
        return StreamDelta(
            content=format_llm_error(exc, classification, provider=self.provider_name),
            finish_reason="error",
            error_classification=classification,
        )


def build_pi_provider(config: "Config", model: str, provider_name: str) -> PiModelProvider:
    """The provider for one configured model, on the process-wide service."""
    from opendde_harness.providers import model_id
    from opendde_harness.providers.pi_service import get_service

    prefix, bare = model_id.split(model)
    return PiModelProvider(
        lambda: get_service(config),
        provider_name or prefix,
        bare or model,
        stored_model=model,
        providers=config.providers,
    )


def pi_compaction_provider(provider: str) -> str:
    """What a provider built for this id would write into a marker.

    The deferred wrapper the TUI builds has to answer for a stored marker
    before it has built anything -- the history selector asks while budgeting
    the first prompt, and both a wrong yes and a wrong no cost the whole
    history. Empty for every backend that compacts nothing.
    """
    return provider if provider in PI_COMPACTING else ""


__all__ = [
    "PI_COMPACTING",
    "PiModelProvider",
    "build_pi_provider",
    "pi_compaction_provider",
]
