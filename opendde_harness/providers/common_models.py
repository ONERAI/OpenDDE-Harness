"""What a provider can be offered as serving: the model service, then a shortlist.

The model service is the catalogue. It answers ``models`` with one row per
model the configured set serves -- pi's own rows for a vendor it carries, the
entry's own declared list for one it does not -- and that is the source the
picker and the wizard both read, so the two cannot disagree about what a
provider offers.

:data:`COMMON_MODELS` is the offline fallback and nothing else. When the
service cannot be reached at all -- no Node, no built bundle, a configuration
it refused -- a curated shortlist is better than an empty list, and it is the
only thing left that can be answered from this process. A service that answers
with no rows for a provider is not that case: it means the provider serves
nothing, and saying so is the honest answer.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from opendde_harness.config.schema import Config

COMMON_MODELS: dict[str, list[str]] = {
    # Keyed by pi provider id, and every id carries that same prefix: a stored
    # model id names its provider, and a shortlist entry is stored as written.
    # Bare OpenRouter ids start with the upstream vendor ("anthropic/..."),
    # which is the model id OpenRouter itself serves -- the prefix in front of
    # it is the gateway.
    "openrouter": [
        "openrouter/anthropic/claude-opus-4.8",
        "openrouter/anthropic/claude-opus-4.7",
        "openrouter/anthropic/claude-sonnet-5",
        "openrouter/anthropic/claude-fable-5",
        "openrouter/openai/gpt-5.5",
        "openrouter/openai/gpt-5.4-mini",
        "openrouter/google/gemini-3.5-flash",
        "openrouter/google/gemini-3-flash-preview",
        "openrouter/x-ai/grok-4.3",
        "openrouter/meta-llama/llama-4-maverick",
        "openrouter/mistralai/mistral-medium-3-5",
        "openrouter/deepseek/deepseek-v4-flash",
        "openrouter/deepseek/deepseek-v4-pro",
        "openrouter/xiaomi/mimo-v2.5",
        "openrouter/minimax/minimax-m3",
        "openrouter/z-ai/glm-5.2",
        "openrouter/tencent/hy3",
        "openrouter/moonshotai/kimi-k2.6",
        "openrouter/qwen/qwen3.7-max",
    ],
    "openai": [
        "openai/gpt-5.5",
        "openai/gpt-5.5-pro",
        "openai/gpt-5.4",
        "openai/gpt-5.4-mini",
        "openai/gpt-5.4-nano",
        "openai/gpt-5.3-codex",
        "openai/gpt-4.1",
        "openai/gpt-4o-mini",
    ],
    "anthropic": [
        "anthropic/claude-sonnet-5",
        "anthropic/claude-opus-4-8",
        "anthropic/claude-opus-4-7",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-haiku-4-5",
        "anthropic/claude-fable-5",
    ],
    "google": [
        "google/gemini-3.5-flash",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-flash-lite",
        "google/gemini-3.1-pro-preview",
        "google/gemini-3.1-flash-lite",
        "google/gemini-3-flash-preview",
    ],
    "groq": [
        "groq/openai/gpt-oss-120b",
        "groq/openai/gpt-oss-20b",
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.1-8b-instant",
        "groq/qwen/qwen3.6-27b",
    ],
    "deepseek": [
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro",
    ],
    "minimax": [
        "minimax/MiniMax-M3",
        "minimax/MiniMax-M2.7",
        "minimax/MiniMax-M2.7-highspeed",
    ],
    "minimax-cn": [
        "minimax-cn/MiniMax-M3",
        "minimax-cn/MiniMax-M2.7",
        "minimax-cn/MiniMax-M2.7-highspeed",
    ],
    "moonshotai": [
        "moonshotai/kimi-k2.6",
        "moonshotai/kimi-k2.5",
    ],
    "zai": [
        "zai/glm-5.2",
        "zai/glm-5.1",
        "zai/glm-5",
        "zai/glm-4.7",
        "zai/glm-4.6",
        "zai/glm-4.5-air",
        "zai/glm-4.5",
        "zai/glm-4.7-flash",
        "zai/glm-4.5-flash",
    ],
    "xai": [
        "xai/grok-4.3",
        "xai/grok-4-fast",
    ],
}


def common_models_for(provider: str) -> list[str]:
    """The curated shortlist for ``provider``, used only when the service is silent.

    Hand-maintained on purpose, and read only on the fallback path below: a
    provider's own ``/v1/models`` returns its whole catalogue (OpenRouter alone
    ships 300+ rows) with no "popular" flag, so a short recognizable set has to
    be curated rather than derived. Model ids drift as vendors ship releases.
    """
    return list(COMMON_MODELS.get(provider, []))


async def service_rows(config: "Config") -> list[dict[str, Any]]:
    """Every model the model service serves, pi's own rows, unconverted.

    One request for the whole set rather than one per provider: the picker asks
    about forty-odd providers at once, and grouping the answer here is cheaper
    than asking forty times. Raises whatever the service raised -- a caller
    that has a fallback decides what a failure means.
    """
    from opendde_harness.providers.pi_service import get_service

    service = await get_service(config)
    return list(await service.models())


async def refresh_models(
    config: "Config", providers: "Sequence[str] | None" = None, *, force: bool = False
) -> dict[str, str]:
    """Ask the declared endpoints for their lists again, through the service.

    What ``/models`` publishes, fetched now rather than read from the service's
    last answer. Returns the failures by provider id; the rows are read back
    with :func:`service_rows`. Raises whatever the service raised.
    """
    from opendde_harness.providers.pi_service import get_service

    service = await get_service(config)
    return await service.refresh(list(providers) if providers else None, force=force)


async def provider_auth(config: "Config") -> list[dict[str, Any]]:
    """How each provider signs in, as pi declares it. pi's rows, unconverted.

    One request for the whole set, like :func:`service_rows`, and for the same
    reason: the picker asks about every provider at once. Raises whatever the
    service raised, so a caller with a fallback decides what a failure means.
    """
    from opendde_harness.providers.pi_service import get_service

    service = await get_service(config)
    return list(await service.providers())


async def _rows_and_close(config: "Config") -> list[dict[str, Any]]:
    """:func:`service_rows`, then end the service it started.

    Only for :func:`models_for_provider`, which runs its own loop and then lets
    that loop close. The service is process-wide and would outlive it: the child
    keeps running with its pipes attached to a loop nobody can reach, the next
    caller on a live loop has to signal it and start another, and the transport
    is left to be finalised after its loop has closed -- which raises inside
    ``__del__``, printing an "Event loop is closed" traceback about an object
    nobody can still reach. A loop that starts a service closes it.
    """
    from opendde_harness.providers.pi_service import shutdown_service

    try:
        return await service_rows(config)
    finally:
        await shutdown_service()


def rows_for_provider(rows: "Sequence[dict[str, Any]]", provider: str) -> list[dict[str, Any]]:
    """The rows of ``rows`` this provider serves.

    Matched on the provider id outright: the config is keyed by pi's own ids,
    and so are the service's rows, so there is nothing between the two.

    The ids in these rows are bare: the service was configured with the id the
    endpoint serves, so a caller that stores one must qualify it first
    (``providers.model_id.join``).
    """
    return [row for row in rows if row.get("provider") == provider]


def models_for_provider(config: "Config | None", provider: str) -> list[dict[str, Any]]:
    """What this provider serves, asked of the service, from synchronous code.

    The one answer the wizard's suggestions and the CLI's model list both read,
    so neither offers a model the other would refuse. Rows are pi's own dicts;
    ``id`` is the bare id the endpoint serves and ``name`` its display name.

    Runs its own event loop, so it cannot be called from inside one -- the
    picker is async and asks :func:`service_rows` directly -- and it ends the
    service before that loop closes (:func:`_rows_and_close`). Where the
    question cannot be put to the service at all (no config, a loop already
    running, a service that will not start) the curated shortlist answers
    instead, which is the only offline answer there is.
    """

    def fallback() -> list[dict[str, Any]]:
        from opendde_harness.providers import model_id

        return [{"id": model_id.bare(model)} for model in common_models_for(provider)]

    if config is None:
        return fallback()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        logger.debug("models_for_provider: a loop is already running; answering {} offline", provider)
        return fallback()
    try:
        return rows_for_provider(asyncio.run(_rows_and_close(config)), provider)
    except Exception as exc:  # noqa: BLE001 - an unreachable service is a fallback, not a failure
        logger.debug("models_for_provider: the model service could not answer for {} ({})", provider, exc)
        return fallback()
