"""One model service per process, started on first use.

The child is expensive to start and holds the credential store, so every
provider in a process shares one. It is started inside the running loop rather
than at import or at provider construction, because both of those happen on a
thread with no loop to attach a subprocess transport to.

The configuration reaches it exactly once per shape: ``configure`` hands over
the credential store's path, every built-in key and every declared provider, and
the service writes the keys into pi's store itself -- pi's store is the one
writer of that file, so a login or a token refresh running at the same moment
is serialised with them rather than overwritten from a stale snapshot. An edited
config is noticed by fingerprint and re-sent; the service replaces its whole
provider set rather than adding to it, so nothing accumulates.

The service belongs to the loop that started it. Another loop in the same
process -- a worker thread running ``asyncio.run`` -- can neither use it nor
end it: ``get_service`` refuses while the owner is still running, and
``shutdown_service`` leaves it alone. Only a loop that has stopped hands its
child over to be signalled and replaced.
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Coroutine, TypeVar

from loguru import logger

from opendde_harness.providers.model_service import ModelService
from opendde_harness.providers.pi_auth import configure_payload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opendde_harness.config.schema import Config

#: pi's credential store, beside this project's own token files. Owned by pi:
#: ``configure`` writes the keys into it, a login run through the service
#: (``ModelService.login``) writes its grant, and a refresh rewrites it. Read
#: here (``pi_credentials``), never written.
_STORE_FILE = "pi-credentials.json"

_service: ModelService | None = None
_service_loop: asyncio.AbstractEventLoop | None = None
_fingerprint: str | None = None
_lock: asyncio.Lock | None = None


def token_dir() -> Path:
    """This project's token directory, resolved and not created.

    ``CHATGPT_TOKEN_DIR`` else ``~/.opendde_harness/oauth/chatgpt`` -- the same
    two the Codex document already honoured, so the override that moves that
    file moves pi's store with it and the two never end up in different places.
    Home-relative even when a different config file is selected: a sign-in has
    to land where every other process looks.
    """
    configured = os.environ.get("CHATGPT_TOKEN_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".opendde_harness" / "oauth" / "chatgpt"


def credential_store_path() -> Path:
    """Where pi keeps its credentials: beside this project's own token files."""
    return token_dir() / _STORE_FILE


def _fingerprint_of(config: "Config") -> str:
    """Everything ``configure`` was built from: the provider sections.

    Not the default model and not the generation settings -- neither reaches
    the service's configuration, and folding them in would re-send the whole
    provider set every time a temperature changed.
    """
    try:
        material = config.providers.model_dump(exclude_none=True)
    except Exception:  # noqa: BLE001 - an unreadable config re-configures every time, which is safe
        return ""
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()


async def _declared_catalog(service: ModelService, config: "Config") -> dict[str, list[dict]]:
    """pi's own rows for every bare model id a declared section names.

    Gathered before the payload is built, because building it is synchronous
    while each lookup is a request. ``catalog`` is answered independently of
    ``configure``, which is what makes this askable while the configuration for
    this very call is still being assembled.

    A failed lookup is not a failed configure: the section is declared with
    whatever its overlay says, which is the same answer it had before pi carried
    a catalogue at all.
    """
    from opendde_harness.providers.pi_auth import declared_model_ids

    rows: dict[str, list[dict]] = {}
    for model_id in declared_model_ids(config):
        try:
            rows[model_id] = await service.catalog(model_id)
        except Exception as exc:  # noqa: BLE001 - an unsized model is not a failed configure
            logger.debug("model service: no catalogue row for {} ({})", model_id, exc)
    return rows


async def _configure(service: ModelService, config: "Config") -> None:
    rows = await _declared_catalog(service, config)
    reply = await service.configure(configure_payload(config, credential_store_path(), catalog=rows.get))
    # Debug, not info: a CLI command's stdout is what the terminal shows, and a
    # count of providers printed into it lands in the middle of a sign-in.
    logger.debug("model service: {} providers, {} models", len(reply.get("providers") or ()), reply.get("models"))


async def get_service(config: "Config") -> ModelService:
    """The process's model service, started and configured.

    Re-configured when the provider sections change, which is how an added key
    or an edited address reaches a service that is already running.
    """
    global _service, _service_loop, _fingerprint, _lock

    loop = asyncio.get_running_loop()
    if _lock is None or _service_loop is not loop:
        _lock = asyncio.Lock()
    async with _lock:
        if _service is not None and _service_loop is not loop:
            if _service_loop is not None and _service_loop.is_running():
                # Another thread's loop is driving it right now: a worker thread
                # that ran ``asyncio.run`` while the TUI's loop streams through
                # the child. Taking it over here would end that stream.
                raise RuntimeError("the model service belongs to a loop that is still running; use it from that loop")
            # The loop that owns the child's pipes is gone. Nothing on this
            # loop can read or close it, so it is let go of and signalled.
            logger.warning("model service: the loop that started it is gone; starting another")
            _service.abandon()
            _service, _fingerprint = None, None
        if _service is None:
            service = ModelService()
            await service.start()
            _service, _service_loop, _fingerprint = service, loop, None
        fingerprint = _fingerprint_of(config)
        if fingerprint != _fingerprint:
            await _configure(_service, config)
            _fingerprint = fingerprint
        return _service


async def shutdown_service() -> None:
    """End the child. Safe to call when there is none, and on any loop.

    Called from a fixture's teardown, from a command that is finished with the
    service, and from the process's own exit hook, so it has to answer for a
    service it may not be able to reach: the loop that owns the child's pipes
    may not be the one running now, and it may be closed. Closing pipes needs
    that loop, and a signal does not, so a service that cannot be closed is
    abandoned instead -- signalled, and let go of.

    Letting go is the part that matters even when the close works. What still
    holds a subprocess transport is finalised by the garbage collector, and a
    finaliser that runs after its loop has closed raises inside ``__del__``: an
    "Event loop is closed" traceback printed after the last line of output,
    about an object nobody can still reach.
    """
    global _service, _service_loop, _fingerprint

    service, owner = _service, _service_loop
    if service is None:
        return
    running = asyncio.get_running_loop()
    if owner is not None and owner is not running and owner.is_running():
        # Somebody else's, and in use: a worker thread's loop ending while the
        # main loop streams. Not this loop's to end, so it is left as it is.
        return
    _service, _service_loop, _fingerprint = None, None, None
    if owner is not None and owner is not running:
        service.abandon()
        return
    await service.close()


T = TypeVar("T")


def run_then_shutdown(coro: "Coroutine[Any, Any, T]") -> "T":
    """``asyncio.run`` for a command that may start the service: end it first.

    ``asyncio.run`` closes its loop on return. A service started inside that
    loop and left there outlives it: the next command's loop finds the child
    unreachable and abandons it (a warning naming its pid), and the transport
    the old loop still holds is finalised by the garbage collector under some
    later line of output, raising "Event loop is closed" inside ``__del__``.
    Every command that runs a coroutine which can reach the service goes
    through here, so the child ends on the loop that owns its pipes.
    """

    async def run() -> "T":
        try:
            return await coro
        finally:
            await shutdown_service()

    return asyncio.run(run())


@atexit.register
def _stop_at_exit() -> None:
    """Signal a surviving child on the way out, and let go of it.

    There is no loop left to close pipes on at interpreter exit, so this is
    ``abandon``: a signal, which needs none. Letting go here also stops the
    child's transport being finalised against a loop that has already closed,
    which is an "Event loop is closed" traceback after the final line of output.
    """
    global _service, _service_loop, _fingerprint

    service, _service, _service_loop, _fingerprint = _service, None, None, None
    if service is not None:
        service.abandon()


__all__ = ["credential_store_path", "get_service", "shutdown_service", "token_dir"]
