"""The pi-ai model service, spoken to over stdio.

The model layer is ``@earendil-works/pi-ai`` running in a Node child
(``ui-tui/src/model-service/main.ts``); this is the Python end of that pipe.
One JSON object per line each way. A ``stream`` request is answered by one
``event`` line per pi-ai ``AssistantMessageEvent`` (without its ``partial``
snapshot) and ends with the ``done`` or ``error`` event. A request carrying a
retry budget can also carry ``retry`` events, one per attempt the service is
about to run again; the events that follow one are the new attempt's, and a
per-gap deadline the request named is what ends an attempt nothing is arriving
on. A ``compact``
request is answered once, by the items a later request replays in place of the
history they stand for. ``models`` and ``catalog`` answer with the rows that
carry every fact this project decides anything with -- window, output ceiling,
input modalities, price -- and ``providers`` with how each one is signed in to,
in pi's own words. The dicts yielded here are pi-ai's own types,
unconverted: the JSON is the contract.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from loguru import logger

#: A pi-ai event can carry a whole tool call or message; lines are not short.
_LINE_LIMIT = 64 * 1024 * 1024
_BUNDLE = "model-service.js"


class ModelServiceError(RuntimeError):
    """The service refused a request (unknown model, malformed call) or is gone."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _terminal_error(message: str) -> dict[str, Any]:
    """The event a stream ends with when the service itself is the failure."""
    return {"type": "error", "reason": "error", "error": {"stopReason": "error", "errorMessage": message}}


def resolve_bundle() -> Path | None:
    """Where the built service lives: the wheel's copy, else the source tree's."""
    from opendde_harness.node_runtime import resolve_dist

    return resolve_dist(_BUNDLE)


def resolve_node() -> str | None:
    """The Node the TUI launcher would use, or whatever is on PATH."""
    from opendde_harness.node_runtime import find_node

    path, _version = find_node()
    return path or shutil.which("node")


class ModelService:
    """One child process, many concurrent streams, multiplexed by request id."""

    def __init__(self, *, node: str | None = None, bundle: Path | None = None, env: dict[str, str] | None = None):
        self._node = node
        self._bundle = bundle
        self._env = env
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stderr: asyncio.Task[None] | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Queue[dict[str, Any]]] = {}

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    @property
    def running(self) -> bool:
        """Is there still a child of ours to end?"""
        return self._proc is not None

    async def start(self) -> None:
        node = self._node or resolve_node()
        bundle = self._bundle or resolve_bundle()
        if node is None:
            raise ModelServiceError("no_node", "no Node runtime found (set OPENDDE_HARNESS_NODE)")
        if bundle is None:
            raise ModelServiceError("no_bundle", f"{_BUNDLE} is not built; run `npm run build` in ui-tui/")
        self._proc = await asyncio.create_subprocess_exec(
            node,
            str(bundle),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Node's built-in fetch ignores HTTP(S)_PROXY unless told to; pi-ai
            # reaches the vendors through that fetch, so a proxied host would
            # otherwise fail with a bare "fetch failed".
            env={**os.environ, "NODE_USE_ENV_PROXY": "1", **(self._env or {})},
            limit=_LINE_LIMIT,
        )
        # The tasks are handed the process rather than reading ``self._proc``:
        # ending a service clears that field, and a reader still draining the
        # last of a dying child's output would then be reading from nothing.
        self._reader = asyncio.create_task(self._read(self._proc), name="model-service-reader")
        self._stderr = asyncio.create_task(self._log_stderr(self._proc), name="model-service-stderr")

    async def close(self) -> None:
        """End the child and let go of it. Safe to call twice, and from anywhere.

        Idempotent because more than one thing ends a service: a fixture's
        teardown, ``pi_service.shutdown_service``, and the process's own exit
        hook can all reach the same object, and the second call must not raise
        over a child that has already gone.

        The references are dropped whatever happens, including on the way out of
        a failure. What is left holding a subprocess transport gets finalised by
        the garbage collector, and if the loop that owns the transport has
        closed by then the finaliser's own cleanup raises inside ``__del__`` --
        an "Event loop is closed" traceback after the last line of output, about
        an object nobody can still reach. Letting go here is what stops that.

        A task belonging to another loop is abandoned rather than awaited: the
        loop that could run it is gone, so awaiting it would block or raise, and
        the child it was reading from is already being ended.
        """
        proc, reader, stderr = self._proc, self._reader, self._stderr
        if proc is None:
            return
        try:
            if proc.stdin is not None and not proc.stdin.is_closing():
                proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except TimeoutError:
                proc.kill()
                await proc.wait()
            running = asyncio.get_running_loop()
            for task in (reader, stderr):
                if task is not None and task.get_loop() is running:
                    await task
        except (ProcessLookupError, RuntimeError) as exc:
            # The child is gone, or this is not the loop that owns its pipes.
            # Either way there is nothing left to end cleanly, so say so once
            # rather than failing a teardown.
            logger.debug("model service: could not close cleanly ({})", exc)
        finally:
            # Last, not first: the readers drain the dying child's output while
            # the close runs, and they are handed the process, so clearing the
            # fields here cannot pull it out from under them.
            self._proc, self._reader, self._stderr = None, None, None

    def abandon(self) -> None:
        """Let go of a child whose loop can no longer be reached, signalling it.

        The last resort, for the two callers that have no loop to work on: the
        process's exit hook, and ``pi_service`` finding that the loop which owned
        the pipes is gone. Nothing here touches the transport -- only a signal
        goes out, which needs no loop -- and the references are dropped so the
        service reads as ended to everything that asks.
        """
        proc, self._proc, self._reader, self._stderr = self._proc, None, None, None
        pid = proc.pid if proc is not None else None
        if pid is None:
            return
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)

    async def models(self) -> list[dict[str, Any]]:
        """Every model the configured providers serve, with its facts.

        One row per model: ``{provider, id, name, contextWindow, maxTokens,
        input, reasoning, cost}``, pi's own figures. ``cost`` is pi's
        ``ModelCost`` -- rates per *million* tokens, with the long-context
        ``tiers`` a vendor publishes.
        """
        return list(await self._request({"method": "models"}))

    async def refresh(self, providers: list[str] | None = None, *, force: bool = False) -> dict[str, str]:
        """Ask the declared endpoints what they serve, now. pi's own refresh.

        Each endpoint's ``/models`` is fetched with its credential, the list is
        kept in the service's models store and overlaid on the declared rows, so
        the next :meth:`models` lists what the endpoint publishes. Answers the
        failures by provider id -- an endpoint that could not be asked keeps
        its last list, and its sentence is here rather than a failed request.
        ``providers`` restricts the ask; every declared endpoint when absent.
        Without ``force`` a list checked within the service's freshness window
        (a day) is left as it is, the way pi's own catalogs skip the network
        for a while after a check.
        """
        params: dict[str, Any] = {}
        if providers:
            params["providers"] = list(providers)
        if force:
            params["force"] = True
        reply = await self._request({"method": "refresh", "params": params})
        errors = reply.get("errors") if isinstance(reply, dict) else None
        return {str(k): str(v) for k, v in (errors or {}).items()}

    async def providers(self) -> list[dict[str, Any]]:
        """How each configured provider can be signed in to, as pi declares it.

        One row per provider: ``{id, name, methods, keyLogin, keyName,
        loginLabel, oauthName, subscription}``. ``methods`` is pi's own pair of
        auth method names (``"oauth"``, ``"api_key"``), and the labels are the
        sentences pi itself shows for each -- read off pi's provider objects
        rather than kept in a table here, so a menu offering the choice offers
        pi's words and cannot drift from them.

        Static: nothing is resolved, no file is read and no vendor is asked.
        """
        return list(await self._request({"method": "providers"}))

    async def catalog(self, model_id: str) -> list[dict[str, Any]]:
        """pi's own built-in rows for one bare model id, in the same shape.

        Independent of :meth:`configure`, and that is the point: a declared
        relay serving a vendor's model is sized from the vendor's row, and the
        question is asked while the configuration is still being built. A
        built-in catalogue nothing reconfigures answers it.

        Several providers can serve the same id (the vendor and every relay
        that resells it); every matching row is returned and the caller decides.
        Empty when pi carries no row for the id.
        """
        if not model_id:
            return []
        return list(await self._request({"method": "catalog", "params": {"id": model_id}}))

    async def configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Hand the service this project's credentials and custom providers.

        ``payload`` is the ``configure`` params -- build it with
        :func:`opendde_harness.providers.pi_auth.configure_payload`. Calling it
        again replaces the whole set rather than adding to it, and requests
        that arrive afterwards see the new one: the service holds its line
        queue until a configure has finished.
        """
        return dict(await self._request({"method": "configure", "params": payload}))

    async def debug(self) -> dict[str, Any]:
        """The last request the service's faux mode built, and nothing else.

        Answered only under ``OPENDDE_MODEL_SERVICE_FAUX=1``. The scripted
        providers send nothing, so this is the one place an offline test can see
        the request a replayed compaction actually produced rather than the
        arguments it was asked with.
        """
        return dict(await self._request({"method": "debug"}))

    async def auth(self) -> dict[str, Any]:
        """What is stored and what each provider resolves. Never a secret value."""
        return dict(await self._request({"method": "auth"}))

    async def login(self, provider: str, *, mode: str = "device_code") -> AsyncIterator[dict[str, Any]]:
        """Run pi's own login flow, yielding each step and then the result.

        ``mode`` settles pi's login-method menu without a round trip, which is
        what a caller with nobody to ask needs. ``device_code`` is the one that
        needs no browser on this machine; ``browser`` completes on the service's
        own localhost callback. To show the menu instead, use
        :meth:`login_with_id` and answer it with :meth:`login_answer`.
        """
        _login_id, steps = await self.login_with_id(provider, mode=mode)
        async for step in steps:
            yield step

    async def login_with_id(
        self, provider: str, *, mode: str | None = None
    ) -> tuple[int, AsyncIterator[dict[str, Any]]]:
        """Like :meth:`login`, also returning the id its answers are addressed to.

        Every yielded dict but the last is one ``login_prompt`` event:
        ``notify`` carries something to show (a device code and its URL, a
        sign-in link), ``ask`` something pi is waiting for -- the login-method
        menu, or the authorization code a browser login falls back to asking
        for. An ``ask`` is answered by :meth:`login_answer` naming this id, and
        :meth:`abort` on the same id cancels the sign-in. The last dict is the
        reply, ``{"provider", "type"}``.

        ``mode`` absent is what leaves the menu to be answered here; naming one
        answers it inside the service. The menu's ``ask`` still arrives either
        way -- the service reports every prompt before deciding who answers it
        -- so a caller that named a mode sees it and owes nothing for it.
        """
        request_id, queue = self._open()
        params: dict[str, Any] = {"provider": provider}
        if mode:
            params["mode"] = mode
        await self._send({"id": request_id, "method": "login", "params": params})

        async def steps() -> AsyncIterator[dict[str, Any]]:
            try:
                while True:
                    reply = await queue.get()
                    if "error" in reply:
                        raise ModelServiceError(reply["error"]["code"], reply["error"]["message"])
                    event = reply.get("event")
                    if event is not None:
                        if event.get("type") != "login_prompt":
                            # The reader's own terminal event: the child is
                            # gone, and no result will ever arrive on this id.
                            raise ModelServiceError("gone", str(event.get("error", {}).get("errorMessage") or event))
                        yield event
                        continue
                    yield dict(reply["result"])
                    return
            finally:
                self._pending.pop(request_id, None)

        return request_id, steps()

    async def login_answer(self, login_id: int, answer: str) -> dict[str, Any]:
        """Hand a waiting login the string it asked for.

        The option id for a menu, the pasted code for a browser login. Answers
        ``{"answered"}``; False when nothing was waiting under that id -- the
        login ended, or pi's own callback server got there first.
        """
        return dict(await self._request({"method": "login_answer", "params": {"loginId": login_id, "answer": answer}}))

    async def logout(self, provider: str) -> dict[str, Any]:
        """Forget this provider's stored credential. The sign-out.

        Run through the service rather than against the file, so the delete is
        serialised with the writes a refresh or a login may be making at the
        same moment. Answers ``{"provider", "forgotten"}``; ``forgotten`` is
        False when there was nothing stored, which is not a failure.
        """
        return dict(await self._request({"method": "logout", "params": {"provider": provider}}))

    async def compact(
        self,
        provider: str,
        model: str,
        context: dict[str, Any],
        *,
        replay: dict[str, Any] | None = None,
        session_id: str | None = None,
        reasoning: str | None = None,
    ) -> dict[str, Any]:
        """Ask the backend to compact this conversation itself.

        One request, one reply: ``{"items", "usage"}``, where ``items`` is what
        a later request replays in place of everything before it and ``usage``
        is pi's own ``Usage`` for the billed call. Anything else raises, because
        the caller keeps its history on a failure and that is only safe while a
        compaction that did not happen cannot look like one that did.

        The tool catalogue rides in ``context`` like it does on a stream, so the
        compacted request carries the same body as the requests it stands for.
        """
        params: dict[str, Any] = {"provider": provider, "model": model, "context": context}
        if replay:
            params["replay"] = replay
        if session_id:
            params["sessionId"] = session_id
        if reasoning:
            params["reasoning"] = reasoning
        result = await self._request({"method": "compact", "params": params})
        if not isinstance(result, dict):
            raise ModelServiceError("compaction_failed", f"the service answered compact with {result!r}")
        return dict(result)

    async def abort(self, stream_id: int) -> None:
        """Cancel one request by its id: a stream, or a login waiting on an answer."""
        request_id, queue = self._open()
        await self._send({"id": request_id, "method": "abort", "params": {"id": stream_id}})
        await queue.get()
        self._pending.pop(request_id, None)

    async def stream(
        self,
        provider: str,
        model: str,
        context: dict[str, Any],
        options: dict[str, Any] | None = None,
        *,
        replay: dict[str, Any] | None = None,
        retry: dict[str, Any] | None = None,
        timeouts: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """pi-ai's events for one request, ending with ``done`` or ``error``.

        Closing the iterator before that end aborts the stream. To abort it
        from elsewhere, use :meth:`stream_with_id` and :meth:`abort`.
        """
        _stream_id, events = await self.stream_with_id(
            provider, model, context, options, replay=replay, retry=retry, timeouts=timeouts
        )
        async for event in events:
            yield event

    async def stream_with_id(
        self,
        provider: str,
        model: str,
        context: dict[str, Any],
        options: dict[str, Any] | None = None,
        *,
        replay: dict[str, Any] | None = None,
        retry: dict[str, Any] | None = None,
        timeouts: dict[str, Any] | None = None,
    ) -> tuple[int, AsyncIterator[dict[str, Any]]]:
        """Like :meth:`stream`, also returning the id an :meth:`abort` names.

        ``replay`` is a backend's own compaction, ``{"items": [...]}``, which
        the service sends ahead of ``context``'s messages and in place of
        everything they no longer carry.

        ``retry`` is this request's retry budget, ``{"maxRetries": n}``, spent
        inside the service by pi's own retry loop. One request still ends once:
        an attempt that will be run again is announced by a ``retry`` event and
        followed by the next attempt's events, and only the last attempt's
        ``done`` or ``error`` arrives.

        ``timeouts`` bounds each attempt per silence,
        ``{"firstTokenMs", "idleMs"}``. An expired budget aborts that attempt at
        the service -- an abandoned stream is still being generated and still
        being billed -- and fails it transiently, so the retry budget covers a
        stalled stream like any other failure. Absent means no deadline, and a
        stream that keeps delivering is never cut however long it runs.
        """
        request_id, queue = self._open()
        params: dict[str, Any] = {"provider": provider, "model": model, "context": context}
        if options:
            params["options"] = options
        if replay:
            params["replay"] = replay
        if retry:
            params["retry"] = retry
        if timeouts:
            params["timeouts"] = timeouts
        await self._send({"id": request_id, "method": "stream", "params": params})

        async def events() -> AsyncIterator[dict[str, Any]]:
            finished = False
            try:
                while True:
                    reply = await queue.get()
                    if "error" in reply:
                        raise ModelServiceError(reply["error"]["code"], reply["error"]["message"])
                    event = reply["event"]
                    if event.get("type") in {"done", "error"}:
                        finished = True
                        yield event
                        return
                    yield event
            finally:
                if not finished and self._alive():
                    try:
                        await self.abort(request_id)
                    except ModelServiceError:
                        pass  # gone between the check and the send: nothing left to abort
                self._pending.pop(request_id, None)

        return request_id, events()

    # -- plumbing -----------------------------------------------------------

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def _request(self, request: dict[str, Any]) -> Any:
        """One request, one reply. Raises what the service refused."""
        request_id, queue = self._open()
        try:
            await self._send({"id": request_id, **request})
            reply = await queue.get()
        finally:
            self._pending.pop(request_id, None)
        if "error" in reply:
            raise ModelServiceError(reply["error"]["code"], reply["error"]["message"])
        if "result" not in reply:
            raise ModelServiceError("gone", f"the service answered {request['method']} with no result")
        return reply["result"]

    def _open(self) -> tuple[int, asyncio.Queue[dict[str, Any]]]:
        if not self._alive():
            raise ModelServiceError("gone", "the model service is not running")
        self._next_id += 1
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending[self._next_id] = queue
        return self._next_id, queue

    async def _send(self, request: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None  # noqa: S101 - _open() checked
        try:
            self._proc.stdin.write(json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n")
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise ModelServiceError("gone", f"the model service closed its input: {exc}") from exc

    async def _read(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None  # noqa: S101 - start() built it
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    reply = json.loads(line)
                except ValueError:
                    logger.warning("model-service: unreadable line {!r}", line[:200])
                    continue
                queue = self._pending.get(reply.get("id"))
                if queue is None:
                    # Normal after an abort or an early close: the stream's
                    # last events land after its reader left.
                    logger.debug("model-service: reply for a closed id {!r}", reply.get("id"))
                    continue
                queue.put_nowait(reply)
        finally:
            # The process is gone. Every waiter learns it now rather than
            # hanging on a queue nothing will fill.
            code = proc.returncode
            message = f"the model service exited (code {code})"
            for queue in self._pending.values():
                queue.put_nowait({"id": None, "event": _terminal_error(message)})

    async def _log_stderr(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr is not None  # noqa: S101 - start() built it
        while True:
            line = await proc.stderr.readline()
            if not line:
                return
            logger.warning("model-service: {}", line.decode("utf-8", "replace").rstrip())


__all__ = ["ModelService", "ModelServiceError", "resolve_bundle", "resolve_node"]
