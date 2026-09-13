"""A model service whose sign-in is a script, for the tests that run one.

Both doors to pi's sign-in -- the TUI's ``model.login`` and the wizard's first
step, which calls the same handler -- are tested against this rather than
against Node: what pi's flow reports is the service's business, and what each
door does with a step is theirs.
"""

from __future__ import annotations

import asyncio

#: The one provider these tests sign in to. One of pi's own, and one of the six
#: pi actually signs in to, which is what the handler gates on.
OAUTH_PROVIDER = "anthropic"

#: pi's own login-method menu and device code, as the service reports them.
MENU_STEP = {
    "type": "login_prompt",
    "ask": {
        "type": "select",
        "message": "Select OpenAI Codex login method:",
        "options": [
            {"id": "browser", "label": "Browser login (default)"},
            {"id": "device_code", "label": "Device code login (headless)"},
        ],
    },
}
CODE_STEP = {
    "type": "login_prompt",
    "notify": {"type": "device_code", "userCode": "ABCD-1234", "verificationUri": "https://example.test/device"},
}

#: The service's own reply when a login stored a grant.
GRANT = {"provider": OAUTH_PROVIDER, "type": "oauth"}


class FakeLoginService:
    """A model service whose login is a script this test drives.

    ``login_with_id`` yields the scripted steps and then waits: the flow is
    still running, which is the only state in which an answer or a cancellation
    can reach it. ``release`` lets it finish and ``abort`` fails it, the way the
    real service's own abort does; ``release_on_answer`` lets the first answer
    do the releasing, for a caller that answers from the same thread it waits on.
    """

    def __init__(
        self, steps: list[dict], result: dict | None = GRANT, *, failure: str = "", release_on_answer: bool = False
    ):
        self._steps = steps
        self._result = result
        self._go = asyncio.Event()
        self._failed = False
        #: What the flow fails with when released, instead of a grant.
        self._failure = failure
        self._release_on_answer = release_on_answer
        self.answers: list[tuple[int, str]] = []
        self.aborted: list[int] = []
        self.asked: dict | None = None
        #: Set once every scripted step has been yielded.
        self.streamed = asyncio.Event()
        #: The service takes the request only once this is set: the moment
        #: between a client's ``model.login`` and the flow existing.
        self.take = asyncio.Event()
        self.take.set()

    async def login_with_id(self, provider: str, *, mode: str | None = None):
        from opendde_harness.providers.model_service import ModelServiceError

        self.asked = {"provider": provider, "mode": mode}
        await self.take.wait()

        async def steps():
            for step in self._steps:
                yield step
            self.streamed.set()
            await self._go.wait()
            if self._failed:
                raise ModelServiceError("login_failed", "the login ended before its select step was answered")
            if self._failure:
                raise ModelServiceError("login_failed", self._failure)
            yield dict(self._result or {})

        return 11, steps()

    async def login_answer(self, login_id: int, answer: str) -> dict:
        self.answers.append((login_id, answer))
        if self._release_on_answer:
            self._go.set()
        return {"answered": True}

    async def abort(self, login_id: int) -> None:
        self.aborted.append(login_id)
        self._failed = True
        self._go.set()

    def release(self) -> None:
        self._go.set()
