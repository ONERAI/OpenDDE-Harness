"""Shared pytest fixtures."""

import os
import tempfile

import httpcore
import httpx
import pytest
from httpcore._backends import auto as auto_backend

# litellm setup exports the OAuth token directories from the real user's home the
# first time a provider module is imported, which happens while pytest collects.
# Claiming them here keeps provider discovery away from the developer's own tokens,
# which otherwise made the startup gate report a configured provider under a
# temporary HOME.
_OAUTH_SANDBOX = tempfile.mkdtemp(prefix="opendde-harness-tests-oauth-")
os.environ["CHATGPT_TOKEN_DIR"] = os.path.join(_OAUTH_SANDBOX, "chatgpt")
os.environ["MINIMAX_OAUTH_TOKEN_DIR"] = os.path.join(_OAUTH_SANDBOX, "minimax")

# And nothing writes a credential unless a test has pointed the store at its
# own directory first (``credential_fixtures.use_token_dir`` lifts this). A
# resolve against a real token directory once converted an owner's file to a
# format the rest of their machine could not read; a test run must not be able
# to do that, whatever the environment it inherits.
os.environ["OPENDDE_HARNESS_CREDENTIALS_READONLY"] = "1"


@pytest.fixture(autouse=True)
async def no_model_service_outlives_its_loop():
    """End the process-wide model service on the loop that owns its pipes.

    ``pi_service`` keeps one child per process and pytest-asyncio gives each test
    its own loop (``asyncio_mode = "auto"``, function-scoped), so a service
    started inside a test has pipes belonging to a loop that closes when the test
    ends. What survives is worse than a stray process: its subprocess transport
    is finalised later by the garbage collector, and the finaliser's own cleanup
    calls ``loop.call_soon`` on a closed loop -- "RuntimeError: Event loop is
    closed" raised inside ``__del__``, printed after the last line of the run and
    naming an object no test can be blamed for.

    Async on purpose, and that is the whole point: an async teardown runs on the
    test's own loop while it is still open, which is the only place the pipes can
    actually be closed. A synchronous teardown can only signal the child and let
    go of it, which leaves the transport unclosed and the finaliser still to
    come. A test that ends its own service leaves this nothing to do.
    """
    yield
    from opendde_harness.providers import pi_service

    if pi_service._service is not None:
        await pi_service.shutdown_service()


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    """Give every test a home of its own, so none of them reads the real one.

    ``load_config()`` with no argument reads ``~/.opendde_harness/config.json``,
    and any test that reached it without isolating first read the developer's
    own config -- their providers, their keys, and whatever retired settings
    that file still carries. That made the suite's result depend on the machine
    it ran on, and put a real credential into a failure traceback.

    ``HOME`` rather than ``loader._current_config_path``: the tests that want a
    config of their own write one under a temporary home and let the same
    fallback find it, and a path pinned here would outrank the file they wrote.
    Everything else this project keeps beside the config -- the workspace, the
    history file, the telemetry directory, the sign-in store -- hangs off the
    same ``Path.home()``, so one variable covers all of them.
    ``OPENDDE_HARNESS_HOME`` is claimed as well: it outranks ``Path.home()`` for
    the Node runtime and the compute assets, and an inherited one would put a
    test back on the real directory.
    """
    from opendde_harness.config import loader

    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("OPENDDE_HARNESS_HOME", str(home / ".opendde_harness"))
    loader._cache.clear()
    yield
    loader._cache.clear()


@pytest.fixture(autouse=True)
def restore_environment():
    """Undo environment writes made by the code under test.

    Provider setup exports credentials into ``os.environ`` (``ZHIPUAI_API_KEY``
    and friends). ``monkeypatch`` only restores what a test set itself, so those
    writes leaked into later tests.
    """
    snapshot = dict(os.environ)
    yield
    if os.environ != snapshot:
        os.environ.clear()
        os.environ.update(snapshot)


#: Hosts a test may address. A loopback address is a server the test started
#: itself; anything else is the internet, whatever route it takes to get there.
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost", "0.0.0.0"})


#: Every httpcore backend that can open a socket. A pool holding one of these,
#: or anything derived from one, reaches the network however it is packaged.
_SOCKET_BACKENDS = (
    httpcore.SyncBackend,
    httpcore.AnyIOBackend,
    auto_backend.AutoBackend,
)


def _opens_a_socket(transport) -> bool:
    """Could this transport reach the network, whatever its class is called?

    Asked of what the backend *is*, not where it was defined. The earlier
    version exempted any backend from outside httpcore, which a bare subclass
    of the real backend defeats: it is declared in a test module and inherits
    every socket operation. Inheritance is the question, so inheritance is what
    is checked.
    """
    backend = getattr(getattr(transport, "_pool", None), "_network_backend", None)
    return backend is None or isinstance(backend, _SOCKET_BACKENDS)


def _refuse(request) -> None:
    """Fail the test that is about to leave this machine.

    The suite must run with no network and must prove nothing about a vendor by
    accident. A credential test that forgot its fake transport reached a real
    token endpoint and then passed or failed on what that endpoint said. This
    names the call instead, and raises an error no credential code catches, so
    it cannot be mistaken for the vendor's own refusal.

    Checked on the request's own host rather than on the socket's destination:
    with a proxy configured, every socket goes to loopback and a
    connection-level guard would wave the whole internet through.
    """
    host = request.url.host
    if host in _LOOPBACK:
        return
    raise AssertionError(
        f"this test tried to reach {request.method} {request.url.scheme}://{host}{request.url.path} -- "
        "give it a fake transport (credential_fixtures.scripted) or a loopback server"
    )


@pytest.fixture(autouse=True)
def no_outbound_network(request, monkeypatch):
    """Refuse any request addressed outside this machine.

    A test that drives httpx's real writer over an in-memory backend says so
    with the ``in_memory_transport`` marker. The marker alone is not enough:
    the backend still has to be one that cannot open a socket, so the
    declaration cannot be used to smuggle a real connection past the guard.
    """
    declared = request.node.get_closest_marker("in_memory_transport") is not None

    def check(transport, outgoing):
        if declared and not _opens_a_socket(transport):
            return
        _refuse(outgoing)

    real_async = httpx.AsyncHTTPTransport.handle_async_request
    real_sync = httpx.HTTPTransport.handle_request

    async def guarded_async(self, outgoing):
        check(self, outgoing)
        return await real_async(self, outgoing)

    def guarded_sync(self, outgoing):
        check(self, outgoing)
        return real_sync(self, outgoing)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", guarded_async)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", guarded_sync)
