"""``web_search`` and ``web_fetch`` answer without a key and read pages locally."""

from __future__ import annotations

import httpx
import pytest

from opendde_harness.agent.tools import web
from opendde_harness.agent.tools.base import ToolResult
from opendde_harness.agent.tools.web import (
    EngineRefusedError,
    Page,
    WebFetchTool,
    WebSearchTool,
    _parse_ddg_html,
    extract,
)

DDG = """<div class="result results_links"><h2 class="result__title">
<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.uniprot.org%2Funiprotkb%2FQ15116&amp;rut=abc">PD-1 <b>entry</b></a></h2></div>
<div class="result results_links"><a href="/l/?rut=opaque" class="result__a">Second</a>
<a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">Second <b>snippet</b></a></div>
<div class="result"><a rel="nofollow" class="result__a" href="https://www.rcsb.org/structure/4ZQK">RCSB</a></div>"""

HTML = (
    "<!doctype html><html><head><title>PD-1 notes</title></head><body><nav>menu menu menu</nav>"
    "<article><h1>Programmed cell death protein 1</h1>"
    + "".join(
        f"<p>Paragraph {i} about the PD-1 extracellular domain and its ligand PD-L1 binding interface.</p>"
        for i in range(12)
    )
    + "</article><footer>footer</footer></body></html>"
)


def test_ddg_html_reads_each_result_container_on_its_own() -> None:
    """A result without a snippet must not borrow the next one's, an anchor
    with href before class still counts, and a redirect without uddg is
    resolved against the endpoint."""
    assert _parse_ddg_html(DDG) == [
        ("PD-1 entry", "https://www.uniprot.org/uniprotkb/Q15116", ""),
        ("Second", "https://html.duckduckgo.com/l/?rut=opaque", "Second snippet"),
        ("RCSB", "https://www.rcsb.org/structure/4ZQK", ""),
    ]


def _serve(monkeypatch, answers: dict[str, object]) -> list[str]:
    """Answer ``fetch_page`` from a table keyed by URL prefix; an Exception
    value is raised. Returns the list of URLs asked for."""
    asked: list[str] = []

    async def fake_fetch(url: str, *, proxy=None, timeout=15.0) -> Page:
        asked.append(url)
        for prefix, answer in answers.items():
            if url.startswith(prefix):
                if isinstance(answer, Exception):
                    raise answer
                return Page(url, "text/html", str(answer), 200)
        raise AssertionError(f"unexpected fetch {url}")

    monkeypatch.setattr(web, "fetch_page", fake_fetch)
    return asked


DDG_URL = web._DDG_ENDPOINT

PUZZLE = '<html><body><form id="challenge-form"><div class="anomaly-modal__title">bots use DuckDuckGo too</div></form></body></html>'


async def test_results_are_numbered_and_capped_by_count(monkeypatch) -> None:
    asked = _serve(monkeypatch, {DDG_URL: DDG})

    out = await WebSearchTool().execute("PD-1 structure", count=1)

    assert [u.split("q=")[0] for u in asked] == [DDG_URL.split("q=")[0]]
    assert out.startswith("Results for: PD-1 structure")
    assert "1. PD-1 entry\n   https://www.uniprot.org/uniprotkb/Q15116" in out
    assert "2. Second" not in out, "count caps the list"


def test_a_puzzle_page_is_a_refusal_not_an_empty_answer() -> None:
    with pytest.raises(EngineRefusedError):
        _parse_ddg_html(PUZZLE)


async def test_every_engine_refusing_is_final_for_a_while(monkeypatch) -> None:
    asked = _serve(monkeypatch, {DDG_URL: PUZZLE})
    tool = WebSearchTool()

    first = await tool.execute("x")
    second = await tool.execute("x again")

    assert isinstance(first, ToolResult) and first.retryable is False
    assert first.model_text.startswith("Error: web search is unavailable")
    assert "web_fetch" in first.model_text
    assert second.model_text == first.model_text
    assert len(asked) == 1, "the second call did not reach the network"


async def test_an_engine_answering_nothing_is_no_results(monkeypatch) -> None:
    _serve(monkeypatch, {DDG_URL: "<html>nothing</html>"})
    assert await WebSearchTool().execute("zzz") == "No results for: zzz"


async def test_content_adds_an_excerpt_of_each_page_and_skips_unreadable_ones(monkeypatch) -> None:
    _serve(
        monkeypatch,
        {
            DDG_URL: DDG,
            "https://www.uniprot.org/uniprotkb/Q15116": HTML,
            "https://html.duckduckgo.com/l/?rut=opaque": RuntimeError("timed out"),
            "https://www.rcsb.org/structure/4ZQK": RuntimeError("timed out"),
        },
    )

    out = await WebSearchTool().execute("PD-1", content=True)

    assert "Paragraph 0 about the PD-1 extracellular domain" in out
    assert "menu menu menu" not in out, "the article, not the chrome"
    assert out.index("2. Second") > out.index("---"), "the unreadable page keeps its bare result"
    assert out.count("---") == 2


def test_the_tool_needs_no_key() -> None:
    tool = WebSearchTool(proxy="socks5://127.0.0.1:1080")
    assert tool.proxy == "socks5://127.0.0.1:1080" and tool.name == "web_search"


# ---------------------------------------------------------------------------
# web_fetch
# ---------------------------------------------------------------------------


def test_an_api_answer_is_returned_as_it_is() -> None:
    page = Page(
        "https://rest.uniprot.org/uniprotkb/Q15116.json", "application/json", '{"primaryAccession":"Q15116"}', 200
    )
    assert extract(page) == ('{"primaryAccession":"Q15116"}', "raw")


def test_an_html_page_is_reduced_to_its_article() -> None:
    text, extractor = extract(Page("https://x/notes", "text/html; charset=utf-8", HTML, 200))
    assert extractor == "trafilatura"
    assert "Paragraph 0 about the PD-1" in text and "menu menu menu" not in text
    plain, _ = extract(Page("https://x/notes", "text/html", HTML, 200), "text")
    assert "#" not in plain


def test_an_html_page_with_no_article_falls_back_to_its_text(monkeypatch) -> None:
    monkeypatch.setattr(web.trafilatura, "extract", lambda *a, **k: None)
    text, extractor = extract(Page("https://x/login", "text/html", "<html><body><b>Sign</b> in</body></html>", 200))
    assert (text, extractor) == ("Sign in", "text")


@pytest.fixture
def any_target(monkeypatch):
    """The URL check resolves hostnames; these tests never reach the network."""
    monkeypatch.setattr(web, "validate_url_target", lambda url: (True, ""))


async def test_fetch_returns_a_header_and_truncates_to_the_asked_size(monkeypatch, any_target) -> None:
    async def fake_fetch(url: str, *, proxy=None, timeout=15.0) -> Page:
        return Page("https://x/final", "text/plain", "x" * 1_000, 200)

    monkeypatch.setattr(web, "fetch_page", fake_fetch)
    out = await WebFetchTool().execute("https://x/notes", maxChars=400)
    assert out.startswith("URL: https://x/final (redirected from https://x/notes)\nextractor: raw\n\n")
    assert "[truncated: the first 400 of 1,000 characters are shown]" in out


async def test_fetch_reports_a_failure_instead_of_raising(monkeypatch, any_target) -> None:
    async def boom(url: str, *, proxy=None, timeout=15.0) -> Page:
        raise ValueError("application/pdf content is not readable as text")

    monkeypatch.setattr(web, "fetch_page", boom)
    assert (await WebFetchTool().execute("https://x/file.pdf")).startswith("Error: application/pdf content")


async def test_fetch_refuses_an_unsafe_target() -> None:
    assert (await WebFetchTool().execute("file:///etc/passwd")).startswith("Error: URL validation failed")


def _readers(monkeypatch, local: object, jina: object = "jina text") -> list[str]:
    """``local`` is a Page or an exception for the local read; ``jina`` a
    string or an exception for Jina. Returns the readers used, in order."""
    calls: list[str] = []

    async def fetch(url: str, *, proxy=None, timeout=15.0) -> Page:
        calls.append("local")
        if isinstance(local, Exception):
            raise local
        return local

    async def via_jina(self, url: str, mode: str) -> str:
        calls.append("jina")
        if isinstance(jina, Exception):
            raise jina
        return str(jina)

    monkeypatch.setattr(web, "fetch_page", fetch)
    monkeypatch.setattr(WebFetchTool, "_via_jina", via_jina)
    return calls


def _refusal(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://x/page")
    return httpx.HTTPStatusError(str(status), request=request, response=httpx.Response(status, request=request))


async def test_a_page_the_local_read_can_say_never_goes_to_jina(monkeypatch, any_target) -> None:
    calls = _readers(monkeypatch, Page("https://x/page", "text/html", HTML, 200))
    out = await WebFetchTool().execute("https://x/page")
    assert calls == ["local"] and "extractor: trafilatura" in out


async def test_an_api_answer_never_goes_to_jina_however_short(monkeypatch, any_target) -> None:
    calls = _readers(monkeypatch, Page("https://x/api", "application/json", '{"n":1}', 200))
    out = await WebFetchTool().execute("https://x/api")
    assert calls == ["local"] and out.endswith('{"n":1}')


async def test_a_thin_html_article_is_read_through_jina(monkeypatch, any_target) -> None:
    """A client-side app's shell extracts to a notice; Jina renders the app."""
    shell = "<html><body><div id=app>Enable JavaScript to see this page.</div></body></html>"
    calls = _readers(monkeypatch, Page("https://x/app", "text/html", shell, 200))
    out = await WebFetchTool().execute("https://x/app")
    assert calls == ["local", "jina"] and "extractor: jina-reader" in out and "jina text" in out


async def test_a_thin_html_page_labelled_as_text_is_still_read_through_jina(monkeypatch, any_target) -> None:
    """The extractor saw the HTML doctype behind the wrong label; reader
    selection follows that, not the label."""
    shell = "<!doctype html><html><body><p>Enable JavaScript.</p></body></html>"
    calls = _readers(monkeypatch, Page("https://x/app", "text/plain", shell, 200))
    out = await WebFetchTool().execute("https://x/app")
    assert calls == ["local", "jina"] and "jina text" in out


async def test_a_refused_local_fetch_is_read_through_jina(monkeypatch, any_target) -> None:
    calls = _readers(monkeypatch, _refusal(403))
    out = await WebFetchTool().execute("https://x/page")
    assert calls == ["local", "jina"] and "jina text" in out


async def test_when_jina_fails_too_the_local_result_or_error_is_what_remains(monkeypatch, any_target) -> None:
    shell = "<html><body>Enable JavaScript.</body></html>"
    _readers(monkeypatch, Page("https://x/app", "text/html", shell, 200), jina=RuntimeError("429"))
    out = await WebFetchTool().execute("https://x/app")
    assert "jina-reader" not in out and "Enable JavaScript." in out

    _readers(monkeypatch, _refusal(403), jina=RuntimeError("429"))
    assert (await WebFetchTool().execute("https://x/page")).startswith("Error: 403")


async def test_a_key_a_header_cannot_carry_is_refused_before_any_request(monkeypatch) -> None:
    """The HTTP stack's own rejection quotes the whole header value, key included."""
    _mock_client(monkeypatch, lambda request: (_ for _ in ()).throw(AssertionError("no request")))
    with pytest.raises(ValueError, match="Jina key") as jina:
        await WebFetchTool(api_key="jina\nkey")._via_jina("https://x/page", "text")
    with pytest.raises(ValueError, match="Brave key") as brave:
        await WebSearchTool(api_key="brave\nkey")._brave("x", 1)
    assert "key" not in str(jina.value).replace("Jina key", "") and "brave\n" not in str(brave.value)


def _in_memory_client(monkeypatch, response: bytes) -> list[str]:
    # The tests using this carry ``in_memory_transport``: the suite's offline
    # guard refuses every non-loopback host unless a test declares that its
    # transport cannot open a socket, which this backend cannot.
    """httpx over a network backend that never opens a socket.

    ``MockTransport`` answers above h11, so it never sees a header the protocol
    refuses; these tests are about what h11 does with one, so the real
    connection pool and its real HTTP/1.1 writer stay in the path and only the
    bytes on the wire are faked.
    """
    import httpcore

    connected: list[str] = []

    class _Stream:
        async def read(self, max_bytes, timeout=None):
            return response

        async def write(self, buffer, timeout=None):
            return None

        async def aclose(self):
            return None

        async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            return self

        def get_extra_info(self, info):
            return None

    class _Backend:
        async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
            connected.append(host)
            return _Stream()

    real_client = httpx.AsyncClient

    def build(**kw):
        transport = httpx.AsyncHTTPTransport()
        transport._pool = httpcore.AsyncConnectionPool(network_backend=_Backend())
        return real_client(transport=transport, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", build)
    return connected


@pytest.mark.in_memory_transport
async def test_a_key_pasted_with_surrounding_space_becomes_a_header_the_stack_accepts(monkeypatch) -> None:
    """A key copied out of a dashboard arrives with a space or a newline around
    it. h11 refuses such a header value and quotes the whole value refusing it,
    so the credential reached the log; the fix is to strip where the key is
    acquired, not to report it."""
    monkeypatch.setenv("BRAVE_API_KEY", " brave-key\n")
    monkeypatch.setenv("JINA_API_KEY", "\tjina-key ")
    assert WebSearchTool().api_key == "brave-key"
    assert WebFetchTool().api_key == "jina-key"

    connected = _in_memory_client(monkeypatch, b'HTTP/1.1 200 OK\r\nContent-Length: 22\r\n\r\n{"web":{"results":[]}}')
    assert await WebSearchTool()._brave("pd-1", 1) == [], "the request went out rather than being refused"
    assert connected == ["api.search.brave.com"]


@pytest.mark.in_memory_transport
async def test_a_header_the_stack_refuses_is_reported_as_its_type(monkeypatch) -> None:
    """The last guard: whatever a future acquisition path lets through, the
    refusal names the header value in full and must not be repeated."""
    _in_memory_client(monkeypatch, b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")

    with pytest.raises(Exception) as raised:  # noqa: PT011 - h11's own type, through httpx
        async with httpx.AsyncClient() as client:
            await client.get("https://api.search.brave.com/", headers={"X-Subscription-Token": " secret-key"})

    assert "secret-key" in str(raised.value), "the stack really does quote the value"
    assert web._brief(raised.value) == type(raised.value).__name__
    assert "secret-key" not in web._brief(raised.value)


@pytest.mark.parametrize("key", ["", "jina-key"])
async def test_jina_is_keyless_and_a_key_only_adds_the_header(monkeypatch, key) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        seen["url"] = str(request.url)
        return httpx.Response(200, text="rendered")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.delenv("JINA_API_KEY", raising=False)

    assert await WebFetchTool(api_key=key or None)._via_jina("https://x/page", "text") == "rendered"
    assert seen["url"] == "https://r.jina.ai/https://x/page"
    assert seen["x-return-format"] == "text"
    assert seen.get("authorization") == (f"Bearer {key}" if key else None)


# ---------------------------------------------------------------------------
# fetch_page: every hop is checked
# ---------------------------------------------------------------------------


def _transport(routes: dict[str, httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: routes[str(request.url)])


async def test_a_redirect_to_a_private_address_is_refused(monkeypatch) -> None:
    """The first URL passes the check; the hop it redirects to must be checked too."""
    routes = {
        "https://public.example/go": httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"}),
        "http://169.254.169.254/latest/": httpx.Response(200, text="secret"),
    }
    asked: list[str] = []

    def check(url: str) -> tuple[bool, str]:
        asked.append(url)
        return (False, "Blocked: private address") if "169.254" in url else (True, "")

    monkeypatch.setattr(web, "validate_url_target", check)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=_transport(routes), **kw))

    with pytest.raises(ValueError, match="refusing to fetch http://169.254.169.254/latest/"):
        await web.fetch_page("https://public.example/go")
    assert asked == ["https://public.example/go", "http://169.254.169.254/latest/"]


async def test_a_public_redirect_is_followed_and_the_final_url_reported(monkeypatch) -> None:
    routes = {
        "https://public.example/go": httpx.Response(301, headers={"location": "https://public.example/there"}),
        "https://public.example/there": httpx.Response(
            200, text='{"ok":1}', headers={"content-type": "application/json"}
        ),
    }
    monkeypatch.setattr(web, "validate_url_target", lambda url: (True, ""))
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=_transport(routes), **kw))

    page = await web.fetch_page("https://public.example/go")
    assert (page.url, page.content_type, page.text) == ("https://public.example/there", "application/json", '{"ok":1}')


async def test_binary_content_is_refused(monkeypatch) -> None:
    routes = {
        "https://public.example/f.pdf": httpx.Response(
            200, content=b"%PDF-1.4\x00", headers={"content-type": "application/pdf"}
        )
    }
    monkeypatch.setattr(web, "validate_url_target", lambda url: (True, ""))
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=_transport(routes), **kw))

    with pytest.raises(ValueError, match="application/pdf content is not readable"):
        await web.fetch_page("https://public.example/f.pdf")


async def test_an_excerpt_page_on_a_private_address_is_skipped_not_fetched(monkeypatch) -> None:
    """Search results are attacker-influenced URLs; the excerpt fetch goes
    through the same check as any other fetch."""
    routes = {
        DDG_URL + "x": httpx.Response(
            200, text=DDG.replace("https://www.rcsb.org/structure/4ZQK", "http://10.0.0.5/admin")
        )
    }
    checked: list[str] = []

    def check(url: str) -> tuple[bool, str]:
        checked.append(url)
        return (False, "Blocked") if "10.0.0.5" in url else (True, "")

    monkeypatch.setattr(web, "validate_url_target", check)

    async def fetch(url: str, *, proxy=None, timeout=15.0) -> Page:
        ok, why = web.validate_url_target(url)
        if not ok:
            raise ValueError(why)
        return Page(url, "text/html", routes[url].text if url in routes else HTML, 200)

    monkeypatch.setattr(web, "fetch_page", fetch)
    out = await WebSearchTool().execute("x", content=True)

    assert "http://10.0.0.5/admin" in checked
    assert "3. RCSB\n   http://10.0.0.5/admin" in out and out.count("---") == 4


# ---------------------------------------------------------------------------
# Engines: Brave with a key, DuckDuckGo without; only engine-wide failures cool down
# ---------------------------------------------------------------------------


def _mock_client(monkeypatch, handler) -> None:
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))


def test_the_engine_order_depends_on_the_key(monkeypatch) -> None:
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    assert [name for name, _ in WebSearchTool().engines()] == ["duckduckgo"]
    assert [name for name, _ in WebSearchTool(api_key="brave-key").engines()] == ["brave", "duckduckgo"]


async def test_brave_is_asked_with_the_key_and_its_results_are_read(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["token"] = request.headers.get("x-subscription-token", "")
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "RCSB PDB - 5GGS",
                            "url": "https://www.rcsb.org/structure/5GGS",
                            "description": "PD-1 <b>Fab</b>",
                        },
                        {"title": "no url", "description": "dropped"},
                    ]
                }
            },
        )

    _mock_client(monkeypatch, handler)
    hits = await WebSearchTool(api_key="brave-key")._brave("PD-1 pembrolizumab", 3)

    assert hits == [("RCSB PDB - 5GGS", "https://www.rcsb.org/structure/5GGS", "PD-1 Fab")]
    assert seen["token"] == "brave-key" and "count=3" in seen["url"] and "q=PD-1" in seen["url"]


async def test_brave_over_quota_falls_back_to_duckduckgo(monkeypatch) -> None:
    request = httpx.Request("GET", "https://api.search.brave.com/res/v1/web/search")

    async def brave(self, query, n):
        raise httpx.HTTPStatusError("429", request=request, response=httpx.Response(429, request=request))

    monkeypatch.setattr(WebSearchTool, "_brave", brave)
    _serve(monkeypatch, {DDG_URL: DDG})
    out = await WebSearchTool(api_key="brave-key").execute("x")
    assert isinstance(out, str) and "1. PD-1 entry" in out


async def test_a_refused_key_and_a_bot_check_name_both_engines_and_cool_down(monkeypatch) -> None:
    """Neither failure is the query's; the answer says what each engine said
    (the status only: the HTTP error's text quotes the request URL)."""
    request = httpx.Request("GET", "https://api.search.brave.com/res/v1/web/search?q=x")

    async def brave(self, query, n):
        raise httpx.HTTPStatusError("401", request=request, response=httpx.Response(401, request=request))

    monkeypatch.setattr(WebSearchTool, "_brave", brave)
    _serve(monkeypatch, {DDG_URL: PUZZLE})
    out = await WebSearchTool(api_key="brave-key").execute("x")
    assert isinstance(out, ToolResult) and out.retryable is False
    assert "brave: HTTP 401; duckduckgo: DuckDuckGo asked for a bot check" in out.model_text
    assert "api.search.brave.com" not in out.model_text and "shorten" not in out.model_text


def test_a_result_about_the_challenge_form_is_not_a_bot_check() -> None:
    """The puzzle is its form, not the words: a page whose result mentions
    them is a page of results."""
    page = '<div class="result"><a class="result__a" href="https://x/forms">How to use challenge-form</a></div>'
    assert web._parse_ddg_html(page) == [("How to use challenge-form", "https://x/forms", "")]
    with pytest.raises(web.EngineRefusedError):
        web._parse_ddg_html(PUZZLE)


async def test_a_request_the_engine_rejects_is_not_a_cooldown(monkeypatch) -> None:
    """A 414 is this query's problem; the next call (another session on the
    shared tool) must still reach the engine."""
    request = httpx.Request("GET", DDG_URL)
    rejected = httpx.HTTPStatusError("414", request=request, response=httpx.Response(414, request=request))
    tool = WebSearchTool()
    _serve(monkeypatch, {DDG_URL: rejected})
    first = await tool.execute("x" * 5000)
    assert isinstance(first, str) and first.startswith("Error: the search request was rejected")

    asked = _serve(monkeypatch, {DDG_URL: DDG})
    assert "1. PD-1 entry" in await tool.execute("short")
    assert len(asked) == 1


async def test_a_slow_excerpt_page_is_dropped_at_the_deadline(monkeypatch) -> None:
    import asyncio

    async def slow(url: str, *, proxy=None, timeout=15.0) -> Page:
        if url.startswith(DDG_URL):
            return Page(url, "text/html", DDG, 200)
        await asyncio.sleep(1)
        return Page(url, "text/html", HTML, 200)

    monkeypatch.setattr(web, "fetch_page", slow)
    monkeypatch.setattr(WebSearchTool, "_EXCERPT_DEADLINE_S", 0.05)
    out = await WebSearchTool().execute("x", content=True, count=1)
    assert "1. PD-1 entry" in out and "---" not in out


# ---------------------------------------------------------------------------
# fetch_page: streamed, bounded
# ---------------------------------------------------------------------------


class _Stream(httpx.AsyncByteStream):
    def __init__(self, gen) -> None:
        self._gen = gen

    async def __aiter__(self):
        async for chunk in self._gen():
            yield chunk


async def test_a_declared_binary_type_is_refused_before_its_body_is_read(monkeypatch) -> None:
    read: list[int] = []

    async def body():
        read.append(1)
        yield b"%PDF-1.4"

    routes = {
        "https://public.example/f.pdf": httpx.Response(
            200, stream=_Stream(body), headers={"content-type": "application/pdf"}
        )
    }
    monkeypatch.setattr(web, "validate_url_target", lambda url: (True, ""))
    _mock_client(monkeypatch, lambda request: routes[str(request.url)])

    with pytest.raises(ValueError, match="application/pdf content is not readable"):
        await web.fetch_page("https://public.example/f.pdf")
    assert read == []


async def test_a_body_over_the_budget_is_refused_part_way(monkeypatch) -> None:
    served: list[int] = []

    async def body():
        for _ in range(100):
            served.append(1)
            yield b"x" * 65_536

    routes = {
        "https://public.example/big": httpx.Response(200, stream=_Stream(body), headers={"content-type": "text/plain"})
    }
    monkeypatch.setattr(web, "validate_url_target", lambda url: (True, ""))
    monkeypatch.setattr(web, "_MAX_BYTES", 200_000)
    _mock_client(monkeypatch, lambda request: routes[str(request.url)])

    with pytest.raises(ValueError, match="larger than 200,000 bytes"):
        await web.fetch_page("https://public.example/big")
    assert len(served) < 10, "refused part-way, not after the whole body"


def test_an_xml_answer_with_its_own_doctype_is_returned_as_it_is() -> None:
    xml = "<!DOCTYPE result><result><identifier>Q15116</identifier></result>"
    assert extract(Page("https://x/api", "application/xml", xml, 200)) == (xml, "raw")
    assert (
        extract(Page("https://x/page", "application/octet-stream", "<!DOCTYPE html><html><body>hi</body></html>", 200))[
            1
        ]
        != "raw"
    )
