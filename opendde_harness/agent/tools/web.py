"""Web tools: ``web_search`` and ``web_fetch``, neither needing a key.

Both fetch pages the same way (``fetch_page``) and read them the same way
(``extract``): the page's own text for JSON, plain text and XML -- what a
data source's API answers -- and the article for HTML, extracted here with
trafilatura rather than by a reader service, which is what kept failing.
"""

import asyncio
import os
import re
import time
from dataclasses import dataclass
from html import unescape
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs, quote, urljoin, urlparse

import httpx
import trafilatura
from loguru import logger

from opendde_harness.agent.tools.base import Tool, ToolResult
from opendde_harness.security.network import validate_url_target

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
_TAGS = re.compile(r"<[^>]+>")
#: Content types answered as they are: a data source's own format, which the
#: model reads better than any extraction of it.
_TEXT_TYPES = ("json", "text/plain", "xml", "csv", "yaml", "fasta", "tsv", "javascript", "x-www-form")


@dataclass(frozen=True)
class Page:
    """One fetched page: where it ended up, what it was, and its text."""

    url: str
    content_type: str
    text: str
    status: int


_MAX_REDIRECTS = 5
#: The most of a page that is read. A data source's biggest answers (a UniProt
#: entry, 70KB) and article pages (Wikipedia, 450KB) are well under it; a
#: body still arriving past it is not a page anyone reads whole.
_MAX_BYTES = 2_000_000


async def fetch_page(url: str, *, proxy: str | None = None, timeout: float = 15.0) -> Page:
    """GET ``url`` as a browser would, following redirects.

    Every hop is checked against ``validate_url_target`` before it is
    requested, the search engines' own endpoints and a search result's page
    included: a result or a redirect pointing at a private address is the
    SSRF route the check exists to close, and checking the first URL alone
    would leave the redirect open.

    The body is streamed: a declared binary type is refused from its headers
    before a byte of it is read, and anything over ``_MAX_BYTES`` is refused
    part-way rather than held in memory whole.
    """
    headers = {"User-Agent": _UA, "Accept": "text/html,application/json,text/plain,*/*"}
    async with httpx.AsyncClient(proxy=proxy, follow_redirects=False, timeout=timeout) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            ok, why = validate_url_target(url)
            if not ok:
                raise ValueError(f"refusing to fetch {url}: {why}")
            async with client.stream("GET", url, headers=headers) as r:
                if r.is_redirect:
                    url = (
                        str(r.next_request.url)
                        if r.next_request is not None
                        else urljoin(url, r.headers.get("location", ""))
                    )
                    continue
                r.raise_for_status()
                content_type = r.headers.get("content-type", "").lower()
                if content_type and not _is_textual(content_type, b""):
                    raise ValueError(f"{content_type} content is not readable as text")
                body = bytearray()
                async for chunk in r.aiter_bytes():
                    body += chunk
                    if len(body) > _MAX_BYTES:
                        raise ValueError(f"{url} is larger than {_MAX_BYTES:,} bytes")
                if not content_type and not _is_textual("", bytes(body[:512])):
                    raise ValueError("binary content is not readable as text")
                return Page(
                    str(r.url), content_type, bytes(body).decode(r.encoding or "utf-8", errors="replace"), r.status_code
                )
        raise ValueError(f"too many redirects from {url}")


def _is_textual(content_type: str, head: bytes) -> bool:
    if any(marker in content_type for marker in ("text/", "html", *_TEXT_TYPES)):
        return True
    return not content_type and b"\x00" not in head


def extract(page: Page, mode: str = "markdown") -> tuple[str, str]:
    """The page's readable text and how it was obtained.

    HTML is reduced to its article (``mode`` picks markdown or plain text);
    anything else is answered verbatim. An HTML page trafilatura finds no
    article in -- a listing, a login wall -- falls back to its text with the
    tags stripped, which is still what the page says.
    """
    head = page.text.lstrip()[:15].lower()
    if "html" not in page.content_type and not head.startswith(("<!doctype html", "<html")):
        # An XML answer may open with a doctype of its own; only an HTML one
        # says the label was wrong.
        return page.text, "raw"
    article = trafilatura.extract(
        page.text,
        url=page.url,
        output_format="markdown" if mode == "markdown" else "txt",
        include_links=False,
        include_tables=True,
        favor_recall=True,
    )
    if article:
        return article, "trafilatura"
    return _clean(page.text), "text"


def _clean(fragment: str) -> str:
    """Tags out, entities decoded, whitespace runs collapsed."""
    text = unescape(_TAGS.sub(" ", fragment or ""))
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r" *\n *", "\n", text)).strip()


def header_safe(value: str) -> bool:
    """Can this credential travel in an HTTP header? A key pasted with a
    newline in it is rejected by the HTTP stack with an error that quotes
    the whole header value, so it is refused here, before any header exists.

    Surrounding whitespace is that same refusal, but it is not the user's
    mistake to correct: every acquisition point strips it instead. What is
    left for this to reject is control and non-ASCII characters.
    """
    return value.isascii() and value.isprintable()


def brave_key(configured: str | None = None) -> str:
    """The Brave key in effect: the configured one, else the environment's.

    Stripped here, the one place both config spellings and the environment
    come through: a key copied with a leading space or a trailing newline is
    a header h11 refuses, and its refusal quotes the key.
    """
    return (configured or os.environ.get("BRAVE_API_KEY", "")).strip()


class WebSearchTool(Tool):
    """Search the web.

    Two ways this is served, decided per request. A provider's own hosted
    search is not one of them any more: the model layer is pi-ai, which passes
    no vendor search tool through, so every provider runs the local search
    below and results carry this tool's citations rather than the vendor's.

    - With a Brave Search API key (``tools.web.braveApiKey``,
      free tier 2,000 queries a month; the engine pi's search skill uses):
      Brave's Web Search API, a documented JSON contract.
    - Otherwise DuckDuckGo's HTML endpoint, which needs no key. It is the one
      keyless engine that answered a multi-word query correctly from a
      datacenter address: Bing (RSS and HTML alike) answered the first word
      only, silently, and Mojeek and Brave's web page refused outright.
      DuckDuckGo refuses too after a burst, with a puzzle page; that is a
      refusal, never an empty result. Brave failing (quota, outage) falls
      back to it.

    ``content=true`` also reads each result's page and returns an excerpt,
    the way pi's search skill does with ``--content``: one call answers a
    factual question instead of a search followed by several fetches.

    Every engine failing is reported once as final for a while, so a model
    does not spend its iterations retrying a search that cannot work.
    """

    name = "web_search"
    description = (
        "Search the web. Returns titles, URLs and snippets; content=true adds an excerpt of each page. "
        "For a fixed data source (PDB, UniProt, PubMed) use its API through web_fetch instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10},
            "content": {"type": "boolean", "description": "Also fetch each result and include an excerpt"},
        },
        "required": ["query"],
    }

    #: How long every engine failing is treated as final.
    _UNAVAILABLE_FOR_S = 120.0
    _EXCERPT_CHARS = 3_000
    #: The whole of one excerpt -- fetch and extraction -- not a per-chunk
    #: wait, which is what an HTTP timeout is and what a trickling page defeats.
    _EXCERPT_DEADLINE_S = 12.0

    def __init__(self, api_key: str | None = None, max_results: int = 5, proxy: str | None = None):
        self._init_api_key = api_key
        self.max_results = max_results
        self.proxy = proxy
        self._unavailable_until = 0.0
        self._unavailable_why = ""

    @property
    def api_key(self) -> str:
        """The Brave key, resolved at call time so env/config changes are picked up."""
        return brave_key(self._init_api_key)

    def engines(self) -> list[tuple[str, "Engine"]]:
        """The engines to try, in order: Brave when there is a key, then DuckDuckGo."""
        keyed: list[tuple[str, Engine]] = [("brave", self._brave)] if self.api_key else []
        return [*keyed, ("duckduckgo", self._duckduckgo)]

    async def execute(
        self, query: str, count: int | None = None, content: bool = False, **kwargs: Any
    ) -> str | ToolResult:
        n = min(max(count or self.max_results, 1), 10)
        if time.monotonic() < self._unavailable_until:
            return self._unavailable()
        logger.debug("WebSearch: {}", "proxy enabled" if self.proxy else "direct connection")
        hits: list[tuple[str, str, str]] = []
        failures: list[tuple[str, Exception]] = []
        engines = self.engines()
        for engine, search in engines:
            try:
                hits = await search(query, n)
            except httpx.ProxyError as e:
                logger.error("WebSearch proxy error: {}", e)
                return f"Proxy error: {e}"
            except Exception as e:
                logger.warning("WebSearch: {} failed: {}", engine, e)
                failures.append((engine, e))
                continue
            if hits:
                break
        if not hits and len(failures) == len(engines):
            detail = "; ".join(f"{engine}: {_brief(e)}" for engine, e in failures)
            if all(_engine_wide(e) for _, e in failures):
                # The engines, not this query: hold the answer so the next
                # call (any session, this tool is shared) does not pay for it.
                self._unavailable_until = time.monotonic() + self._UNAVAILABLE_FOR_S
                self._unavailable_why = detail
                return self._unavailable()
            return f"Error: the search request was rejected ({detail}); shorten or simplify the query"
        hits = hits[:n]
        if not hits:
            return f"No results for: {query}"
        excerpts = await self._excerpts([url for _, url, _ in hits]) if content else {}
        lines = [f"Results for: {query}\n"]
        for i, (title, url, snippet) in enumerate(hits, 1):
            lines.append(f"{i}. {title}\n   {url}")
            if snippet:
                lines.append(f"   {snippet}")
            if url in excerpts:
                lines.append("   ---\n   " + excerpts[url].replace("\n", "\n   ") + "\n   ---")
        return "\n".join(lines)

    async def _brave(self, query: str, n: int) -> list[tuple[str, str, str]]:
        """Brave's Web Search API: ``web.results[]`` with title, url, description."""
        if not header_safe(self.api_key):
            raise ValueError("the Brave key holds characters a header cannot carry")
        async with httpx.AsyncClient(proxy=self.proxy, timeout=15.0) as client:
            r = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": n},
                headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
            )
            r.raise_for_status()
            results = ((r.json().get("web") or {}).get("results")) or []
        return [
            (
                _clean(str(item.get("title") or "")),
                str(item.get("url") or ""),
                _clean(str(item.get("description") or "")),
            )
            for item in results
            if item.get("url")
        ]

    async def _duckduckgo(self, query: str, n: int) -> list[tuple[str, str, str]]:
        return _parse_ddg_html((await fetch_page(_DDG_ENDPOINT + quote(query), proxy=self.proxy)).text)

    def _unavailable(self) -> ToolResult:
        # Not retryable: the same call answers the same way until the engines
        # come back, and the registry's change-approach hint would invite a
        # retry. The sentence names the way that does work.
        return ToolResult(
            f"Error: web search is unavailable right now ({self._unavailable_why}). Do not retry it this turn; "
            "read a known URL with web_fetch, or query the source's own API (PDB, UniProt, PubMed).",
            retryable=False,
        )

    async def _excerpts(self, urls: list[str]) -> dict[str, str]:
        """The first few thousand characters of each page, fetched a few at a time.

        A page that cannot be read is simply absent: the search result stands
        on its own, and the model can fetch the page itself if it matters.
        """
        gate = asyncio.Semaphore(3)

        async def read(url: str) -> str:
            page = await fetch_page(url, proxy=self.proxy, timeout=10.0)
            return await asyncio.to_thread(extract, page, "txt")

        async def one(url: str) -> tuple[str, str | None]:
            async with gate:
                try:
                    text, _ = await asyncio.wait_for(read(url), self._EXCERPT_DEADLINE_S)
                except Exception as e:
                    logger.debug("WebSearch: no excerpt for {}: {}", url, e)
                    return url, None
            return url, text[: self._EXCERPT_CHARS] + ("…" if len(text) > self._EXCERPT_CHARS else "")

        return {url: text for url, text in await asyncio.gather(*(one(u) for u in urls)) if text}


#: One engine: a query and a result count in, (title, url, snippet) hits out.
Engine = Callable[[str, int], Awaitable[list[tuple[str, str, str]]]]

_DDG_ENDPOINT = "https://html.duckduckgo.com/html/?q="
_DDG_ANCHOR = re.compile(r'<a\b(?=[^>]*\bclass="result__a")[^>]*\bhref="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.S)
_DDG_SNIPPET = re.compile(r'class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.S)


class EngineRefusedError(RuntimeError):
    """The engine answered with a bot check instead of results."""


def _engine_wide(exc: Exception) -> bool:
    """Does this failure say the engine is unavailable, rather than that it
    disliked this one query? A bot check, a refused credential, a rate limit,
    a server error or a transport failure is the engine's; any other 4xx
    (414, a query too long) is the request's."""
    if isinstance(exc, EngineRefusedError | httpx.TransportError | asyncio.TimeoutError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (401, 403, 429) or exc.response.status_code >= 500
    return not isinstance(exc, ValueError)


def _brief(exc: Exception) -> str:
    """One failure in a few words: the status for an HTTP error (its full text
    quotes the request URL), the type alone for a header the HTTP stack refused
    locally, the message otherwise.

    h11 names the offending header value in full when it refuses one, and every
    request on this path carries a credential in a header; that message must
    reach neither a log, nor a tool result, nor the wizard's warning.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, httpx.LocalProtocolError) or type(exc).__module__.split(".")[0] == "h11":
        return type(exc).__name__
    return str(exc) or type(exc).__name__


def _parse_ddg_html(text: str) -> list[tuple[str, str, str]]:
    """DuckDuckGo's HTML endpoint, one result container at a time, so a
    result without a snippet cannot borrow the next one's. Links are
    redirects carrying the target in ``uddg``; anything else is resolved
    against the endpoint. A puzzle page is a refusal, not an empty answer."""
    if 'id="challenge-form"' in text or 'class="anomaly-modal' in text:
        raise EngineRefusedError("DuckDuckGo asked for a bot check")
    hits = []
    for container in re.split(r'<div[^>]+class="result\b', text)[1:]:
        anchor = _DDG_ANCHOR.search(container)
        if anchor is None:
            continue
        href = unescape(anchor.group("href"))
        target = parse_qs(urlparse(href).query).get("uddg", [""])[0] or urljoin(_DDG_ENDPOINT, href)
        snippet = _DDG_SNIPPET.search(container)
        hits.append((_clean(anchor.group("title")), target, _clean(snippet.group("snippet")) if snippet else ""))
    return hits


class WebFetchTool(Tool):
    """Fetch a URL and return what it says.

    Two readers, in a fixed order:

    1. Locally: the page is fetched here and an API's JSON, TSV, FASTA or XML
       is returned verbatim; an HTML page is reduced to its article with
       trafilatura. Fast, no third party, no rate limit, and what a data
       source's API needs.
    2. Jina Reader (``r.jina.ai``), only when the local read cannot say what
       the page says: the page is HTML and its article is thin -- a
       JavaScript app rendering client-side, a consent or login wall -- or the
       site refused the local fetch (403, 429, a bot check). Jina renders the
       page in a browser and is free without a key; ``tools.web.jinaApiKey``
       only raises its rate limit. If Jina fails too, the local result is
       what the model gets.
    """

    name = "web_fetch"
    description = (
        "Fetch a URL. Returns an API's JSON/text as is, or an HTML page's readable article. "
        "Use it for a known page or a data source's API (RCSB PDB, UniProt, NCBI E-utilities need no key)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100},
        },
        "required": ["url"],
    }

    #: An HTML article shorter than this is not the page: a client-side app's
    #: shell, or a wall. Measured on uniprot.org, whose shell extracts to a
    #: ~400-character "enable JavaScript" notice.
    _THIN_ARTICLE_CHARS = 500

    def __init__(self, api_key: str | None = None, max_chars: int = 12_000, proxy: str | None = None):
        self._init_api_key = api_key
        self.max_chars = max_chars
        self.proxy = proxy

    @property
    def api_key(self) -> str:
        """Resolve API key at call time so env/config changes are picked up;
        stripped for the reason ``brave_key`` strips."""
        return (self._init_api_key or os.environ.get("JINA_API_KEY", "")).strip()

    async def execute(self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any) -> str:  # noqa: N803  (LLM tool schema uses camelCase)
        max_chars = maxChars or self.max_chars
        is_valid, error_msg = validate_url_target(url)
        if not is_valid:
            return f"Error: URL validation failed: {error_msg}"
        logger.debug("WebFetch: {}", "proxy enabled" if self.proxy else "direct connection")

        local: tuple[str, str, str] | None = None  # text, extractor, final url
        refused: Exception | None = None
        try:
            page = await fetch_page(url, proxy=self.proxy)
        except httpx.ProxyError as e:
            logger.error("WebFetch proxy error for {}: {}", url, e)
            return f"Error: proxy error: {e}"
        except httpx.HTTPStatusError as e:
            refused = e
        except Exception as e:
            logger.error("WebFetch error for {}: {}", url, e)
            return f"Error: {e}"
        else:
            text, extractor = extract(page, extractMode)
            local = (text, extractor, page.url)
            if extractor == "raw" or len(text) >= self._THIN_ARTICLE_CHARS:
                return self._render(url, *local, max_chars)

        # Thin article or refused: a reader with a browser of its own.
        try:
            rendered = await self._via_jina(url, extractMode)
        except Exception as e:
            logger.warning(
                "WebFetch: Jina could not read {} ({}); {}",
                url,
                _brief(e),
                "local result kept" if local else "no result",
            )
            if local is None:
                return f"Error: {refused}"
            return self._render(url, *local, max_chars)
        return self._render(url, rendered, "jina-reader", url, max_chars)

    @staticmethod
    def _render(url: str, text: str, extractor: str, final_url: str, max_chars: int) -> str:
        header = (
            f"URL: {final_url}"
            + (f" (redirected from {url})" if final_url != url else "")
            + f"\nextractor: {extractor}"
        )
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[truncated: the first {max_chars:,} of {len(text):,} characters are shown]"
        return f"{header}\n\n{text}"

    async def _via_jina(self, url: str, mode: str) -> str:
        headers = {"Accept": "text/plain", "X-Return-Format": "markdown" if mode == "markdown" else "text"}
        if self.api_key:
            if not header_safe(self.api_key):
                raise ValueError("the Jina key holds characters a header cannot carry")
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(timeout=45.0, proxy=self.proxy) as client:
            r = await client.get(f"https://r.jina.ai/{url}", headers=headers)
            r.raise_for_status()
            return r.text
