"""Discover the pages a listing points at, so a whole site can be imported.

The web connector reads *one* page. That is the right default for a single
article, but a forum, a blog index, or a documentation table of contents holds
many pages behind links, and fetching them one URL at a time by hand is what
makes a site-wide import impractical. This module is the small, dependency-free
reader that closes that gap: given a listing page it returns the *distinct*
in-page links, resolved to absolute URLs and optionally confined to the same
host, so the caller can feed each one back through the existing web connector.

It parses HTML with the standard library only (``html.parser``), never executes
anything it reads, and is deliberately conservative:

* a link is returned once, in the order it first appears, so a page's repeated
  navigation is not fetched a hundred times;
* ``same_host`` (the default) keeps the walk on the site that was named, so a
  listing full of advertisements and social links does not silently expand the
  import to the whole internet;
* ``fragment``-only links, ``mailto:``/``javascript:``/``tel:`` targets, and
  binary assets (images, stylesheets, scripts) are dropped;
* pagination links (``page-2``, ``?page=2``, ``/page/2``) are recognised so a
  multi-page listing is followed in order.

Nothing here reaches the network by itself: :func:`fetch_links` takes an
injectable ``opener``, and the governed callers pass the same permission- and
approval-gated fetch the web connector uses.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

#: Schemes a discovered link may use; everything else (``mailto:``,
#: ``javascript:``, ``tel:``, ``data:``) is not a page to fetch.
FETCHABLE_SCHEMES = frozenset({"http", "https"})

#: Suffixes that are assets rather than pages, so a listing's images and scripts
#: are not offered as pages to import.
ASSET_SUFFIXES = frozenset(
    {
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp",
        ".css", ".js", ".mjs", ".json", ".xml", ".rss", ".pdf", ".zip",
        ".gz", ".tar", ".mp3", ".mp4", ".avi", ".mov", ".woff", ".woff2",
        ".ttf", ".eot",
    }
)


class _LinkExtractor(HTMLParser):
    """Collect ``href`` targets from ``<a>`` tags, in document order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.links.append(value)


def extract_links(html: str, base_url: str, *, same_host: bool = True) -> tuple[str, ...]:
    """The distinct, absolute, fetchable links a page points at, in order.

    ``base_url`` resolves a relative ``href`` and (with ``same_host``) decides
    which host a link must share to be kept. A malformed document is tolerated:
    whatever links were read before the error are returned rather than raising.
    """
    parser = _LinkExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup: use what was read before the error
        pass
    host = urlparse(base_url).netloc.casefold()
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in parser.links:
        absolute = _resolve(raw, base_url)
        if absolute is None:
            continue
        if same_host and urlparse(absolute).netloc.casefold() != host:
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        ordered.append(absolute)
    return tuple(ordered)


def _resolve(raw: str, base_url: str) -> str | None:
    """An absolute fetchable URL for ``raw``, or ``None`` when it is not one."""
    target = raw.strip()
    if not target or target.startswith(("#", "mailto:", "javascript:", "tel:", "data:")):
        return None
    absolute = urljoin(base_url, target)
    parsed = urlparse(absolute)
    if parsed.scheme.casefold() not in FETCHABLE_SCHEMES or not parsed.netloc:
        return None
    if parsed.path.casefold().endswith(tuple(ASSET_SUFFIXES)):
        return None
    # Drop a trailing fragment so the same page linked twice is one page.
    return canonical_url(parsed._replace(fragment="").geturl())


#: A XenForo (and similar) per-post permalink segment, e.g. ``/post-112804``.
_PER_POST_RE = re.compile(r"/post-\d+/?$", re.IGNORECASE)

#: Alternate views of the same page: ``/latest`` and ``/unread`` land on the
#: thread's own page, so they are the same document as the canonical thread.
_VIEW_SUFFIXES = ("/latest", "/unread", "/new")


def canonical_url(url: str) -> str:
    """The one URL a page is fetched by, so its alternate views collapse to it.

    A forum links the same thread three ways -- the thread's own URL, a
    ``/post-NNNN`` permalink to a particular reply, and a ``/latest`` view -- and
    fetching all three would import one page three times. Only the shape is
    normalised (the query is kept, because ``?page=2`` is a *different* page);
    nothing is invented.
    """
    parsed = urlparse(url)
    path = _PER_POST_RE.sub("/", parsed.path)
    lowered = path.casefold()
    for suffix in _VIEW_SUFFIXES:
        if lowered.endswith(suffix):
            path = path[: -len(suffix)] + "/"
            lowered = path.casefold()
            break
    if not path:
        path = "/"
    return parsed._replace(path=path).geturl()


@dataclass(frozen=True)
class PageLink:
    """One discovered page: its URL and the anchor text that named it."""

    url: str
    title: str = ""

    @property
    def host(self) -> str:
        return urlparse(self.url).netloc.casefold()


def is_pagination(url: str) -> bool:
    """Whether a link looks like a "next page" of a listing."""
    lowered = url.casefold()
    markers = ("page-", "/page/", "page=", "&page=", "?page=", "start=")
    return any(marker in lowered for marker in markers)


def discover_links(
    html: str,
    base_url: str,
    *,
    same_host: bool = True,
    include_pagination: bool = True,
) -> tuple[str, ...]:
    """The pages to follow from a listing page, pagination last.

    Content links come first (in document order) and pagination links after, so a
    caller that caps the number of pages still fetches real content before it
    spends its budget walking the listing's own pages.
    """
    content, paging = split_links(html, base_url, same_host=same_host)
    if include_pagination:
        return (*content, *paging)
    return content


def split_links(
    html: str,
    base_url: str,
    *,
    same_host: bool = True,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The page's own links as ``(content, pagination)``.

    A board lists its threads (content) *and* its own further pages (pagination);
    keeping the two apart is what lets :func:`fetch_board` walk every page of a
    listing without mistaking a "next page" link for a thread.
    """
    links = extract_links(html, base_url, same_host=same_host)
    content: list[str] = []
    paging: list[str] = []
    for link in links:
        if link.rstrip("/") == base_url.rstrip("/"):
            continue
        (paging if is_pagination(link) else content).append(link)
    return tuple(content), tuple(paging)


def _read_text(opener: Any, url: str, timeout: float, max_bytes: int) -> str:
    from urllib.request import Request

    request = Request(url, headers={"User-Agent": "fayd-al-ruh-ingest/1.0"}, method="GET")
    with opener(request, timeout=timeout) as response:
        raw = response.read(max_bytes)
        charset = response.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def fetch_links(
    url: str,
    *,
    opener: Any = None,
    timeout: float = 15.0,
    max_bytes: int = 2_000_000,
    same_host: bool = True,
    include_pagination: bool = True,
) -> tuple[str, ...]:
    """Fetch a listing page and return the pages it links to.

    The fetch is the caller's responsibility (``opener`` is injectable and
    defaults to the stdlib opener), so a governed caller can pass the same
    permission- and approval-gated fetch the web connector uses. A fetch failure
    is reported as a :class:`~ingest.errors.ConnectorError`, never raised raw.
    """
    return _fetch_page_links(
        url,
        opener=opener,
        timeout=timeout,
        max_bytes=max_bytes,
        same_host=same_host,
        include_pagination=include_pagination,
    )


def _fetch_page_links(
    url: str,
    *,
    opener: Any,
    timeout: float,
    max_bytes: int,
    same_host: bool,
    include_pagination: bool,
) -> tuple[str, ...]:
    html = _fetch_html(url, opener=opener, timeout=timeout, max_bytes=max_bytes)
    return discover_links(
        html, url, same_host=same_host, include_pagination=include_pagination
    )


def _fetch_html(url: str, *, opener: Any, timeout: float, max_bytes: int) -> str:
    from urllib.error import HTTPError, URLError
    from urllib.request import urlopen

    from ingest.errors import ConnectorError

    opener = opener or urlopen
    try:
        return _read_text(opener, url, timeout, max_bytes)
    except HTTPError as error:
        raise ConnectorError(f"HTTP {error.code} while discovering {url!r}.") from error
    except URLError as error:
        raise ConnectorError(f"Could not reach {url!r}: {error.reason}") from error
    except (OSError, ValueError) as error:
        raise ConnectorError(f"Could not fetch {url!r}: {error}") from error


def fetch_board(
    url: str,
    *,
    opener: Any = None,
    timeout: float = 15.0,
    max_bytes: int = 2_000_000,
    same_host: bool = True,
    max_pages: int = 10,
    max_links: int = 1000,
    match: str | None = None,
    page_selector: Callable[[str], bool] | None = None,
    first_html: str | None = None,
) -> tuple[str, ...]:
    """Walk a listing's own pages and return every content link found.

    A board -- a forum, a blog index, a documentation table of contents -- spreads
    its entries over many pages (``page-2``, ``?page=2``, ``.../page/2``). This
    walks those pages breadth-first, following the pagination links a page names,
    and collects the **content** links from each (a link that is itself pagination
    is followed, never returned as content). The walk is bounded by ``max_pages``
    (how many listing pages to read) and ``max_links`` (how many content links to
    return), so a large board is read in a bounded time and never crawls the whole
    internet by accident.

    ``match``, when given, keeps only content links whose URL contains that
    substring -- the honest way to ask for just the *stories* on a board that also
    links navigation and category pages (``match="/threads/"``). It is a plain
    substring test, never a pattern that could be read as a command.

    ``page_selector``, when given, decides which discovered links are the listing's
    own further pages; the default follows the built-in :func:`is_pagination`
    heuristic. ``first_html``, when given, is the already-read text of ``url`` --
    a caller that fetched it to discover the board's child boards passes it here
    so the same page is not read twice. Every URL is de-duplicated and returned in
    first-seen order, so the same story linked twice is one entry. A page that
    cannot be fetched is skipped (a dead pagination link must not abort the walk);
    a failure on the *first* page is raised, because then there is nothing to walk.
    """
    from ingest.errors import ConnectorError

    is_page = page_selector or is_pagination
    seen_pages: set[str] = {url}
    queue: list[str] = [url]
    content: list[str] = []
    seen_content: set[str] = set()
    read = 0
    first = True
    while queue and read < max_pages and len(content) < max_links:
        current = queue.pop(0)
        read += 1
        if first_html is not None and current == url:
            html = first_html
        else:
            try:
                html = _fetch_html(current, opener=opener, timeout=timeout, max_bytes=max_bytes)
            except ConnectorError:
                if first:
                    raise
                continue
        first = False
        for link in extract_links(html, current, same_host=same_host):
            if link.rstrip("/") == current.rstrip("/"):
                continue
            if is_page(link):
                if link not in seen_pages:
                    seen_pages.add(link)
                    queue.append(link)
                continue
            if match is not None and match not in link:
                continue
            if link not in seen_content:
                seen_content.add(link)
                content.append(link)
                if len(content) >= max_links:
                    break
    return tuple(content)


def is_board(url: str, *, base_url: str | None = None) -> bool:
    """Whether a link names a forum *board* (a listing) rather than a thread.

    A board is exactly ``/forums/<slug>.<id>/`` -- one path segment after
    ``/forums/`` carrying the board's numeric id -- so a thread
    (``/threads/...``), a board's own ``post-thread`` action, and a pagination
    view are all excluded. ``base_url``, when given, drops the board the link was
    found on (a page always links itself).
    """
    parsed = urlparse(url)
    if parsed.netloc and base_url:
        if parsed.netloc.casefold() != urlparse(base_url).netloc.casefold():
            return False
    path = parsed.path.rstrip("/")
    prefix = "/forums/"
    if not path.startswith(prefix):
        return False
    tail = path[len(prefix):]
    if "/" in tail or "." not in tail:
        return False
    if base_url is not None and canonical_url(url) == canonical_url(base_url):
        return False
    return True


def board_links(
    html: str, base_url: str, *, same_host: bool = True
) -> tuple[str, ...]:
    """The child boards a listing page links to, in document order (distinct).

    A category page lists its sub-forums beside its own threads; keeping them
    apart is what lets :func:`crawl_forum` descend into the category without
    mistaking a thread for a board or the board for its own child.
    """
    seen: set[str] = set()
    boards: list[str] = []
    for link in extract_links(html, base_url, same_host=same_host):
        if is_pagination(link) or not is_board(link, base_url=base_url):
            continue
        key = canonical_url(link)
        if key not in seen:
            seen.add(key)
            boards.append(link)
    return tuple(boards)


def crawl_forum(
    url: str,
    *,
    opener: Any = None,
    timeout: float = 15.0,
    max_bytes: int = 2_000_000,
    same_host: bool = True,
    max_pages: int = 10,
    max_links: int = 1000,
    max_boards: int = 10,
    match: str | None = None,
    page_selector: Callable[[str], bool] | None = None,
) -> tuple[str, ...]:
    """Walk a forum board **and its child boards**, returning every story link.

    A category page (``/forums/<slug>.<id>/``) holds a few threads of its own and
    links its sub-forums, where the bulk of the stories live. This breadth-first
    walk reads each board's own pages (through :func:`fetch_board`) and descends
    into the child boards it links, bounded by ``max_boards`` (how many boards to
    read) and ``max_pages``/``max_links`` per board, so "all the stories in this
    category" is one bounded action rather than a hand-assembled list of boards.

    ``match`` is passed through to every board, so a walk can still be narrowed to
    real stories (``match="/threads/"``). Boards are de-duplicated by
    :func:`canonical_url`, so a child linking its parent back never re-reads it.
    The links are returned in first-seen board order, each story once.
    """
    from ingest.errors import ConnectorError

    roots: list[str] = []
    seen_boards: set[str] = {canonical_url(url)}
    queue: list[str] = [url]
    content: list[str] = []
    seen_content: set[str] = set()
    first = True
    while queue and len(roots) < max_boards and len(content) < max_links:
        board = queue.pop(0)
        roots.append(board)
        try:
            html = _fetch_html(board, opener=opener, timeout=timeout, max_bytes=max_bytes)
        except ConnectorError:
            if first:
                raise
            continue
        first = False
        for child in board_links(html, board, same_host=same_host):
            key = canonical_url(child)
            if key not in seen_boards:
                seen_boards.add(key)
                queue.append(child)
        for link in fetch_board(
            board,
            opener=opener,
            timeout=timeout,
            max_bytes=max_bytes,
            same_host=same_host,
            max_pages=max_pages,
            max_links=max_links - len(content),
            match=match,
            page_selector=page_selector,
            first_html=html,
        ):
            if is_board(link):
                # A child board is walked, never returned as a story.
                continue
            key = canonical_url(link)
            if key not in seen_content:
                seen_content.add(key)
                content.append(link)
                if len(content) >= max_links:
                    break
    return tuple(content)


def iter_links(links: tuple[str, ...], *, limit: int | None = None) -> Iterator[str]:
    """Yield ``links``, capped at ``limit`` when one is given."""
    for index, link in enumerate(links):
        if limit is not None and index >= limit:
            return
        yield link


__all__ = [
    "ASSET_SUFFIXES",
    "FETCHABLE_SCHEMES",
    "PageLink",
    "board_links",
    "canonical_url",
    "crawl_forum",
    "discover_links",
    "extract_links",
    "fetch_board",
    "fetch_links",
    "is_board",
    "is_pagination",
    "iter_links",
    "split_links",
]
