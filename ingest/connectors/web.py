"""Import a single authorized web page as a source, with the network denied by default.

Internet import is the one connector that leaves the machine, so it carries more
guards than the others and is **off by default** in three independent places:

* :func:`ingest.registry.build_default_registry` does not register it unless the
  caller passes ``allow_web=True``;
* resolving a URL here still requires the host to be on an explicit
  ``allowed_hosts`` allow-list, or an ``authorized=True`` opt-in, otherwise the
  connector refuses with :class:`ingest.errors.ConnectorError`;
* the CLI additionally runs the fetch through the project's
  :class:`core.permissions.PermissionPolicy` and
  :class:`core.human_approval.HumanApprovalGate`, the same gate
  :mod:`core.web_source` uses.

*Local operation never depends on this*: nothing here is imported on a normal
run, the fetch uses only the standard library (``urllib``), and a failure to
reach the network is reported as a source error rather than raised.

Only the page text is kept. Credentials (an authorization header, a token query
parameter) are read from the source options for the request and are never
written to a record, a log, or an error message.
"""

from __future__ import annotations

from collections.abc import Iterator
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from core.acceptance_policy import is_trusted
from core.web_source import estimate_credibility

from ingest.errors import ConnectorError
from ingest.hashing import source_key
from ingest.records import (
    ConnectorInfo,
    IngestItem,
    SourceRecord,
    SourceSpec,
)

#: Hard cap on bytes read from one page, so a huge response cannot exhaust memory.
DEFAULT_MAX_BYTES = 2_000_000
DEFAULT_TIMEOUT = 15.0

_BLOCK_TAGS = frozenset(
    {"p", "div", "section", "article", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "pre"}
)
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})


class _TextExtractor(HTMLParser):
    """Turn HTML into readable text: keep visible words, drop tags and scripts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS and self._parts:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._parts.append(text + " ")

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


class WebConnector:
    """Fetches an allow-listed web page, once, as a governed source."""

    info = ConnectorInfo(
        name="web",
        description="Authorized import of an allow-listed web page (denied by default).",
        schemes=frozenset({"http", "https"}),
        remote=True,
    )

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_bytes: int = DEFAULT_MAX_BYTES,
        opener=None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive.")
        if max_bytes < 1:
            raise ValueError("max_bytes must be at least 1.")
        self.timeout = timeout
        self.max_bytes = max_bytes
        # Injectable for tests; defaults to the stdlib opener.
        self._opener = opener or urlopen

    # -- planning ---------------------------------------------------------

    def plan(self, spec: SourceSpec) -> Iterator[IngestItem]:
        host = urlparse(spec.uri).netloc.casefold()
        if not host:
            raise ConnectorError(f"Not a valid web URL: {spec.uri!r}")
        self._authorise(spec, host)
        yield IngestItem(
            source_key=source_key(spec.uri),
            uri=spec.uri,
            title=spec.option("title", host),
            fingerprint="",  # a remote page has no cheap local fingerprint
            metadata={"host": host, "url": spec.uri},
        )

    def _authorise(self, spec: SourceSpec, host: str) -> None:
        if bool(spec.option("authorized", False)):
            return
        allowed = {str(entry).casefold() for entry in (spec.option("allowed_hosts") or ())}
        if host in allowed:
            return
        raise ConnectorError(
            f"Refusing to fetch {host!r}: internet import is denied unless the host "
            "is on allowed_hosts or the source sets authorized=True."
        )

    # -- reading ----------------------------------------------------------

    def stream_item(self, item: IngestItem) -> Iterator[SourceRecord]:
        yield from self._read_item(item, token="")

    def stream(self, spec: SourceSpec) -> Iterator[SourceRecord]:
        token = str(spec.option("token", "") or "")
        for item in self.plan(spec):
            yield from self._read_item(item, token=token)

    def _read_item(self, item: IngestItem, *, token: str) -> Iterator[SourceRecord]:
        url = str(item.metadata.get("url", item.uri))
        text = self._fetch_text(url, token)
        if not text.strip():
            raise ConnectorError(f"{url!r} yielded no readable text.")
        yield SourceRecord(
            title=item.title,
            text=text,
            source="web",
            url=url,
            license="web",
            source_key=item.source_key,
            metadata={
                "kind": "web",
                "host": item.metadata.get("host", ""),
                "credibility": estimate_credibility(url),
                "trusted": is_trusted(url),
            },
        )

    def _fetch_text(self, url: str, token: str) -> str:
        # The token is read from options only; it is placed in the request header
        # and never returned, logged, or stored.
        headers = {"User-Agent": "fayd-al-ruh-ingest/1.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(url, headers=headers, method="GET")
        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read(self.max_bytes)
                charset = response.headers.get_content_charset() or "utf-8"
        except HTTPError as error:
            raise ConnectorError(f"HTTP {error.code} while fetching the source.") from error
        except URLError as error:
            raise ConnectorError(f"Could not reach the source: {error.reason}") from error
        except (OSError, ValueError) as error:
            raise ConnectorError(f"Could not fetch the source: {error}") from error
        return _html_to_text(raw.decode(charset, errors="replace"))


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup: fall back to a tag strip
        return " ".join(html.split())
    return parser.text()


__all__ = ["DEFAULT_MAX_BYTES", "DEFAULT_TIMEOUT", "WebConnector"]
