"""Governed web search: find sources on the internet, safely.

The cognitive core reasons over what it already knows. This module lets it
reach *outside* that knowledge, but only through the project's existing
governance: an internet search is an **external tool**, so it is denied by
default and runs only when a :class:`core.permissions.PermissionPolicy` grants
the capability, the tool name is on the allow-list, and a live
:class:`core.human_approval.HumanApprovalGate` approval exists. Every decision
is written to the append-only :class:`core.audit_log.AuditLog`.

Two providers are provided:

* :class:`StaticSearchProvider` -- deterministic, offline, used by tests and the
  demo so the whole path works on a fresh clone with no network and no key;
* :class:`HttpSearchProvider` -- a thin JSON client for a caller-supplied
  endpoint (the application owns the URL and the API key; none is stored here).

A provider returns :class:`WebResult` items, each carrying a URL, a title, a
snippet, and a credibility estimate. The credibility is a transparent heuristic
over the domain's top-level suffix (government, academic, and standards bodies
rank above generic blogs), never a claim of truth.
"""

from __future__ import annotations


from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

from core.audit_log import AuditLog, AuditOutcome, AuditRecord
from core.human_approval import HumanApprovalGate
from core.permissions import Capability, PermissionDecision, PermissionPolicy

_SEARCH_TOOL = "web_search"

# Top-level suffixes (and a few well-known hosts) ranked by how much a claim
# from them is worth. The scale is deliberately coarse and inspectable.
_HIGH_CREDIBILITY_SUFFIXES = (".gov", ".edu", ".mil", ".ac.uk", ".gov.uk")
_STANDARDS_HOSTS = ("spe.org", "onepetro.org", "api.org", "iso.org", "astm.org")
_REFERENCE_HOSTS = ("wikipedia.org", "britannica.com")
_MEDIUM_CREDIBILITY_SUFFIXES = (".org",)


@dataclass(frozen=True)
class WebResult:
    """One search result: a source the answer can be checked against."""

    url: str
    title: str
    snippet: str
    credibility: float = 0.5
    provider: str = "static"

    def __post_init__(self) -> None:
        if not self.url.strip():
            raise ValueError("Web result URL cannot be empty.")
        if not self.title.strip():
            raise ValueError("Web result title cannot be empty.")
        if not 0.0 <= self.credibility <= 1.0:
            raise ValueError("Web result credibility must be between 0 and 1.")

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc.casefold()


class WebSearchProvider(Protocol):
    """The contract every search backend satisfies."""

    name: str

    def search(self, query: str, *, limit: int = 5) -> tuple[WebResult, ...]: ...


class StaticSearchProvider:
    """An offline provider over a fixed corpus of results (tests and demo)."""

    name = "static"

    def __init__(self, results: tuple[WebResult, ...] = ()) -> None:
        self._results = tuple(results)

    def search(self, query: str, *, limit: int = 5) -> tuple[WebResult, ...]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        from core.rag import _query_terms

        terms = _query_terms(query)
        scored: list[tuple[float, WebResult]] = []
        for result in self._results:
            overlap = len(terms & _query_terms(f"{result.title} {result.snippet}"))
            if overlap:
                scored.append((overlap, result))
        scored.sort(key=lambda item: (-item[0], item[1].url))
        return tuple(result for _score, result in scored[:limit])


class HttpSearchProvider:
    """A JSON search client for a caller-supplied endpoint.

    The application owns the endpoint URL and any API key; nothing is stored in
    the project. The endpoint must answer with a JSON object holding a
    ``results`` list of ``{url, title, snippet}`` objects.
    """

    name = "http"

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("Search endpoint cannot be empty.")
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, *, limit: int = 5) -> tuple[WebResult, ...]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        import requests

        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.get(
            self.endpoint,
            params={"q": query, "limit": str(limit)},
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        raw_results = payload.get("results", []) if isinstance(payload, dict) else []
        results: list[WebResult] = []
        for item in raw_results[:limit]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            title = str(item.get("title", "")).strip()
            if not url or not title:
                continue
            results.append(
                WebResult(
                    url=url,
                    title=title,
                    snippet=str(item.get("snippet", "")).strip(),
                    credibility=estimate_credibility(url),
                    provider=self.name,
                )
            )
        return tuple(results)


class WikipediaSearchProvider:
    """A keyless provider over the Wikipedia search API.

    It needs no API key, so the project can answer from the real internet out of
    the box. Any network failure raises and is contained by
    :class:`WebSearchService`, which lets the caller fall back to an offline
    provider.
    """

    name = "wikipedia"
    _API = "https://en.wikipedia.org/w/api.php"
    # Wikipedia blocks requests without a descriptive User-Agent.
    _USER_AGENT = "FaydAlRuh/2.2 (educational; +https://github.com/pt5ri-debug/fayd-al-ruh)"

    def __init__(self, *, timeout: float = 15.0) -> None:
        self.timeout = timeout

    def search(self, query: str, *, limit: int = 5) -> tuple[WebResult, ...]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        import requests

        response = requests.get(
            self._API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": str(limit),
            },
            headers={"Accept": "application/json", "User-Agent": self._USER_AGENT},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        raw_results = payload.get("query", {}).get("search", []) if isinstance(payload, dict) else []
        results: list[WebResult] = []
        for item in raw_results[:limit]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            page_id = item.get("pageid")
            url = (
                f"https://en.wikipedia.org/?curid={page_id}"
                if page_id is not None
                else f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            )
            results.append(
                WebResult(
                    url=url,
                    title=title,
                    snippet=_strip_html(str(item.get("snippet", ""))),
                    credibility=estimate_credibility(url),
                    provider=self.name,
                )
            )
        return tuple(results)


def _strip_html(text: str) -> str:
    """Remove the HTML tags Wikipedia wraps around highlighted search terms."""
    out: list[str] = []
    in_tag = False
    for char in text:
        if char == "<":
            in_tag = True
        elif char == ">":
            in_tag = False
        elif not in_tag:
            out.append(char)
    return "".join(out).replace("&quot;", '"').replace("&amp;", "&").strip()


class FallbackSearchProvider:
    """Try a primary provider, then a backup if it fails or finds nothing.

    Used to reach the real internet first and drop back to the offline corpus
    when there is no network, so the governed path always has something to work
    with.
    """

    def __init__(self, primary: WebSearchProvider, backup: WebSearchProvider) -> None:
        self.primary = primary
        self.backup = backup
        self.name = f"{primary.name}+{backup.name}"

    def search(self, query: str, *, limit: int = 5) -> tuple[WebResult, ...]:
        try:
            results = self.primary.search(query, limit=limit)
        except Exception:
            results = ()
        if results:
            return results
        return self.backup.search(query, limit=limit)


@dataclass(frozen=True)
class SearchOutcome:
    """What one governed search did."""

    query: str
    allowed: bool
    reason: str
    results: tuple[WebResult, ...] = ()
    approval_id: str | None = None

    @property
    def result_count(self) -> int:
        return len(self.results)


def estimate_credibility(url: str) -> float:
    """A transparent credibility heuristic over a source's host and suffix."""
    host = urlparse(url).netloc.casefold()
    if not host:
        return 0.2
    if host.endswith(_HIGH_CREDIBILITY_SUFFIXES):
        return 0.9
    if any(host == standard or host.endswith(f".{standard}") for standard in _STANDARDS_HOSTS):
        return 0.9
    if any(host == reference or host.endswith(f".{reference}") for reference in _REFERENCE_HOSTS):
        return 0.7
    if host.endswith(_MEDIUM_CREDIBILITY_SUFFIXES):
        return 0.6
    if host.endswith((".com", ".net", ".io", ".co")):
        return 0.4
    return 0.3


class WebSearchService:
    """Runs searches through the project's permission, approval, and audit gates."""

    def __init__(
        self,
        provider: WebSearchProvider,
        *,
        permissions: PermissionPolicy | None = None,
        approvals: HumanApprovalGate | None = None,
        audit: AuditLog | None = None,
        actor: str = "assistant",
    ) -> None:
        self.provider = provider
        self.permissions = permissions or PermissionPolicy()
        self.approvals = approvals or HumanApprovalGate()
        self.audit = audit or AuditLog()
        self.actor = actor

    def _authorise(self) -> tuple[bool, str, str | None]:
        decision = self.permissions.decide(Capability.EXTERNAL_TOOL, _SEARCH_TOOL)
        if not decision.allowed:
            return False, decision.reason, None
        approval = self.approvals.decide(_SEARCH_TOOL)
        if not approval.approved:
            return False, approval.reason, approval.approval_id
        return True, "Search allowed.", approval.approval_id

    def search(self, query: str, *, limit: int = 5) -> SearchOutcome:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if limit < 1:
            raise ValueError("Search limit must be at least 1.")

        decision = self.permissions.decide(Capability.EXTERNAL_TOOL, _SEARCH_TOOL)
        allowed, reason, approval_id = self._authorise()
        self.audit.record_decision(
            PermissionDecision(decision.capability, decision.target, allowed, reason),
            actor=self.actor,
            metadata={"query": query, "approval_id": approval_id or ""},
        )
        if not allowed:
            return SearchOutcome(query, False, reason, approval_id=approval_id)

        try:
            results = self.provider.search(query, limit=limit)
        except Exception as error:  # a provider must never crash the caller
            message = f"Search provider failed: {error}"
            self.audit.append(
                AuditRecord(
                    Capability.EXTERNAL_TOOL,
                    _SEARCH_TOOL,
                    AuditOutcome.DENIED,
                    message,
                    actor=self.actor,
                    metadata={"query": query, "phase": "provider"},
                )
            )
            return SearchOutcome(query, False, message, approval_id=approval_id)

        return SearchOutcome(query, True, "Search allowed.", results, approval_id)


__all__ = [
    "FallbackSearchProvider",
    "HttpSearchProvider",
    "SearchOutcome",
    "StaticSearchProvider",
    "WebResult",
    "WebSearchProvider",
    "WebSearchService",
    "WikipediaSearchProvider",
    "estimate_credibility",
]
