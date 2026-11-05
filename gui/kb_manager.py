"""The Knowledge Base Manager engine: one character, four ways to fill its corpus.

Each character (Jarvis, Ultron, Vision, Banner, Natasha, Hakeem, Cloud) and the
project's own model may have its **own** knowledge corpus under
``data/characters/<owner>/knowledge_base`` (see
:mod:`core.character_knowledge`). Until now the only way to fill one was the
CLI's bulk build or a hand-written ``python -m cli knowledge --store-dir ...``
command. This module is the engine behind the **Knowledge Base Manager** window:
it lets a user pick a character and add to *that character's* corpus in the four
ways the task asks for, all through the **same** ingestion pipeline the Import
Center already uses.

1. **Add documents** -- choose files or a folder; each is read through
   :class:`gui.import_center.KnowledgeImporter` (files, digital PDF, Word,
   optional OCR) and written to the selected character's store.
2. **Search the web and fetch full pages** -- a governed search returns results
   (title, URL, snippet), and each result's **whole page** can be fetched and
   imported. A listing page can be walked to discover its own links (a forum, a
   blog index), so a *site* is imported rather than one URL at a time.
3. **Paste text** -- a window-specific chat box whose text is content-addressed
   and imported directly (see :class:`ingest.connectors.text.TextConnector`).
4. **Choose the character** -- every source flows into the selected character's
   own corpus, resolved from the owner key so isolation holds structurally
   (:func:`core.character_knowledge.character_knowledge_dir`).

Honest by construction: the web path runs only through the project's permission
**and** human-approval gates (both denied by default), a search with no approval
returns an honest refusal rather than reaching the network, an empty paste is a
reported failure rather than a silent no-op, and ``stats()`` reports the real
corpus or the real next step.

The class is Tk-free on purpose, so the whole workflow is exercised headless with
no display, no model, and no network (the search provider and the page fetch are
injectable seams).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.character_identity import CharacterIdentity, all_identities
from core.character_knowledge import CharacterKnowledge, character_knowledge_dir
from core.human_approval import HumanApprovalGate
from core.knowledge_index import build_index
from core.web_source import SearchOutcome, WebResult, WebSearchService
from ingest.discovery import crawl_forum, fetch_board, fetch_links

from gui.import_center import ImportOutcome, ImportReport, KnowledgeImporter

#: The external-tool name the web search runs under, so the permission policy
#: and the audit trail name it the same way the rest of the project does.
WEB_TOOL = "web_search"

#: How many search results a query asks for by default.
DEFAULT_SEARCH_LIMIT = 8

#: The most pages one "follow the links" run will import, so a listing with
#: thousands of links cannot start an unbounded crawl by accident.
DEFAULT_PAGE_LIMIT = 25

#: The refusal a caller sees when the web path has not been approved.
WEB_OFF_REASON = (
    "Web access is off. Turn on 'Search the web' to allow the governed search "
    "and page fetch, then try again."
)


@dataclass(frozen=True)
class KbCharacter:
    """One selectable corpus owner, as the window lists it."""

    owner: str
    mouth: str
    name: str
    private: bool

    @property
    def label(self) -> str:
        return f"{self.name} ({self.owner})" + ("  ·  private" if self.private else "")

    @classmethod
    def from_identity(cls, identity: CharacterIdentity) -> KbCharacter:
        return cls(
            owner=identity.owner,
            mouth=identity.mouth,
            name=identity.name,
            private=identity.private,
        )


@dataclass(frozen=True)
class WebPageOutcome:
    """What happened to one fetched page, for the window's per-page results."""

    url: str
    title: str
    ok: bool
    chunks: int = 0
    records: int = 0
    message: str = ""


@dataclass
class WebFetchReport:
    """The aggregate result of one web search / fetch run."""

    query: str = ""
    allowed: bool = True
    reason: str = ""
    results: tuple[WebResult, ...] = ()
    pages: list[WebPageOutcome] = field(default_factory=list)
    records: int = 0
    chunks: int = 0

    @property
    def fetched(self) -> int:
        return sum(1 for page in self.pages if page.ok)

    @property
    def failed(self) -> int:
        return sum(1 for page in self.pages if not page.ok)

    def summary(self) -> str:
        if not self.allowed:
            return self.reason or "The web path is not approved."
        if not self.pages:
            return self.reason or "Nothing was fetched."
        parts = [f"{self.fetched} page(s) fetched"]
        if self.failed:
            parts.append(f"{self.failed} failed")
        parts.append(f"{self.chunks} chunk(s)")
        return ", ".join(parts) + "."


class KnowledgeBaseManagerService:
    """Fill one character's knowledge corpus: files, web pages, and pasted text.

    The character is selected once and every subsequent operation targets that
    character's own store, so the window cannot accidentally write one
    character's sources into another's corpus. The web path is denied until
    :meth:`set_web_approval` records the user's explicit approval.
    """

    def __init__(
        self,
        *,
        data_root: str | Path | None = None,
        provider: Any | None = None,
        opener: Any | None = None,
        page_limit: int = DEFAULT_PAGE_LIMIT,
        search_limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> None:
        self._data_root = Path(data_root) if data_root is not None else None
        self._characters = tuple(
            KbCharacter.from_identity(identity) for identity in all_identities()
        )
        self._selected = self._default_owner()
        self.page_limit = page_limit
        self.search_limit = search_limit
        # Injectable seams: a test supplies its own search provider and page
        # opener, so no real network is touched.
        self._provider = provider
        self._opener = opener
        # The approval gate the web path runs through, denied until approved.
        self._approvals = HumanApprovalGate()
        self._approval_id: str | None = None
        self._web_approved = False
        self._importers: dict[str, KnowledgeImporter] = {}
        self._web_importer: KnowledgeImporter | None = None

    # ------------------------------------------------------------ characters

    def characters(self) -> tuple[KbCharacter, ...]:
        """Every selectable owner: the seven characters and the project model."""
        return self._characters

    def _default_owner(self) -> str:
        # Default to the first character (Jarvis), so the window always has a
        # valid target the moment it opens.
        return self._characters[0].owner if self._characters else "jarvis"

    @property
    def selected(self) -> str:
        return self._selected

    def selected_character(self) -> KbCharacter:
        for character in self._characters:
            if character.owner == self._selected:
                return character
        raise KeyError(self._selected)

    def set_character(self, key: str) -> KbCharacter:
        """Select the corpus owner ``key`` (an owner, a mouth, or a name)."""
        for character in self._characters:
            if key in (character.owner, character.mouth, character.name):
                if character.owner != self._selected:
                    self._selected = character.owner
                    # The web importer is bound to one store, so a switch to
                    # another character must rebuild it (else pages land in the
                    # character that was selected first).
                    self._web_importer = None
                return character
        raise KeyError(f"Unknown character: {key!r}")

    def character_dir(self, owner: str | None = None) -> Path:
        """The selected (or named) character's own knowledge-corpus folder."""
        return character_knowledge_dir(owner or self._selected, root=self._data_root)

    # ------------------------------------------------------------- importers

    def importer(self, owner: str | None = None) -> KnowledgeImporter:
        """The file importer for one character's store (cached per owner)."""
        target = owner or self._selected
        existing = self._importers.get(target)
        if existing is None:
            existing = KnowledgeImporter(store_dir=self.character_dir(target))
            self._importers[target] = existing
        return existing

    def _web_importer_for(self) -> KnowledgeImporter:
        """The importer used for web pages: it has the web connector registered."""
        if self._web_importer is None:
            self._web_importer = KnowledgeImporter(
                store_dir=self.character_dir(), allow_web=True, web_opener=self._opener
            )
        return self._web_importer

    # ------------------------------------------------------------ documents

    def import_files(
        self,
        paths: Iterable[str | Path],
        *,
        progress: Callable[[str], None] | None = None,
    ) -> ImportReport:
        """Add files to the selected character's corpus."""
        return self.importer().import_files(paths, progress=progress)

    def import_folder(
        self,
        folder: str | Path,
        *,
        recursive: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> ImportReport:
        """Add every supported file under ``folder`` to the selected corpus."""
        return self.importer().import_folder(folder, recursive=recursive, progress=progress)

    def import_text(
        self,
        text: str,
        *,
        title: str = "",
        progress: Callable[[str], None] | None = None,
    ) -> ImportReport:
        """Add pasted text to the selected character's corpus."""
        return self.importer().import_text(text, title=title, progress=progress)

    # ------------------------------------------------------------------ web

    def web_available(self) -> bool:
        """Whether the web path has the user's approval right now."""
        return self._web_approved

    def set_web_approval(self, approved: bool) -> None:
        """Record or withdraw the user's approval for the web path.

        Approving records a live approval for :data:`WEB_TOOL` on the service's
        own gate; withdrawing rejects the request it recorded. Nothing reaches the
        network until this is true, so the window's checkbox is the single switch.
        """
        self._web_approved = bool(approved)
        if approved:
            request = self._approvals.request(WEB_TOOL, requested_by="knowledge-base-manager")
            request.approve("local-user")
            self._approval_id = request.id
        elif self._approval_id is not None:
            existing = self._approvals.get(self._approval_id)
            if existing is not None:
                existing.reject("local-user")
            self._approval_id = None

    def _search_service(self) -> WebSearchService:
        from core.web_verifier import granted_permissions, internet_provider

        provider = self._provider if self._provider is not None else internet_provider()
        return WebSearchService(
            provider,
            permissions=granted_permissions(),
            approvals=self._approvals,
            actor="knowledge-base-manager",
        )

    def search_web(self, query: str, *, limit: int | None = None) -> WebFetchReport:
        """Search the web for ``query``; denied until the user approves.

        Returns the results (title, URL, snippet) without importing anything, so
        the user chooses which pages to fetch.
        """
        limit = limit or self.search_limit
        if not self._web_approved:
            return WebFetchReport(query=query, allowed=False, reason=WEB_OFF_REASON)
        try:
            outcome: SearchOutcome = self._search_service().search(query, limit=limit)
        except (ValueError, OSError) as error:
            return WebFetchReport(query=query, allowed=False, reason=str(error))
        return WebFetchReport(
            query=query,
            allowed=outcome.allowed,
            reason=outcome.reason,
            results=outcome.results,
        )

    def discover_site(
        self,
        url: str,
        *,
        same_host: bool = True,
        limit: int | None = None,
    ) -> WebFetchReport:
        """Read a listing page and return the pages it links to (not imported).

        Denied until the user approves. This is the "fetch a whole site" step:
        the caller inspects the discovered links, then imports the ones it wants
        with :meth:`import_urls`.
        """
        if not self._web_approved:
            return WebFetchReport(allowed=False, reason=WEB_OFF_REASON)
        cap = limit or self.page_limit
        try:
            links = fetch_links(url, opener=self._opener, same_host=same_host)
        except Exception as error:  # a bad page must not crash the window
            return WebFetchReport(allowed=False, reason=f"{type(error).__name__}: {error}")
        results = tuple(
            WebResult(url=link, title=link, snippet="", credibility=0.5, provider="discovery")
            for link in links[:cap]
        )
        return WebFetchReport(allowed=True, reason=f"Found {len(results)} link(s).", results=results)

    def walk_board(
        self,
        url: str,
        *,
        same_host: bool = True,
        max_pages: int = 10,
        limit: int | None = None,
        match: str | None = None,
        descend: bool = False,
        max_boards: int = 10,
    ) -> WebFetchReport:
        """Walk a board's own pages and return every story link found.

        The forum / index case the task asks for: a board spreads its threads over
        many pages, so this follows the listing's pagination links and collects the
        content links from each (bounded by ``max_pages`` and ``limit``). ``match``
        keeps only links whose URL contains that substring, so a board that also
        links navigation can be narrowed to its stories (``match="/threads/"``).

        ``descend`` is the *category* case: a category page holds a few threads and
        links its sub-forums, where most stories live, so with ``descend=True`` the
        walk also enters those child boards (bounded by ``max_boards``) -- "all the
        stories in this category" in one bounded action. Nothing is imported yet --
        the caller reviews the discovered stories, then imports them with
        :meth:`import_urls`.
        """
        if not self._web_approved:
            return WebFetchReport(allowed=False, reason=WEB_OFF_REASON)
        cap = limit or self.page_limit
        try:
            if descend:
                links = crawl_forum(
                    url,
                    opener=self._opener,
                    same_host=same_host,
                    max_pages=max_pages,
                    max_links=cap,
                    max_boards=max_boards,
                    match=match or None,
                )
                scope = f"{max_boards} board(s)"
            else:
                links = fetch_board(
                    url,
                    opener=self._opener,
                    same_host=same_host,
                    max_pages=max_pages,
                    max_links=cap,
                    match=match or None,
                )
                scope = f"up to {max_pages} page(s)"
        except Exception as error:  # a bad page must not crash the window
            return WebFetchReport(allowed=False, reason=f"{type(error).__name__}: {error}")
        results = tuple(
            WebResult(url=link, title=link, snippet="", credibility=0.5, provider="board")
            for link in links
        )
        return WebFetchReport(
            allowed=True,
            reason=f"Found {len(results)} story link(s) across {scope}.",
            results=results,
        )

    def import_urls(
        self,
        urls: Sequence[str],
        *,
        progress: Callable[[str], None] | None = None,
    ) -> WebFetchReport:
        """Fetch each URL's **whole page** and import it into the selected corpus.

        Runs only through the web connector's own authorization (the service sets
        ``authorized`` because the *user* approved the web path) so the same
        guards the CLI applies with ``--allow-web --approve`` hold here. Each
        page's outcome is reported per URL.
        """
        if not self._web_approved:
            return WebFetchReport(allowed=False, reason=WEB_OFF_REASON)
        report = WebFetchReport(allowed=True, reason="Fetched and imported.")
        importer = self._web_importer_for()
        for url in urls:
            outcome = importer.import_url(url, progress=progress)
            page = _page_outcome(url, outcome)
            report.pages.append(page)
            report.records += page.records
            report.chunks += page.chunks
        return report

    def fetch_and_import(
        self,
        url: str,
        *,
        follow_links: bool = False,
        same_host: bool = True,
        limit: int | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> WebFetchReport:
        """Fetch one page (and optionally the pages it links to) and import them.

        With ``follow_links`` the page is treated as a listing: its in-page links
        are discovered first (bounded by ``limit``), then each is fetched and
        imported, so a forum or an index becomes a corpus in one action.
        """
        if not self._web_approved:
            return WebFetchReport(allowed=False, reason=WEB_OFF_REASON)
        if not follow_links:
            return self.import_urls([url], progress=progress)
        discovered = self.discover_site(url, same_host=same_host, limit=limit)
        if not discovered.allowed:
            return discovered
        targets = [url, *(result.url for result in discovered.results)]
        report = self.import_urls(targets, progress=progress)
        report.reason = f"Fetched the page and {len(discovered.results)} linked page(s)."
        return report

    # --------------------------------------------------------------- reports

    def knowledge(self, owner: str | None = None) -> CharacterKnowledge:
        """A read-only view over one character's corpus."""
        return CharacterKnowledge(owner or self._selected, root=self._data_root)

    def stats(self, owner: str | None = None) -> dict[str, Any]:
        """The selected character's corpus statistics (honest, may be empty)."""
        return self.knowledge(owner).stats()

    def describe(self, owner: str | None = None) -> str:
        """One honest line about the selected corpus, or the real next step."""
        character = self.selected_character()
        stats = self.stats(owner)
        if not stats.get("present"):
            return (
                f"{character.name}: no knowledge corpus yet. Add documents, fetch "
                "web pages, or paste text to build one."
            )
        indexed = "indexed" if stats.get("indexed") else "not indexed"
        return (
            f"{character.name}: {stats['chunks']} chunk(s), {stats['size_mb']} MB "
            f"({indexed})."
        )

    def build_index(
        self, owner: str | None = None, *, progress: Any | None = None
    ) -> dict[str, Any]:
        """Build the full-text index for a character's corpus (fast retrieval)."""
        return build_index(self.character_dir(owner), progress=progress)


def _page_outcome(url: str, outcome: ImportOutcome) -> WebPageOutcome:
    return WebPageOutcome(
        url=url,
        title=outcome.filename or url,
        ok=outcome.ok,
        chunks=outcome.chunks,
        records=outcome.records,
        message=outcome.message,
    )


__all__ = [
    "DEFAULT_PAGE_LIMIT",
    "DEFAULT_SEARCH_LIMIT",
    "WEB_OFF_REASON",
    "WEB_TOOL",
    "KbCharacter",
    "KnowledgeBaseManagerService",
    "WebFetchReport",
    "WebPageOutcome",
]
