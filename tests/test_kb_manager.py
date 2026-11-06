"""The Knowledge Base Manager: engine, CLI, and the window.

The feature this pins: a **Knowledge Base Manager** entry in the main window's
sidebar opens a window that lets a user pick a character and fill *that
character's* own knowledge corpus four ways -- add documents, search the web and
fetch whole pages, paste text into a window-specific chat box, and switch the
target character -- all through the same ingestion pipeline the Import Center
uses.

The headless parts (the service, the text connector, the link discovery, the CLI)
run with no display and no network (the search provider and the page fetch are
injectable seams). The window parts are skipped unless a display (or Xvfb) is
available, exactly like ``tests/test_model_chat.py``.
"""

from __future__ import annotations

import json
from email.message import Message
from pathlib import Path
from typing import Any, Literal

import pytest
from core.web_source import StaticSearchProvider, WebResult
from gui.kb_manager import (
    KbCharacter,
    KnowledgeBaseManagerService,
    WebFetchReport,
)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Point the whole program's data folder at a fresh temp directory."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    return root


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


class _FakeResponse:
    """A minimal urlopen-style response for the injectable opener seam."""

    def __init__(self, html: str) -> None:
        self._body = html.encode("utf-8")

    def read(self, _n: int = -1) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_a: object) -> Literal[False]:
        return False

    @property
    def headers(self) -> Message:
        message = Message()
        message["Content-Type"] = "text/html; charset=utf-8"
        return message


def _opener(html: str):
    def opener(_request: Any, timeout: float = 15.0) -> _FakeResponse:
        return _FakeResponse(html)

    return opener


# --- the text connector ------------------------------------------------------


def test_text_connector_is_always_registered_and_local():
    from ingest.registry import build_default_registry

    registry = build_default_registry()
    assert "text" in registry.names()
    assert "web" not in registry.names()


def test_text_connector_ingests_supplied_text(tmp_path: Path) -> None:
    from core.knowledge_base import KnowledgeBaseStore
    from ingest.dedup import DedupLedger
    from ingest.pipeline import IngestionPipeline, ledger_path
    from ingest.records import SourceSpec
    from ingest.registry import build_default_registry

    store = KnowledgeBaseStore(tmp_path / "kb")
    pipeline = IngestionPipeline(store, DedupLedger(ledger_path(store.root)))
    registry = build_default_registry()
    spec = SourceSpec("text:abc", options={"text": "Permeability governs flow. " * 20, "title": "Note"})
    report = pipeline.ingest(spec, connector=registry.resolve(spec), incremental=True)
    assert report.records == 1
    assert report.chunks >= 1


def test_text_connector_refuses_empty_text():
    from ingest.errors import ConnectorError
    from ingest.records import SourceSpec
    from ingest.registry import build_default_registry

    connector = build_default_registry().get("text")
    assert connector is not None
    with pytest.raises(ConnectorError):
        list(connector.plan(SourceSpec("text:x", options={"text": "   "})))


# --- link discovery ----------------------------------------------------------


def test_extract_links_resolves_dedupes_and_confines_host():
    from ingest.discovery import extract_links

    html = (
        '<a href="/threads/one.1/">One</a>'
        '<a href="/threads/one.1/">One again</a>'
        '<a href="https://other.example/x">Ext</a>'
        '<a href="/img.png">Image</a>'
        '<a href="mailto:a@b.c">Mail</a>'
        '<a href="#frag">Frag</a>'
    )
    links = extract_links(html, "https://site.example/forums/board.5/")
    assert links == ("https://site.example/threads/one.1/",)
    # Off-host links are kept when the caller asks for them.
    all_links = extract_links(html, "https://site.example/forums/board.5/", same_host=False)
    assert "https://other.example/x" in all_links


def test_discover_links_orders_content_before_pagination():
    from ingest.discovery import discover_links

    html = (
        '<a href="/page-2">Next</a>'
        '<a href="/threads/a.1/">A</a>'
        '<a href="/threads/b.2/">B</a>'
    )
    links = discover_links(html, "https://site.example/board/")
    assert links == (
        "https://site.example/threads/a.1/",
        "https://site.example/threads/b.2/",
        "https://site.example/page-2",
    )


def test_discover_links_can_drop_pagination():
    from ingest.discovery import discover_links

    html = '<a href="/page-2">Next</a><a href="/threads/a.1/">A</a>'
    links = discover_links(html, "https://site.example/board/", include_pagination=False)
    assert links == ("https://site.example/threads/a.1/",)


def test_fetch_links_uses_the_injected_opener():
    from ingest.discovery import fetch_links

    html = '<a href="/threads/a.1/">A</a><a href="/threads/b.2/">B</a>'
    links = fetch_links("https://site.example/board/", opener=_opener(html))
    assert links == ("https://site.example/threads/a.1/", "https://site.example/threads/b.2/")


def test_fetch_links_reports_an_unreachable_page():
    from ingest.discovery import fetch_links
    from ingest.errors import ConnectorError

    def broken(_request: Any, timeout: float = 15.0) -> Any:
        from urllib.error import URLError

        raise URLError("no network")

    with pytest.raises(ConnectorError):
        fetch_links("https://site.example/board/", opener=broken)


def test_canonical_url_collapses_forum_views_of_one_thread():
    from ingest.discovery import canonical_url

    assert canonical_url("https://f.example/threads/x.7314/post-112804") == (
        "https://f.example/threads/x.7314/"
    )
    assert canonical_url("https://f.example/threads/x.81851/latest") == (
        "https://f.example/threads/x.81851/"
    )
    assert canonical_url("https://f.example/threads/x.81851/") == (
        "https://f.example/threads/x.81851/"
    )
    # A query string is a *different* page (a pagination page), so it is kept.
    assert canonical_url("https://f.example/board/?page=2") == "https://f.example/board/?page=2"


def test_extract_links_collapses_alternate_thread_views():
    from ingest.discovery import extract_links

    html = (
        '<a href="/threads/x.7314/post-1">A reply</a>'
        '<a href="/threads/x.7314/">The thread</a>'
        '<a href="/threads/x.7314/latest">Latest</a>'
    )
    assert extract_links(html, "https://f.example/board/") == ("https://f.example/threads/x.7314/",)


def test_split_links_separates_content_from_pagination():
    from ingest.discovery import split_links

    html = (
        '<a href="/page-2">Next</a>'
        '<a href="/threads/a.1/">A</a>'
        '<a href="/threads/b.2/">B</a>'
    )
    content, paging = split_links(html, "https://f.example/board/")
    assert content == ("https://f.example/threads/a.1/", "https://f.example/threads/b.2/")
    assert paging == ("https://f.example/page-2",)


def test_fetch_board_walks_every_page():
    from ingest.discovery import fetch_board

    pages = {
        "https://f.example/board/": (
            '<a href="/threads/a.1/">A</a><a href="/page-2">Next</a>'
        ),
        "https://f.example/page-2": (
            '<a href="/threads/b.2/">B</a><a href="/threads/c.3/">C</a>'
            '<a href="/page-3">Next</a>'
        ),
        "https://f.example/page-3": '<a href="/threads/d.4/">D</a>',
    }

    def opener(request: Any, timeout: float = 15.0) -> _FakeResponse:
        return _FakeResponse(pages[request.full_url])

    links = fetch_board("https://f.example/board/", opener=opener, max_pages=10)
    assert links == (
        "https://f.example/threads/a.1/",
        "https://f.example/threads/b.2/",
        "https://f.example/threads/c.3/",
        "https://f.example/threads/d.4/",
    )


def test_fetch_board_respects_the_page_bound():
    from ingest.discovery import fetch_board

    pages = {
        "https://f.example/board/": '<a href="/threads/a.1/">A</a><a href="/page-2">Next</a>',
        "https://f.example/page-2": '<a href="/threads/b.2/">B</a>',
    }

    def opener(request: Any, timeout: float = 15.0) -> _FakeResponse:
        return _FakeResponse(pages[request.full_url])

    links = fetch_board("https://f.example/board/", opener=opener, max_pages=1)
    assert links == ("https://f.example/threads/a.1/",)


def test_fetch_board_respects_the_link_bound():
    from ingest.discovery import fetch_board

    html = "".join(f'<a href="/threads/t{i}.{i}/">T</a>' for i in range(20))
    links = fetch_board("https://f.example/board/", opener=_opener(html), max_links=5)
    assert len(links) == 5


def test_fetch_board_skips_a_dead_pagination_page():
    from ingest.discovery import fetch_board
    from ingest.errors import ConnectorError

    def opener(request: Any, timeout: float = 15.0) -> _FakeResponse:
        if request.full_url.endswith("/page-2"):
            raise ConnectorError("dead link")
        return _FakeResponse('<a href="/threads/a.1/">A</a><a href="/page-2">Next</a>')

    links = fetch_board("https://f.example/board/", opener=opener, max_pages=5)
    assert links == ("https://f.example/threads/a.1/",)


def test_fetch_board_filters_links_by_match():
    from ingest.discovery import fetch_board

    html = (
        '<a href="/threads/a.1/">A</a>'
        '<a href="/categories/x.9/">A category</a>'
        '<a href="/members/bob.7/">Bob</a>'
    )
    links = fetch_board("https://f.example/board/", opener=_opener(html), match="/threads/")
    assert links == ("https://f.example/threads/a.1/",)


def test_is_board_recognises_a_board_and_rejects_a_thread():
    from ingest.discovery import is_board

    assert is_board("https://f.example/forums/stories.5/")
    assert is_board("https://f.example/forums/stories.5", base_url="https://f.example/")
    # A thread, a board action, a deeper path, and the board itself are not boards.
    assert not is_board("https://f.example/threads/a.1/")
    assert not is_board("https://f.example/forums/stories.5/post-thread")
    assert not is_board("https://f.example/forums/stories.5/page-2")
    assert not is_board("https://f.example/forums/stories.5/", base_url="https://f.example/forums/stories.5/")


def test_board_links_lists_child_boards_not_threads():
    from ingest.discovery import board_links

    html = (
        '<a href="/forums/child.6/">Child</a>'
        '<a href="/threads/a.1/">A thread</a>'
        '<a href="/forums/parent.5/">Parent</a>'
        '<a href="/forums/child.6/post-thread">Post</a>'
    )
    boards = board_links(html, "https://f.example/forums/child.6/")
    assert boards == ("https://f.example/forums/parent.5/",)


def test_crawl_forum_descends_into_child_boards():
    from ingest.discovery import crawl_forum

    pages = {
        "https://f.example/forums/cat.5/": (
            '<a href="/threads/top.1/">Top</a>'
            '<a href="/forums/child.6/">Child</a>'
        ),
        "https://f.example/forums/child.6/": '<a href="/threads/deep.2/">Deep</a>',
    }

    def opener(request: Any, timeout: float = 15.0) -> _FakeResponse:
        return _FakeResponse(pages[request.full_url])

    links = crawl_forum("https://f.example/forums/cat.5/", opener=opener, match="/threads/")
    assert links == (
        "https://f.example/threads/top.1/",
        "https://f.example/threads/deep.2/",
    )


def test_crawl_forum_does_not_re_read_a_board_that_links_back():
    from ingest.discovery import crawl_forum

    calls: list[str] = []

    def opener(request: Any, timeout: float = 15.0) -> _FakeResponse:
        calls.append(request.full_url)
        return _FakeResponse(
            '<a href="/forums/child.6/">Child</a><a href="/threads/a.1/">A</a>'
        )

    crawl_forum("https://f.example/forums/cat.5/", opener=opener, match="/threads/")
    assert calls.count("https://f.example/forums/cat.5/") == 1
    assert calls.count("https://f.example/forums/child.6/") == 1


def test_crawl_forum_respects_the_board_bound():
    from ingest.discovery import crawl_forum

    def opener(request: Any, timeout: float = 15.0) -> _FakeResponse:
        return _FakeResponse(
            '<a href="/forums/child.6/">Child</a><a href="/threads/a.1/">A</a>'
        )

    links = crawl_forum("https://f.example/forums/cat.5/", opener=opener, max_boards=1)
    assert links == ("https://f.example/threads/a.1/",)


def test_fetch_board_raises_when_the_first_page_is_unreachable():
    from ingest.discovery import fetch_board
    from ingest.errors import ConnectorError

    def broken(_request: Any, timeout: float = 15.0) -> Any:
        raise ConnectorError("no board")

    with pytest.raises(ConnectorError):
        fetch_board("https://f.example/board/", opener=broken)


# --- the service: characters -------------------------------------------------


def test_service_lists_the_seven_characters_and_the_model(data_root) -> None:
    service = KnowledgeBaseManagerService(data_root=data_root)
    owners = [c.owner for c in service.characters()]
    assert owners == ["jarvis", "ultron", "vision", "banner", "natasha", "hakeem", "cloud", "fayd_al_ruh"]
    assert service.selected == "jarvis"


def test_service_selects_a_character_by_owner_mouth_or_name(data_root) -> None:
    service = KnowledgeBaseManagerService(data_root=data_root)
    assert service.set_character("natasha").owner == "natasha"
    assert service.selected == "natasha"
    assert service.set_character("sources").owner == "jarvis"
    assert service.set_character("Ultron").owner == "ultron"
    with pytest.raises(KeyError):
        service.set_character("nobody")


def test_service_targets_each_characters_own_store(data_root) -> None:
    service = KnowledgeBaseManagerService(data_root=data_root)
    service.set_character("natasha")
    natasha_dir = service.character_dir()
    service.set_character("hakeem")
    hakeem_dir = service.character_dir()
    assert natasha_dir != hakeem_dir
    assert natasha_dir.parts[-3:] == ("characters", "natasha", "knowledge_base")
    assert hakeem_dir.parts[-3:] == ("characters", "hakeem", "knowledge_base")


def test_kb_character_label_marks_a_private_character() -> None:
    public = KbCharacter("jarvis", "sources", "Jarvis", False)
    private = KbCharacter("natasha", "natasha", "Natasha", True)
    assert "private" not in public.label
    assert "private" in private.label


# --- the service: documents and paste ----------------------------------------


def test_service_imports_pasted_text_into_the_selected_corpus(data_root) -> None:
    service = KnowledgeBaseManagerService(data_root=data_root)
    service.set_character("natasha")
    report = service.import_text("A story about the sea. " * 30, title="Sea story")
    assert report.ok
    assert report.chunks >= 1
    assert service.stats()["present"] is True


def test_service_imports_files_into_the_selected_corpus(data_root, tmp_path) -> None:
    service = KnowledgeBaseManagerService(data_root=data_root)
    service.set_character("hakeem")
    doc = tmp_path / "note.txt"
    doc.write_text("History of the region. " * 30, encoding="utf-8")
    report = service.import_files([doc])
    assert report.ok
    assert service.stats()["chunks"] >= 1


def test_service_imports_a_folder(data_root, tmp_path) -> None:
    service = KnowledgeBaseManagerService(data_root=data_root)
    service.set_character("jarvis")
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("First. " * 40, encoding="utf-8")
    (folder / "b.txt").write_text("Second. " * 40, encoding="utf-8")
    report = service.import_folder(folder)
    assert report.succeeded == 2


def test_service_paste_refuses_empty_text(data_root) -> None:
    service = KnowledgeBaseManagerService(data_root=data_root)
    report = service.import_text("   ")
    assert not report.ok
    assert "empty" in report.failures[0].message.lower()


def test_service_describe_names_the_next_step_when_empty(data_root) -> None:
    service = KnowledgeBaseManagerService(data_root=data_root)
    service.set_character("cloud")
    line = service.describe()
    assert "Cloud" in line
    assert "no knowledge corpus yet" in line


def test_service_builds_the_index(data_root) -> None:
    service = KnowledgeBaseManagerService(data_root=data_root)
    service.set_character("banner")
    service.import_text("Slide content about drilling. " * 30, title="Deck")
    result = service.build_index()
    assert result["indexed"] >= 1
    assert service.stats()["indexed"] is True


# --- the service: the web path -----------------------------------------------


def test_service_web_is_off_by_default(data_root) -> None:
    service = KnowledgeBaseManagerService(data_root=data_root)
    assert service.web_available() is False
    report = service.search_web("porosity")
    assert report.allowed is False
    assert "off" in report.reason.lower()


def test_service_refuses_a_fetch_before_approval(data_root) -> None:
    service = KnowledgeBaseManagerService(data_root=data_root)
    report = service.import_urls(["https://site.example/a"])
    assert report.allowed is False


def test_service_approval_toggles_the_web_path(data_root) -> None:
    service = KnowledgeBaseManagerService(data_root=data_root)
    service.set_web_approval(True)
    assert service.web_available() is True
    service.set_web_approval(False)
    assert service.web_available() is False


def test_service_search_returns_results_with_the_provider(data_root) -> None:
    provider = StaticSearchProvider(
        (WebResult(url="https://site.example/a", title="Porosity", snippet="pore space"),)
    )
    service = KnowledgeBaseManagerService(data_root=data_root, provider=provider)
    service.set_web_approval(True)
    report = service.search_web("porosity")
    assert report.allowed is True
    assert [r.url for r in report.results] == ["https://site.example/a"]


def test_service_fetches_a_whole_page_into_the_corpus(data_root) -> None:
    html = "<html><body><h1>Story</h1><p>" + "A long tale. " * 40 + "</p></body></html>"
    service = KnowledgeBaseManagerService(data_root=data_root, opener=_opener(html))
    service.set_character("natasha")
    service.set_web_approval(True)
    report = service.import_urls(["https://site.example/a"])
    assert report.allowed is True
    assert report.fetched == 1
    assert report.chunks >= 1
    assert service.stats()["present"] is True


def test_service_discovers_the_links_of_a_listing_page(data_root) -> None:
    listing = (
        '<a href="/threads/one.1/">One</a>'
        '<a href="/threads/two.2/">Two</a>'
        '<a href="https://other.example/x">Ext</a>'
    )
    service = KnowledgeBaseManagerService(data_root=data_root, opener=_opener(listing))
    service.set_web_approval(True)
    report = service.discover_site("https://site.example/board/")
    assert report.allowed is True
    urls = [r.url for r in report.results]
    assert urls == ["https://site.example/threads/one.1/", "https://site.example/threads/two.2/"]


def test_service_fetch_and_import_follows_links(data_root) -> None:
    html = (
        "<html><body><p>" + "Story text. " * 30 + "</p>"
        '<a href="/threads/one.1/">One</a><a href="/threads/two.2/">Two</a>'
        "</body></html>"
    )
    service = KnowledgeBaseManagerService(data_root=data_root, opener=_opener(html))
    service.set_character("natasha")
    service.set_web_approval(True)
    report = service.fetch_and_import("https://site.example/board/", follow_links=True, limit=5)
    assert report.allowed is True
    # The listing page itself plus its two linked pages.
    assert len(report.pages) == 3


def test_service_walk_board_collects_story_links(data_root) -> None:
    pages = {
        "https://f.example/board/": '<a href="/threads/a.1/">A</a><a href="/page-2">Next</a>',
        "https://f.example/page-2": '<a href="/threads/b.2/">B</a>',
    }

    def opener(request: Any, timeout: float = 15.0) -> _FakeResponse:
        return _FakeResponse(pages[request.full_url])

    service = KnowledgeBaseManagerService(data_root=data_root, opener=opener)
    service.set_web_approval(True)
    report = service.walk_board("https://f.example/board/", max_pages=5)
    assert report.allowed is True
    assert [r.url for r in report.results] == [
        "https://f.example/threads/a.1/",
        "https://f.example/threads/b.2/",
    ]


def test_service_walk_board_refuses_before_approval(data_root) -> None:
    service = KnowledgeBaseManagerService(data_root=data_root)
    report = service.walk_board("https://f.example/board/")
    assert report.allowed is False


def test_service_walk_board_honours_match(data_root) -> None:
    html = (
        '<a href="/threads/a.1/">A</a><a href="/categories/x.9/">A category</a>'
    )
    service = KnowledgeBaseManagerService(data_root=data_root, opener=_opener(html))
    service.set_web_approval(True)
    report = service.walk_board("https://f.example/board/", match="/threads/")
    assert [r.url for r in report.results] == ["https://f.example/threads/a.1/"]


def test_service_web_importer_rebuilds_when_the_character_changes(data_root) -> None:
    """A fetch after switching characters must land in the *new* character's store."""
    html = "<html><body><p>" + "Story. " * 60 + "</p></body></html>"
    service = KnowledgeBaseManagerService(data_root=data_root, opener=_opener(html))
    service.set_character("natasha")
    service.set_web_approval(True)
    service.import_urls(["https://site.example/a"])
    service.set_character("hakeem")
    service.import_urls(["https://site.example/b"])
    assert service.knowledge("natasha").stats()["present"] is True
    assert service.knowledge("hakeem").stats()["present"] is True


def test_web_fetch_report_summary_is_honest_when_refused() -> None:
    report = WebFetchReport(allowed=False, reason="Web access is off.")
    assert report.summary() == "Web access is off."


# --- the CLI -----------------------------------------------------------------


def _cli(argv: list[str], capsys, monkeypatch=None) -> tuple[int, str]:
    from cli import kb_manager

    code = kb_manager.main(argv)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_cli_lists_characters(data_root, capsys) -> None:
    code, out = _cli(["characters"], capsys)
    assert code == 0
    assert "natasha" in out
    assert "private" in out


def test_cli_status_reports_the_next_step_when_empty(data_root, capsys) -> None:
    code, out = _cli(["status", "natasha"], capsys)
    assert code == 3
    assert "no knowledge corpus yet" in out


def test_cli_paste_then_status(data_root, capsys) -> None:
    code, out = _cli(["paste", "natasha", "--text", "A story. " * 30, "--title", "Story"], capsys)
    assert code == 0
    code, out = _cli(["status", "natasha"], capsys)
    assert code == 0
    assert "chunk" in out


def test_cli_paste_requires_text(data_root, capsys) -> None:
    code, out = _cli(["paste", "natasha"], capsys)
    assert code == 2
    assert "nothing to add" in out.lower()


def test_cli_add_files(data_root, tmp_path, capsys) -> None:
    doc = tmp_path / "note.txt"
    doc.write_text("Content. " * 40, encoding="utf-8")
    code, out = _cli(["add-files", "hakeem", str(doc)], capsys)
    assert code == 0
    assert "Added" in out


def test_cli_search_requires_approval(data_root, capsys) -> None:
    code, out = _cli(["search", "natasha", "porosity"], capsys)
    assert code == 4
    assert "off" in out.lower()


def test_cli_fetch_requires_approval(data_root, capsys) -> None:
    code, _out = _cli(["fetch", "natasha", "https://site.example/a"], capsys)
    assert code == 4


def test_cli_board_requires_approval(data_root, capsys) -> None:
    code, out = _cli(["board", "natasha", "https://f.example/board/"], capsys)
    assert code == 4
    assert "off" in out.lower()


def test_cli_board_lists_then_imports(data_root, capsys, monkeypatch) -> None:
    from gui import kb_manager as engine

    monkeypatch.setattr(
        engine,
        "fetch_board",
        lambda *a, **k: (
            "https://f.example/threads/a.1/",
            "https://f.example/threads/b.2/",
        ),
    )
    code, out = _cli(["board", "natasha", "https://f.example/board/", "--list", "--approve"], capsys)
    assert code == 0
    assert "2 story link(s)" in out
    assert "threads/a.1" in out


def test_cli_unknown_character(data_root, capsys) -> None:
    code, out = _cli(["status", "nobody"], capsys)
    assert code == 3
    assert "unknown character" in out.lower()


def test_cli_json_after_subcommand(data_root, capsys) -> None:
    _cli(["paste", "natasha", "--text", "A story. " * 30], capsys)
    code, out = _cli(["status", "natasha", "--json"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["present"] is True


def test_cli_index(data_root, capsys) -> None:
    _cli(["paste", "banner", "--text", "Slide. " * 30], capsys)
    code, out = _cli(["index", "banner"], capsys)
    assert code == 0
    assert "Index built" in out


def test_cli_is_registered_in_the_menu() -> None:
    from cli import menu

    assert "run_kb_manager" in menu.__all__
    labels = [label for label, _fn, _argv in menu.MENU]
    assert any("Knowledge Base Manager" in label for label in labels)


# --- the sidebar action and window wiring ------------------------------------


def test_sidebar_action_is_visible_and_grouped() -> None:
    from gui import actions

    visible = {action.key for action in actions.visible_actions()}
    assert "kb-manager" in visible
    grouped = {group.key: {a.key for a in members} for group, members in actions.grouped_visible_actions()}
    assert "kb-manager" in grouped["knowledge"]
    assert actions.SIDEBAR_LABELS["kb-manager"] == "Knowledge Base Manager"


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_main_window_routes_the_action_to_its_window() -> None:
    import inspect

    from gui import app as gui_app

    source = inspect.getsource(gui_app.FaydGUI._on_action)
    assert 'action.key == "kb-manager"' in source
    assert "_open_kb_manager" in source
    assert "_kb_manager_dialog" in inspect.getsource(gui_app.FaydGUI._open_dialogs)


# --- the window --------------------------------------------------------------


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_shows_the_character_the_sources_and_the_composer(data_root) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.kb_manager_dialog import KnowledgeBaseManagerDialog

    root = tk.Tk()
    try:
        service = KnowledgeBaseManagerService(data_root=data_root)
        dialog = KnowledgeBaseManagerDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        # The character selector, the documents controls, the web controls, and a
        # window-specific composer are all present.
        assert dialog._character_var.get() == "jarvis"
        assert dialog._web_var.get() is False
        assert dialog._composer is not None
        assert dialog._results_list is not None
        assert dialog._url_entry is not None
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_paste_adds_to_the_selected_character(data_root) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.kb_manager_dialog import KnowledgeBaseManagerDialog

    root = tk.Tk()
    try:
        service = KnowledgeBaseManagerService(data_root=data_root)
        dialog = KnowledgeBaseManagerDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        dialog._on_character("natasha")
        dialog._composer.set_text("A story about the sea. " * 30)
        dialog._on_paste()
        # Drain the worker queue deterministically. The import runs on a worker
        # thread, so the loop must yield to it (`time.sleep`) as well as pump the
        # UI queue; a bare `root.update()` loop spins without ever letting the
        # thread run and the test then reads a corpus that was never written.
        import time

        for _ in range(400):
            root.update()
            if not dialog._busy:
                break
            time.sleep(0.01)
        assert service.knowledge("natasha").stats()["present"] is True
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_web_toggle_sets_the_service_approval(data_root) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.kb_manager_dialog import KnowledgeBaseManagerDialog

    root = tk.Tk()
    try:
        service = KnowledgeBaseManagerService(data_root=data_root)
        dialog = KnowledgeBaseManagerDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        dialog._web_var.set(True)
        dialog._on_web_toggle()
        assert service.web_available() is True
        dialog._web_var.set(False)
        dialog._on_web_toggle()
        assert service.web_available() is False
    finally:
        root.destroy()
