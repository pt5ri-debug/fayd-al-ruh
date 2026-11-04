"""Tests for the character-corpus connectors (Wikipedia ar/en, Gutenberg, Wikisource).

Everything here is offline: each connector is handed a fake session, so the
parsing, the language switch, the Gutenberg boilerplate stripping, the citation
labels, and the multi-source build are exercised without touching the network.
"""

from __future__ import annotations

from core.knowledge_base import (
    GutenbergFetcher,
    KnowledgeBaseStore,
    WikipediaFetcher,
    WikisourceFetcher,
    build_knowledge_base,
)

_ARTICLE = (
    "Love is a strong affection and a range of positive emotional states. "
    "It differs from liking and from mere attraction. "
) * 30


class _Response:
    def __init__(self, *, text: str = "", payload: dict | None = None, status: int = 200) -> None:
        self.text = text
        self._payload = payload or {}
        self.status_code = status
        self.headers: dict[str, str] = {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _WikiSession:
    """A fake MediaWiki session: one response per request, keyed by action."""

    def __init__(self, *, titles: list[str], text: str = _ARTICLE) -> None:
        self.titles = titles
        self.text = text
        self.requests: list[dict] = []

    def get(self, url: str, *, params: dict | None = None, timeout: float = 0.0) -> _Response:
        params = params or {}
        self.requests.append({"url": url, **params})
        action = params.get("action")
        if action == "query" and params.get("list") == "categorymembers":
            return _Response(
                payload={"query": {"categorymembers": [{"title": t} for t in self.titles]}}
            )
        if action == "query" and params.get("prop") == "revisions":
            return _Response(
                payload={
                    "query": {
                        "pages": [
                            {
                                "title": t,
                                "revisions": [{"slots": {"main": {"content": self.text}}}],
                            }
                            for t in self.titles
                        ]
                    }
                }
            )
        if action == "query" and params.get("prop") == "extracts":
            title = params.get("titles", "")
            return _Response(payload={"query": {"pages": [{"title": title, "extract": self.text}]}})
        return _Response(payload={})


def test_wikipedia_fetcher_selects_the_edition():
    en = WikipediaFetcher(session=_WikiSession(titles=["Love"]))
    ar = WikipediaFetcher(lang="ar", session=_WikiSession(titles=["حب"]))
    assert en.api == "https://en.wikipedia.org/w/api.php"
    assert ar.api == "https://ar.wikipedia.org/w/api.php"
    assert ar.base_url == "https://ar.wikipedia.org/wiki/"
    assert en.lang == "en" and ar.lang == "ar"


def test_wikipedia_fetcher_rejects_an_unknown_language():
    import pytest

    with pytest.raises(ValueError):
        WikipediaFetcher(lang="fr")


def test_wikipedia_fetcher_reads_titles_and_extracts():
    session = _WikiSession(titles=["Love", "Emotion"])
    fetch = WikipediaFetcher(session=session)
    titles = fetch.category_titles("Love")
    assert titles == ["Love", "Emotion"]
    extracts = fetch.extracts(titles)
    assert [t for t, _ in extracts] == ["Love", "Emotion"]
    assert all(text for _, text in extracts)


class _GutenbergSession:
    """A fake Gutenberg site: a search page, then one text per book id."""

    SEARCH = (
        "<html>"
        '<span class="title">Authors</span><a href="/ebooks/4300">a</a>'
        '<span class="title">Memoirs of Fanny Hill</span><a href="/ebooks/5225">b</a>'
        '<span class="title">Venus in Furs</span><a href="/ebooks/73144">c</a>'
        "</html>"
    )
    BOOK = (
        "Header text before the work. "
        "*** START OF THE PROJECT GUTENBERG EBOOK VENUS IN FURS *** "
        + ("A sensual tale of desire and surrender. " * 40)
        + " *** END OF THE PROJECT GUTENBERG EBOOK VENUS IN FURS *** "
        "Footer licence text."
    )

    def get(self, url: str, *, params: dict | None = None, timeout: float = 0.0) -> _Response:
        if "search" in url:
            return _Response(text=self.SEARCH)
        return _Response(text=self.BOOK)


def test_gutenberg_fetcher_parses_ids_titles_and_drops_navigation():
    fetch = GutenbergFetcher(session=_GutenbergSession())
    titles = fetch.category_titles("erotic literature")
    assert "5225::Memoirs of Fanny Hill" in titles
    assert "73144::Venus in Furs" in titles
    # The navigation link ("Authors") is not a book and is dropped.
    assert not any(t.startswith("4300::") for t in titles)


def test_gutenberg_fetcher_strips_boilerplate():
    fetch = GutenbergFetcher(session=_GutenbergSession())
    extracts = fetch.extracts(["73144::Venus in Furs"])
    assert len(extracts) == 1
    title, text = extracts[0]
    assert title == "Venus in Furs"
    assert "START OF THE PROJECT GUTENBERG" not in text
    assert "END OF THE PROJECT GUTENBERG" not in text
    assert "sensual tale" in text


def test_gutenberg_fetcher_declares_its_source_and_license():
    fetch = GutenbergFetcher(session=_GutenbergSession())
    assert fetch.source == "gutenberg"
    assert "public domain" in fetch.license.lower()
    # A query names its own subject, so there is no subcategory walk.
    assert fetch.subcategories("romance") == []


class _WikisourceSession:
    def __init__(self, *, text: str = _ARTICLE) -> None:
        self.text = text
        self.urls: list[str] = []

    def get(self, url: str, *, params: dict | None = None, timeout: float = 0.0) -> _Response:
        self.urls.append(url)
        params = params or {}
        if params.get("list") == "categorymembers":
            return _Response(
                payload={"query": {"categorymembers": [{"title": "Category:Love poetry"}]}}
            )
        if params.get("prop") == "extracts":
            return _Response(
                payload={"query": {"pages": [{"title": params.get("titles"), "extract": self.text}]}}
            )
        return _Response(payload={})


def test_wikisource_fetcher_points_at_wikisource():
    session = _WikisourceSession()
    fetch = WikisourceFetcher(lang="en", session=session)
    assert fetch.api == "https://en.wikisource.org/w/api.php"
    assert fetch.source == "wikisource"
    extracts = fetch.extracts(["The Bad wife"])
    assert extracts and extracts[0][0] == "The Bad wife"
    assert session.urls and "wikisource.org" in session.urls[0]


def test_wikisource_fetcher_rejects_an_unknown_language():
    import pytest

    with pytest.raises(ValueError):
        WikisourceFetcher(lang="de")


def test_build_records_the_source_and_language(tmp_path):
    store_dir = tmp_path / "kb"
    manifest = build_knowledge_base(
        store_dir,
        target_mb=1,
        categories=["Love"],
        fetcher=WikipediaFetcher(lang="ar", session=_WikiSession(titles=["حب", "عاطفة"])),
        batch=5,
        sleep=0,
    )
    assert manifest["source"] == "wikipedia"
    assert manifest["lang"] == "ar"
    chunk = next(KnowledgeBaseStore(store_dir).iter_chunks())
    assert chunk.source == "wikipedia"
    assert chunk.url.startswith("https://ar.wikipedia.org/wiki/")


def test_multiple_sources_share_one_store_without_checkpoint_collision(tmp_path):
    store_dir = tmp_path / "kb"
    # Same category name on two sources must not mark each other done.
    build_knowledge_base(
        store_dir,
        target_mb=1,
        categories=["Love"],
        fetcher=WikipediaFetcher(session=_WikiSession(titles=["Love"])),
        batch=5,
        sleep=0,
        source_tag="wikipedia:en",
    )
    manifest = build_knowledge_base(
        store_dir,
        target_mb=1,
        categories=["Love"],
        fetcher=GutenbergFetcher(session=_GutenbergSession()),
        batch=5,
        sleep=0,
        source_tag="gutenberg",
    )
    sources = {chunk.source for chunk in KnowledgeBaseStore(store_dir).iter_chunks()}
    assert sources == {"wikipedia", "gutenberg"}
    assert manifest["source"] == "gutenberg"


def test_the_romance_seed_is_non_empty_and_bilingual():
    from core.knowledge_base import (
        ROMANCE_CATEGORIES_AR,
        ROMANCE_CATEGORIES_EN,
        ROMANCE_GUTENBERG_QUERIES,
    )

    assert len(ROMANCE_CATEGORIES_EN) >= 10
    assert len(ROMANCE_CATEGORIES_AR) >= 6
    assert any(any("\u0600" <= ch <= "\u06ff" for ch in c) for c in ROMANCE_CATEGORIES_AR)
    assert "love poetry" in ROMANCE_GUTENBERG_QUERIES


def test_cli_source_selection_builds_the_right_fetcher():
    from cli.knowledge import _make_fetcher

    assert isinstance(_make_fetcher("wikipedia")[0], WikipediaFetcher)
    assert _make_fetcher("wikipedia-ar")[0].lang == "ar"
    assert isinstance(_make_fetcher("gutenberg")[0], GutenbergFetcher)
    assert _make_fetcher("wikisource")[0].source == "wikisource"
    assert _make_fetcher("wikisource-ar")[0].lang == "ar"
    # Each source gets its own checkpoint namespace.
    tags = {_make_fetcher(s)[1] for s in
            ("wikipedia", "wikipedia-ar", "gutenberg", "wikisource", "wikisource-ar")}
    assert len(tags) == 5


def test_cli_romance_seed_is_selected_per_source(tmp_path, monkeypatch):
    from cli import knowledge as cli_knowledge

    captured: dict = {}

    def fake_build(store_dir, **kwargs):
        captured.update(kwargs)
        return {
            "source": "x", "total_mb": 0.0, "total_chunks": 0,
            "total_articles": 0, "complete": False,
        }

    monkeypatch.setattr(cli_knowledge, "build_knowledge_base", fake_build)
    import argparse

    args = argparse.Namespace(
        store_dir=str(tmp_path / "kb"), source="wikipedia-ar", romance=True,
        categories=None, target_mb=1, batch=5, chunk_chars=1200, overlap=150,
        depth=0, subcat_limit=500, max_articles=1,
    )
    cli_knowledge.run_build(args)
    from core.knowledge_base import ROMANCE_CATEGORIES_AR

    assert captured["categories"] == ROMANCE_CATEGORIES_AR
    assert captured["source_tag"] == "wikipedia:ar"


def test_store_dir_is_accepted_before_and_after_the_subcommand(tmp_path):
    # Regression: `--store-dir` used to be a *top-level only* option, so the
    # natural `knowledge build --store-dir X` failed with "unrecognized
    # arguments". It must parse in both positions, for every subcommand.
    from cli.knowledge import _build_parser

    parser = _build_parser()
    after = parser.parse_args(["build", "--store-dir", str(tmp_path), "--target-mb", "1"])
    before = parser.parse_args(["--store-dir", str(tmp_path), "build", "--target-mb", "1"])
    assert after.store_dir == str(tmp_path)
    assert before.store_dir == str(tmp_path)

    for sub in ("stats", "build-index"):
        assert parser.parse_args([sub, "--store-dir", str(tmp_path)]).store_dir == str(tmp_path)
        assert parser.parse_args(["--store-dir", str(tmp_path), sub]).store_dir == str(tmp_path)
    assert (
        parser.parse_args(["search", "--store-dir", str(tmp_path), "porosity"]).store_dir
        == str(tmp_path)
    )


def test_store_dir_defaults_to_none_when_not_given(tmp_path):
    from cli.knowledge import _build_parser

    assert _build_parser().parse_args(["build", "--target-mb", "1"]).store_dir is None
