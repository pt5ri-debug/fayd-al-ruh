"""A large, buildable, citation-carrying knowledge base.

The seeds in ``core/petroleum_corpus.py`` and ``core/curriculum_knowledge.py``
are the project's *authored* knowledge: small, curated, and planted into the
ontology. This module is the other half -- a **bulk reference corpus** fetched
from a public, keyless source (Wikipedia) and stored on disk as chunked JSONL.
It exists so the project can answer across every field, not only petroleum, and
so the knowledge base can grow to a target size without bloating the repository
or the in-memory graph.

Design rules, all deliberate:

* **Separate from governed learning.** This text is *reference material*. It is
  never fed to the acceptance policy or the belief layers; it is retrieved and
  cited, and every chunk carries its source URL and licence. The governed
  internet learner stays the only path that can change internal knowledge.
* **Out of the repository.** The store lives under the writable data directory
  (``data/knowledge_base`` by default), which is git-ignored, and is *built on
  demand* with ``python -m cli knowledge build``. Nothing large is committed.
* **Lazy and bounded.** The retriever loads chunks in batches on demand and
  keeps only what it needs in memory, so a 100 MB+ store does not have to fit
  in RAM at once.
* **Network-tolerant.** The fetcher is plain ``requests`` against the MediaWiki
  API with a descriptive User-Agent, batched, paced, and resumable through a
  checkpoint so a long build can be stopped and restarted.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.documents import Document

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_USER_AGENT = (
    "FaydAlRuh/2.2 (educational knowledge base; +https://github.com/pt5ri-debug/fayd-al-ruh)"
)
WIKI_LICENSE = "CC BY-SA 4.0 (Wikipedia)"
WIKI_BASE_URL = "https://en.wikipedia.org/wiki/"

# A broad seed across the fields a petroleum mind reasons with and around.
# Titles are Wikipedia category names; each expands to its members.
DEFAULT_CATEGORIES: tuple[str, ...] = (
    "Petroleum engineering",
    "Petroleum geology",
    "Petroleum production",
    "Drilling technology",
    "Oil wells",
    "Natural gas",
    "Reservoir engineering",
    "Geophysics",
    "Geology",
    "Sedimentology",
    "Stratigraphy",
    "Mineralogy",
    "Petrology",
    "Structural geology",
    "Hydrogeology",
    "Chemistry",
    "Organic chemistry",
    "Physical chemistry",
    "Analytical chemistry",
    "Physics",
    "Classical mechanics",
    "Thermodynamics",
    "Fluid dynamics",
    "Materials science",
    "Mathematics",
    "Statistics",
    "Numerical analysis",
    "Computer science",
    "Electrical engineering",
    "Mechanical engineering",
    "Chemical engineering",
    "Civil engineering",
    "Environmental science",
    "Economics",
    "Project management",
    "Occupational safety and health",
    "Energy",
    "Climate change",
    "Earth sciences",
    "Metallurgy",
)

# Defaults tuned for a small, fast, and polite build.
DEFAULT_BATCH = 50
DEFAULT_CHUNK_CHARS = 1200
DEFAULT_OVERLAP = 150
DEFAULT_MIN_CHUNK = 200
DEFAULT_SLEEP = 0.05
DEFAULT_TARGET_MB = 100.0
# Most Wikipedia content sits in subcategories, so walking two levels down is
# what turns a thin seed into a broad corpus.
DEFAULT_DEPTH = 2
DEFAULT_SUBCAT_LIMIT = 500


@dataclass(frozen=True)
class KnowledgeChunk:
    """One citable slice of bulk reference text."""

    chunk_id: str
    title: str
    text: str
    source: str
    url: str
    license: str = WIKI_LICENSE
    category: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "chunk_id": self.chunk_id,
                "title": self.title,
                "text": self.text,
                "source": self.source,
                "url": self.url,
                "license": self.license,
                "category": self.category,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, line: str) -> KnowledgeChunk:
        data = json.loads(line)
        return cls(
            chunk_id=str(data["chunk_id"]),
            title=str(data["title"]),
            text=str(data["text"]),
            source=str(data.get("source", "wikipedia")),
            url=str(data.get("url", "")),
            license=str(data.get("license", WIKI_LICENSE)),
            category=str(data.get("category", "")),
        )

    def to_document(self) -> Document:
        """Adapt to the project's citable :class:`~core.documents.Document`."""
        return Document(
            doc_id=self.chunk_id,
            title=self.title,
            text=self.text,
            source=self.source,
            metadata={
                "url": self.url,
                "license": self.license,
                "category": self.category,
                "origin": "knowledge_base",
            },
        )


_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_REF_RE = re.compile(r"<ref\b[^>]*?/>|<ref\b[^>]*?>.*?</ref>", flags=re.DOTALL | re.IGNORECASE)
_TABLE_RE = re.compile(r"\{\|.*?\|\}", flags=re.DOTALL)
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}", flags=re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_RE = re.compile(r"^=+\s*(.*?)\s*=+\s*$", flags=re.MULTILINE)
_FILE_RE = re.compile(r"\[\[(?:File|Image|Category)\s*:[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]", flags=re.IGNORECASE)
_LINK_RE = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]|]*)\]\]")
_URL_RE = re.compile(r"\[(?:https?://\S+)\s+([^\]]*)\]")
_BOLD_RE = re.compile(r"'{2,5}")
_MAGIC_RE = re.compile(r"__[A-Z]+__")


def clean_wikitext(text: str) -> str:
    """Reduce MediaWiki wikitext to readable plain prose.

    A pragmatic cleaner, not a parser: it strips comments, references, tables,
    nested templates, and file/category links, unwraps internal and external
    links to their display text, and removes heading and emphasis markup. The
    goal is quotable prose, so leftover artifacts are acceptable but structure
    is not.
    """
    if not text:
        return ""
    text = _COMMENT_RE.sub(" ", text)
    text = _REF_RE.sub(" ", text)
    text = _TABLE_RE.sub(" ", text)
    # Templates can nest; a few passes collapse the common cases.
    for _ in range(6):
        new = _TEMPLATE_RE.sub(" ", text)
        if new == text:
            break
        text = new
    text = _FILE_RE.sub(" ", text)
    text = _HEADING_RE.sub(r"\1. ", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(r"\1", text)
    text = _TAG_RE.sub(" ", text)
    text = _BOLD_RE.sub("", text)
    text = _MAGIC_RE.sub(" ", text)
    return " ".join(text.split())


def chunk_text(
    text: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    min_chunk: int = DEFAULT_MIN_CHUNK,
) -> list[str]:
    """Split ``text`` into overlapping slices of roughly ``chunk_chars``.

    Prefers paragraph boundaries, then sentence boundaries, then a hard cut, so
    a chunk rarely breaks mid-sentence. Slices shorter than ``min_chunk`` are
    dropped except when they are the only content.
    """
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be at least 1.")
    if overlap < 0 or overlap >= chunk_chars:
        raise ValueError("overlap must be in [0, chunk_chars).")
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= chunk_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(start + chunk_chars, length)
        if end < length:
            window = cleaned[start:end]
            for boundary in ("\n\n", ". ", "? ", "! "):
                cut = window.rfind(boundary)
                if cut > chunk_chars // 2:
                    end = start + cut + len(boundary)
                    break
        piece = cleaned[start:end].strip()
        if len(piece) >= min_chunk or not chunks:
            chunks.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _safe_filename(name: str) -> str:
    """A filesystem-safe token derived from a title."""
    keep = [ch if ch.isalnum() else "_" for ch in name]
    return "".join(keep).strip("_")[:80] or "item"


class WikipediaFetcher:
    """Fetches article full text from Wikipedia's MediaWiki API (keyless).

    One instance reuses a ``requests`` session. It is intentionally small: it
    resolves a category to article titles and resolves titles to plain-text
    extracts, nothing more.
    """

    def __init__(self, *, timeout: float = 30.0, session: Any | None = None) -> None:
        self.timeout = timeout
        if session is not None:
            self._session = session
        else:  # pragma: no cover - network path
            import requests

            self._session = requests.Session()
            self._session.headers.update({"User-Agent": WIKI_USER_AGENT})

    def _get(
        self, params: dict[str, Any], *, retries: int = 6
    ) -> dict[str, Any]:  # pragma: no cover - network
        params = {**params, "format": "json", "maxlag": "5"}
        delay = 1.0
        for _attempt in range(retries):
            response = self._session.get(WIKI_API, params=params, timeout=self.timeout)
            if response.status_code in (429, 503):
                # Wikipedia rate-limits bursts; honour Retry-After, then back off.
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if (retry_after or "").replace(".", "", 1).isdigit() else delay
                time.sleep(min(max(wait, delay), 60.0))
                delay = min(delay * 2, 60.0)
                continue
            response.raise_for_status()
            return response.json()
        response.raise_for_status()
        return {}

    def subcategories(self, category: str, *, limit: int = 500) -> list[str]:
        """Return the subcategory names of a category (namespace 14)."""
        names: list[str] = []
        cont: str | None = None
        while True:
            params: dict[str, Any] = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category if category.startswith("Category:") else f"Category:{category}",
                "cmlimit": "500",
                "cmtype": "subcat",
                "cmnamespace": "14",
            }
            if cont:
                params["cmcontinue"] = cont
            payload = self._get(params)
            for member in payload.get("query", {}).get("categorymembers", []):
                name = member["title"].split(":", 1)[-1]
                names.append(name)
                if len(names) >= limit:
                    return names
            cont = payload.get("continue", {}).get("cmcontinue")
            if not cont:
                break
        return names

    def category_titles(self, category: str, *, limit: int | None = None) -> list[str]:
        """Return the article titles in a category (namespace 0 only)."""
        titles: list[str] = []
        cont: str | None = None
        while True:
            params: dict[str, Any] = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category if category.startswith("Category:") else f"Category:{category}",
                "cmlimit": "500",
                "cmtype": "page",
                "cmnamespace": "0",
            }
            if cont:
                params["cmcontinue"] = cont
            payload = self._get(params)
            titles.extend(m["title"] for m in payload.get("query", {}).get("categorymembers", []))
            if limit is not None and len(titles) >= limit:
                return titles[:limit]
            cont = payload.get("continue", {}).get("cmcontinue")
            if not cont:
                break
        return titles

    def extracts(self, titles: Sequence[str]) -> list[tuple[str, str]]:
        """Return ``(title, plain_text)`` for many titles per request.

        Uses ``prop=revisions`` (raw wikitext) rather than ``prop=extracts``:
        the latter returns only one full extract per request, while revisions
        return the whole batch (up to 50 pages) at once. The wikitext is reduced
        to plain prose with :func:`clean_wikitext`.
        """
        out: list[tuple[str, str]] = []
        for index in range(0, len(titles), 50):
            batch = [t for t in titles[index : index + 50] if t]
            if not batch:
                continue
            payload = self._get(
                {
                    "action": "query",
                    "prop": "revisions",
                    "rvprop": "content",
                    "rvslots": "main",
                    "titles": "|".join(batch),
                    "formatversion": "2",
                }
            )
            for page in payload.get("query", {}).get("pages", []):
                revisions = page.get("revisions") or []
                if not revisions:
                    continue
                content = revisions[0].get("slots", {}).get("main", {}).get("content", "")
                text = clean_wikitext(content)
                if text:
                    out.append((page.get("title", ""), text))
        return out


def _category_queue(
    fetch: Any,
    categories: Sequence[str],
    *,
    depth: int,
    subcat_limit: int,
    progress: Any | None,
) -> Iterator[tuple[str, int]]:
    """Breadth-first walk of a category and its subcategories to ``depth``.

    Most of Wikipedia's content lives in subcategories, so a seed category alone
    is a thin sample. Yields ``(category_name, level)`` in BFS order. Depth 0
    means the seed categories only.
    """
    seen: set[str] = set()
    queue: list[tuple[str, int]] = [(name, 0) for name in categories]
    while queue:
        name, level = queue.pop(0)
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        yield name, level
        if level >= depth:
            continue
        try:
            children = fetch.subcategories(name, limit=subcat_limit)
        except Exception as error:  # pragma: no cover - network path
            if progress:
                progress(f"subcategories for {name!r} failed: {error}")
            continue
        queue.extend((child, level + 1) for child in children)


def build_knowledge_base(
    store_dir: str | Path,
    *,
    target_mb: float = DEFAULT_TARGET_MB,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    fetcher: Any | None = None,
    batch: int = DEFAULT_BATCH,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    min_chunk: int = DEFAULT_MIN_CHUNK,
    sleep: float = DEFAULT_SLEEP,
    depth: int = DEFAULT_DEPTH,
    subcat_limit: int = DEFAULT_SUBCAT_LIMIT,
    max_articles: int | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Fetch, chunk, and store a bulk knowledge base until ``target_mb`` is met.

    Categories are walked breadth-first through their subcategories (``depth``),
    which is what makes the corpus broad rather than a thin sample. Resumable: a
    ``checkpoint.json`` records finished categories and the byte total, and the
    shards already written are kept. Passing an explicit ``fetcher`` (any object
    with ``subcategories`` / ``category_titles`` / ``extracts``) makes the whole
    build offline and testable.
    """
    store = KnowledgeBaseStore(store_dir)
    store.ensure()
    target_bytes = int(target_mb * 1024 * 1024)
    checkpoint = store.load_checkpoint()
    done_categories: set[str] = set(checkpoint.get("done_categories", []))
    seen_titles: set[str] = set(checkpoint.get("seen_titles", []))
    total_bytes = int(checkpoint.get("total_bytes", 0))
    total_chunks = int(checkpoint.get("total_chunks", 0))
    total_articles = int(checkpoint.get("total_articles", 0))
    shard_index = int(checkpoint.get("shard_index", store.next_shard_index()))

    fetch = fetcher if fetcher is not None else WikipediaFetcher()
    stop = False

    def save() -> None:
        store.save_checkpoint(
            {
                "done_categories": sorted(done_categories),
                "seen_titles": sorted(seen_titles),
                "total_bytes": total_bytes,
                "total_chunks": total_chunks,
                "total_articles": total_articles,
                "shard_index": shard_index,
            }
        )

    for category, level in _category_queue(
        fetch, categories, depth=depth, subcat_limit=subcat_limit, progress=progress
    ):
        if stop:
            break
        if category in done_categories:
            continue
        try:
            titles = fetch.category_titles(category)
        except Exception as error:  # pragma: no cover - network path
            if progress:
                progress(f"category {category!r} failed: {error}")
            continue
        unique = [t for t in titles if t and t not in seen_titles]
        for index in range(0, len(unique), batch):
            if max_articles is not None and total_articles >= max_articles:
                stop = True
                break
            slice_titles = unique[index : index + batch]
            try:
                articles = fetch.extracts(slice_titles)
            except Exception as error:  # pragma: no cover - network path
                if progress:
                    progress(f"extracts failed for {slice_titles[:1]}: {error}")
                continue
            chunks: list[KnowledgeChunk] = []
            for title, text in articles:
                seen_titles.add(title)
                if len(text) < min_chunk:
                    continue
                url = WIKI_BASE_URL + title.replace(" ", "_")
                for n, piece in enumerate(chunk_text(text, chunk_chars=chunk_chars, overlap=overlap, min_chunk=min_chunk)):
                    chunks.append(
                        KnowledgeChunk(
                            chunk_id=f"kb:{_safe_filename(title)}:{n}",
                            title=title,
                            text=piece,
                            source="wikipedia",
                            url=url,
                            category=category,
                        )
                    )
            if chunks:
                written = store.append(shard_index, chunks)
                total_bytes += written
                total_chunks += len(chunks)
                total_articles += len(articles)
                shard_index = store.next_shard_index()
            if progress:
                progress(
                    f"{category} (d{level}): {total_articles} articles, "
                    f"{total_bytes / 1024 / 1024:.1f}/{target_mb:.0f} MB"
                )
            if total_bytes >= target_bytes:
                stop = True
                break
            if sleep:
                time.sleep(sleep)
        done_categories.add(category)
        save()

    manifest = {
        "source": "wikipedia",
        "license": WIKI_LICENSE,
        "target_mb": target_mb,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1024 / 1024, 2),
        "total_chunks": total_chunks,
        "total_articles": total_articles,
        "categories": list(categories),
        "depth": depth,
        "done_categories": sorted(done_categories),
        "complete": total_bytes >= target_bytes,
    }
    store.write_manifest(manifest)
    return manifest


def _fsync_dir(path: Path) -> None:
    """Flush a directory entry so a just-created shard survives a crash.

    Best effort: a filesystem that will not let us open the directory for
    reading (or has no ``O_DIRECTORY``) is not a reason to fail an otherwise
    durable write.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class KnowledgeBaseStore:
    """On-disk JSONL storage for knowledge chunks, sharded for lazy reading."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "checkpoint.json"

    def shard_paths(self) -> list[Path]:
        return sorted(self.root.glob("shard-*.jsonl"))

    def next_shard_index(self) -> int:
        existing = self.shard_paths()
        if not existing:
            return 0
        indices = []
        for path in existing:
            try:
                indices.append(int(path.stem.split("-")[1]))
            except (IndexError, ValueError):
                continue
        return (max(indices) + 1) if indices else len(existing)

    def append(
        self,
        shard_index: int,
        chunks: Iterable[KnowledgeChunk],
        *,
        record_ids: set[str] | None = None,
        sync: bool = False,
    ) -> int:
        """Append chunks to a shard; return the number of bytes written.

        ``record_ids`` (optional) collects each appended ``chunk_id`` into the
        given set, so a caller that must only commit a checkpoint after the data
        is on disk can learn exactly which ids a write covered. ``sync`` flushes
        the file and its directory before returning, so a crash afterwards cannot
        lose the just-written batch -- resumable imports turn it on, the bulk
        wiki build leaves it off for speed.
        """
        self.ensure()
        path = self.root / f"shard-{shard_index:05d}.jsonl"
        written = 0
        with path.open("a", encoding="utf-8") as handle:
            for chunk in chunks:
                line = chunk.to_json() + "\n"
                handle.write(line)
                written += len(line.encode("utf-8"))
                if record_ids is not None:
                    record_ids.add(chunk.chunk_id)
            if sync:
                handle.flush()
                os.fsync(handle.fileno())
        if sync:
            _fsync_dir(self.root)
        return written

    def iter_chunks(self) -> Iterator[KnowledgeChunk]:
        for path in self.shard_paths():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield KnowledgeChunk.from_json(line)

    def read_chunks(self, chunk_ids: Iterable[str]) -> dict[str, KnowledgeChunk]:
        """Read specific chunks by id, scanning shards once (used by the index).

        A single full pass is cheaper than open-seek per id, and it keeps memory
        bounded: only the requested chunks are returned.
        """
        wanted = set(chunk_ids)
        if not wanted:
            return {}
        found: dict[str, KnowledgeChunk] = {}
        for chunk in self.iter_chunks():
            if chunk.chunk_id in wanted:
                found[chunk.chunk_id] = chunk
                if len(found) == len(wanted):
                    break
        return found

    def iter_documents(self, *, limit: int | None = None) -> Iterator[Document]:
        for count, chunk in enumerate(self.iter_chunks()):
            if limit is not None and count >= limit:
                return
            yield chunk.to_document()

    def size_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.shard_paths())

    def count(self) -> int:
        return sum(1 for _ in self.iter_chunks())

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.ensure()
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def load_checkpoint(self) -> dict[str, Any]:
        if not self.checkpoint_path.exists():
            return {}
        return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))

    def save_checkpoint(self, state: dict[str, Any]) -> None:
        self.ensure()
        self.checkpoint_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def stats(self) -> dict[str, Any]:
        manifest = self.load_manifest()
        size = self.size_bytes()
        return {
            "path": str(self.root),
            "shards": len(self.shard_paths()),
            "size_bytes": size,
            "size_mb": round(size / 1024 / 1024, 2),
            "chunks": manifest.get("total_chunks", self.count()),
            "articles": manifest.get("total_articles", 0),
            "complete": manifest.get("complete", False),
            "target_mb": manifest.get("target_mb", DEFAULT_TARGET_MB),
        }


def default_knowledge_base_dir() -> Path:
    """Where a built knowledge base lives by default (under the data directory)."""
    from core.paths import data_dir

    return data_dir() / "knowledge_base"


def iter_local_documents(store_dir: str | Path, *, limit: int | None = None) -> Iterator[Document]:
    """Yield citable documents from a built knowledge base, if one exists."""
    store = KnowledgeBaseStore(store_dir)
    if not store.shard_paths():
        return
    yield from store.iter_documents(limit=limit)


__all__ = [
    "DEFAULT_BATCH",
    "DEFAULT_CATEGORIES",
    "DEFAULT_CHUNK_CHARS",
    "DEFAULT_DEPTH",
    "DEFAULT_OVERLAP",
    "DEFAULT_SUBCAT_LIMIT",
    "DEFAULT_TARGET_MB",
    "WIKI_API",
    "WIKI_LICENSE",
    "WIKI_USER_AGENT",
    "KnowledgeBaseStore",
    "KnowledgeChunk",
    "WikipediaFetcher",
    "build_knowledge_base",
    "chunk_text",
    "clean_wikitext",
    "default_knowledge_base_dir",
    "iter_local_documents",
]
