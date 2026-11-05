"""Ingest text the caller already holds (a paste, an editor, a chat message).

The one source that is neither a file nor a URL: the text *is* the source. It is
how the Knowledge Base Manager's paste box, the CLI's ``text`` command, and any
caller holding a document's text feed the **same** pipeline the file and web
connectors feed -- so a pasted note is chunked, de-duplicated, indexed, and cited
exactly like an imported file.

The text and its title travel in the :class:`~ingest.records.SourceSpec`'s own
options, so nothing is written to a temporary file and no path is involved. The
source is content-addressed (``text:<digest>``), which is what makes pasting the
same text twice one source rather than two.
"""

from __future__ import annotations

from collections.abc import Iterator

from ingest.errors import ConnectorError
from ingest.hashing import content_hash, source_key
from ingest.records import ConnectorInfo, IngestItem, SourceRecord, SourceSpec

#: The title used when the caller supplies none.
DEFAULT_TITLE = "Pasted text"

#: The URI scheme that names a direct-text source (``text:<digest>``).
TEXT_SCHEME = "text"


class TextConnector:
    """A connector over text held in the spec's own options (no file, no URL)."""

    info = ConnectorInfo(
        name="text",
        description="Text supplied directly by the caller (manual paste); no file, no URL.",
        schemes=frozenset({TEXT_SCHEME}),
    )

    def plan(self, spec: SourceSpec) -> Iterator[IngestItem]:
        text = str(spec.option("text", "") or "")
        if not text.strip():
            raise ConnectorError("The text source is empty; nothing to ingest.")
        title = str(spec.option("title", "") or "").strip() or DEFAULT_TITLE
        digest = content_hash(text)
        # The text travels with the item so ``stream_item`` needs no state and the
        # same connector instance can serve any number of sources.
        yield IngestItem(
            source_key=source_key(digest, extra=TEXT_SCHEME),
            uri=f"{TEXT_SCHEME}:{digest[:16]}",
            title=title,
            fingerprint=digest,
            metadata={"kind": "text", "title": title, "text": text},
        )

    def stream_item(self, item: IngestItem) -> Iterator[SourceRecord]:
        text = str(item.metadata.get("text", "") or "")
        if not text.strip():
            raise ConnectorError(f"{item.uri!r} carries no text.")
        yield SourceRecord(
            title=item.title,
            text=text,
            source=TEXT_SCHEME,
            license="user",
            source_key=item.source_key,
            metadata={"kind": "text", "title": item.title},
        )

    def stream(self, spec: SourceSpec) -> Iterator[SourceRecord]:
        for item in self.plan(spec):
            yield from self.stream_item(item)


__all__ = ["DEFAULT_TITLE", "TEXT_SCHEME", "TextConnector"]
