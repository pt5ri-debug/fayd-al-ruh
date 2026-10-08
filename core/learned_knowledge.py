"""Learned knowledge that ships with the project.

Most learned statements live in the user's writable data directory and never
enter version control. Some, though, are reviewed and promoted into the project
itself, so that a fresh clone already knows them: they sit in
``knowledge/learned_petroleum.json`` and are loaded here as ordinary documents.

The file is written by :mod:`ai.internet_learner` when the Continuous Internet
Learner is asked to promote an accepted batch into the project. Each record is a
plain :class:`core.documents.Document` primitive, so the loader adds no new
format -- it just reads the same citable documents the runtime store uses.

Loading is tolerant: a missing or unreadable file yields no documents rather than
breaking store construction, so the project still starts if the file is absent.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from core.acceptance_policy import sanitize_statement
from core.documents import Document
from core.paths import bundle_root

LEARNED_FILENAME = "learned_petroleum.json"


def learned_path() -> Path:
    """Where the promoted knowledge file lives (inside the bundled project)."""
    return bundle_root() / "knowledge" / LEARNED_FILENAME


def learned_documents() -> list[Document]:
    """The promoted learned documents, or an empty list when there are none."""
    path = learned_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    documents: list[Document] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            document = Document.from_primitive(item)
        except (KeyError, ValueError, TypeError):
            continue
        # Repair at read time: a promoted statement that is only a fragment (or
        # has no terminal punctuation) is dropped, so a stale file can never put
        # broken prose back into an answer.
        statement = sanitize_statement(document.text)
        if not statement:
            continue
        if statement != document.text:
            document = replace(document, text=statement)
        documents.append(document)
    return documents


__all__ = ["LEARNED_FILENAME", "learned_documents", "learned_path"]
