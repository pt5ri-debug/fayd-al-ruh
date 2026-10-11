"""Ingestion connectors, one module per source family.

* :mod:`ingest.connectors.files` -- local TXT/Markdown/DOCX/PDF/CSV/JSON/JSONL.
* :mod:`ingest.connectors.sql` -- SQLite (and optional DB-API) databases.
* :mod:`ingest.connectors.web` -- optional, permission-gated internet import.
"""

__all__ = ["files", "sql", "web"]
