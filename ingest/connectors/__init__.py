"""Ingestion connectors, one module per source family.

* :mod:`ingest.connectors.files` -- local TXT/Markdown/DOCX/PDF/CSV/JSON/JSONL.
* :mod:`ingest.connectors.sqlite` -- read-only SQLite (standard library).
* :mod:`ingest.connectors.sql` -- read-only PostgreSQL/MySQL via the DB-API.
* :mod:`ingest.connectors.web` -- optional, permission-gated internet import.
"""

__all__ = ["files", "sql", "sqlite", "web"]
