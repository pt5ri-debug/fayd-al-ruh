from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any


class SqliteStore:
    """Namespaced JSON-value store backed by SQLite from Python's standard library."""
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS records (namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(namespace, key))")
            connection.commit()
        finally:
            connection.close()

    def _connect(self):
        return sqlite3.connect(self.path)

    def save(self, namespace: str, key: str, value: Any) -> None:
        connection = self._connect()
        try:
            connection.execute("INSERT INTO records(namespace,key,value) VALUES(?,?,?) ON CONFLICT(namespace,key) DO UPDATE SET value=excluded.value", (namespace, key, json.dumps(value, ensure_ascii=False)))
            connection.commit()
        finally:
            connection.close()

    def load(self, namespace: str, key: str) -> Any | None:
        connection = self._connect()
        try:
            row = connection.execute("SELECT value FROM records WHERE namespace=? AND key=?", (namespace, key)).fetchone()
        finally:
            connection.close()
        return json.loads(row[0]) if row else None

    def list_keys(self, namespace: str) -> list[str]:
        connection = self._connect()
        try:
            rows = connection.execute("SELECT key FROM records WHERE namespace=? ORDER BY key", (namespace,)).fetchall()
        finally:
            connection.close()
        return [row[0] for row in rows]
