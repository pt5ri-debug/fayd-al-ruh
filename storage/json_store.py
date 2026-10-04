from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonStore:
    """Simple namespaced JSON persistence for local development and snapshots."""
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, namespace: str, key: str) -> Path:
        if not namespace.strip() or not key.strip() or any(part in {".", ".."} for part in (namespace, key)):
            raise ValueError("Namespace and key must be non-empty simple names.")
        return self.root / namespace / f"{key}.json"

    def save(self, namespace: str, key: str, value: Any) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    def load(self, namespace: str, key: str) -> Any | None:
        path = self._path(namespace, key)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def list_keys(self, namespace: str) -> list[str]:
        path = self.root / namespace
        return sorted(item.stem for item in path.glob("*.json")) if path.exists() else []
