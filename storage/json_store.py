from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote


class JsonStore:
    """Simple namespaced JSON persistence for local development and snapshots."""
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def _encode(part: str) -> str:
        # Namespaces and keys are logical names, not filenames: a document id
        # like "learned:5ab188c4" is perfectly valid, but ':' is illegal in a
        # Windows filename, so ``os.replace`` fails there with WinError 87.
        # Percent-encode the parts so any logical name maps to a path every
        # platform accepts; the mapping is reversible, so keys round-trip.
        return quote(part, safe="")

    def _path(self, namespace: str, key: str) -> Path:
        if not namespace.strip() or not key.strip() or any(part in {".", ".."} for part in (namespace, key)):
            raise ValueError("Namespace and key must be non-empty simple names.")
        return self.root / self._encode(namespace) / f"{self._encode(key)}.json"

    def _legacy_path(self, namespace: str, key: str) -> Path:
        # Files written before keys were encoded used the raw name. Identity
        # names ("one", "episode") are unchanged, but a raw "learned:x.json"
        # saved on a POSIX host must still be readable after the fix.
        return self.root / namespace / f"{key}.json"

    def save(self, namespace: str, key: str, value: Any) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    def load(self, namespace: str, key: str) -> Any | None:
        path = self._path(namespace, key)
        if not path.exists():
            legacy = self._legacy_path(namespace, key)
            if not legacy.exists():
                return None
            path = legacy
        return json.loads(path.read_text(encoding="utf-8"))

    def list_keys(self, namespace: str) -> list[str]:
        path = self.root / self._encode(namespace)
        keys = {unquote(item.stem) for item in path.glob("*.json")} if path.exists() else set()
        legacy = self.root / namespace
        if legacy != path and legacy.exists():
            keys |= {item.stem for item in legacy.glob("*.json")}
        return sorted(keys)
