"""The Dataset Studio: a thin front end over the governed ``ml.training`` builder.

The project already owns a *governed* dataset pipeline (Phase 6): a validated,
versioned, provenance-carrying snapshot with train/validation/test splits, a
duplicate/leakage/quality report, and the structural personality-isolation guard.
This module does **not** re-implement any of that -- it composes it so the
Training Center can list, build, validate, inspect, and export dataset versions
through one interface, and so a learning example that was never approved can
never reach a dataset.

Every operation here delegates to :class:`ml.training.datasets.DatasetBuilder`
and its validator, so the CLI, the window, and the tests share one registry.
Dependency-free (stdlib + numpy, already a base dependency), Tk-free, and
offline.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.training.datasets import (
    DatasetBuilder,
    DatasetRecord,
    DatasetValidationError,
    TrainingExample,
)

from core.training_center.records import LearningExample


@dataclass(frozen=True)
class DatasetRow:
    """One dataset version, as a table row the window and the CLI can draw."""

    dataset_id: str
    version: int
    total: int
    train: int
    validation: int
    test: int
    quality: float
    status: str
    eligible: bool
    scope: tuple[str, ...]
    source: str
    created_at: str

    @property
    def key(self) -> str:
        return f"{self.dataset_id}@v{self.version}"

    def describe(self) -> str:
        state = "eligible" if self.eligible else self.status
        scope = ", ".join(self.scope) or "project"
        return (
            f"{self.key} [{state}] {self.total} example(s) "
            f"(train {self.train} / val {self.validation} / test {self.test}) · "
            f"scope {scope} · quality {self.quality:.2f}"
        )


@dataclass(frozen=True)
class ExportReport:
    """What one dataset export wrote."""

    dataset_key: str
    path: str
    examples: int
    ok: bool
    message: str = ""

    def describe(self) -> str:
        if not self.ok:
            return f"Export failed: {self.message}"
        return f"Wrote {self.examples} example(s) of {self.dataset_key} to {self.path}."


class DatasetStudio:
    """List, build, validate, and export governed dataset versions."""

    def __init__(self, *, builder: DatasetBuilder | None = None) -> None:
        self.builder = builder or DatasetBuilder()

    # -- reading -----------------------------------------------------------
    def rows(self) -> tuple[DatasetRow, ...]:
        """Every stored dataset version, newest first."""
        records = list(self.builder.list_datasets())
        records.sort(key=lambda r: (r.dataset_id, -r.version))
        return tuple(self._row(record) for record in records)

    def _row(self, record: DatasetRecord) -> DatasetRow:
        return DatasetRow(
            dataset_id=record.dataset_id,
            version=record.version,
            total=record.total,
            train=record.train_count,
            validation=record.validation_count,
            test=record.test_count,
            quality=record.quality,
            status=record.status,
            eligible=record.eligible,
            scope=tuple(record.personality_scope),
            source=record.source,
            created_at=record.created_at,
        )

    def get(self, dataset_id: str, version: int | None = None) -> DatasetRow | None:
        """One dataset version (the latest when ``version`` is omitted)."""
        for row in self.rows():
            if row.dataset_id == dataset_id and (version is None or row.version == version):
                if version is None:
                    return row
                return row
        return None

    def latest(self, dataset_id: str) -> DatasetRow | None:
        rows = [row for row in self.rows() if row.dataset_id == dataset_id]
        return max(rows, key=lambda r: r.version) if rows else None

    def describe(self, dataset_id: str, version: int | None = None) -> str:
        """A readable report of one stored dataset version."""
        built = (
            self.builder.load(dataset_id, version)
            if version is not None
            else self.builder.latest(dataset_id)
        )
        if built is None:
            return f"No dataset {dataset_id!r} was found."
        record = built.record
        lines = [record.describe(), f"  digest {record.digest[:16]}"]
        if record.notes:
            lines.extend(f"  note: {note}" for note in record.notes)
        report = built.report
        lines.append(f"  validator ok: {report.ok}")
        for issue in report.issues[:20]:
            lines.append(f"    - {issue.describe()}")
        return "\n".join(lines)

    # -- building ----------------------------------------------------------
    def build_from_examples(
        self,
        dataset_id: str,
        examples: Iterable[LearningExample],
        *,
        source: str = "training-center",
        license: str = "approved-learning-example",
        intended_model: str = "fayd_al_ruh",
        intended_task: str = "chat",
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 0,
    ) -> DatasetRow:
        """Build a dataset from **approved** learning examples only.

        An example that is not approved is refused by name, so the review gate is
        structural rather than a convention: nothing reaches a dataset without a
        human having approved or corrected it.
        """
        approved: list[TrainingExample] = []
        for example in examples:
            if not example.approved:
                raise DatasetValidationError(
                    f"Example {example.example_id} is {example.status!r}, not approved; "
                    "approve or correct it before exporting it to a dataset."
                )
            approved.append(
                TrainingExample(
                    text=example.training_text,
                    source=example.provider or source,
                    license=license,
                    tags=(example.status, example.language),
                )
            )
        if not approved:
            raise DatasetValidationError("No approved learning examples were supplied.")
        return self.build_from_texts(
            dataset_id,
            [example.text for example in approved],
            source=source,
            license=license,
            intended_model=intended_model,
            intended_task=intended_task,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    def build_from_texts(
        self,
        dataset_id: str,
        texts: Sequence[str],
        *,
        source: str = "training-center",
        license: str = "project-corpus",
        intended_model: str = "fayd_al_ruh",
        intended_task: str = "chat",
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 0,
    ) -> DatasetRow:
        """Build a dataset version from plain texts (provenance recorded)."""
        examples = [
            TrainingExample(
                text=text,
                source=source,
                license=license,
            )
            for text in texts
            if text.strip()
        ]
        if not examples:
            raise DatasetValidationError("A dataset needs at least one non-empty example.")
        built = self.builder.build(
            dataset_id,
            examples,
            source=source,
            license=license,
            intended_model=intended_model,
            intended_task=intended_task,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        return self._row(built.record)

    def validate(self, dataset_id: str, version: int | None = None) -> str:
        """Re-run the governed validator and return a readable report."""
        return self.describe(dataset_id, version)

    # -- export ------------------------------------------------------------
    def export_jsonl(
        self, dataset_id: str, destination: str | Path, version: int | None = None
    ) -> ExportReport:
        """Write a stored dataset version to JSONL, one example per line."""
        built = (
            self.builder.load(dataset_id, version)
            if version is not None
            else self.builder.latest(dataset_id)
        )
        key = f"{dataset_id}@v{version}" if version is not None else dataset_id
        if built is None:
            return ExportReport(dataset_key=key, path="", examples=0, ok=False, message="no such dataset")
        target = Path(destination)
        if target.is_dir() or target.suffix == "":
            target = target / f"{dataset_id}-v{built.record.version}.jsonl"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            count = 0
            with target.open("w", encoding="utf-8") as handle:
                for example in built.split.all_examples():
                    handle.write(json.dumps(example.to_primitive(), ensure_ascii=False) + "\n")
                    count += 1
        except OSError as error:
            return ExportReport(
                dataset_key=built.record.key, path=str(target), examples=0, ok=False,
                message=str(error),
            )
        return ExportReport(
            dataset_key=built.record.key, path=str(target), examples=count, ok=True
        )

    def stats(self) -> dict[str, Any]:
        """Counts for the dashboard, read from the real registry."""
        rows = self.rows()
        return {
            "datasets": len({row.dataset_id for row in rows}),
            "versions": len(rows),
            "examples": sum(row.total for row in rows),
            "eligible": sum(1 for row in rows if row.eligible),
            "train": sum(row.train for row in rows),
            "validation": sum(row.validation for row in rows),
            "test": sum(row.test for row in rows),
            "storage_bytes": _dir_size(self.builder.root),
        }


def _dir_size(root: Path) -> int:
    """Total bytes under ``root``, or 0 when it does not exist."""
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


__all__ = [
    "DatasetRow",
    "DatasetStudio",
    "ExportReport",
]
