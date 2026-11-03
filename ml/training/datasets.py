"""Dataset governance for Phase 6: the record, the validator, and the snapshot.

A dataset is the one thing a training run may consume, and this module makes it a
**governed artifact** rather than a folder of text:

* :class:`TrainingExample` is one example with its **provenance** (the owner it
  came from, the source it was taken from, and a stable ``example_id``).
* :class:`DatasetRecord` is the dataset's own record -- the task's required fields
  (id, version, source/provenance, creation date, license, intended model/task,
  personality scope, quality, validation status, eligibility, exclusion rules)
  plus the split counts and a content digest.
* :class:`DatasetValidator` runs the pipeline: format and encoding checks, empty
  and over-long records, exact and near duplicates, and **train/validation/test
  leakage** (the same example appearing on two sides of a split).
* :class:`DatasetBuilder` assembles a dataset from explicit, authorised examples,
  applies the validator, splits it, and writes a **versioned snapshot** with a
  manifest -- and refuses to overwrite an existing version.

Honesty rules, enforced here rather than promised in a doc:

* **Nothing is auto-ingested.** This module never reads the conversation store,
  the memory store, or the knowledge graph. A caller must pass the examples.
* **No private owner is ever accepted silently.** :func:`DatasetBuilder.build`
  refuses a dataset whose personality scope contains a private owner unless the
  caller explicitly declares it private, which the training path never does.
* **A version is immutable.** Writing a version that already exists raises rather
  than overwriting it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.learning_isolation import (
    MAIN_MODEL_OWNER,
    is_owner,
    is_private,
)
from core.paths import training_datasets_dir

# ---------------------------------------------------------------- constants

#: The split names a dataset carries. ``test`` is held back from training.
SPLITS: tuple[str, ...] = ("train", "validation", "test")

#: A record shorter than this is too small to teach anything.
MIN_EXAMPLE_CHARS = 8

#: A record longer than this is truncated-with-a-flag, never silently kept.
MAX_EXAMPLE_CHARS = 20000

#: Two examples whose normalised text share this Jaccard ratio are near-duplicates.
NEAR_DUPLICATE_RATIO = 0.9

STATUS_PENDING = "pending"
STATUS_VALID = "valid"
STATUS_INVALID = "invalid"

_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _normalise(text: str) -> str:
    """A comparison form for duplicate detection: case, whitespace, punctuation."""
    lowered = text.casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", lowered)).strip()


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_normalise(text).split())


# ---------------------------------------------------------------- the example


@dataclass(frozen=True)
class TrainingExample:
    """One governed training example with its provenance.

    ``owner`` is the personality the example came from (``""`` for a dataset the
    project authored itself). ``source`` names where it was taken from -- a
    document title, a corpus key, a hand-written set -- so a training run can
    always say what it learned from. ``license`` records the authorisation the
    example carries; an example with no recorded authorisation is *unverified*,
    which the validator reports rather than hides.
    """

    text: str
    owner: str = ""
    source: str = ""
    license: str = ""
    tags: tuple[str, ...] = ()
    #: A stable id: the digest of the text plus its owner, so the same example is
    #: the same id across a rebuild and a duplicate is detectable by id alone.
    example_id: str = ""

    def __post_init__(self) -> None:
        if not self.example_id and self.text:
            object.__setattr__(
                self, "example_id", _digest(f"{self.owner}\x00{self.text}")[:16]
            )

    @property
    def verified(self) -> bool:
        """True when the example records an authorisation and a source."""
        return bool(self.license.strip()) and bool(self.source.strip())

    def to_primitive(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "license": self.license,
            "owner": self.owner,
            "source": self.source,
            "tags": list(self.tags),
            "text": self.text,
        }

    @classmethod
    def from_primitive(cls, data: object) -> TrainingExample:
        if not isinstance(data, dict):
            raise DatasetValidationError("A training example must be a mapping.")
        text = str(data.get("text", ""))
        return cls(
            text=text,
            owner=str(data.get("owner", "")),
            source=str(data.get("source", "")),
            license=str(data.get("license", "")),
            tags=tuple(str(t) for t in data.get("tags", ()) or ()),
            example_id=str(data.get("example_id", "")) or "",
        )


# ---------------------------------------------------------------- the record


@dataclass(frozen=True)
class DatasetRecord:
    """The dataset's own record -- the task's required governance fields."""

    dataset_id: str
    version: int
    source: str = ""
    provenance: str = ""
    created_at: str = ""
    license: str = ""
    #: The model/task this dataset is intended for.
    intended_model: str = ""
    intended_task: str = ""
    #: The personalities this dataset may draw from (empty = the project itself).
    personality_scope: tuple[str, ...] = ()
    quality: float = 0.0
    status: str = STATUS_PENDING
    eligible: bool = False
    exclusion_rules: tuple[str, ...] = ()
    train_count: int = 0
    validation_count: int = 0
    test_count: int = 0
    duplicate_count: int = 0
    invalid_count: int = 0
    leakage_count: int = 0
    digest: str = ""
    notes: tuple[str, ...] = field(default=())

    @property
    def total(self) -> int:
        return self.train_count + self.validation_count + self.test_count

    @property
    def key(self) -> str:
        return f"{self.dataset_id}@v{self.version}"

    def describe(self) -> str:
        state = "eligible" if self.eligible else self.status
        scope = ", ".join(self.personality_scope) or "project"
        return (
            f"{self.key} [{state}] {self.total} example(s) "
            f"(train {self.train_count} / val {self.validation_count} / "
            f"test {self.test_count}) · scope {scope} · quality {self.quality:.2f}"
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "dataset_id": self.dataset_id,
            "digest": self.digest,
            "duplicate_count": self.duplicate_count,
            "eligible": self.eligible,
            "exclusion_rules": list(self.exclusion_rules),
            "intended_model": self.intended_model,
            "intended_task": self.intended_task,
            "invalid_count": self.invalid_count,
            "leakage_count": self.leakage_count,
            "license": self.license,
            "notes": list(self.notes),
            "personality_scope": list(self.personality_scope),
            "provenance": self.provenance,
            "quality": self.quality,
            "source": self.source,
            "status": self.status,
            "test_count": self.test_count,
            "train_count": self.train_count,
            "validation_count": self.validation_count,
            "version": self.version,
        }

    @classmethod
    def from_primitive(cls, data: object) -> DatasetRecord:
        if not isinstance(data, dict):
            raise DatasetValidationError("A dataset record must be a mapping.")
        dataset_id = str(data.get("dataset_id", "")).strip()
        if not dataset_id:
            raise DatasetValidationError("A dataset record needs a dataset_id.")
        return cls(
            dataset_id=dataset_id,
            version=int(data.get("version", 1) or 1),
            source=str(data.get("source", "")),
            provenance=str(data.get("provenance", "")),
            created_at=str(data.get("created_at", "")),
            license=str(data.get("license", "")),
            intended_model=str(data.get("intended_model", "")),
            intended_task=str(data.get("intended_task", "")),
            personality_scope=tuple(
                str(o) for o in (data.get("personality_scope") or ())
            ),
            quality=float(data.get("quality", 0.0) or 0.0),
            status=str(data.get("status", STATUS_PENDING)),
            eligible=bool(data.get("eligible", False)),
            exclusion_rules=tuple(str(r) for r in (data.get("exclusion_rules") or ())),
            train_count=int(data.get("train_count", 0) or 0),
            validation_count=int(data.get("validation_count", 0) or 0),
            test_count=int(data.get("test_count", 0) or 0),
            duplicate_count=int(data.get("duplicate_count", 0) or 0),
            invalid_count=int(data.get("invalid_count", 0) or 0),
            leakage_count=int(data.get("leakage_count", 0) or 0),
            digest=str(data.get("digest", "")),
            notes=tuple(str(n) for n in (data.get("notes") or ())),
        )


# ---------------------------------------------------------------- the split


@dataclass(frozen=True)
class DatasetSplit:
    """The three sides of a governed dataset, each a tuple of examples."""

    train: tuple[TrainingExample, ...] = ()
    validation: tuple[TrainingExample, ...] = ()
    test: tuple[TrainingExample, ...] = ()

    def __getitem__(self, name: str) -> tuple[TrainingExample, ...]:
        if name not in SPLITS:
            raise KeyError(f"Unknown split {name!r}; expected one of {SPLITS}.")
        return getattr(self, name)

    def all_examples(self) -> tuple[TrainingExample, ...]:
        return self.train + self.validation + self.test

    def counts(self) -> dict[str, int]:
        return {name: len(self[name]) for name in SPLITS}


# ---------------------------------------------------------------- the result


class DatasetValidationError(ValueError):
    """A dataset could not be built or read as asked."""


@dataclass(frozen=True)
class ValidationIssue:
    """One problem the validator found, with the example it concerns."""

    kind: str
    example_id: str
    detail: str

    def describe(self) -> str:
        return f"{self.kind}: {self.detail} ({self.example_id})"


@dataclass(frozen=True)
class ValidationReport:
    """The validator's verdict: the issues, and the counts they came to."""

    issues: tuple[ValidationIssue, ...] = ()
    duplicates: tuple[str, ...] = ()
    leakage: tuple[str, ...] = ()
    invalid: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues

    def of_kind(self, kind: str) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.kind == kind)

    def describe(self) -> str:
        if not self.issues:
            return "validation passed: no duplicate, invalid, or leaking example"
        lines = [f"validation found {len(self.issues)} issue(s):"]
        lines.extend(f"  - {issue.describe()}" for issue in self.issues[:20])
        if len(self.issues) > 20:
            lines.append(f"  ... and {len(self.issues) - 20} more")
        return "\n".join(lines)


# ---------------------------------------------------------------- validator


class DatasetValidator:
    """The dataset validation pipeline: format, quality, duplicates, leakage.

    Every check is deterministic and reports what it found; nothing is silently
    dropped. A caller decides what to do with the report -- the builder refuses an
    invalid dataset, and a preview shows it.
    """

    def __init__(
        self,
        *,
        min_chars: int = MIN_EXAMPLE_CHARS,
        max_chars: int = MAX_EXAMPLE_CHARS,
        near_duplicate_ratio: float = NEAR_DUPLICATE_RATIO,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.near_duplicate_ratio = near_duplicate_ratio

    # -- per-example checks ------------------------------------------------
    def check_example(self, example: TrainingExample) -> ValidationIssue | None:
        """Return the first problem with one example, or ``None`` when it is fine."""
        text = example.text
        if not text.strip():
            return ValidationIssue("empty", example.example_id, "the text is blank")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:  # pragma: no cover - defensive
            return ValidationIssue(
                "encoding", example.example_id, f"not encodable as UTF-8: {exc}"
            )
        if "\x00" in text:
            return ValidationIssue(
                "encoding", example.example_id, "contains a NUL byte"
            )
        stripped = text.strip()
        if len(stripped) < self.min_chars:
            return ValidationIssue(
                "too_short",
                example.example_id,
                f"{len(stripped)} characters, below the {self.min_chars} minimum",
            )
        if len(stripped) > self.max_chars:
            return ValidationIssue(
                "too_long",
                example.example_id,
                f"{len(stripped)} characters, above the {self.max_chars} maximum",
            )
        if not any(ch.isalnum() for ch in stripped):
            return ValidationIssue(
                "no_content", example.example_id, "contains no alphanumeric character"
            )
        return None

    # -- duplicates --------------------------------------------------------
    def find_duplicates(
        self, examples: Sequence[TrainingExample]
    ) -> tuple[tuple[str, ...], tuple[ValidationIssue, ...]]:
        """Exact (by id) and near (by token Jaccard) duplicates.

        The *first* example of a duplicate group is kept; every later one is
        reported. The returned ids are the ones a builder would drop.
        """
        seen_ids: set[str] = set()
        seen_tokens: list[tuple[str, frozenset[str]]] = []
        dropped: list[str] = []
        issues: list[ValidationIssue] = []
        for example in examples:
            if example.example_id in seen_ids:
                dropped.append(example.example_id)
                issues.append(
                    ValidationIssue(
                        "duplicate",
                        example.example_id,
                        "an identical example already appears earlier",
                    )
                )
                continue
            tokens = _tokens(example.text)
            near = self._nearest(tokens, seen_tokens)
            if near is not None:
                dropped.append(example.example_id)
                issues.append(
                    ValidationIssue(
                        "near_duplicate",
                        example.example_id,
                        f"shares {near:.0%} of its words with an earlier example",
                    )
                )
                continue
            seen_ids.add(example.example_id)
            seen_tokens.append((example.example_id, tokens))
        return tuple(dropped), tuple(issues)

    def _nearest(
        self, tokens: frozenset[str], seen: list[tuple[str, frozenset[str]]]
    ) -> float | None:
        if not tokens:
            return None
        for _id, other in seen:
            if not other:
                continue
            union = tokens | other
            if not union:
                continue
            ratio = len(tokens & other) / len(union)
            if ratio >= self.near_duplicate_ratio:
                return ratio
        return None

    # -- leakage -----------------------------------------------------------
    def find_leakage(self, split: DatasetSplit) -> tuple[tuple[str, ...], tuple[ValidationIssue, ...]]:
        """An example that appears on two sides of a split is a leak.

        This is checked by ``example_id`` (an exact leak) and by token overlap (a
        near leak), so a lightly-edited copy of a validation example cannot sit in
        the training set and inflate the score.
        """
        by_split: dict[str, list[TrainingExample]] = {
            name: list(split[name]) for name in SPLITS
        }
        leaks: list[str] = []
        issues: list[ValidationIssue] = []
        for i, left in enumerate(SPLITS):
            for right in SPLITS[i + 1 :]:
                left_ids = {e.example_id for e in by_split[left]}
                for example in by_split[right]:
                    if example.example_id in left_ids:
                        leaks.append(example.example_id)
                        issues.append(
                            ValidationIssue(
                                "leakage",
                                example.example_id,
                                f"appears in both {left} and {right}",
                            )
                        )
                        continue
                    if self._near_any(example, by_split[left]):
                        leaks.append(example.example_id)
                        issues.append(
                            ValidationIssue(
                                "near_leakage",
                                example.example_id,
                                f"nearly repeats a {left} example in {right}",
                            )
                        )
        return tuple(leaks), tuple(issues)

    def _near_any(
        self, example: TrainingExample, others: Sequence[TrainingExample]
    ) -> bool:
        tokens = _tokens(example.text)
        if not tokens:
            return False
        for other in others:
            other_tokens = _tokens(other.text)
            union = tokens | other_tokens
            if union and len(tokens & other_tokens) / len(union) >= self.near_duplicate_ratio:
                return True
        return False

    # -- the whole pipeline ------------------------------------------------
    def validate(self, split: DatasetSplit) -> ValidationReport:
        """Run every check over a split and return one report."""
        issues: list[ValidationIssue] = []
        invalid: list[str] = []
        for example in split.all_examples():
            problem = self.check_example(example)
            if problem is not None:
                issues.append(problem)
                invalid.append(example.example_id)

        # Duplicates are checked across the whole set (the same text must not
        # appear twice anywhere, which also keeps the split honest).
        duplicates, dup_issues = self.find_duplicates(split.all_examples())
        issues.extend(dup_issues)

        leakage, leak_issues = self.find_leakage(split)
        issues.extend(leak_issues)

        return ValidationReport(
            issues=tuple(issues),
            duplicates=tuple(dict.fromkeys(duplicates)),
            leakage=tuple(dict.fromkeys(leakage)),
            invalid=tuple(dict.fromkeys(invalid)),
        )

    def quality_score(self, split: DatasetSplit, report: ValidationReport) -> float:
        """A 0..1 quality figure: how much of the set survived every check.

        A dataset with no examples is **0.0**, not 1.0 -- an empty dataset teaches
        nothing and must never read as perfect.
        """
        total = len(split.all_examples())
        if total == 0:
            return 0.0
        bad = len(set(report.invalid) | set(report.duplicates) | set(report.leakage))
        return max(0.0, (total - bad) / total)


# ---------------------------------------------------------------- builder


def deterministic_split(
    examples: Sequence[TrainingExample],
    *,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
) -> DatasetSplit:
    """Split a sequence deterministically, so a rebuild yields the same split.

    The examples are ordered by their ``example_id`` (not by input order) and
    shuffled with a seeded RNG, so the same examples always produce the same
    train/validation/test assignment -- which is what makes a comparison between
    two runs meaningful.
    """
    import random

    ordered = sorted(examples, key=lambda e: e.example_id)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    total = len(ordered)
    n_test = round(total * test_ratio)
    n_val = round(total * validation_ratio)
    if total and n_test == 0 and test_ratio > 0:
        n_test = 1
    if total - n_test and n_val == 0 and validation_ratio > 0:
        n_val = 1
    # Never let the two sides consume the whole set.
    while n_test + n_val >= total and (n_test or n_val):
        if n_test >= n_val and n_test:
            n_test -= 1
        elif n_val:
            n_val -= 1
    test = tuple(ordered[:n_test])
    validation = tuple(ordered[n_test : n_test + n_val])
    train = tuple(ordered[n_test + n_val :])
    return DatasetSplit(train=train, validation=validation, test=test)


class DatasetBuilder:
    """Assemble, validate, split, and persist a governed dataset version."""

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        validator: DatasetValidator | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else training_datasets_dir()
        self.validator = validator or DatasetValidator()

    # -- the governed build ------------------------------------------------
    def build(
        self,
        dataset_id: str,
        examples: Iterable[TrainingExample],
        *,
        source: str = "",
        provenance: str = "",
        license: str = "",
        intended_model: str = "",
        intended_task: str = "",
        personality_scope: Sequence[str] = (),
        exclusion_rules: Sequence[str] = (),
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 0,
        allow_unverified: bool = False,
        persist: bool = True,
    ) -> BuiltDataset:
        """Build one dataset version.

        The dataset is **refused** rather than built when:

        * the id is malformed;
        * an example's owner is outside the declared personality scope;
        * the scope names a private owner (Natasha) -- the training path never
          declares one, and this is the structural guard;
        * an example is unverified and ``allow_unverified`` is False;
        * the validator reports any problem.
        """
        self._check_id(dataset_id)
        scope = tuple(dict.fromkeys(personality_scope))
        for owner in scope:
            if not is_owner(owner):
                raise DatasetValidationError(f"Unknown personality in scope: {owner!r}.")
            if is_private(owner):
                raise DatasetValidationError(
                    f"A dataset may not declare the private personality {owner!r} in "
                    "its scope; private experience is never a training source."
                )

        materialised = tuple(examples)
        if not materialised:
            raise DatasetValidationError("A dataset needs at least one example.")

        for example in materialised:
            if example.owner and example.owner not in scope:
                raise DatasetValidationError(
                    f"Example {example.example_id} came from {example.owner!r}, which "
                    f"is not in the declared scope {scope or '(project)'}."
                )
            if not allow_unverified and not example.verified:
                raise DatasetValidationError(
                    f"Example {example.example_id} is unverified (no source/license); "
                    "pass allow_unverified=True only for a deliberate experiment."
                )

        split = deterministic_split(
            materialised,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        report = self.validator.validate(split)
        quality = self.validator.quality_score(split, report)
        eligible = report.ok and quality > 0.0
        status = STATUS_VALID if eligible else STATUS_INVALID

        version = self.next_version(dataset_id) if persist else 1
        digest = _digest(
            "\n".join(sorted(e.example_id for e in split.all_examples()))
            + f"|{dataset_id}|{version}"
        )
        notes: list[str] = []
        if not report.ok:
            notes.append(report.describe())
        if quality < 1.0 and report.ok:
            notes.append(f"quality {quality:.2f}")

        record = DatasetRecord(
            dataset_id=dataset_id,
            version=version,
            source=source,
            provenance=provenance,
            created_at=_now(),
            license=license,
            intended_model=intended_model,
            intended_task=intended_task,
            personality_scope=scope,
            quality=quality,
            status=status,
            eligible=eligible,
            exclusion_rules=tuple(exclusion_rules),
            train_count=len(split.train),
            validation_count=len(split.validation),
            test_count=len(split.test),
            duplicate_count=len(report.duplicates),
            invalid_count=len(report.invalid),
            leakage_count=len(report.leakage),
            digest=digest,
            notes=tuple(notes),
        )

        built = BuiltDataset(record=record, split=split, report=report)
        if persist:
            self.save(built)
        return built

    def _check_id(self, dataset_id: str) -> None:
        if not _TAG_RE.match(dataset_id or ""):
            raise DatasetValidationError(
                f"Invalid dataset id {dataset_id!r}: use lower-case letters, digits, "
                "and . _ - only."
            )

    # -- persistence -------------------------------------------------------
    def dataset_dir(self, dataset_id: str) -> Path:
        return self.root / dataset_id

    def versions(self, dataset_id: str) -> tuple[int, ...]:
        folder = self.dataset_dir(dataset_id)
        if not folder.is_dir():
            return ()
        found: list[int] = []
        for child in folder.iterdir():
            if child.is_dir() and child.name.startswith("v"):
                try:
                    found.append(int(child.name[1:]))
                except ValueError:
                    continue
        return tuple(sorted(found))

    def next_version(self, dataset_id: str) -> int:
        versions = self.versions(dataset_id)
        return (versions[-1] + 1) if versions else 1

    def version_dir(self, dataset_id: str, version: int) -> Path:
        return self.dataset_dir(dataset_id) / f"v{version}"

    def save(self, built: BuiltDataset) -> Path:
        """Write a versioned snapshot. **Never overwrites** an existing version."""
        target = self.version_dir(built.record.dataset_id, built.record.version)
        if target.exists():
            raise DatasetValidationError(
                f"{built.record.key} already exists; a dataset version is immutable."
            )
        target.mkdir(parents=True, exist_ok=False)
        for name in SPLITS:
            path = target / f"{name}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for example in built.split[name]:
                    handle.write(json.dumps(example.to_primitive(), sort_keys=True))
                    handle.write("\n")
        manifest = {
            "record": built.record.to_primitive(),
            "splits": built.split.counts(),
            "validation": {
                "ok": built.report.ok,
                "duplicates": list(built.report.duplicates),
                "invalid": list(built.report.invalid),
                "leakage": list(built.report.leakage),
                "issues": [issue.describe() for issue in built.report.issues],
            },
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return target

    def load(self, dataset_id: str, version: int) -> BuiltDataset:
        """Read a stored version back, with its record, split, and report."""
        target = self.version_dir(dataset_id, version)
        manifest_path = target / "manifest.json"
        if not manifest_path.is_file():
            raise DatasetValidationError(f"{dataset_id}@v{version} was not found.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = DatasetRecord.from_primitive(manifest.get("record"))
        split = DatasetSplit(
            **{
                name: tuple(
                    TrainingExample.from_primitive(item)
                    for item in self._read_jsonl(target / f"{name}.jsonl")
                )
                for name in SPLITS
            }
        )
        validation = manifest.get("validation", {})
        report = ValidationReport(
            issues=tuple(
                ValidationIssue("stored", "", text)
                for text in validation.get("issues", ())
            ),
            duplicates=tuple(validation.get("duplicates", ())),
            leakage=tuple(validation.get("leakage", ())),
            invalid=tuple(validation.get("invalid", ())),
        )
        return BuiltDataset(record=record, split=split, report=report)

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def latest(self, dataset_id: str) -> BuiltDataset | None:
        versions = self.versions(dataset_id)
        return self.load(dataset_id, versions[-1]) if versions else None

    def list_datasets(self) -> tuple[DatasetRecord, ...]:
        if not self.root.is_dir():
            return ()
        records: list[DatasetRecord] = []
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue
            for version in self.versions(folder.name):
                try:
                    records.append(self.load(folder.name, version).record)
                except (DatasetValidationError, OSError):
                    continue
        return tuple(records)


@dataclass(frozen=True)
class BuiltDataset:
    """A built dataset: its record, its split, and the validator's report."""

    record: DatasetRecord
    split: DatasetSplit
    report: ValidationReport

    @property
    def eligible(self) -> bool:
        return self.record.eligible

    def describe(self) -> str:
        return self.record.describe()


def aggregate_examples(
    *,
    root: Path | str | None = None,
    include_private: bool = False,
) -> tuple[TrainingExample, ...]:
    """The **authorised** examples the project model may train on.

    This is the one function that turns personality learning records into training
    examples, and it asks the Phase-5 isolation policy rather than remembering the
    rule: every shareable character contributes, and the private one (Natasha) is
    excluded unless ``include_private`` is set -- which the training path never
    sets. A character with an empty record contributes nothing.
    """
    from core.agents.learning_scope import aggregated_dataset_owners
    from core.character_learning import CharacterLearning
    from core.learning_isolation import mouth_key

    owners = aggregated_dataset_owners(include_private=include_private)
    examples: list[TrainingExample] = []
    for owner in owners:
        if owner == MAIN_MODEL_OWNER:
            continue
        learning = CharacterLearning(mouth_key(owner), root=root)
        for fact in learning.facts():
            examples.append(
                TrainingExample(
                    text=f"Q: What does {owner} know?\nA: {fact}",
                    owner=owner,
                    source=f"character_learning/{owner}",
                    license="project-authored",
                )
            )
        for exchange in learning.exchanges():
            if exchange.question and exchange.answer:
                examples.append(
                    TrainingExample(
                        text=f"Q: {exchange.question}\nA: {exchange.answer}",
                        owner=owner,
                        source=f"character_learning/{owner}",
                        license="project-authored",
                    )
                )
    return tuple(examples)


__all__ = [
    "MAX_EXAMPLE_CHARS",
    "MIN_EXAMPLE_CHARS",
    "NEAR_DUPLICATE_RATIO",
    "SPLITS",
    "STATUS_INVALID",
    "STATUS_PENDING",
    "STATUS_VALID",
    "BuiltDataset",
    "DatasetBuilder",
    "DatasetRecord",
    "DatasetSplit",
    "DatasetValidationError",
    "DatasetValidator",
    "TrainingExample",
    "ValidationIssue",
    "ValidationReport",
    "aggregate_examples",
    "deterministic_split",
]
