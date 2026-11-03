"""The perception result validator (section 3).

A validator is what keeps a ``PerceptionResult`` honest: it checks the shape's
invariants (a non-``OK`` status must have a reason; content and segments must
agree with the status; a confidence must be in range), and it reports what is
*misleading* rather than merely malformed -- an ``OK`` result whose content is
whitespace, a confidence of ``1.0`` on OCR text, a status of ``FAILED`` with no
error message.

It also answers the availability question every surface needs: what can this
build actually perceive *right now*? :func:`perception_status` reads the optional
features (:mod:`core.optional_deps`) and the speech/image engine probes, and
returns a :class:`PerceptionStatus` -- a report, never an exception -- so a window
or a CLI can tell a user the real next step instead of failing.

Dependency-free and Tk-free: it imports no engine, only asks each one whether it
is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from perception.schema import (
    ContentType,
    PerceptionResult,
)
from perception.schema import (
    PerceptionStatus as ResultStatus,
)

#: A quality ceiling for extracted-by-machine content. OCR and ASR are good but
#: never certain, so a result that claims 1.0 for either is flagged: the number
#: is a quality signal, and machine reading is never perfect.
MACHINE_QUALITY_CEILING = 0.95


@dataclass(frozen=True)
class ValidationFinding:
    """One problem the validator found, with its severity."""

    message: str
    blocking: bool = False
    """A blocking finding means the result must not be used as evidence."""

    def describe(self) -> str:
        return f"{'blocking' if self.blocking else 'warning'}: {self.message}"


@dataclass(frozen=True)
class ValidationVerdict:
    """The validator's conclusion about one result."""

    ok: bool
    findings: tuple[ValidationFinding, ...] = ()

    @property
    def blocking(self) -> tuple[ValidationFinding, ...]:
        return tuple(finding for finding in self.findings if finding.blocking)

    @property
    def warnings(self) -> tuple[ValidationFinding, ...]:
        return tuple(finding for finding in self.findings if not finding.blocking)

    def describe(self) -> str:
        if self.ok and not self.findings:
            return "valid"
        head = "valid with warnings" if self.ok else "invalid"
        return head + ": " + "; ".join(finding.describe() for finding in self.findings)


def validate_result(result: PerceptionResult) -> ValidationVerdict:
    """Check one perception result for honesty and internal consistency.

    Returns a verdict (never raises), so a caller can log a warning or refuse to
    publish a bad result. A result is *invalid* only when it is internally
    contradictory; a merely suspicious value is a warning.
    """
    findings: list[ValidationFinding] = []

    if result.status is ResultStatus.OK and not result.has_content:
        findings.append(
            ValidationFinding("status OK but the result carries no content", blocking=True)
        )
    if result.status is ResultStatus.FAILED and not result.errors:
        findings.append(
            ValidationFinding("status FAILED but no error explains it", blocking=True)
        )
    if result.status is ResultStatus.UNSUPPORTED and not result.errors:
        findings.append(
            ValidationFinding("status UNSUPPORTED but no reason names the format", blocking=True)
        )
    if result.status is ResultStatus.PARTIAL and not result.warnings and not result.errors:
        findings.append(
            ValidationFinding("status PARTIAL but nothing names what was skipped", blocking=True)
        )
    if result.content.strip() and result.content.strip() != result.content:
        findings.append(
            ValidationFinding("content has leading/trailing whitespace; normalise it")
        )
    if result.content_type is ContentType.MODEL and result.status is ResultStatus.OK:
        findings.append(
            ValidationFinding(
                "a model description is not ground truth; keep it labelled as a model's reading"
            )
        )
    if (
        result.content_type in (ContentType.OCR, ContentType.ASR)
        and result.confidence > MACHINE_QUALITY_CEILING
    ):
        findings.append(
            ValidationFinding(
                f"{result.content_type.value} confidence {result.confidence:.2f} exceeds the "
                f"machine-reading ceiling {MACHINE_QUALITY_CEILING:.2f}"
            )
        )
    if result.content and not result.provenance:
        findings.append(
            ValidationFinding("content carries no provenance; a result must name its source")
        )
    if result.segments and result.content:
        joined = "\n\n".join(segment.text for segment in result.segments if segment.has_text)
        if joined.strip() and joined.strip() not in result.content and result.content.strip() not in joined:
            findings.append(
                ValidationFinding("the content and the segments disagree about what was read")
            )

    blocking = any(finding.blocking for finding in findings)
    return ValidationVerdict(ok=not blocking, findings=tuple(findings))


# --------------------------------------------------------------------------- #
# What this build can perceive right now
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EngineAvailability:
    """Whether one perception engine is usable, and the real next step if not."""

    name: str
    available: bool
    detail: str = ""
    hint: str = ""

    def describe(self) -> str:
        if self.available:
            return f"{self.name}: available. {self.detail}".strip()
        return f"{self.name}: not available. {self.detail} {self.hint}".strip()


@dataclass(frozen=True)
class PerceptionStatus:
    """An honest report of every perception engine this build can use."""

    engines: tuple[EngineAvailability, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def get(self, name: str) -> EngineAvailability | None:
        for engine in self.engines:
            if engine.name == name:
                return engine
        return None

    def available_names(self) -> tuple[str, ...]:
        return tuple(engine.name for engine in self.engines if engine.available)

    def describe(self) -> str:
        lines = ["Perception engines:"]
        lines.extend(f"  - {engine.describe()}" for engine in self.engines)
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def perception_status() -> PerceptionStatus:
    """Probe every optional perception engine and report what really works here.

    Never raises: a missing feature is reported as unavailable with its next step.
    The probe imports no heavy module -- it only asks each layer's own availability
    function, which the project already exposes. The model-backed engines (vision,
    OCR, speech) come from :func:`perception.models.perception_model_status` so the
    two surfaces can never disagree; the document reader is added here.
    """
    engines: list[EngineAvailability] = []

    from core.pdf_extract import pdf_available

    engines.append(
        EngineAvailability(
            "pdf-text",
            pdf_available(),
            "digital PDF text extraction" if pdf_available() else "the [pdf] extra is not installed.",
            "python -m pip install -r requirements-pdf.txt" if not pdf_available() else "",
        )
    )

    from perception.models import perception_model_status

    for model in perception_model_status():
        engines.append(EngineAvailability(model.name, model.available, model.detail, model.hint))

    notes = [
        "No engine is mandatory: document text extraction works on a base install; "
        "OCR, vision, and speech each report their own next step."
    ]
    return PerceptionStatus(engines=tuple(engines), notes=tuple(notes))


__all__ = [
    "MACHINE_QUALITY_CEILING",
    "EngineAvailability",
    "PerceptionStatus",
    "ValidationFinding",
    "ValidationVerdict",
    "perception_status",
    "validate_result",
]
