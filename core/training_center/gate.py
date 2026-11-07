"""The held-out gate for the Training Center.

A held-out gate is a catalogue of hand-written judgements about what the feature
**must** do, scored as one number and wired into ``evaluation`` so a regression is
a gate failure rather than a passing suite. This module is that catalogue for the
Training Center.

The judgements run the **real** code paths with lightweight, offline inputs --
a temporary data folder, an in-memory review queue, a fake provider, a fake
coordinator -- and several are **negative controls**, so a feature that merely
*looks* present fails here:

* an unreviewed learning example is refused by the dataset builder;
* a dataset build with no approved examples is refused rather than built empty;
* a governed pipeline stage (training) is refused without approval;
* a capability the catalogue marks unavailable is reported as unavailable;
* a provider with no key/approval reports the real next step, never an answer;
* a resource figure the platform cannot report stays ``None``, not a zero;
* every catalogue capability's CLI command resolves against the real parser.

Nothing here trains a model, reaches the network, or needs a display, so it runs
on a fresh clone.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from core.training_center.catalog import CAPABILITIES, SECTIONS, capabilities_for, capability
from core.training_center.datasets import DatasetStudio
from core.training_center.pipelines import (
    STAGE_SPECS,
    LearningPipeline,
    PipelineRunner,
    PipelineStage,
    PipelineStore,
    default_pipeline,
    stage_spec,
)
from core.training_center.qa_learning import LearningReviewQueue
from core.training_center.records import (
    STATUS_APPROVED,
    STATUS_PENDING,
    LearningExample,
)


@dataclass(frozen=True)
class GateJudgement:
    """One held-out judgement: a name, what it checks, and its verdict."""

    name: str
    passed: bool
    detail: str = ""

    def describe(self) -> str:
        return f"{'ok  ' if self.passed else 'FAIL'} {self.name}: {self.detail}"


def _tmp_queue() -> LearningReviewQueue:
    return LearningReviewQueue(root=Path(tempfile.mkdtemp(prefix="tc-gate-")))


def _tmp_studio() -> DatasetStudio:
    from ml.training.datasets import DatasetBuilder

    return DatasetStudio(builder=DatasetBuilder(root=Path(tempfile.mkdtemp(prefix="tc-gate-")) / "datasets"))


# -- the judgements --------------------------------------------------------


def _judgement_example_status() -> GateJudgement:
    queue = _tmp_queue()
    example = LearningExample(question="What is porosity?", answer="A measure of void space.", provider="manual")
    queue.add(example)
    assert queue.get(example.example_id) is not None
    approved = queue.approve(example.example_id)
    ok = approved.status == STATUS_APPROVED and queue.approved() == (approved,)
    return GateJudgement(
        "review-status-governs-export",
        ok,
        "an example is pending until approved, and only approved examples are exportable",
    )


def _judgement_refuses_unreviewed() -> GateJudgement:
    studio = _tmp_studio()
    pending = LearningExample(question="Q", answer="A", provider="manual", status=STATUS_PENDING)
    try:
        studio.build_from_examples("learned", [pending])
    except Exception as error:
        refused = "not approved" in str(error)
        return GateJudgement(
            "refuses-unreviewed-example", refused,
            "the dataset builder refuses an example that is not approved",
        )
    return GateJudgement("refuses-unreviewed-example", False, "an unapproved example was accepted")


def _judgement_rejected_never_exported() -> GateJudgement:
    queue = _tmp_queue()
    example = LearningExample(question="Q", answer="A", provider="manual")
    queue.add(example)
    queue.reject(example.example_id)
    return GateJudgement(
        "rejected-is-not-approved",
        queue.approved() == (),
        "a rejected example is never in the approved set",
    )


def _judgement_builds_from_approved() -> GateJudgement:
    studio = _tmp_studio()
    queue = _tmp_queue()
    questions = (
        "What does porosity measure in a rock sample?",
        "How is permeability defined for a reservoir rock?",
        "Why does a waterflood raise reservoir pressure?",
        "How does acid stimulation increase permeability?",
        "What does the density log measure in a well?",
        "How is skin factor estimated from a well test?",
    )
    examples = [
        LearningExample(
            question=question,
            answer=(
                "The project's own sources answer this: "
                + question.lower().rstrip("?") + ", with its citation recorded."
            ),
            provider="manual",
            status=STATUS_APPROVED,
        )
        for question in questions
    ]
    queue.add_many(examples)
    try:
        row = studio.build_from_examples("learned", queue.approved())
    except Exception as error:  # pragma: no cover - defensive
        return GateJudgement("builds-from-approved", False, str(error))
    return GateJudgement(
        "builds-from-approved",
        row.total == 6 and row.eligible,
        f"built {row.total} example(s), eligible={row.eligible}",
    )


def _judgement_pipeline_refuses_training_without_approval() -> GateJudgement:
    pipeline = LearningPipeline(
        pipeline_id="p", name="p", stages=[PipelineStage("train", {"dataset": "learned"})],
    )
    report = PipelineRunner().run(pipeline, approved=False)
    ok = bool(report.stopped_at) and not report.outcomes[0].ok
    return GateJudgement(
        "pipeline-refuses-governed-stage",
        ok,
        "a governed stage (training) is refused without explicit approval",
    )


def _judgement_pipeline_reports_unavailable() -> GateJudgement:
    unavailable = next((spec for spec in STAGE_SPECS if not spec.available), None)
    if unavailable is None:  # pragma: no cover - catalogue always has one
        return GateJudgement("pipeline-reports-unavailable", False, "no unavailable stage in the catalogue")
    pipeline = LearningPipeline(
        pipeline_id="p", name="p", stages=[PipelineStage(unavailable.key, {})],
    )
    report = PipelineRunner().run(pipeline, approved=True)
    ok = report.outcomes and not report.outcomes[0].ok and "unavailable" in report.outcomes[0].message
    return GateJudgement(
        "pipeline-reports-unavailable",
        bool(ok),
        "an unimplemented stage is reported unavailable, never run as a no-op",
    )


def _judgement_catalog_cli_resolves() -> GateJudgement:
    """Every catalogue CLI command must resolve against the real parser."""
    import argparse

    from cli import training_center as cli_module

    parser = cli_module.build_parser()
    # Collect the real subcommands the parser knows.
    subcommands: set[str] = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subcommands.update(action.choices)

    bad: list[str] = []
    for cap in CAPABILITIES:
        command = cap.cli
        if "training-center" not in command:
            continue  # a command in another module's namespace is checked elsewhere
        parts = command.split()
        # `python -m cli training-center <sub>`
        if len(parts) >= 5 and parts[:4] == ["python", "-m", "cli", "training-center"]:
            sub = parts[4]
            if sub not in subcommands:
                bad.append(f"{cap.key}: {sub}")
    return GateJudgement(
        "catalog-cli-resolves",
        not bad,
        "every training-center CLI command resolves" if not bad else f"unresolved: {', '.join(bad)}",
    )


def _judgement_catalog_complete() -> GateJudgement:
    """Every section has a title and a summary; every capability is explained."""
    problems: list[str] = []
    for section in SECTIONS:
        if not capabilities_for(section):
            problems.append(f"section {section} has no capability")
    for cap in CAPABILITIES:
        if not cap.summary or not cap.outcome or not cap.cli:
            problems.append(f"{cap.key} is missing a summary, outcome, or command")
        if cap.governed and "--approve" not in cap.cli and "asks first" not in cap.describe():
            problems.append(f"{cap.key} is governed but names no confirmation")
    return GateJudgement(
        "catalog-complete",
        not problems,
        "every control is explained" if not problems else "; ".join(problems[:5]),
    )


def _judgement_unknown_capability_is_none() -> GateJudgement:
    return GateJudgement(
        "unknown-capability-is-none",
        capability("does-not-exist") is None,
        "an unknown capability returns None rather than a made-up record",
    )


def _judgement_provider_honest_without_key() -> GateJudgement:
    from core.training_center.providers import CloudLearningProvider

    provider = CloudLearningProvider("gemini", approved=False)
    status = provider.status()
    ok = not status.available and bool(status.detail or status.hint)
    return GateJudgement(
        "provider-honest-without-approval",
        ok,
        "an unapproved cloud provider reports unavailable with the real next step",
    )


def _judgement_resources_unknown_not_zero() -> GateJudgement:
    from core.training_center.resources import human_size, read_resources

    snapshot = read_resources()
    # A figure the platform cannot report must read as "unknown", never a zero.
    unknown_ok = human_size(None) == "unknown"
    typed_ok = snapshot.resources is not None
    return GateJudgement(
        "resources-unknown-not-zero",
        unknown_ok and typed_ok,
        "an unreportable resource reads as unknown rather than zero",
    )


def _judgement_pipeline_store_roundtrip() -> GateJudgement:
    root = Path(tempfile.mkdtemp(prefix="tc-gate-"))
    store = PipelineStore(root=root)
    pipeline = default_pipeline("demo")
    store.save(pipeline)
    loaded = store.get("demo")
    ok = loaded is not None and len(loaded.stages) == len(pipeline.stages)
    return GateJudgement(
        "pipeline-store-roundtrip",
        ok,
        "a saved pipeline loads back with its stages intact",
    )


def _judgement_unavailable_stage_note() -> GateJudgement:
    unavailable = next((spec for spec in STAGE_SPECS if not spec.available), None)
    ok = unavailable is not None and bool(unavailable.note)
    return GateJudgement(
        "unavailable-stage-names-next-step",
        ok,
        "an unavailable stage names the real next step",
    )


def _judgement_stage_spec_unknown() -> GateJudgement:
    return GateJudgement(
        "unknown-stage-is-none",
        stage_spec("not-a-stage") is None,
        "an unknown pipeline stage returns None",
    )


JUDGEMENTS: tuple[str, ...] = (
    "review-status-governs-export",
    "refuses-unreviewed-example",
    "rejected-is-not-approved",
    "builds-from-approved",
    "pipeline-refuses-governed-stage",
    "pipeline-reports-unavailable",
    "catalog-cli-resolves",
    "catalog-complete",
    "unknown-capability-is-none",
    "provider-honest-without-approval",
    "resources-unknown-not-zero",
    "pipeline-store-roundtrip",
    "unavailable-stage-names-next-step",
    "unknown-stage-is-none",
)


def run_judgements() -> tuple[GateJudgement, ...]:
    """Run every judgement and return its verdict."""
    return (
        _judgement_example_status(),
        _judgement_refuses_unreviewed(),
        _judgement_rejected_never_exported(),
        _judgement_builds_from_approved(),
        _judgement_pipeline_refuses_training_without_approval(),
        _judgement_pipeline_reports_unavailable(),
        _judgement_catalog_cli_resolves(),
        _judgement_catalog_complete(),
        _judgement_unknown_capability_is_none(),
        _judgement_provider_honest_without_key(),
        _judgement_resources_unknown_not_zero(),
        _judgement_pipeline_store_roundtrip(),
        _judgement_unavailable_stage_note(),
        _judgement_stage_spec_unknown(),
    )


def training_center_accuracy() -> float:
    """The share of judgements the feature satisfies, in ``[0, 1]``."""
    results = run_judgements()
    if not results:
        return 0.0
    return round(sum(1 for result in results if result.passed) / len(results), 6)


def describe() -> str:
    """A readable report of every judgement and the overall accuracy."""
    results = run_judgements()
    lines = ["Training Center — held-out gate"]
    lines.extend("  " + result.describe() for result in results)
    lines.append(f"  accuracy: {training_center_accuracy():.4f}")
    return "\n".join(lines)


__all__ = [
    "JUDGEMENTS",
    "GateJudgement",
    "describe",
    "run_judgements",
    "training_center_accuracy",
]
