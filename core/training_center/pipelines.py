"""The visual pipeline model: saved stages run in order, bounded and resumable.

A learning workflow is a sequence of stages -- search, download, extract, clean,
deduplicate, generate questions, ask a model, review, build a dataset, validate,
train, evaluate, export. The Training Center lets a user assemble such a sequence
and run it, so the whole workflow is one action instead of a dozen.

The model is honest by construction:

* every stage names whether it is **really available** here; a stage the project
  does not implement is reported as unavailable with the real next step, never
  run as a no-op that pretends to have worked;
* a stage that starts something expensive (a training run) requires explicit
  approval, so a pipeline can never silently launch a long job;
* a stage's failure is recorded and, by default, stops the run -- a partial
  pipeline reports exactly where it stopped and what it produced.

The stages delegate to the project's own subsystems (the importer, the retriever,
the governed dataset builder, the controlled trainer, the evaluator), so nothing
is re-implemented. The store is durable through
:class:`storage.json_store.JsonStore`. Dependency-free and Tk-free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from storage.json_store import JsonStore

from core.paths import data_dir

#: The JsonStore namespace and key the saved pipelines live under.
NAMESPACE = "training_center"
KEY = "pipelines"

# -- the stage catalogue --------------------------------------------------

STAGE_IMPORT_FOLDER = "import_folder"
STAGE_IMPORT_URLS = "import_urls"
STAGE_BUILD_DATASET = "build_dataset"
STAGE_VALIDATE_DATASET = "validate_dataset"
STAGE_TRAIN = "train"
STAGE_EVALUATE = "evaluate"
STAGE_EXPORT_DATASET = "export_dataset"
STAGE_KNOWLEDGE_SEARCH = "knowledge_search"
# Stages the project folds into ingestion (they run, but inside the importer) --
# kept as named, honest stages so a pipeline reads the way a user thinks.
STAGE_EXTRACT = "extract"
STAGE_CLEAN = "clean"
STAGE_DEDUPLICATE = "deduplicate"
# Stages the project does not implement as a standalone action here.
STAGE_GENERATE_QUESTIONS = "generate_questions"
STAGE_ASK_MODEL = "ask_model"
STAGE_EXPORT_MODEL = "export_model"


@dataclass(frozen=True)
class StageSpec:
    """One pipeline stage, described once so no surface invents it."""

    key: str
    title: str
    summary: str
    available: bool
    governed: bool = False
    note: str = ""


STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(
        STAGE_IMPORT_FOLDER, "Import a folder",
        "Import every supported file under a folder into the knowledge base.",
        available=True,
    ),
    StageSpec(
        STAGE_IMPORT_URLS, "Download URLs",
        "Fetch user-supplied URLs and import the extracted text.",
        available=True, governed=True,
        note="needs the web path approved",
    ),
    StageSpec(
        STAGE_EXTRACT, "Extract text",
        "Extract the text layer from each imported document.",
        available=True,
        note="runs inside the importer, per file",
    ),
    StageSpec(
        STAGE_CLEAN, "Clean and normalise",
        "Normalise Unicode and whitespace and (optionally) fold Arabic forms.",
        available=True,
        note="runs inside the importer, per file",
    ),
    StageSpec(
        STAGE_DEDUPLICATE, "Deduplicate",
        "Skip a source whose content hash was already imported.",
        available=True,
        note="runs inside the importer, per source",
    ),
    StageSpec(
        STAGE_KNOWLEDGE_SEARCH, "Search the knowledge base",
        "Run the project's own retrieval over what was just imported.",
        available=True,
    ),
    StageSpec(
        STAGE_BUILD_DATASET, "Build a dataset",
        "Build a governed dataset version from the approved learning examples.",
        available=True,
    ),
    StageSpec(
        STAGE_VALIDATE_DATASET, "Validate the dataset",
        "Run the governed validator and report every problem.",
        available=True,
    ),
    StageSpec(
        STAGE_TRAIN, "Train the model",
        "Start a governed training run on the built dataset.",
        available=True, governed=True,
        note="needs explicit approval and a compatible model",
    ),
    StageSpec(
        STAGE_EVALUATE, "Evaluate the model",
        "Measure the model on the dataset's held-out test split.",
        available=True,
    ),
    StageSpec(
        STAGE_EXPORT_DATASET, "Export the dataset",
        "Write the built dataset version to JSONL.",
        available=True,
    ),
    StageSpec(
        STAGE_GENERATE_QUESTIONS, "Generate questions",
        "Generate questions on a topic with a learning provider.",
        available=False,
        note="use Question-and-Answer Learning, which records the review status",
    ),
    StageSpec(
        STAGE_ASK_MODEL, "Ask a model",
        "Ask a learning provider a question and record the answer for review.",
        available=False,
        note="use Question-and-Answer Learning, which records the review status",
    ),
    StageSpec(
        STAGE_EXPORT_MODEL, "Export the model",
        "Write the trained adapter out for use elsewhere.",
        available=False,
        note="an adapter is already a portable directory; copy it from data/training/adapters",
    ),
)

_STAGE_BY_KEY: dict[str, StageSpec] = {spec.key: spec for spec in STAGE_SPECS}


def stage_spec(key: str) -> StageSpec | None:
    return _STAGE_BY_KEY.get(key)


def stage_names() -> tuple[str, ...]:
    return tuple(spec.key for spec in STAGE_SPECS)


# -- the pipeline model ---------------------------------------------------


@dataclass
class PipelineStage:
    """One stage in a pipeline, with its options and whether it is enabled."""

    stage: str
    options: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.stage not in _STAGE_BY_KEY:
            raise ValueError(f"Unknown pipeline stage {self.stage!r}; expected one of {stage_names()}.")

    @property
    def spec(self) -> StageSpec:
        return _STAGE_BY_KEY[self.stage]

    def to_primitive(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "options": dict(self.options), "stage": self.stage}

    @classmethod
    def from_primitive(cls, data: object) -> PipelineStage:
        if not isinstance(data, dict):
            raise ValueError("A pipeline stage must be a mapping.")
        return cls(
            stage=str(data.get("stage", "")),
            options=dict(data.get("options", {}) or {}),
            enabled=bool(data.get("enabled", True)),
        )

    def describe(self) -> str:
        state = "" if self.enabled else " (disabled)"
        return f"{self.spec.title}{state}"


@dataclass
class LearningPipeline:
    """A saved sequence of stages."""

    pipeline_id: str
    name: str
    stages: list[PipelineStage] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.pipeline_id.strip():
            raise ValueError("A pipeline needs an id.")
        if not self.created_at:
            from core.training_center.records import _now

            self.created_at = _now()

    @property
    def enabled_stages(self) -> tuple[PipelineStage, ...]:
        return tuple(stage for stage in self.stages if stage.enabled)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "name": self.name,
            "pipeline_id": self.pipeline_id,
            "stages": [stage.to_primitive() for stage in self.stages],
        }

    @classmethod
    def from_primitive(cls, data: object) -> LearningPipeline:
        if not isinstance(data, dict):
            raise ValueError("A pipeline must be a mapping.")
        return cls(
            pipeline_id=str(data.get("pipeline_id", "")),
            name=str(data.get("name", "")),
            stages=[PipelineStage.from_primitive(s) for s in data.get("stages", []) or []],
            created_at=str(data.get("created_at", "")),
        )

    def describe(self) -> str:
        steps = " → ".join(stage.spec.title for stage in self.enabled_stages) or "(no stages)"
        return f"{self.name}: {steps}"


# -- results --------------------------------------------------------------


@dataclass(frozen=True)
class StageOutcome:
    """What one stage did."""

    stage: str
    ok: bool
    message: str
    artifacts: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        mark = "ok" if self.ok else "failed"
        return f"[{mark}] {self.stage}: {self.message}"


@dataclass(frozen=True)
class PipelineReport:
    """The whole run's outcome, stage by stage."""

    pipeline_id: str
    outcomes: tuple[StageOutcome, ...]
    stopped_at: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.outcomes) and all(outcome.ok for outcome in self.outcomes) and not self.stopped_at

    def describe(self) -> str:
        lines = [f"Pipeline {self.pipeline_id}: {'ok' if self.ok else 'stopped'}"]
        lines.extend("  " + outcome.describe() for outcome in self.outcomes)
        if self.stopped_at:
            lines.append(f"  stopped at {self.stopped_at}")
        return "\n".join(lines)


# -- the runner -----------------------------------------------------------

#: A stage runner receives ``(stage, context)`` and returns a ``StageOutcome``.
StageRunner = Callable[[PipelineStage, dict[str, Any]], StageOutcome]


class PipelineRunner:
    """Run a pipeline's enabled stages in order, bounded and honest.

    ``runners`` maps a stage key to its implementation. A stage with no runner, or
    whose spec is unavailable, is reported as unavailable -- never silently
    skipped -- and stops the run. A governed stage (training) is refused unless
    ``approved`` is set.
    """

    def __init__(self, *, runners: dict[str, StageRunner] | None = None, context: dict[str, Any] | None = None) -> None:
        self._runners: dict[str, StageRunner] = dict(DEFAULT_RUNNERS)
        if runners:
            self._runners.update(runners)
        self.context: dict[str, Any] = dict(context or {})

    def run(
        self,
        pipeline: LearningPipeline,
        *,
        approved: bool = False,
        on_stage: Callable[[StageOutcome], None] | None = None,
        stop_on_failure: bool = True,
    ) -> PipelineReport:
        outcomes: list[StageOutcome] = []
        stopped_at = ""
        for stage in pipeline.enabled_stages:
            spec = stage.spec
            if not spec.available:
                outcome = StageOutcome(
                    stage=stage.stage, ok=False,
                    message=f"unavailable: {spec.note or 'not implemented here'}",
                )
            elif spec.governed and not approved:
                outcome = StageOutcome(
                    stage=stage.stage, ok=False,
                    message="needs explicit approval (re-run with --approve)",
                )
            else:
                runner = self._runners.get(stage.stage)
                if runner is None:
                    outcome = StageOutcome(
                        stage=stage.stage, ok=False, message="no runner is registered for this stage",
                    )
                else:
                    try:
                        outcome = runner(stage, self.context)
                    except Exception as error:
                        outcome = StageOutcome(
                            stage=stage.stage, ok=False,
                            message=f"{type(error).__name__}: {error}",
                        )
            outcomes.append(outcome)
            if on_stage is not None:
                on_stage(outcome)
            if not outcome.ok and stop_on_failure:
                stopped_at = stage.stage
                break
        return PipelineReport(pipeline_id=pipeline.pipeline_id, outcomes=tuple(outcomes), stopped_at=stopped_at)


# -- the real stage runners ----------------------------------------------


def _runner_import_folder(stage: PipelineStage, context: dict[str, Any]) -> StageOutcome:
    folder = str(stage.options.get("folder") or context.get("folder") or "")
    if not folder:
        return StageOutcome(stage=stage.stage, ok=False, message="no folder was given")
    from gui.import_center import KnowledgeImporter

    report = KnowledgeImporter(store_dir=context.get("store_dir")).import_folder(folder)
    context["last_import"] = report
    return StageOutcome(
        stage=stage.stage, ok=report.ok, message=report.summary(),
        artifacts={"chunks": report.chunks, "records": report.records},
    )


def _runner_import_urls(stage: PipelineStage, context: dict[str, Any]) -> StageOutcome:
    urls = list(stage.options.get("urls") or context.get("urls") or [])
    if not urls:
        return StageOutcome(stage=stage.stage, ok=False, message="no URLs were given")
    from gui.import_center import KnowledgeImporter

    importer = KnowledgeImporter(store_dir=context.get("store_dir"), allow_web=True)
    ok = 0
    for url in urls:
        outcome = importer.import_url(url)
        ok += 1 if outcome.ok else 0
    context["last_urls"] = ok
    return StageOutcome(
        stage=stage.stage, ok=ok == len(urls),
        message=f"imported {ok}/{len(urls)} URL(s)", artifacts={"imported": ok},
    )


def _runner_knowledge_search(stage: PipelineStage, context: dict[str, Any]) -> StageOutcome:
    query = str(stage.options.get("query") or context.get("query") or "")
    if not query:
        return StageOutcome(stage=stage.stage, ok=False, message="no query was given")
    from core.documents import build_document_store
    from core.rag import RagRetriever

    passages = RagRetriever(build_document_store()).retrieve(query, limit=4)
    context["last_passages"] = passages
    return StageOutcome(
        stage=stage.stage, ok=bool(passages),
        message=f"{len(passages)} passage(s) found" if passages else "no passage matched",
        artifacts={"passages": len(passages)},
    )


def _runner_build_dataset(stage: PipelineStage, context: dict[str, Any]) -> StageOutcome:
    dataset_id = str(stage.options.get("dataset") or context.get("dataset") or "")
    if not dataset_id:
        return StageOutcome(stage=stage.stage, ok=False, message="no dataset id was given")
    from core.training_center.datasets import DatasetStudio
    from core.training_center.qa_learning import LearningReviewQueue

    approved = LearningReviewQueue().approved()
    if not approved:
        return StageOutcome(
            stage=stage.stage, ok=False,
            message="no approved learning examples; approve some in Question-and-Answer Learning first",
        )
    row = DatasetStudio().build_from_examples(dataset_id, approved)
    context["dataset"] = dataset_id
    context["dataset_version"] = row.version
    return StageOutcome(
        stage=stage.stage, ok=row.eligible, message=row.describe(),
        artifacts={"dataset": dataset_id, "version": row.version},
    )


def _runner_validate_dataset(stage: PipelineStage, context: dict[str, Any]) -> StageOutcome:
    dataset_id = str(stage.options.get("dataset") or context.get("dataset") or "")
    if not dataset_id:
        return StageOutcome(stage=stage.stage, ok=False, message="no dataset id was given")
    from core.training_center.datasets import DatasetStudio

    report = DatasetStudio().validate(dataset_id, context.get("dataset_version"))
    ok = "eligible" in report
    return StageOutcome(stage=stage.stage, ok=ok, message=report.splitlines()[0] if report else "")


def _runner_export_dataset(stage: PipelineStage, context: dict[str, Any]) -> StageOutcome:
    dataset_id = str(stage.options.get("dataset") or context.get("dataset") or "")
    destination = str(stage.options.get("destination") or context.get("destination") or "")
    if not dataset_id or not destination:
        return StageOutcome(stage=stage.stage, ok=False, message="a dataset id and destination are required")
    from core.training_center.datasets import DatasetStudio

    report = DatasetStudio().export_jsonl(dataset_id, destination, context.get("dataset_version"))
    return StageOutcome(stage=stage.stage, ok=report.ok, message=report.describe())


def _runner_train(stage: PipelineStage, context: dict[str, Any]) -> StageOutcome:
    dataset_id = str(stage.options.get("dataset") or context.get("dataset") or "")
    if not dataset_id:
        return StageOutcome(stage=stage.stage, ok=False, message="no dataset id was given")
    from core.training_center.jobs import TrainingJobManager

    manager = TrainingJobManager()
    model = _pipeline_model(stage.options.get("model") or context.get("model"))
    config = manager.build_config(
        model_id=str(stage.options.get("model") or context.get("model") or "fayd-inprocess-model"),
        dataset_id=dataset_id,
        dataset_version=int(context.get("dataset_version", 0) or 0),
        preset=str(stage.options.get("preset") or "cpu-safe"),
    )
    handle = manager.start_and_wait(model, dataset_id, config, version=context.get("dataset_version"))
    if handle.error is not None:
        return StageOutcome(stage=stage.stage, ok=False, message=str(handle.error))
    result = handle.result
    ok = bool(result is not None and getattr(result, "ok", False))
    context["last_run_id"] = handle.run_id
    return StageOutcome(
        stage=stage.stage, ok=ok,
        message=result.describe() if result is not None else "the run produced no result",
        artifacts={"run_id": handle.run_id},
    )


def _runner_evaluate(stage: PipelineStage, context: dict[str, Any]) -> StageOutcome:
    run_id = str(stage.options.get("run_id") or context.get("last_run_id") or "")
    dataset_id = str(stage.options.get("dataset") or context.get("dataset") or "")
    if not dataset_id:
        return StageOutcome(stage=stage.stage, ok=False, message="no dataset id was given")
    from core.training_center.jobs import TrainingJobManager

    manager = TrainingJobManager()
    run = manager.coordinator.trainer.get(run_id) if run_id else None
    dataset = manager.coordinator.datasets.load(dataset_id, context.get("dataset_version"))
    if dataset is None:
        return StageOutcome(stage=stage.stage, ok=False, message=f"no dataset {dataset_id!r}")
    from ml.llm.tokenizer import ByteTokenizer

    model = _pipeline_model(stage.options.get("model") or context.get("model"))
    adapter_dir = getattr(run, "adapter_path", None) if run is not None else None
    report = manager.coordinator.evaluate(model, dataset, tokenizer=ByteTokenizer(), adapter_dir=adapter_dir)
    context["last_evaluation"] = report
    return StageOutcome(stage=stage.stage, ok=True, message=report.describe())


def _pipeline_model(model_id: Any) -> Any:
    """The model a pipeline stage trains: the in-project checkpoint, or a record."""
    from ml.llm.transformer import TransformerConfig, TransformerLM

    from core.paths import default_checkpoint

    key = str(model_id or "")
    if not key or key in ("fayd", "fayd-llm", "fayd-inprocess-model"):
        checkpoint = default_checkpoint()
        if checkpoint.exists():
            return TransformerLM.load(checkpoint)
        return TransformerLM(TransformerConfig.preset("small"))
    from core.model_manager import ModelManager

    return ModelManager().get(key)


DEFAULT_RUNNERS: dict[str, StageRunner] = {
    STAGE_IMPORT_FOLDER: _runner_import_folder,
    STAGE_IMPORT_URLS: _runner_import_urls,
    STAGE_KNOWLEDGE_SEARCH: _runner_knowledge_search,
    STAGE_BUILD_DATASET: _runner_build_dataset,
    STAGE_VALIDATE_DATASET: _runner_validate_dataset,
    STAGE_EXPORT_DATASET: _runner_export_dataset,
    STAGE_TRAIN: _runner_train,
    STAGE_EVALUATE: _runner_evaluate,
    # The stages the importer folds in are honest no-ops with a note, because the
    # work really happened inside the import stage above.
    STAGE_EXTRACT: lambda stage, ctx: StageOutcome(
        stage=stage.stage, ok=True, message="text extraction runs inside the importer, per file",
    ),
    STAGE_CLEAN: lambda stage, ctx: StageOutcome(
        stage=stage.stage, ok=True, message="normalisation runs inside the importer, per file",
    ),
    STAGE_DEDUPLICATE: lambda stage, ctx: StageOutcome(
        stage=stage.stage, ok=True, message="deduplication runs inside the importer, per source",
    ),
}


# -- the store ------------------------------------------------------------


class PipelineStore:
    """Save and load pipelines through the project's own JSON store."""

    def __init__(self, *, root: Path | str | None = None, store: JsonStore | None = None) -> None:
        self.root = Path(root) if root is not None else data_dir()
        self._store = store or JsonStore(self.root)

    def all(self) -> tuple[LearningPipeline, ...]:
        raw = self._store.load(NAMESPACE, KEY)
        if not isinstance(raw, list):
            return ()
        pipelines: list[LearningPipeline] = []
        for item in raw:
            try:
                pipelines.append(LearningPipeline.from_primitive(item))
            except (ValueError, TypeError):
                continue
        return tuple(pipelines)

    def get(self, pipeline_id: str) -> LearningPipeline | None:
        for pipeline in self.all():
            if pipeline.pipeline_id == pipeline_id:
                return pipeline
        return None

    def save(self, pipeline: LearningPipeline) -> LearningPipeline:
        pipelines = [p for p in self.all() if p.pipeline_id != pipeline.pipeline_id]
        pipelines.append(pipeline)
        self._save(pipelines)
        return pipeline

    def remove(self, pipeline_id: str) -> bool:
        pipelines = self.all()
        kept = [p for p in pipelines if p.pipeline_id != pipeline_id]
        if len(kept) == len(pipelines):
            return False
        self._save(kept)
        return True

    def _save(self, pipelines: list[LearningPipeline]) -> None:
        self._store.save(NAMESPACE, KEY, [pipeline.to_primitive() for pipeline in pipelines])


def default_pipeline(pipeline_id: str = "learn-and-train") -> LearningPipeline:
    """A small, honest starter pipeline a user can run or edit."""
    return LearningPipeline(
        pipeline_id=pipeline_id,
        name="Learn from a folder, then build and validate a dataset",
        stages=[
            PipelineStage(STAGE_IMPORT_FOLDER, {"folder": ""}),
            PipelineStage(STAGE_KNOWLEDGE_SEARCH, {"query": ""}),
            PipelineStage(STAGE_BUILD_DATASET, {"dataset": "learned"}),
            PipelineStage(STAGE_VALIDATE_DATASET, {"dataset": "learned"}),
            PipelineStage(STAGE_TRAIN, {"dataset": "learned", "preset": "cpu-safe"}, enabled=False),
        ],
    )


__all__ = [
    "DEFAULT_RUNNERS",
    "KEY",
    "NAMESPACE",
    "STAGE_SPECS",
    "LearningPipeline",
    "PipelineReport",
    "PipelineRunner",
    "PipelineStage",
    "PipelineStore",
    "StageOutcome",
    "StageSpec",
    "default_pipeline",
    "stage_names",
    "stage_spec",
]
