"""The Training Center service: the engine behind the window and the CLI.

One Tk-free service drives the whole Training Center surface, so the window,
``python -m cli training-center``, and any other caller share one review queue,
one dataset registry, one run log, and one history -- no figure is computed twice
and no surface can disagree with another.

It is deliberately a **thin** layer: it composes the pieces the project already
has (the importer, the governed dataset builder, the controlled trainer, the
review queue, the learning providers, the pipeline model, the history) behind the
shape a window and a CLI both want -- plain reports, a saved selection, a
progress callback for a slow operation, and a background-thread seam so a long
import, crawl, or training run never freezes the interface.

Everything here is exercised headless: no display, no network, no weights. Every
collaborator may be injected, so a test drives the whole workflow with stand-ins.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from core.best_effort import best_effort
from core.paths import data_dir
from core.training_center import catalog
from core.training_center.dashboard import DashboardSnapshot, build_dashboard
from core.training_center.datasets import DatasetRow, DatasetStudio, ExportReport
from core.training_center.history import HistoryLine, TrainingHistory
from core.training_center.jobs import JobHandle, JobView, TrainingJobManager
from core.training_center.pipelines import (
    LearningPipeline,
    PipelineReport,
    PipelineRunner,
    PipelineStore,
    StageOutcome,
    default_pipeline,
)
from core.training_center.providers import (
    LearningProvider,
    ProviderAnswer,
    ProviderStatus,
    build_provider,
)
from core.training_center.qa_learning import LearningReviewQueue, ReviewSummary
from core.training_center.records import LearningExample
from core.training_center.resources import (
    ResourceSnapshot,
    TrainingEstimate,
    estimate_training,
    read_resources,
)
from ml.training.datasets import DatasetBuilder
from storage.json_store import JsonStore

#: The JsonStore namespace/key the service's own preferences live under.
_NAMESPACE = "training_center"
_SETTINGS_KEY = "settings"

#: The shipped preferences. A review is required before export by default, which
#: is the project's hard rule made a setting the user can see (and turn off only
#: deliberately).
DEFAULT_SETTINGS: dict[str, Any] = {
    "language": "en",
    "chunk_chars": 1200,
    "review_required": True,
    "default_provider": "manual",
    "cloud_provider": "gemini",
    "default_preset": "cpu-safe",
    "include_documents": True,
}


class TrainingCenterService:
    """Compose the project's learning machinery behind one interface."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        importer: Any = None,
        datasets: DatasetStudio | None = None,
        queue: LearningReviewQueue | None = None,
        jobs: TrainingJobManager | None = None,
        history: TrainingHistory | None = None,
        pipelines: PipelineStore | None = None,
        providers: dict[str, LearningProvider] | None = None,
        coordinator: Any = None,
    ) -> None:
        self.root = Path(root) if root is not None else data_dir()
        self._store = JsonStore(self.root)
        self._importer = importer
        self.datasets = datasets or DatasetStudio(
            builder=DatasetBuilder(root=self.root / "training" / "datasets")
        )
        self.queue = queue or LearningReviewQueue(root=self.root)
        self.jobs = jobs or TrainingJobManager(coordinator, root=self.root / "training")
        self.history = history or TrainingHistory(root=self.root)
        self.pipelines = pipelines or PipelineStore(root=self.root)
        self._providers: dict[str, LearningProvider] = dict(providers or {})

    # -- settings ----------------------------------------------------------
    def settings(self) -> dict[str, Any]:
        """The saved preferences merged over the shipped defaults."""
        saved = self._store.load(_NAMESPACE, _SETTINGS_KEY)
        merged = dict(DEFAULT_SETTINGS)
        if isinstance(saved, dict):
            merged.update(saved)
        return merged

    def setting(self, key: str) -> Any:
        return self.settings().get(key, DEFAULT_SETTINGS.get(key))

    def set_setting(self, key: str, value: Any) -> dict[str, Any]:
        """Persist one preference; an unknown key is refused rather than stored."""
        if key not in DEFAULT_SETTINGS:
            raise KeyError(f"Unknown setting {key!r}; expected one of {tuple(DEFAULT_SETTINGS)}.")
        current = self.settings()
        current[key] = value
        self._store.save(_NAMESPACE, _SETTINGS_KEY, current)
        self.history.record("settings", f"{key} = {value}")
        return current

    def settings_text(self) -> str:
        lines = ["Training Center — settings"]
        for key, value in sorted(self.settings().items()):
            lines.append(f"  {key}: {value}")
        lines.append("")
        lines.append(
            "A review is required before export: "
            + ("yes" if self.setting("review_required") else "no")
        )
        return "\n".join(lines)

    # -- dashboard ---------------------------------------------------------
    def dashboard(self) -> DashboardSnapshot:
        return build_dashboard(self.root)

    # -- catalog -----------------------------------------------------------
    def catalog_text(self) -> str:
        return catalog.catalog_text()

    def capabilities(self) -> tuple[catalog.Capability, ...]:
        return catalog.CAPABILITIES

    # -- knowledge ---------------------------------------------------------
    def _importer_for(self, *, allow_web: bool = False) -> Any:
        if self._importer is not None:
            return self._importer
        from gui.import_center import KnowledgeImporter

        return KnowledgeImporter(store_dir=self.knowledge_store_dir(), allow_web=allow_web)

    def import_files(self, paths: Sequence[str | Path]) -> Any:
        report = self._importer_for().import_files(paths)
        self.history.record("import", f"imported {len(list(paths))} file(s)", outcome=report.summary())
        return report

    def import_folder(self, folder: str | Path, *, recursive: bool = True) -> Any:
        report = self._importer_for().import_folder(folder, recursive=recursive)
        self.history.record("import", f"imported folder {folder}", outcome=report.summary())
        return report

    def import_urls(self, urls: Sequence[str], *, approved: bool = False) -> tuple[Any, ...]:
        """Import user-supplied URLs through the governed web path.

        The web path is the project's own permission + approval gate: without
        ``approved`` this returns an empty tuple and records nothing, so a caller
        must explicitly consent (the same click the chat's web toggle is).
        """
        if not approved:
            return ()
        importer = self._importer_for(allow_web=True)
        outcomes = tuple(importer.import_url(url) for url in urls)
        ok = sum(1 for outcome in outcomes if outcome.ok)
        self.history.record("crawl", f"imported {ok}/{len(urls)} URL(s)")
        return outcomes

    def knowledge_stats(self) -> dict[str, Any]:
        try:
            from core.knowledge_base import KnowledgeBaseStore

            return KnowledgeBaseStore(self.knowledge_store_dir()).stats()
        except Exception as error:  # pragma: no cover - defensive
            return {"error": str(error)}

    def knowledge_search(self, query: str, *, limit: int = 4) -> tuple[Any, ...]:
        from core.documents import build_document_store
        from core.rag import RagRetriever

        return tuple(RagRetriever(build_document_store()).retrieve(query, limit=limit))

    def reindex(self, *, progress: Any = None) -> dict[str, Any]:
        from core.knowledge_index import build_index

        summary = build_index(self.knowledge_store_dir(), progress=progress)
        self.history.record("knowledge", "rebuilt the knowledge index")
        return summary

    def knowledge_store_dir(self) -> Path:
        """The knowledge base this service reads and writes (its own data root).

        A service built with a root keeps its knowledge base *inside* that root,
        so the CLI's ``--root``, a test, and a portable build never touch the
        machine-wide store; the default service uses the project's own folder.
        """
        return self.root / "knowledge_base"

    def export_knowledge(self, destination: str | Path) -> str:
        """Copy the knowledge base's shards and manifest to ``destination``."""
        import shutil

        source = self.knowledge_store_dir()
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        copied = 0
        for path in source.glob("*"):
            if path.is_file():
                shutil.copy2(path, target / path.name)
                copied += 1
        self.history.record("knowledge", f"backed up {copied} file(s) to {target}")
        return f"Copied {copied} file(s) of the knowledge base to {target}."

    # -- providers and Q&A -------------------------------------------------
    def provider(self, kind: str, **kwargs: Any) -> LearningProvider:
        """The provider for ``kind`` (cached, so a local model loads once)."""
        if kind not in self._providers:
            if kind == "gemini":
                kwargs.setdefault("provider", self.setting("cloud_provider"))
            self._providers[kind] = build_provider(kind, **kwargs)
        return self._providers[kind]

    def provider_statuses(self) -> tuple[ProviderStatus, ...]:
        """The status of every provider the Training Center offers."""
        statuses: list[ProviderStatus] = []
        for kind in ("manual", "document", "gemini"):
            with best_effort(f"read the {kind} provider status"):
                statuses.append(self.provider(kind).status())
        for mouth in ("ultron", "vision"):
            from core.training_center.providers import LocalModelLearningProvider

            with best_effort(f"read the {mouth} provider status"):
                statuses.append(LocalModelLearningProvider(mouth).status())
        return tuple(statuses)

    def ask(
        self,
        question: str,
        *,
        provider: str = "",
        topic: str = "",
        system: str = "",
    ) -> LearningExample:
        """Ask a provider one question and record the answer as *pending*.

        The example is recorded with the provider's own name and the answer's
        citations, and its status is ``pending`` -- a human must approve it before
        it can reach a dataset.
        """
        kind = provider or str(self.setting("default_provider"))
        answer: ProviderAnswer = self.provider(kind).ask(question, system=system)
        example = LearningExample(
            question=question,
            answer=answer.text,
            provider=answer.provider or kind,
            model=answer.model,
            topic=topic,
            citations=answer.citations,
            note=answer.note,
        )
        self.queue.add(example)
        self.history.record("ask", f"asked {kind}: {question[:60]}")
        return example

    def generate_questions(self, topic: str, *, count: int = 5, provider: str = "") -> tuple[str, ...]:
        """Generate questions on a topic from a provider."""
        kind = provider or str(self.setting("default_provider"))
        return self.provider(kind).generate(topic, count=count)

    def ask_many(
        self, questions: Sequence[str], *, provider: str = "", topic: str = ""
    ) -> tuple[LearningExample, ...]:
        examples: list[LearningExample] = []
        for question in questions:
            with best_effort(f"ask about {question[:40]}"):
                examples.append(self.ask(question, provider=provider, topic=topic))
        return tuple(examples)

    def examples(self, *, status: str = "", query: str = "") -> tuple[LearningExample, ...]:
        if query:
            return self.queue.search(query)
        if status:
            return self.queue.by_status(status)
        return self.queue.all()

    def review_summary(self) -> ReviewSummary:
        return self.queue.summary()

    def approve(self, example_id: str, *, note: str = "") -> LearningExample:
        example = self.queue.approve(example_id, note=note)
        self.history.record("review", f"approved {example_id}")
        return example

    def reject(self, example_id: str, *, note: str = "") -> LearningExample:
        example = self.queue.reject(example_id, note=note)
        self.history.record("review", f"rejected {example_id}")
        return example

    def correct(self, example_id: str, answer: str, *, note: str = "") -> LearningExample:
        example = self.queue.correct(example_id, answer, note=note)
        self.history.record("review", f"corrected {example_id}")
        return example

    # -- datasets ----------------------------------------------------------
    def dataset_rows(self) -> tuple[DatasetRow, ...]:
        return self.datasets.rows()

    def build_dataset_from_approved(self, dataset_id: str) -> DatasetRow:
        """Build a governed dataset from the **approved** learning examples."""
        approved = self.queue.approved()
        if not approved:
            raise ValueError(
                "No approved learning examples; approve some in Question-and-Answer "
                "Learning first, or import a document into the knowledge base."
            )
        row = self.datasets.build_from_examples(dataset_id, approved)
        self.queue.mark_exported(example.example_id for example in approved)
        self.history.record("dataset", f"built {row.key} from approved examples", outcome=row.status)
        return row

    def build_dataset_from_texts(self, dataset_id: str, texts: Sequence[str]) -> DatasetRow:
        row = self.datasets.build_from_texts(dataset_id, texts)
        self.history.record("dataset", f"built {row.key}", outcome=row.status)
        return row

    def validate_dataset(self, dataset_id: str, version: int | None = None) -> str:
        return self.datasets.validate(dataset_id, version)

    def export_dataset(self, dataset_id: str, destination: str | Path, version: int | None = None) -> ExportReport:
        report = self.datasets.export_jsonl(dataset_id, destination, version)
        self.history.record("dataset", report.describe(), outcome="ok" if report.ok else "failed")
        return report

    # -- jobs --------------------------------------------------------------
    def job_rows(self) -> tuple[JobView, ...]:
        return self.jobs.rows()

    def feasibility(self, model: Any, dataset_id: str, config: Any, *, version: int | None = None) -> Any:
        return self.jobs.feasibility(model, dataset_id, config, version=version)

    def build_config(self, **kwargs: Any) -> Any:
        return self.jobs.build_config(**kwargs)

    def start_job(self, model: Any, dataset_id: str, config: Any, **kwargs: Any) -> JobHandle:
        handle = self.jobs.start(model, dataset_id, config, **kwargs)
        self.history.record("training", f"started {handle.run_id}")
        return handle

    def cancel_job(self, run_id: str) -> bool:
        cancelled = self.jobs.cancel(run_id)
        if cancelled:
            self.history.record("training", f"cancelled {run_id}")
        return cancelled

    def evaluate_run(self, run_id: str) -> Any:
        run = self.jobs.coordinator.trainer.get(run_id)
        if run is None:
            raise KeyError(f"No training run {run_id!r} was found.")
        dataset = self.jobs.coordinator.datasets.load(run.dataset_id, run.dataset_version)
        if dataset is None:
            raise KeyError(f"The run's dataset {run.dataset_id!r} is no longer stored.")
        from ml.llm.tokenizer import ByteTokenizer

        model = _load_model_for_eval(run.model_id)
        report = self.jobs.coordinator.evaluate(
            model, dataset, tokenizer=ByteTokenizer(),
            adapter_dir=getattr(run, "adapter_path", None) or None,
        )
        self.history.record("evaluation", f"evaluated {run_id}", outcome=report.describe().splitlines()[0])
        return report

    # -- pipelines ---------------------------------------------------------
    def pipeline_list(self) -> tuple[LearningPipeline, ...]:
        return self.pipelines.all()

    def save_pipeline(self, pipeline: LearningPipeline) -> LearningPipeline:
        saved = self.pipelines.save(pipeline)
        self.history.record("pipeline", f"saved {pipeline.pipeline_id}")
        return saved

    def create_pipeline(self, pipeline_id: str) -> LearningPipeline:
        return self.save_pipeline(default_pipeline(pipeline_id))

    def run_pipeline(
        self,
        pipeline_id: str,
        *,
        approved: bool = False,
        on_stage: Callable[[StageOutcome], None] | None = None,
        context: dict[str, Any] | None = None,
    ) -> PipelineReport:
        pipeline = self.pipelines.get(pipeline_id)
        if pipeline is None:
            raise KeyError(f"No pipeline {pipeline_id!r} was found.")
        report = PipelineRunner(context=context).run(pipeline, approved=approved, on_stage=on_stage)
        self.history.record(
            "pipeline", f"ran {pipeline_id}", outcome="ok" if report.ok else f"stopped at {report.stopped_at}"
        )
        return report

    # -- history -----------------------------------------------------------
    def history_lines(self, *, kind: str = "", query: str = "", limit: int = 100) -> tuple[HistoryLine, ...]:
        return self.history.search(kind=kind, query=query, limit=limit)

    # -- resources ---------------------------------------------------------
    def resources(self) -> ResourceSnapshot:
        return read_resources()

    def estimate(
        self, *, parameter_count: int | None, rank: int = 8, batch_size: int = 4, grad_accum: int = 2
    ) -> TrainingEstimate:
        return estimate_training(
            parameter_count=parameter_count, rank=rank, batch_size=batch_size, grad_accum=grad_accum,
        )

    def model_choices(self) -> tuple[tuple[str, str], ...]:
        """The models a training run may use here, as ``(id, label)`` rows.

        Only the in-project checkpoint is offered as a *training* base by default
        -- a GGUF file is an inference file and is refused by the compatibility
        validator, so it is not offered as one that would work.
        """
        rows: list[tuple[str, str]] = [("fayd-inprocess-model", "the project's own in-process transformer")]
        with best_effort("read the local model catalog"):
            from core.language_models import all_language_models

            for _, model in all_language_models():
                rows.append((model.key, f"{model.title} ({model.engine})"))
        return tuple(rows)

    # -- background --------------------------------------------------------
    def run_in_background(
        self,
        work: Callable[[], Any],
        *,
        on_done: Callable[[Any, Exception | None], None],
        on_progress: Callable[[str], None] | None = None,
    ) -> threading.Thread:
        """Run ``work`` on a worker thread and hand the result to ``on_done``.

        Tk is not thread-safe, so nothing here touches a widget: the window's own
        queue-draining pump is what calls ``on_done`` on the interface thread.
        """

        def _run() -> None:
            if on_progress is not None:
                with best_effort("report progress"):
                    on_progress("working…")
            try:
                result = work()
            except Exception as error:  # handed back, never swallowed
                on_done(None, error)
                return
            on_done(result, None)

        thread = threading.Thread(target=_run, name="training-center", daemon=True)
        thread.start()
        return thread


def _load_model_for_eval(model_id: str) -> Any:
    from core.paths import default_checkpoint
    from ml.llm.transformer import TransformerConfig, TransformerLM

    if not model_id or model_id in ("fayd", "fayd-llm", "fayd-inprocess-model"):
        checkpoint = default_checkpoint()
        if checkpoint.exists():
            return TransformerLM.load(checkpoint)
        return TransformerLM(TransformerConfig.preset("small"))
    from core.model_manager import ModelManager

    return ModelManager().get(model_id)


def training_center_service(root: Path | str | None = None, **kwargs: Any) -> TrainingCenterService:
    """A service over the project's own data folder (or ``root``)."""
    return TrainingCenterService(root, **kwargs)


__all__ = [
    "DEFAULT_SETTINGS",
    "TrainingCenterService",
    "training_center_service",
]
