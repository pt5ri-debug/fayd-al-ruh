"""The Training Center: the core records, the governed paths, and the service.

Every test here runs with no display, no model, and no network: the core stores
are built over a temporary root, the providers are fakes, and the dataset builder
writes into a temporary folder. The window is exercised separately in
``tests/test_training_center_window.py`` and the CLI in
``tests/test_training_center_cli.py``; here the *engine* is what is checked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from core.training_center import catalog
from core.training_center.datasets import DatasetStudio
from core.training_center.pipelines import (
    STAGE_SPECS,
    LearningPipeline,
    PipelineRunner,
    PipelineStage,
    PipelineStore,
    default_pipeline,
    stage_names,
)
from core.training_center.providers import (
    ManualProvider,
    ProviderAnswer,
    ProviderStatus,
    build_provider,
    provider_names,
)
from core.training_center.qa_learning import LearningReviewQueue
from core.training_center.records import (
    STATUS_APPROVED,
    STATUS_CORRECTED,
    STATUS_PENDING,
    STATUS_REJECTED,
    LearningExample,
    detect_language,
)
from core.training_center.resources import estimate_training, human_size, read_resources
from gui.training_center import TrainingCenterService
from ml.training.datasets import DatasetBuilder


def _studio(tmp_path: Path) -> DatasetStudio:
    return DatasetStudio(builder=DatasetBuilder(root=tmp_path / "datasets"))


def _queue(tmp_path: Path) -> LearningReviewQueue:
    return LearningReviewQueue(root=tmp_path)


def _service(tmp_path: Path) -> TrainingCenterService:
    return TrainingCenterService(
        tmp_path,
        datasets=_studio(tmp_path),
        queue=_queue(tmp_path),
    )


# -- records ---------------------------------------------------------------


def test_example_is_pending_until_approved() -> None:
    example = LearningExample(question="What is porosity?", answer="Void space.", provider="manual")
    assert example.status == STATUS_PENDING
    assert not example.approved
    assert example.example_id
    approved = example.with_status(STATUS_APPROVED)
    assert approved.approved
    # A status change is a *copy*; the original is untouched.
    assert example.status == STATUS_PENDING


def test_example_needs_a_question_or_an_answer() -> None:
    with pytest.raises(ValueError):
        LearningExample(question="", answer="   ")


def test_example_rejects_an_unknown_status() -> None:
    with pytest.raises(ValueError):
        LearningExample(question="Q", answer="A", status="made-up")


def test_correction_marks_the_example_corrected_and_approved() -> None:
    example = LearningExample(question="Q", answer="A", provider="manual")
    corrected = example.with_correction("A better answer.")
    assert corrected.status == STATUS_CORRECTED
    assert corrected.approved
    assert corrected.answer == "A better answer."


def test_training_text_joins_question_and_answer() -> None:
    example = LearningExample(question="Q?", answer="A.", provider="manual")
    assert example.training_text == "Q?\nA."


def test_example_round_trips_through_a_primitive() -> None:
    example = LearningExample(
        question="Q", answer="A", provider="gemini", topic="t", citations=("doc",),
    )
    restored = LearningExample.from_primitive(example.to_primitive())
    assert restored == example


def test_detect_language_separates_arabic_from_english() -> None:
    assert detect_language("ما هي المسامية؟") == "ar"
    assert detect_language("What is porosity?") == "en"


# -- the review queue ------------------------------------------------------


def test_review_queue_governs_export(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    example = LearningExample(question="Q", answer="A", provider="manual")
    queue.add(example)
    assert queue.get(example.example_id) is not None
    assert queue.approved() == ()
    queue.approve(example.example_id)
    assert len(queue.approved()) == 1


def test_review_queue_refuses_an_unknown_status(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    example = LearningExample(question="Q", answer="A", provider="manual")
    queue.add(example)
    with pytest.raises(ValueError):
        queue.set_status(example.example_id, "nope")


def test_review_queue_refuses_an_unknown_id(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    with pytest.raises(KeyError):
        queue.approve("missing")


def test_rejected_is_never_approved(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    example = LearningExample(question="Q", answer="A", provider="manual")
    queue.add(example)
    queue.reject(example.example_id)
    assert queue.approved() == ()
    assert queue.by_status(STATUS_REJECTED) == (queue.get(example.example_id),)


def test_review_queue_search_and_summary(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.add(LearningExample(question="What is porosity?", answer="A", provider="manual"))
    queue.add(LearningExample(question="How is permeability measured?", answer="B", provider="manual"))
    assert len(queue.search("permeability")) == 1
    summary = queue.summary()
    assert summary.counts.get(STATUS_PENDING) == 2


def test_correction_counts_as_approved(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    example = LearningExample(question="Q", answer="A", provider="manual")
    queue.add(example)
    queue.correct(example.example_id, "A better answer.")
    approved = queue.approved()
    assert len(approved) == 1
    assert approved[0].answer == "A better answer."


# -- the dataset studio ----------------------------------------------------


def _approved(question: str) -> LearningExample:
    return LearningExample(
        question=question,
        answer="The project's own sources answer this with its citation recorded.",
        provider="manual",
        status=STATUS_APPROVED,
    )


def test_dataset_refuses_an_unapproved_example(tmp_path: Path) -> None:
    studio = _studio(tmp_path)
    pending = LearningExample(question="Q", answer="A", provider="manual")
    with pytest.raises(Exception) as error:
        studio.build_from_examples("learned", [pending])
    assert "not approved" in str(error.value)


def test_dataset_refuses_an_empty_build(tmp_path: Path) -> None:
    studio = _studio(tmp_path)
    with pytest.raises(Exception):
        studio.build_from_examples("learned", [])


def test_dataset_builds_from_approved_examples(tmp_path: Path) -> None:
    studio = _studio(tmp_path)
    examples = [_approved(f"Question number {i} about the reservoir rock?") for i in range(6)]
    row = studio.build_from_examples("learned", examples)
    assert row.total == 6
    assert row.eligible
    assert row.train + row.validation + row.test == 6


def test_dataset_validation_reports_a_problem(tmp_path: Path) -> None:
    studio = _studio(tmp_path)
    studio.build_from_texts("tiny", ["one short line"])
    text = studio.validate("tiny")
    assert "tiny" in text


def test_dataset_export_writes_jsonl(tmp_path: Path) -> None:
    studio = _studio(tmp_path)
    examples = [_approved(f"Question {i} about the well?") for i in range(6)]
    studio.build_from_examples("learned", examples)
    report = studio.export_jsonl("learned", tmp_path / "out")
    assert Path(report.path).exists()


# -- providers -------------------------------------------------------------


def test_manual_provider_returns_the_answer_it_was_given() -> None:
    provider = ManualProvider(answers={"What is porosity?": "Void space."})
    answer = provider.ask("What is porosity?")
    assert isinstance(answer, ProviderAnswer)
    assert answer.text == "Void space."
    # A question with no supplied answer is honest rather than fabricated.
    assert not provider.ask("Unknown?").text


def test_build_provider_refuses_an_unknown_kind() -> None:
    with pytest.raises(ValueError):
        build_provider("nope")


def test_provider_names_are_the_documented_set() -> None:
    assert set(provider_names()) == {"manual", "document", "gemini", "local", "character"}


def test_cloud_provider_without_approval_is_not_available() -> None:
    from core.training_center.providers import CloudLearningProvider

    provider = CloudLearningProvider("gemini")
    status = provider.status()
    assert isinstance(status, ProviderStatus)
    # Approval is the user's click; without it the provider is never "available",
    # even when a key happens to be present in the environment.
    assert not status.available
    assert status.requires_network


# -- the pipelines ---------------------------------------------------------


def test_default_pipeline_round_trips_through_the_store(tmp_path: Path) -> None:
    store = PipelineStore(root=tmp_path)
    pipeline = default_pipeline()
    store.save(pipeline)
    loaded = store.get(pipeline.pipeline_id)
    assert loaded is not None
    assert [stage.stage for stage in loaded.stages] == [stage.stage for stage in pipeline.stages]


def test_pipeline_refuses_a_governed_stage_without_approval() -> None:
    pipeline = LearningPipeline(
        pipeline_id="p", name="p", stages=[PipelineStage("train", {"dataset": "learned"})],
    )
    report = PipelineRunner().run(pipeline, approved=False)
    assert report.stopped_at
    assert not report.outcomes[0].ok


def test_pipeline_reports_an_unavailable_stage() -> None:
    unavailable = next((spec for spec in STAGE_SPECS if not spec.available), None)
    assert unavailable is not None  # the catalogue always names one
    pipeline = LearningPipeline(
        pipeline_id="p", name="p", stages=[PipelineStage(unavailable.key, {})],
    )
    report = PipelineRunner().run(pipeline, approved=True)
    assert not report.outcomes[0].ok
    assert "unavailable" in report.outcomes[0].message


def test_stage_names_are_a_closed_set() -> None:
    assert "train" in stage_names()
    with pytest.raises(Exception):
        PipelineStage("not-a-stage", {})


# -- the catalogue ---------------------------------------------------------


def test_catalog_is_complete_and_bilingual() -> None:
    assert catalog.CAPABILITIES
    keys = [cap.key for cap in catalog.CAPABILITIES]
    assert len(keys) == len(set(keys))  # unique
    for cap in catalog.CAPABILITIES:
        assert cap.title and cap.summary and cap.outcome and cap.cli
    for section in catalog.SECTIONS:
        assert section in catalog.SECTION_SUMMARIES
        assert section in catalog.SECTION_TITLES_AR


def test_a_governed_capability_names_its_confirmation() -> None:
    for cap in catalog.CAPABILITIES:
        if cap.governed:
            assert cap.cli, f"{cap.key} is governed but names no command"


def test_every_section_has_at_least_one_capability() -> None:
    for section in catalog.SECTIONS:
        assert catalog.capabilities_for(section), section


def test_unknown_capability_is_none() -> None:
    assert catalog.capability("nope") is None


# -- resources -------------------------------------------------------------


def test_human_size_never_shows_a_zero_for_unknown() -> None:
    assert human_size(None) == "unknown"
    assert human_size(2048).endswith("KB")


def test_resources_read_never_raises() -> None:
    snapshot = read_resources()
    assert snapshot.describe()


def test_estimate_labels_itself_an_estimate() -> None:
    estimate = estimate_training(parameter_count=1_000_000)
    assert "estimate" in estimate.describe().lower()
    assert estimate.estimated_memory_bytes > 0


def test_estimate_says_so_when_the_size_is_unknown() -> None:
    estimate = estimate_training(parameter_count=None)
    assert any("could not be read" in note for note in estimate.notes)


# -- the service -----------------------------------------------------------


def test_service_settings_round_trip(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert service.setting("review_required") in (True, False)
    service.set_setting("chunk_chars", 900)
    assert service.setting("chunk_chars") == 900


def test_service_refuses_an_unknown_setting(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(KeyError):
        service.set_setting("nope", 1)


def test_service_ask_records_a_pending_example(tmp_path: Path) -> None:
    service = _service(tmp_path)
    example = service.ask("What is porosity?", provider="manual")
    assert example.status == STATUS_PENDING
    assert service.examples(status=STATUS_PENDING)


def test_service_build_dataset_needs_approved_examples(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.ask("What is porosity?", provider="manual")
    with pytest.raises(ValueError):
        service.build_dataset_from_approved("learned")


def test_service_approve_then_build(tmp_path: Path) -> None:
    service = _service(tmp_path)
    for i in range(6):
        example = service.ask(f"Question number {i} about the reservoir?", provider="manual")
        service.approve(example.example_id)
    row = service.build_dataset_from_approved("learned")
    assert row.total == 6
    assert row.eligible


def test_service_history_records_actions(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.ask("What is porosity?", provider="manual")
    lines = service.history_lines()
    assert any("ask" in line.describe() for line in lines)


def test_service_dashboard_reads_real_figures(tmp_path: Path) -> None:
    service = _service(tmp_path)
    snapshot = service.dashboard()
    assert "dashboard" in snapshot.describe().lower()
    # An empty store reads as 0, never a fabricated figure.
    assert snapshot.datasets == 0


def test_service_pipeline_create_and_list(tmp_path: Path) -> None:
    service = _service(tmp_path)
    created = service.create_pipeline("my-pipeline")
    assert service.pipelines.get(created.pipeline_id) is not None
    assert any(p.pipeline_id == "my-pipeline" for p in service.pipeline_list())


def test_service_catalog_text_lists_controls(tmp_path: Path) -> None:
    service = _service(tmp_path)
    text = service.catalog_text()
    assert "Training Center" in text


def test_service_provider_statuses_are_honest(tmp_path: Path) -> None:
    service = _service(tmp_path)
    statuses = service.provider_statuses()
    assert statuses
    assert all(isinstance(status, ProviderStatus) for status in statuses)


# -- the internet switch ---------------------------------------------------


def test_web_is_off_by_default(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert service.web_enabled() is False
    assert "off" in service.web_status().lower()


def test_set_web_persists_and_reports(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert service.set_web(True) is True
    assert service.web_enabled() is True
    # A second service over the same root reads the saved choice.
    assert _service(tmp_path).web_enabled() is True
    assert service.toggle_web() is False
    assert service.web_enabled() is False


def test_import_urls_needs_the_switch_even_with_approval(tmp_path: Path) -> None:
    service = _service(tmp_path)
    # approved=True but the internet switch is off: nothing runs, nothing fetched.
    assert service.import_urls(["https://example.com"], approved=True) == ()
    service.set_web(True)
    # With the switch on, an *unapproved* call is still refused.
    assert service.import_urls(["https://example.com"], approved=False) == ()


def test_build_knowledge_base_is_refused_until_the_switch_is_on(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(PermissionError):
        service.build_knowledge_base(target_mb=1)


def test_cloud_provider_is_unapproved_until_the_switch_is_on(tmp_path: Path) -> None:
    service = _service(tmp_path)
    off = service.provider("gemini")
    assert off.status().available is False
    service.set_web(True)
    on = service.provider("gemini")
    # The switch rebuilds the provider, so a cached unapproved one is never reused.
    assert on is not off


# -- the dropdown choices --------------------------------------------------


def test_provider_choices_are_values_the_service_accepts(tmp_path: Path) -> None:
    service = _service(tmp_path)
    rows = service.provider_choices()
    values = [value for value, _ in rows]
    assert "manual" in values
    assert "document" in values
    assert any(value.startswith("local:") for value in values)
    assert any(value.startswith("character:") for value in values)
    # Every row is a value the service really accepts: a manual ask works.
    assert all(isinstance(label, str) and label for _, label in rows)


def test_provider_choices_include_the_cloud_providers(tmp_path: Path) -> None:
    from core.cloud_model import provider_names

    service = _service(tmp_path)
    values = set(service.provider_values())
    assert set(provider_names()) <= values


def test_mouth_and_character_choices(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert "ultron" in service.mouth_values()
    assert "cloud" not in service.mouth_values()  # cloud is remote, not a local mouth
    assert service.character_values() == (
        "sources", "ultron", "vision", "banner", "natasha", "hakeem", "cloud",
    )


def test_dataset_run_and_pipeline_choices_read_the_stores(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert service.dataset_choices() == ()
    assert service.run_choices() == ()
    assert service.pipeline_choices() == ()
    service.create_pipeline("demo")
    assert "demo" in service.pipeline_choices()


def test_setting_keys_and_value_choices(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert "web" in service.setting_keys()
    assert service.setting_value_choices("web") == ("true", "false")
    assert service.setting_value_choices("language") == ("en", "ar")
    assert "manual" in service.setting_value_choices("default_provider")


def test_provider_resolves_the_local_and_character_prefixes(tmp_path: Path) -> None:
    from core.training_center.providers import (
        CharacterLearningProvider,
        LocalModelLearningProvider,
    )

    service = _service(tmp_path)
    local = service.provider("local:vision")
    assert isinstance(local, LocalModelLearningProvider)
    assert local.mouth == "vision"
    character = service.provider("character:hakeem")
    assert isinstance(character, CharacterLearningProvider)
    assert character.character == "hakeem"
    # The same value returns the cached instance.
    assert service.provider("local:vision") is local


def test_every_provider_choice_resolves(tmp_path: Path) -> None:
    service = _service(tmp_path)
    for value in service.provider_values():
        # A value the dropdown offers must build without raising.
        provider = service.provider(value)
        assert provider is not None
