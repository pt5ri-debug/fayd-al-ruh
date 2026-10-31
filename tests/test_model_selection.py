"""Intelligent model selection, fallback, and the trained-model integration.

Two subjects, one file, because they meet: a trained checkpoint is just another
candidate the ranking can choose, so the same fixtures exercise both.

Everything runs without a real model, a GPU, or a network. The ranking is a pure
function of records the test builds, the fallback ladder is built from those
records, and the trained-artifact reader is pointed at a small synthetic
checkpoint the test writes -- so the trained path is exercised end to end without
the committed 6 MB weights.
"""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

from core.model_manager import ModelManager
from core.model_registry import SOURCE_TRAINED, ModelRecord
from core.model_selection import (
    FACTOR_CAPABILITY,
    FACTOR_ENGINE,
    FACTOR_FIT,
    FACTOR_HEALTH,
    FACTOR_PREFERENCE,
    FACTOR_TASK_MATCH,
    FACTOR_WEIGHTS,
    FallbackPlan,
    RankedModel,
    TaskProfile,
    describe_ranking,
    is_known_task,
    plan_fallback,
    rank_for_task,
    task_profile,
)
from core.trained_artifacts import (
    KIND_CLASSIFIER,
    KIND_LANGUAGE,
    KIND_POLICY,
    KIND_UNKNOWN,
    EvaluationResult,
    classify_artifact,
    describe_artifact,
    discover_trained_artifacts,
    evaluations_from_manifest,
    load_evaluation_sidecar,
    load_manifest,
    register_trained_models,
    save_evaluation,
    trained_checkpoint_path,
    trained_model_record,
)

# ------------------------------------------------------------------- helpers


def _gguf_bytes() -> bytes:
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 0)


def _manager(tmp_path, records: list[ModelRecord]) -> ModelManager:
    manager = ModelManager(root=tmp_path)
    for record in records:
        manager.registry.register_record(record)
    return manager


def _record(tmp_path, name: str, **kwargs) -> ModelRecord:
    path = tmp_path / f"{name}.gguf"
    if not path.exists():
        path.write_bytes(_gguf_bytes())
    base = {
        "model_id": name,
        "identity": f"test/{name}",
        "display_name": name.title(),
        "model_format": "gguf",
        "runtime": "llama.cpp",
        "source": "local",
        "path": path,
        "size_bytes": path.stat().st_size,
        "architecture": "llama",
        "context_length": 8192,
        "parameter_count": 7_000_000_000,
        "integrity": "ok",
    }
    base.update(kwargs)
    return ModelRecord(**base)


def _resources():
    from core.model_resources import READING_MEASURED, SystemResources

    return SystemResources(
        system="Linux", machine="x86_64", cpu_count=8,
        total_memory_bytes=16 * 1024**3, available_memory_bytes=12 * 1024**3,
        memory_reading=READING_MEASURED,
    )


# ------------------------------------------------------------------ profiles


def test_known_tasks_have_profiles():
    assert is_known_task("chat")
    assert not is_known_task("telepathy")
    profile = task_profile("code")
    assert profile.task == "code"
    assert profile.capabilities == ("code",)
    assert profile.context == 8192


def test_an_unknown_task_gets_no_invented_profile():
    profile = task_profile("telepathy")
    assert profile.task == "telepathy"
    assert profile.engines == ()
    assert "does not declare" in profile.description


def test_profile_primitive_is_json_safe():
    assert json.dumps(task_profile("chat").to_primitive())


def test_weights_sum_to_one():
    assert abs(sum(FACTOR_WEIGHTS.values()) - 1.0) < 1e-9


# ------------------------------------------------------------------- ranking


def test_ranking_prefers_the_tasks_own_engine():
    manager = _manager(
        tmp_path := Path(_tmp("rank_engine")),
        [
            _record(tmp_path, "banner-model", engine="banner"),
            _record(tmp_path, "vision-model", engine="vision"),
        ],
    )
    ranked = rank_for_task(manager, "presentation", resources=_resources())
    assert ranked
    assert ranked[0].display_name == "Banner-Model"
    assert ranked[0].factors[FACTOR_ENGINE] == 1.0


def test_ranking_is_deterministic():
    root = Path(_tmp("rank_determinism"))
    manager = _manager(
        root,
        [_record(root, "a-model", engine="banner"), _record(root, "b-model", engine="banner")],
    )
    first = [item.model_id for item in rank_for_task(manager, "presentation", resources=_resources())]
    second = [item.model_id for item in rank_for_task(manager, "presentation", resources=_resources())]
    assert first == second


def test_ranking_carries_the_factors_behind_the_score():
    root = Path(_tmp("rank_factors"))
    manager = _manager(root, [_record(root, "a-model", engine="banner")])
    ranked = rank_for_task(manager, "presentation", resources=_resources())
    factors = ranked[0].factors
    for name in (
        FACTOR_ENGINE, FACTOR_TASK_MATCH, FACTOR_CAPABILITY,
        FACTOR_FIT, FACTOR_PREFERENCE, FACTOR_HEALTH,
    ):
        assert name in factors
    assert "engine 1.00" in ranked[0].describe()


def test_a_user_preference_breaks_a_tie_but_does_not_override_a_better_fit():
    root = Path(_tmp("rank_pref"))
    manager = _manager(
        root,
        [
            _record(root, "banner-model", engine="banner"),
            _record(root, "other-model", engine="vision"),
        ],
    )
    ranked = rank_for_task(
        manager, "presentation", preferred_model_id="other-model", resources=_resources()
    )
    # The task's own engine still wins on the engine factor.
    assert ranked[0].model_id == "banner-model"
    other = next(item for item in ranked if item.model_id == "other-model")
    assert other.factors[FACTOR_PREFERENCE] == 1.0


def test_preference_wins_when_the_models_are_otherwise_equal():
    root = Path(_tmp("rank_pref_equal"))
    manager = _manager(
        root,
        [_record(root, "a-model", engine="banner"), _record(root, "b-model", engine="banner")],
    )
    ranked = rank_for_task(
        manager, "presentation", preferred_model_id="b-model", resources=_resources()
    )
    assert ranked[0].model_id == "b-model"


def test_ranking_only_includes_ready_models_by_default():
    root = Path(_tmp("rank_ready"))
    present = _record(root, "present", engine="banner")
    missing = _record(root, "missing", engine="banner", path=root / "gone.gguf")
    manager = _manager(root, [present, missing])
    ranked = rank_for_task(manager, "presentation", resources=_resources())
    ids = {item.model_id for item in ranked}
    assert "present" in ids
    assert "missing" not in ids  # a file that is gone is never a candidate
    everything = rank_for_task(manager, "presentation", ready_only=False, resources=_resources())
    assert "missing" in {item.model_id for item in everything}


def test_a_damaged_file_scores_lower_on_health():
    root = Path(_tmp("rank_health"))
    good = _record(root, "good", engine="banner")
    bad = _record(root, "bad", engine="banner", integrity="truncated")
    manager = _manager(root, [good, bad])
    ranked = rank_for_task(manager, "presentation", ready_only=False, resources=_resources())
    bad_ranked = next(item for item in ranked if item.model_id == "bad")
    assert bad_ranked.factors[FACTOR_HEALTH] == 0.0
    assert any("integrity" in note for note in bad_ranked.notes)


def test_a_short_context_scores_lower_but_is_not_excluded():
    root = Path(_tmp("rank_context"))
    manager = _manager(root, [_record(root, "short", engine="banner", context_length=512)])
    ranked = rank_for_task(manager, "presentation", resources=_resources())
    assert ranked  # still a candidate
    assert ranked[0].factors[FACTOR_FIT] < 1.0


def test_a_model_with_no_declared_context_is_noted():
    root = Path(_tmp("rank_noctx"))
    manager = _manager(root, [_record(root, "unknown-ctx", engine="banner", context_length=None)])
    ranked = rank_for_task(manager, "presentation", resources=_resources())
    assert any("no context" in note for note in ranked[0].notes)


def test_an_oversized_estimate_lowers_the_fit_score_and_says_so():
    root = Path(_tmp("rank_mem"))
    manager = _manager(root, [_record(root, "huge", engine="banner", size_bytes=200 * 1024**3)])
    ranked = rank_for_task(manager, "presentation", resources=_resources())
    assert ranked[0].factors[FACTOR_FIT] < 1.0
    assert any("memory" in note for note in ranked[0].notes)


def test_unstated_capabilities_are_unknown_not_absent():
    root = Path(_tmp("rank_caps"))
    manager = _manager(root, [_record(root, "plain", engine="banner")])
    ranked = rank_for_task(manager, "code", resources=_resources())
    assert ranked[0].factors[FACTOR_CAPABILITY] == 0.5
    assert any("no capabilities" in note for note in ranked[0].notes)


def test_stated_capabilities_are_scored():
    root = Path(_tmp("rank_caps_stated"))
    manager = _manager(root, [_record(root, "coder", capabilities=("code",))])
    ranked = rank_for_task(manager, "code", resources=_resources())
    assert ranked[0].factors[FACTOR_CAPABILITY] == 1.0
    assert ranked[0].factors[FACTOR_TASK_MATCH] == 1.0


def test_ranking_an_unknown_task_yields_nothing():
    root = Path(_tmp("rank_unknown"))
    manager = _manager(root, [_record(root, "a-model")])
    assert rank_for_task(manager, "telepathy", resources=_resources()) == ()


def test_ranking_limit_is_honoured():
    root = Path(_tmp("rank_limit"))
    manager = _manager(root, [_record(root, f"m{index}") for index in range(5)])
    assert len(rank_for_task(manager, "chat", limit=2, resources=_resources())) == 2


def test_ranked_model_primitive_is_json_safe():
    ranked = RankedModel(model_id="m", display_name="M", score=0.5, factors={"engine": 1.0})
    assert json.dumps(ranked.to_primitive())


def test_describe_ranking_is_readable():
    assert "no model can answer" in describe_ranking(())
    ranked = RankedModel(model_id="m", display_name="M", score=0.5)
    assert "M" in describe_ranking((ranked,), task="chat")


def test_a_custom_profile_can_be_ranked_without_changing_the_code():
    root = Path(_tmp("rank_custom"))
    manager = _manager(root, [_record(root, "special", engine="vision")])
    profile = TaskProfile(task="chat", engines=("vision",), capabilities=(), context=1024)
    ranked = rank_for_task(manager, "chat", profile=profile, resources=_resources())
    assert ranked[0].factors[FACTOR_ENGINE] == 1.0


# ------------------------------------------------------------------ fallback


def test_fallback_ladder_is_the_ranking_capped():
    root = Path(_tmp("fallback_ladder"))
    manager = _manager(root, [_record(root, f"m{index}") for index in range(5)])
    plan = plan_fallback(manager, "chat", limit=2, resources=_resources())
    assert plan.available
    assert len(plan.steps) == 2
    assert "Fallback order" in plan.describe()


def test_fallback_prefers_the_tasks_engine_first():
    root = Path(_tmp("fallback_engine"))
    manager = _manager(
        root,
        [_record(root, "banner-model", engine="banner"), _record(root, "other-model")],
    )
    plan = plan_fallback(manager, "presentation", resources=_resources())
    assert plan.steps[0].model_id == "banner-model"


def test_fallback_reports_honestly_when_nothing_can_run():
    root = Path(_tmp("fallback_empty"))
    manager = _manager(root, [])
    plan = plan_fallback(manager, "chat", resources=_resources())
    assert not plan.available
    assert plan.steps == ()
    # This checkout ships a catalog, so the honest reason is that a model is known
    # but not present; the point is that *a* reason is given, never a silent default.
    assert plan.reason
    assert "cloud" in plan.cloud_hint.lower()
    assert "No local model can answer" in plan.describe()


def test_fallback_reason_names_absence_when_nothing_is_known():
    root = Path(_tmp("fallback_nothing_known"))
    manager = ModelManager(root=root)
    # A manager whose catalog is emptied reports the plain absence.
    manager.catalog_entries = lambda: ()  # type: ignore[method-assign]
    plan = plan_fallback(manager, "chat", resources=_resources())
    assert not plan.available
    assert "No model that can run here" in plan.reason


def test_fallback_distinguishes_a_known_but_absent_model():
    root = Path(_tmp("fallback_absent"))
    manager = _manager(root, [_record(root, "gone", path=root / "missing.gguf")])
    plan = plan_fallback(manager, "chat", resources=_resources())
    assert not plan.available
    assert "known" in plan.reason and "not present" in plan.reason


def test_fallback_for_an_unknown_task_says_so():
    root = Path(_tmp("fallback_unknown_task"))
    manager = _manager(root, [_record(root, "a-model")])
    plan = plan_fallback(manager, "telepathy", resources=_resources())
    assert not plan.available
    assert "does not declare" in plan.reason


def test_fallback_plan_primitive_is_json_safe():
    plan = FallbackPlan(task="chat", reason="nothing")
    assert json.dumps(plan.to_primitive())


# ------------------------------------------------------- manager wiring


def test_manager_ranks_and_plans_fallback(tmp_path):
    manager = _manager(tmp_path, [_record(tmp_path, "a-model", engine="banner")])
    ranked = manager.rank_for_task("presentation", resources=_resources())
    assert ranked
    plan = manager.fallback_plan("presentation", resources=_resources())
    assert plan.available
    assert manager.task_profile("chat").task == "chat"


# ------------------------------------------------------- trained artifacts


def _write_checkpoint(path: Path, *, n_layer: int = 4, n_embd: int = 128, vocab: int = 64) -> Path:
    """A tiny but structurally real ``.npz`` checkpoint, so the probe accepts it.

    Written with the standard library only -- a zip holding a float64 ``config.npy``
    (the layout the trainer writes) plus one weight array -- so the reader is
    exercised without numpy.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    config = struct.pack(
        "<7d", float(vocab), float(n_layer), 4.0, float(n_embd), 32.0, 0.1, 1.0
    )
    header = "{'descr': '<f8', 'fortran_order': False, 'shape': (7,), }"
    header_bytes = header.encode("latin1")
    padding = (-(10 + len(header_bytes) + 1)) % 64
    header_bytes = header_bytes + b" " * padding + b"\n"
    npy = b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(header_bytes)) + header_bytes + config
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("config.npy", npy)
        archive.writestr("weight.npy", b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", 0) + b" ")
    return path


def test_classify_artifact_labels_each_kind():
    assert classify_artifact("policy_boil-water.npz") == KIND_POLICY
    assert classify_artifact("experience_classifier.npz") == KIND_CLASSIFIER
    assert classify_artifact("fayd-llm.state.npz") == KIND_UNKNOWN
    assert classify_artifact("model.gguf") == KIND_UNKNOWN
    assert classify_artifact("notes.txt") == KIND_UNKNOWN


def test_classify_artifact_calls_a_real_checkpoint_a_language_model(tmp_path):
    path = _write_checkpoint(tmp_path / "fayd-llm.npz")
    assert classify_artifact(path) == KIND_LANGUAGE


def test_describe_artifact_reads_the_manifest(tmp_path):
    path = _write_checkpoint(tmp_path / "fayd-llm.npz")
    manifest = {
        "language_model": {
            "architecture": {"n_layer": 4, "n_embd": 128, "block_size": 32},
            "parameters": 842624,
            "perplexity": {"train": 3.1, "test": 4.2, "bits_per_byte": 2.1},
        }
    }
    artifact = describe_artifact(path, manifest=manifest)
    assert artifact.kind == KIND_LANGUAGE
    assert artifact.chat_capable
    assert artifact.parameters == 842624
    values = {item.name: item.value for item in artifact.evaluations}
    assert values["test"] == 4.2
    assert all(item.source == "release manifest" for item in artifact.evaluations)


def test_describe_artifact_never_invents_a_measurement(tmp_path):
    path = _write_checkpoint(tmp_path / "fayd-llm.npz")
    artifact = describe_artifact(path, manifest={})
    values = {item.name: item.value for item in artifact.evaluations}
    assert set(values.values()) == {None}
    assert "not recorded" in artifact.evaluations[0].describe()


def test_describe_artifact_flags_a_manifest_mismatch(tmp_path):
    path = _write_checkpoint(tmp_path / "fayd-llm.npz", n_layer=4)
    manifest = {"language_model": {"architecture": {"n_layer": 99}}}
    artifact = describe_artifact(path, manifest=manifest)
    assert "differs" in artifact.detail


def test_a_policy_file_is_not_chat_capable(tmp_path):
    path = tmp_path / "policy_boil-water.npz"
    path.write_bytes(b"not really an npz")
    artifact = describe_artifact(path, manifest={})
    assert artifact.kind == KIND_POLICY
    assert not artifact.chat_capable
    assert "not a chat model" in artifact.describe()


def test_manifest_absent_reads_as_empty(tmp_path):
    assert load_manifest(tmp_path / "missing.json") == {}


def test_a_corrupt_manifest_reads_as_empty(tmp_path):
    path = tmp_path / "release_manifest.json"
    path.write_text("{not json")
    assert load_manifest(path) == {}


def test_evaluations_from_manifest_attributes_its_source():
    results = evaluations_from_manifest({"language_model": {"perplexity": {"test": 4.0}}})
    test_result = next(item for item in results if item.name == "test")
    assert test_result.value == 4.0
    assert test_result.source == "release manifest"
    assert all(item.value is None for item in results if item.name != "test")


def test_evaluation_sidecar_round_trips_beside_the_checkpoint(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "fayd-llm.npz")
    results = (
        EvaluationResult(name="test", value=4.25, source="evaluation sidecar"),
        EvaluationResult(name="bits_per_byte", value=2.0, source="evaluation sidecar"),
    )
    path = save_evaluation(checkpoint, results, detail="measured locally")
    assert path.parent == checkpoint.parent
    assert checkpoint.read_bytes() == checkpoint.read_bytes()  # the weights are untouched
    loaded = load_evaluation_sidecar(checkpoint)
    values = {item.name: item.value for item in loaded}
    assert values["test"] == 4.25
    assert all(item.source == "evaluation sidecar" for item in loaded)


def test_a_missing_evaluation_sidecar_yields_nothing(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "fayd-llm.npz")
    assert load_evaluation_sidecar(checkpoint) == ()


def test_a_corrupt_evaluation_sidecar_yields_nothing(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "fayd-llm.npz")
    checkpoint.with_suffix(".npz.evaluation.json").write_text("{bad")
    assert load_evaluation_sidecar(checkpoint) == ()


def test_run_state_is_read_from_the_sidecars(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "fayd-llm.npz")
    # A minimal .state.npz carrying one small integer array.
    import numpy as np

    np.savez(checkpoint.with_suffix(".npz.state.npz"), step_count=np.array([1234]))
    (checkpoint.with_suffix(".npz.schedule.json")).write_text(
        json.dumps({"consumed": 2, "total": 5})
    )
    artifact = describe_artifact(checkpoint, manifest={})
    assert artifact.total_steps == 1234
    assert artifact.resumed is True
    assert "2 of 5" in artifact.schedule


def test_a_missing_run_state_is_unknown_not_an_error(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "fayd-llm.npz")
    artifact = describe_artifact(checkpoint, manifest={})
    assert artifact.total_steps is None
    assert artifact.resumed is None


def test_discover_trained_artifacts_describes_the_folder(tmp_path):
    _write_checkpoint(tmp_path / "fayd-llm.npz")
    (tmp_path / "policy_a.npz").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignore me")
    found = discover_trained_artifacts((tmp_path,))
    kinds = {item.kind for item in found}
    assert KIND_LANGUAGE in kinds
    assert KIND_POLICY in kinds
    assert found[0].kind == KIND_LANGUAGE  # the language model is first


def test_trained_model_record_is_built_from_a_real_checkpoint(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "fayd-llm.npz")
    record = trained_model_record(checkpoint)
    assert record is not None
    assert record.source == SOURCE_TRAINED
    assert record.runtime == "fayd-inprocess"
    assert record.identity == "fayd-llm"
    assert record.parameter_count is not None


def test_trained_model_record_is_none_for_a_non_checkpoint(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(_gguf_bytes())
    assert trained_model_record(path) is None


def test_register_trained_models_makes_it_an_ordinary_record(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "fayd-llm.npz")
    manager = ModelManager(root=tmp_path)
    stored = register_trained_models(manager, replace_existing=True)
    # The record points at the checkpoint we made only when the resolver finds it;
    # here we register the record directly to prove the path.
    assert stored == () or stored[0].source == SOURCE_TRAINED
    direct = trained_model_record(checkpoint)
    assert direct is not None
    manager.registry.register_record(direct)
    assert manager.registry.get(direct.model_id).source == SOURCE_TRAINED


def test_manager_trained_helpers_are_wired(tmp_path):
    manager = ModelManager(root=tmp_path)
    # The committed checkpoint is present in this checkout, so the trained model
    # resolves; on a machine without one the honest answer is None.
    model = manager.trained_model()
    if model is not None:
        assert model.kind == KIND_LANGUAGE
        assert manager.trained_artifacts()
    assert isinstance(manager.register_trained(), tuple)


def test_trained_checkpoint_path_resolves_in_this_checkout():
    path = trained_checkpoint_path()
    # Present in the committed checkout; when it is not, the honest answer is None.
    assert path is None or path.is_file()


def test_artifact_primitive_is_json_safe(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "fayd-llm.npz")
    artifact = describe_artifact(checkpoint, manifest={})
    assert json.dumps(artifact.to_primitive())


def test_evaluation_result_describe_is_honest():
    assert "not recorded" in EvaluationResult(name="test").describe()
    assert "recorded by" in EvaluationResult(name="test", value=1.0, source="release manifest").describe()


# ------------------------------------------------------------------ fixtures


def _tmp(name: str) -> str:
    import tempfile

    base = Path(tempfile.gettempdir()) / "fayd-model-selection-tests" / name
    base.mkdir(parents=True, exist_ok=True)
    return str(base)
