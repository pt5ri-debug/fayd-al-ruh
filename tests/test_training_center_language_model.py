"""The Language Model section of the Training Center: every training surface in one place.

The project trains its own language model through :mod:`ml.llm`, and it used to be
reachable from several scattered commands. This file pins the *one* home that now
drives all of it -- :class:`core.training_center.language_model.LanguageModelEngine`,
the :class:`gui.training_center.TrainingCenterService` delegation, the catalogue,
and the headless CLI -- with a real (tiny, fast) training run so the wiring is
proved end to end and not merely asserted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from cli import training_center as cli
from core.training_center import catalog
from core.training_center.language_model import (
    LanguageModelEngine,
    LanguageModelSettings,
)
from gui.training_center import TrainingCenterService


def _tiny() -> LanguageModelSettings:
    """A fast, self-contained run: no knowledge base, no characters, no resume."""
    return LanguageModelSettings(
        preset="small",
        steps=4,
        repeats=2,
        structured_repeats=2,
        kb_max_chunks=0,
        include_characters=False,
        include_runtime=False,
        resume=False,
        log_every=2,
        val_every=2,
        weight_decay=0.0,
        restore_best=False,
    )


# -- the settings ---------------------------------------------------------


def test_settings_defaults_are_a_serious_release_run() -> None:
    settings = LanguageModelSettings()
    assert settings.preset == "release-1"
    assert settings.resume is True  # continuing is the default
    assert settings.weight_decay == 0.01
    assert settings.lr_schedule == "cosine"
    assert settings.restore_best is True
    assert settings.shard_count == 0  # whole corpus


def test_settings_round_trip_and_describe() -> None:
    settings = _tiny()
    assert LanguageModelSettings.from_primitive(settings.to_primitive()) == settings
    text = settings.describe()
    assert "small" in text
    assert "resume" in text


def test_from_primitive_ignores_unknown_keys() -> None:
    settings = LanguageModelSettings.from_primitive({"steps": 7, "nonsense": 1})
    assert settings.steps == 7
    assert LanguageModelSettings.from_primitive("not a dict") == LanguageModelSettings()


def test_set_setting_persists_and_reset_restores(tmp_path: Path) -> None:
    engine = LanguageModelEngine(tmp_path)
    engine.set_setting("steps", 12)
    assert engine.settings().steps == 12
    assert LanguageModelEngine(tmp_path).settings().steps == 12  # persisted
    engine.reset_settings()
    assert engine.settings().steps == 4000


def test_set_setting_refuses_an_unknown_option(tmp_path: Path) -> None:
    engine = LanguageModelEngine(tmp_path)
    with pytest.raises(KeyError):
        engine.set_setting("does_not_exist", 1)


# -- the checkpoint -------------------------------------------------------


def test_checkpoint_report_is_honest_when_absent(tmp_path: Path) -> None:
    report = LanguageModelEngine(tmp_path).checkpoint_report()
    assert report.exists is False
    assert "Train one from scratch" in report.next_step


def test_checkpoint_report_reads_a_trained_file(tmp_path: Path) -> None:
    engine = LanguageModelEngine(tmp_path)
    outcome = engine.train(_tiny())
    assert outcome.ok
    report = engine.checkpoint_report()
    assert report.exists is True
    assert report.steps == 4
    assert report.parameters and report.parameters > 0
    assert report.preset == "small"
    assert report.architecture["n_layer"] == 3


# -- the corpus -----------------------------------------------------------


def test_corpus_summary_reports_the_real_documents(tmp_path: Path) -> None:
    engine = LanguageModelEngine(tmp_path)
    summary = engine.corpus_summary(_tiny())
    assert summary["curriculum_sentences"] > 0
    assert summary["corpus_sentences"] > 0
    assert "balance" in summary
    text = engine.describe_corpus(_tiny())
    assert "curriculum_sentences" in text


# -- training -------------------------------------------------------------


def test_train_streams_lines_and_steps(tmp_path: Path) -> None:
    engine = LanguageModelEngine(tmp_path)
    lines: list[str] = []
    steps: list[tuple[int, float, int]] = []
    outcome = engine.train(_tiny(), on_line=lines.append, on_step=lambda s, loss, t: steps.append((s, loss, t)))
    assert outcome.ok, outcome.error
    assert outcome.steps == 4
    assert outcome.final_loss < outcome.initial_loss
    assert outcome.parameters > 0
    assert lines  # the trainer's own output reached the sink
    assert steps and steps[-1][0] == 4
    assert all(total >= step for step, _loss, total in steps)


def test_train_resumes_and_raises_the_total(tmp_path: Path) -> None:
    engine = LanguageModelEngine(tmp_path)
    first = engine.train(_tiny())
    assert first.total_steps == 4
    resumed = engine.train(_tiny().with_values(resume=True, steps=3))
    assert resumed.ok
    assert resumed.resumed is True
    assert resumed.total_steps == 7


def test_train_reports_a_failure_honestly(tmp_path: Path) -> None:
    def boom(**kwargs: object) -> None:
        raise RuntimeError("no trainer here")

    engine = LanguageModelEngine(tmp_path, trainer=boom)
    outcome = engine.train(_tiny())
    assert outcome.ok is False
    assert "RuntimeError" in outcome.error


def test_train_reports_the_outcome_as_a_primitive(tmp_path: Path) -> None:
    outcome = LanguageModelEngine(tmp_path).train(_tiny())
    payload = outcome.to_primitive()
    assert payload["ok"] is True
    assert payload["steps"] == 4


# -- generation -----------------------------------------------------------


def test_generate_requires_a_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LanguageModelEngine(tmp_path).generate("porosity")


def test_generate_samples_from_the_trained_model(tmp_path: Path) -> None:
    engine = LanguageModelEngine(tmp_path)
    engine.train(_tiny())
    text = engine.generate("porosity", max_new_tokens=8, seed=1)
    assert isinstance(text, str)
    assert text  # even a byte model continues the prompt


# -- the plan -------------------------------------------------------------


def test_plan_names_the_real_headless_command(tmp_path: Path) -> None:
    engine = LanguageModelEngine(tmp_path)
    plan = engine.plan(_tiny())
    assert plan.startswith("python -m cli training-center lm-train")
    assert "--steps 4" in plan
    assert "--fresh" in plan  # resume is off in the tiny settings


def test_plan_omits_fresh_when_resuming(tmp_path: Path) -> None:
    plan = LanguageModelEngine(tmp_path).plan(LanguageModelSettings(resume=True))
    assert "--fresh" not in plan


# -- choices --------------------------------------------------------------


def test_choices_are_read_from_the_real_catalogs(tmp_path: Path) -> None:
    engine = LanguageModelEngine(tmp_path)
    presets = dict(engine.preset_choices())
    assert "small" in presets and "release-1" in presets
    assert dict(engine.tokenizer_choices())["byte"]
    assert "cosine" in engine.lr_schedule_choices()
    assert "learned-large" in engine.learned_tier_choices()
    pack = dict(engine.pack_choices())
    assert "fayd-llm" in pack


# -- the release run ------------------------------------------------------


def test_release_maps_every_setting_onto_the_config(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def runner(settings: LanguageModelSettings) -> dict[str, object]:
        captured["settings"] = settings
        return {"manifest_path": str(tmp_path / "release_manifest.json")}

    engine = LanguageModelEngine(tmp_path, release_runner=runner)
    outcome = engine.release(_tiny())
    assert outcome.ok
    assert outcome.manifest_path.endswith("release_manifest.json")
    assert captured["settings"] == _tiny()


def test_release_reports_a_failure_honestly(tmp_path: Path) -> None:
    def boom(_settings: LanguageModelSettings) -> dict[str, object]:
        raise RuntimeError("no release here")

    outcome = LanguageModelEngine(tmp_path, release_runner=boom).release(_tiny())
    assert outcome.ok is False
    assert "RuntimeError" in outcome.error


# -- the service ----------------------------------------------------------


def test_service_delegates_and_records_history(tmp_path: Path) -> None:
    service = TrainingCenterService(tmp_path)
    assert service.lm_settings().preset == "release-1"
    service.lm_set_setting("steps", 9)
    assert service.lm_settings().steps == 9
    assert "language_model.steps" in "\n".join(
        line.describe() for line in service.history_lines(limit=10)
    )
    assert service.lm_settings_text()
    assert service.lm_settings_keys()
    assert "small" in service.lm_settings_values("preset")


def test_service_refuses_an_unknown_option(tmp_path: Path) -> None:
    service = TrainingCenterService(tmp_path)
    with pytest.raises(KeyError):
        service.lm_set_setting("nope", 1)


def test_service_trains_through_the_engine(tmp_path: Path) -> None:
    service = TrainingCenterService(tmp_path)
    outcome = service.lm_train(settings=_tiny())
    assert outcome.ok, outcome.error
    assert service.lm_checkpoint().exists
    assert "Checkpoint" in service.lm_status()


def test_service_records_a_training_run_in_history(tmp_path: Path) -> None:
    service = TrainingCenterService(tmp_path)
    service.lm_train(settings=_tiny())
    lines = "\n".join(line.describe() for line in service.history_lines(limit=20))
    assert "language model" in lines.lower()


# -- the catalogue --------------------------------------------------------


def test_catalogue_has_the_language_model_section() -> None:
    assert catalog.SECTION_LANGUAGE_MODEL in catalog.SECTIONS
    caps = catalog.capabilities_for(catalog.SECTION_LANGUAGE_MODEL)
    assert len(caps) >= 15
    keys = {cap.key for cap in caps}
    for expected in (
        "lm.status", "lm.settings.show", "lm.settings.set", "lm.train",
        "lm.train.fresh", "lm.train.sharded", "lm.corpus", "lm.release",
        "lm.evaluate", "lm.generate", "lm.weights.install", "lm.weights.import",
        "lm.weights.fetch", "lm.weights.build", "lm.weights.learned", "lm.plan",
    ):
        assert expected in keys, expected


def test_every_language_model_capability_is_explained_bilingually() -> None:
    for cap in catalog.capabilities_for(catalog.SECTION_LANGUAGE_MODEL):
        assert cap.summary and cap.outcome and cap.cli
        assert cap.title_ar and cap.summary_ar


def test_governed_language_model_actions_name_their_confirmation() -> None:
    governed = [cap for cap in catalog.capabilities_for(catalog.SECTION_LANGUAGE_MODEL) if cap.governed]
    assert governed
    for cap in governed:
        assert "--approve" in cap.cli or "asks first" in cap.describe()


# -- the CLI --------------------------------------------------------------


def _run(capsys, *argv: str, root: Path) -> tuple[int, str]:
    code = cli.main(["--root", str(root), *argv])
    return code, capsys.readouterr().out


def test_cli_settings_set_and_read(tmp_path: Path, capsys) -> None:
    code, _ = _run(capsys, "lm-settings", "--set", "steps=6", root=tmp_path)
    assert code == cli.EXIT_OK
    code, out = _run(capsys, "lm-settings", "--json", root=tmp_path)
    assert code == cli.EXIT_OK
    import json

    assert json.loads(out)["steps"] == 6


def test_cli_settings_refuses_an_unknown_option(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "lm-settings", "--set", "nope=1", root=tmp_path)
    assert code == cli.EXIT_USAGE
    assert "invalid" in out


def test_cli_settings_refuses_a_malformed_pair(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "lm-settings", "--set", "steps", root=tmp_path)
    assert code == cli.EXIT_USAGE
    assert "invalid" in out


def test_cli_plan_prints_the_command(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "lm-plan", root=tmp_path)
    assert code == cli.EXIT_OK
    assert "lm-train" in out


def test_cli_status_and_corpus(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "lm-status", root=tmp_path)
    assert code == cli.EXIT_OK
    assert "No checkpoint" in out
    code, out = _run(capsys, "lm-corpus", root=tmp_path)
    assert code == cli.EXIT_OK
    assert "curriculum_sentences" in out


def test_cli_train_is_a_dry_run_without_approval(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "lm-train", "--steps", "4", root=tmp_path)
    assert code == cli.EXIT_OK
    assert "Dry run" in out


def test_cli_train_runs_with_approval(tmp_path: Path, capsys) -> None:
    _run(capsys, "lm-settings", "--set", "preset=small", "--set", "kb_max_chunks=0",
         "--set", "include_characters=false", "--set", "include_runtime=false",
         "--set", "resume=false", root=tmp_path)
    code, out = _run(capsys, "lm-train", "--approve", "--steps", "4", root=tmp_path)
    assert code == cli.EXIT_OK
    assert "Trained 4 steps" in out


def test_cli_generate_needs_a_checkpoint(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "lm-generate", "--prompt", "porosity", root=tmp_path)
    assert code == cli.EXIT_UNAVAILABLE
    assert "not available" in out


def test_cli_generate_samples_after_training(tmp_path: Path, capsys) -> None:
    _run(capsys, "lm-settings", "--set", "preset=small", "--set", "kb_max_chunks=0",
         "--set", "include_characters=false", "--set", "include_runtime=false",
         "--set", "resume=false", root=tmp_path)
    _run(capsys, "lm-train", "--approve", "--steps", "4", root=tmp_path)
    code, out = _run(capsys, "lm-generate", "--prompt", "porosity", "--seed", "1",
                      "--max-new-tokens", "8", root=tmp_path)
    assert code == cli.EXIT_OK
    assert out.strip()


def test_cli_weights_lists_the_learned_tiers(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "lm-weights", "--list", root=tmp_path)
    assert code == cli.EXIT_OK
    assert "learned-large" in out


def test_cli_weights_requires_an_action(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "lm-weights", root=tmp_path)
    assert code == cli.EXIT_USAGE
    assert "invalid" in out


def test_cli_weights_installs_the_committed_checkpoint(tmp_path: Path, capsys) -> None:
    code, _out = _run(capsys, "lm-weights", "--install", "fayd-llm", root=tmp_path)
    assert code == cli.EXIT_OK
    assert (tmp_path / "ml" / "llm" / "checkpoints" / "fayd-llm.npz").exists()


# -- the one home (Phase E) ----------------------------------------------


def test_every_language_model_menu_entry_routes_through_the_training_center() -> None:
    """The scattered trainers are gone; the menu now points into the Training Center."""
    from cli import menu

    training = {
        label: (action, tuple(argv))
        for label, action, argv in menu.MENU
        if action is menu.run_training_center
    }
    # The visible entries carry the language-model controls.
    visible = {label for label, _action, _argv in menu.visible_menu()}
    assert "Training Center: train or continue the language model (فيض الروح)" in visible
    assert "Training Center: the language-model training settings (all options)" in visible
    assert "Training Center: train everything for a release" in visible
    # The old labels stay reachable but are hidden and route into the Center.
    for label in (
        "Train the language model from all repositories",
        "Train the language model on the local mouths' own chats",
        "Train the language model on the characters' own learning",
        "Train everything for a release",
        "Train everything for a release (including the local mouths' chats)",
    ):
        assert label in training, label
        assert label in menu.HIDDEN_MENU_LABELS, label
        assert label not in visible, label


def test_no_menu_entry_runs_the_scattered_trainers() -> None:
    """No menu entry points at the old ``ml.llm``/``ml.release`` runners directly."""
    from cli import menu

    actions = {action for _label, action, _argv in menu.MENU}
    # The aliases still exist (a caller may resolve them) but no entry uses them.
    assert menu.run_llm not in actions
    assert menu.run_release not in actions


def test_every_gui_training_action_routes_through_the_training_center() -> None:
    from gui import actions

    by_key = {action.key: action for action in actions.ACTIONS}
    for key in ("llm", "llm-fresh", "llm-settings", "llm-weights", "release"):
        assert by_key[key].command is actions.run_training_center, key
    # The GUI no longer imports the scattered trainers at all.
    assert not hasattr(actions, "run_llm")
    assert not hasattr(actions, "run_release")
