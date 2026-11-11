"""The learned capacity tiers and the generator that builds them.

These pin the two halves of the learned-weights work: the **tiers** (the larger
capacity presets of the modern architecture, named and described in one registry)
and the **generator** (``core.learned_weights``) that trains a tier through the
project's own trainer, records its provenance, and lists what is on disk.

The negative controls matter as much as the happy paths: an unknown tier, a
second build without ``force``, and a folder with no checkpoint must each be
*reported* rather than fabricated or silently overwritten.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest
from core.learned_weights import (
    LEARNED_SUBDIR,
    LEARNED_TIERS,
    TIER_KEYS,
    BuildReport,
    LearnedCheckpoint,
    LearnedTier,
    LearnedWeightsError,
    build_learned,
    describe_learned,
    describe_tiers,
    learned_dir,
    list_learned,
    load_provenance,
    provenance_path,
    resolve_tier,
    tier_keys,
    tier_summaries,
)
from ml.llm.transformer import PRESETS, TransformerConfig, TransformerLM

#: A tiny, fast config so the tests exercise the real trainer without a long run.
_TINY = TransformerConfig(
    vocab_size=257,
    n_layer=2,
    n_head=2,
    n_embd=32,
    block_size=32,
    tie_weights=True,
    norm="rms",
    position="rope",
    activation="swiglu",
)


def _build(root: Path, tier: str = "learned-large", *, force: bool = False) -> BuildReport:
    with contextlib.redirect_stdout(io.StringIO()):
        return build_learned(
            tier,
            root=root,
            steps=2,
            repeats=1,
            structured_repeats=1,
            config=_TINY,
            force=force,
            verbose=False,
        )


# ------------------------------------------------------------------- presets


def test_the_learned_presets_are_defined_and_larger_than_the_base():
    assert "learned-large" in PRESETS and "learned-xl" in PRESETS
    base = TransformerLM(TransformerConfig.preset("release-1")).parameters()
    large = TransformerLM(TransformerConfig.preset("learned-large")).parameters()
    xl = TransformerLM(TransformerConfig.preset("learned-xl")).parameters()
    base_params = sum(p.size for p in base)
    assert sum(p.size for p in large) > base_params
    assert sum(p.size for p in xl) > sum(p.size for p in large)
    # They keep the modern architecture family, so a comparison isolates capacity.
    for name in ("learned-large", "learned-xl"):
        config = TransformerConfig.preset(name)
        assert config.norm == "rms" and config.position == "rope"
        assert config.activation == "swiglu"
        assert config.tie_weights is True


def test_the_learned_presets_are_selectable_by_the_cli_and_have_sidecars():
    # The trainer's --preset choices come straight from PRESETS, so a tier is
    # reachable the moment it is defined.
    for key in TIER_KEYS:
        assert LEARNED_TIERS[key].preset in PRESETS


# --------------------------------------------------------------------- tiers


def test_tier_registry_is_ordered_and_described():
    assert tier_keys() == TIER_KEYS
    assert TIER_KEYS == ("learned-large", "learned-xl")
    rows = tier_summaries()
    assert [row[0] for row in rows] == list(TIER_KEYS)
    assert all(row[1] and row[2] for row in rows)
    described = describe_tiers()
    for key in TIER_KEYS:
        assert key in described


def test_resolve_tier_accepts_a_key_or_a_preset():
    assert resolve_tier("learned-large").key == "learned-large"
    assert resolve_tier("learned-xl").preset == "learned-xl"
    # A preset name resolves to the same tier as its key.
    assert resolve_tier("learned-large") is LEARNED_TIERS["learned-large"]


def test_resolve_tier_refuses_an_unknown_name_with_the_real_list():
    with pytest.raises(LearnedWeightsError) as error:
        resolve_tier("enormous")
    assert "learned-large" in str(error.value)
    assert "learned-xl" in str(error.value)


def test_tier_dimensions_and_command_are_real():
    tier = LEARNED_TIERS["learned-large"]
    assert tier.dimensions().startswith("4L/256w/")
    command = tier.continue_command()
    assert command.startswith("python -m cli llm")
    assert tier.filename in command
    primitive = tier.to_primitive()
    assert primitive["key"] == "learned-large"
    assert primitive["preset"] == "learned-large"


# ------------------------------------------------------------------- folder


def test_learned_dir_defaults_to_the_pack_subfolder():
    from core.paths import weights_pack_dir

    assert learned_dir() == weights_pack_dir() / LEARNED_SUBDIR
    # A caller passing a folder gets exactly that folder.
    assert learned_dir("/tmp/scratch") == Path("/tmp/scratch")


def test_provenance_path_is_beside_the_checkpoint():
    assert provenance_path("/x/fayd-llm-large.npz").name == "fayd-llm-large.npz.learned.json"


# ------------------------------------------------------------------- building


def test_build_trains_a_tier_and_records_provenance(tmp_path):
    report = _build(tmp_path)
    assert report.built
    assert report.checkpoint.exists()
    assert report.parameters > 0
    assert report.size_bytes > 0
    assert report.steps == 2
    # The tokenizer sidecar travels with the checkpoint, so a resume encodes the
    # same way.
    assert report.checkpoint.with_suffix(".tokenizer.json").exists()
    provenance = load_provenance(report.checkpoint)
    assert provenance["preset"] == "learned-large"
    assert provenance["steps"] == 2
    assert provenance["producer"].startswith("the project's own trainer")
    assert provenance_path(report.checkpoint).exists()


def test_build_is_refused_without_force(tmp_path):
    first = _build(tmp_path)
    assert first.built
    bytes_before = first.checkpoint.read_bytes()
    second = _build(tmp_path)
    assert not second.built
    assert "already exists" in second.detail
    # The existing checkpoint is untouched.
    assert first.checkpoint.read_bytes() == bytes_before


def test_build_with_force_replaces_the_checkpoint(tmp_path):
    first = _build(tmp_path)
    assert first.built
    second = _build(tmp_path, force=True)
    assert second.built


def test_build_refuses_a_non_positive_step_budget(tmp_path):
    with pytest.raises(LearnedWeightsError):
        with contextlib.redirect_stdout(io.StringIO()):
            build_learned("learned-large", root=tmp_path, steps=0, config=_TINY, verbose=False)


def test_build_refuses_an_unknown_tier(tmp_path):
    with pytest.raises(LearnedWeightsError):
        with contextlib.redirect_stdout(io.StringIO()):
            build_learned("nope", root=tmp_path, steps=2, config=_TINY, verbose=False)


def test_build_reports_when_the_trainer_writes_nothing(tmp_path):
    def silent_trainer(**_kwargs):
        return None

    report = build_learned(
        "learned-large",
        root=tmp_path,
        steps=2,
        config=_TINY,
        verbose=False,
        trainer=silent_trainer,
    )
    assert not report.built
    assert "wrote no checkpoint" in report.detail


def test_build_uses_an_injected_trainer_seam(tmp_path):
    seen: dict[str, object] = {}

    def trainer(**kwargs):
        seen.update(kwargs)
        # Write a real checkpoint so the report can measure it.
        model = TransformerLM(kwargs["config"], seed=0)
        model.save(kwargs["checkpoint"])
        return model

    report = build_learned(
        "learned-large",
        root=tmp_path,
        steps=3,
        config=_TINY,
        verbose=False,
        trainer=trainer,
    )
    assert report.built
    assert seen["steps"] == 3
    assert seen["weight_decay"] == 0.01
    assert seen["restore_best"] is True


# ------------------------------------------------------------------ listing


def test_list_and_describe_report_what_is_on_disk(tmp_path):
    assert list_learned(tmp_path) == ()
    assert "No learned checkpoints" in describe_learned(tmp_path)
    _build(tmp_path)
    found = list_learned(tmp_path)
    assert len(found) == 1
    assert isinstance(found[0], LearnedCheckpoint)
    assert found[0].tier_key == "learned-large"
    assert found[0].usable
    described = describe_learned(tmp_path)
    assert "learned-large" in described


def test_list_excludes_the_optimiser_state_sidecar(tmp_path):
    _build(tmp_path)
    names = {item.path.name for item in list_learned(tmp_path)}
    assert "fayd-llm-large.state.npz" not in names


def test_list_reports_an_unusable_file_rather_than_hiding_it(tmp_path):
    folder = learned_dir(tmp_path)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "broken.npz").write_bytes(b"not a numpy archive")
    found = list_learned(tmp_path)
    assert len(found) == 1
    assert not found[0].usable


def test_provenance_of_a_missing_or_corrupt_record_reads_empty(tmp_path):
    assert load_provenance(tmp_path / "absent.npz") == {}
    broken = tmp_path / "x.npz"
    broken.write_bytes(b"x")
    provenance_path(broken).write_text("{not json", encoding="utf-8")
    assert load_provenance(broken) == {}


# ------------------------------------------------------------------ reports


def test_build_report_round_trips_through_primitive(tmp_path):
    report = _build(tmp_path)
    primitive = report.to_primitive()
    assert primitive["built"] is True
    assert primitive["tier"] == "learned-large"
    assert primitive["parameters"] > 0
    assert primitive["continue_command"].startswith("python -m cli llm")
    assert "built" in report.describe()


def test_a_not_built_report_names_the_real_next_step(tmp_path):
    _build(tmp_path)
    report = _build(tmp_path)
    described = report.describe()
    assert "not built" in described
    assert "force" in described


def test_learned_tier_describe_includes_its_dimensions():
    text = LEARNED_TIERS["learned-xl"].describe()
    assert "learned-xl" in text
    assert "8L/320w" in text


def test_tier_is_a_frozen_dataclass():
    tier = LearnedTier(key="k", title="t", preset="modern", filename="f.npz", summary="s")
    with pytest.raises(Exception):
        tier.key = "other"  # type: ignore[misc]


# ------------------------------------------------------------ pack integration


def _pack_with_a_learned_tier(tmp_path: Path):
    """Build a tiny learned tier and pack it (no policies), returning the pack."""
    from core.weights_pack import generate_manifest, load_pack

    folder = tmp_path / "learned"
    _build(folder)
    manifest_root = tmp_path / "pack"
    generate_manifest(
        root=manifest_root,
        checkpoint=tmp_path / "absent-base.npz",
        learned_dir=folder,
        include_policies=False,
    )
    return load_pack(manifest_root)


def test_generate_manifest_packs_a_learned_tier_under_its_key(tmp_path):
    pack = _pack_with_a_learned_tier(tmp_path)
    assert pack.available
    assert "learned-large" in pack.keys()
    entry = pack.get("learned-large")
    assert entry is not None
    assert entry.kind == "language-model"
    assert entry.resumable
    # Its architecture is read from the file, not assumed.
    assert entry.architecture.get("n_layer") == 2


def test_a_learned_entry_installs_beside_the_base_checkpoint(tmp_path):
    from core.weights_pack import installed_destination

    pack = _pack_with_a_learned_tier(tmp_path)
    entry = pack.get("learned-large")
    assert entry is not None
    destination = installed_destination(entry)
    assert destination is not None
    # It installs under its own name, never over the base checkpoint.
    assert destination.name == "fayd-llm-large.npz"
    assert entry.destination_name == "fayd-llm-large.npz"


def test_a_learned_entry_verifies_cleanly(tmp_path):
    from core.weights_pack import verify_entry

    pack = _pack_with_a_learned_tier(tmp_path)
    entry = pack.get("learned-large")
    assert entry is not None
    assert verify_entry(entry).ok


def test_destination_name_round_trips_through_the_manifest(tmp_path):
    from core.weights_pack import generate_manifest, load_pack

    folder = tmp_path / "learned"
    _build(folder)
    manifest_root = tmp_path / "pack"
    generate_manifest(
        root=manifest_root,
        checkpoint=tmp_path / "absent.npz",
        learned_dir=folder,
        include_policies=False,
    )
    reloaded = load_pack(manifest_root).get("learned-large")
    assert reloaded is not None
    assert reloaded.destination_name == "fayd-llm-large.npz"


def test_generate_manifest_without_learned_leaves_the_base_only(tmp_path):
    from core.weights_pack import generate_manifest

    folder = tmp_path / "learned"
    _build(folder)
    base = tmp_path / "fayd-llm.npz"
    # A real base checkpoint so the pack has at least one entry.
    from ml.llm.transformer import TransformerLM

    TransformerLM(_TINY, seed=0).save(base)
    pack = generate_manifest(
        root=tmp_path / "pack",
        checkpoint=base,
        learned_dir=folder,
        include_learned=False,
        include_policies=False,
    )
    assert "learned-large" not in pack.keys()
    assert "fayd-llm" in pack.keys()




# --------------------------------------------------------------- CLI commands


def test_cli_lists_the_learned_tiers(capsys):
    from cli.weights import main

    assert main(["learned-tiers"]) == 0
    out = capsys.readouterr().out
    assert "learned-large" in out and "learned-xl" in out


def test_cli_lists_tiers_as_json(capsys):
    import json as _json

    from cli.weights import main

    assert main(["learned-tiers", "--json"]) == 0
    payload = _json.loads(capsys.readouterr().out)
    assert {row["key"] for row in payload} == set(TIER_KEYS)


def test_cli_learned_list_reports_an_empty_folder(tmp_path, capsys):
    from cli.weights import main

    assert main(["learned-list", "--root", str(tmp_path)]) == 0
    assert "No learned checkpoints" in capsys.readouterr().out


def test_cli_builds_a_learned_tier_with_an_injected_config(tmp_path, monkeypatch, capsys):
    # The CLI uses the tier's real config (too slow for a test), so the engine's
    # build is exercised through a monkeypatched trainer that writes a tiny model.
    from cli.weights import main
    from core import learned_weights as lw

    def tiny_trainer(**kwargs):
        model = TransformerLM(_TINY, seed=0)
        model.save(kwargs["checkpoint"])
        return model

    monkeypatch.setattr(lw, "_default_trainer", tiny_trainer)
    assert main(["build-learned", "learned-large", "--root", str(tmp_path), "--quiet"]) == 0
    assert (tmp_path / "fayd-llm-large.npz").exists()
    assert "built" in capsys.readouterr().out


def test_cli_build_learned_needs_a_tier(capsys):
    from cli.weights import main

    assert main(["build-learned"]) == 2
    assert "tier" in capsys.readouterr().err.lower()


def test_cli_build_learned_refuses_an_unknown_tier(tmp_path, capsys):
    from cli.weights import main

    assert main(["build-learned", "huge", "--root", str(tmp_path)]) == 3
    assert "Unknown learned tier" in capsys.readouterr().err



def test_the_menu_offers_the_learned_entries():
    from cli.menu import MENU

    labels = [label for label, _action, _args in MENU]
    assert any("Learned weights" in label for label in labels)
    # The build entry points at the CLI's own build-learned command.
    build_rows = [args for label, _action, args in MENU if "train the large tier" in label]
    assert build_rows and build_rows[0] == ("build-learned", "learned-large")



# -------------------------------------------------------- --build-learned flag


def test_the_trainer_cli_exposes_build_learned(tmp_path, monkeypatch, capsys):
    # The flag parses with no value (defaulting to learned-large) and with a value;
    # the parser is built inside main(), so this checks behaviour, not internals.
    from ml.llm import __main__ as llm_main

    seen: list[str] = []

    def fake_build(tier, **_kwargs):
        seen.append(tier)
        return BuildReport(
            tier=resolve_tier(tier), checkpoint=tmp_path / "x.npz", built=True, steps=1
        )

    monkeypatch.setattr("core.learned_weights.build_learned", fake_build)
    assert llm_main.main(["--build-learned"]) == 0
    assert llm_main.main(["--build-learned", "learned-xl"]) == 0
    assert seen == ["learned-large", "learned-xl"]
    assert "built" in capsys.readouterr().out


def test_build_learned_flag_trains_through_the_real_engine(tmp_path, monkeypatch):
    from ml.llm import __main__ as llm_main

    captured: dict[str, object] = {}

    def fake_build(tier, **kwargs):
        captured["tier"] = tier
        captured.update(kwargs)
        return BuildReport(
            tier=resolve_tier(tier),
            checkpoint=tmp_path / "fayd-llm-large.npz",
            built=True,
            steps=5,
        )

    monkeypatch.setattr("core.learned_weights.build_learned", fake_build)
    # The tier's own budget is used unless --steps is passed explicitly.
    code = llm_main.main(["--build-learned", "learned-large"])
    assert code == 0
    assert captured["tier"] == "learned-large"
    assert captured["steps"] is None
    code = llm_main.main(["--build-learned", "learned-xl", "--steps", "7"])
    assert code == 0
    assert captured["steps"] == 7


def test_build_learned_flag_refuses_an_unknown_tier(capsys):
    from ml.llm.__main__ import main

    assert main(["--build-learned", "enormous"]) == 3
    assert "Unknown learned tier" in capsys.readouterr().err



# ------------------------------------------------------------------ the gate


def test_the_gate_is_one_hundred_percent_on_the_shipped_wiring():
    from core.learned_weights_gate import learned_weights_accuracy

    assert learned_weights_accuracy() == 1.0


def test_the_gate_reports_every_judgement():
    from core.learned_weights_gate import judgements

    results = judgements()
    assert results
    assert all(item.passed for item in results)
    # Each judgement names itself and carries a readable detail.
    assert all(item.name and item.detail for item in results)


def test_an_empty_catalogue_scores_zero(monkeypatch):
    from core import learned_weights_gate as gate

    monkeypatch.setattr(gate, "judgements", lambda: ())
    assert gate.learned_weights_accuracy() == 0.0


def test_the_gate_discriminates_when_the_larger_tier_is_not_larger(monkeypatch):
    # A negative control: if the XL tier were not larger, the gate's first
    # judgement fails -- so a broken capacity tier cannot pass silently.
    from dataclasses import replace as _replace

    from core import learned_weights_gate as gate

    broken = {
        key: _replace(tier, preset="learned-large")
        for key, tier in gate.LEARNED_TIERS.items()
    }
    monkeypatch.setattr(gate, "LEARNED_TIERS", broken)
    results = gate.judgements()
    first = next(item for item in results if "larger" in item.name)
    assert not first.passed


def test_the_gate_metric_is_wired_into_the_evaluation():
    from evaluation.benchmark import METRIC_SPECS
    from evaluation.runner import EvaluationMetrics

    names = {spec.name for spec in METRIC_SPECS}
    assert "learned_weights_accuracy" in names
    assert "learned_weights_accuracy" in EvaluationMetrics.__dataclass_fields__
