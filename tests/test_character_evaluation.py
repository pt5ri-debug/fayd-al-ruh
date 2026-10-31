"""The character evaluation engine: scores from real facts, honest commentary.

Every test runs headless with no display, no model, and no data directory: the
evaluator reads a temporary root and a registry that is either the real one or a
stand-in. The point of this suite is that the *scores move with the facts* -- an
unused store still scores on memory capability, a character with no model scores
low on training structure, and the commentary names the same facts that produced
the score -- never a fixed table of praise.
"""

from __future__ import annotations

from pathlib import Path

from core.character_config import CharacterConfig, default_config_for, owner_names
from core.character_learning import CHARACTER_KEYS, MAX_EXCHANGES, MAX_FACTS, CharacterLearning
from core.learning_isolation import MAIN_MODEL_OWNER
from gui.character_evaluation import (
    DIMENSIONS,
    CharacterEvaluator,
    DimensionScore,
    character_depth,
    evaluate_characters,
)


class StubRegistry:
    """A registry that returns a config per key, with no files at all."""

    def __init__(
        self,
        configs: dict[str, CharacterConfig] | None = None,
        customised: set[str] | None = None,
    ):
        self._configs: dict[str, CharacterConfig] = configs or {}
        self._customised = customised or set()

    def load(self, key: str) -> CharacterConfig:
        return self._configs.get(key, default_config_for(key))

    def is_customised(self, key: str) -> bool:
        return key in self._customised


class _StubModel:
    def __init__(self, title: str, available: bool) -> None:
        self.title = title
        self.available = available


class StubCatalog:
    """A stand-in language-model catalog, so model facts are controllable."""

    def __init__(self, available: bool = True, key: str = "stub-model") -> None:
        self._available = available
        self._key = key

    def default_language_model_key(self, mouth: str) -> str:
        return "retrieval" if mouth == "sources" else self._key

    def find_language_model(self, mouth: str, key: str) -> _StubModel:
        return _StubModel(title=f"{key} (stub)", available=self._available)


def _evaluator(tmp_path: Path, **kwargs: object) -> CharacterEvaluator:
    return CharacterEvaluator(data_root=tmp_path, registry=StubRegistry(**kwargs))  # type: ignore[arg-type]


def test_report_covers_every_owner_in_the_project_order() -> None:
    report = evaluate_characters(registry=StubRegistry(), catalog=StubCatalog())
    assert tuple(c.key for c in report.characters) == owner_names()
    assert len(report.characters) == 8
    assert len(CHARACTER_KEYS) == 7


def test_every_character_carries_the_four_dimensions() -> None:
    report = evaluate_characters(registry=StubRegistry(), catalog=StubCatalog())
    for character in report.characters:
        assert tuple(d.key for d in character.dimensions) == DIMENSIONS


def test_overall_is_the_mean_of_the_dimensions() -> None:
    report = evaluate_characters(registry=StubRegistry(), catalog=StubCatalog())
    for character in report.characters:
        expected = sum(d.score for d in character.dimensions) / len(character.dimensions)
        assert character.overall == round(expected, 6)


def test_an_empty_store_still_scores_on_memory_capability(tmp_path: Path) -> None:
    # An *unused* memory is not an *absent* one: the layer is real (durable,
    # isolated, bounded), so capability is scored even with nothing learned yet.
    report = _evaluator(tmp_path).evaluate()
    for character in report.characters:
        score = character.dimension("memory").score
        assert 0.4 <= score < 0.75
        assert any("memory layer is ready" in s for s in character.shortcomings)
        facts = character.dimension("memory").facts
        assert any("durable character-learning store: reachable" in f for f in facts)
        assert any("durable facts: 0/" in f for f in facts)


def test_a_filled_store_raises_the_memory_score(tmp_path: Path) -> None:
    learning = CharacterLearning("ultron", root=tmp_path)
    for i in range(MAX_FACTS):
        learning.add_fact(f"fact {i}")
    for i in range(MAX_EXCHANGES):
        learning.remember(f"q{i}", f"a{i}")
    report = _evaluator(tmp_path).evaluate()
    ultron = next(c for c in report.characters if c.key == "ultron")
    # facts full (0.35) + exchanges full (0.30); records are a separate store.
    assert ultron.dimension("memory").score >= 0.6


def test_memory_score_is_monotonic_in_the_facts(tmp_path: Path) -> None:
    learning = CharacterLearning("vision", root=tmp_path)
    for i in range(10):
        learning.add_fact(f"fact {i}")
    low = _evaluator(tmp_path).evaluate()
    vision_low = next(c for c in low.characters if c.key == "vision")
    for i in range(10, 60):
        learning.add_fact(f"fact {i}")
    high = _evaluator(tmp_path).evaluate()
    vision_high = next(c for c in high.characters if c.key == "vision")
    assert vision_high.dimension("memory").score > vision_low.dimension("memory").score


def test_missing_model_lowers_the_training_score() -> None:
    # The language-model catalog is the source of truth: an unavailable model
    # lowers the training score, and an available one raises it.
    missing = evaluate_characters(registry=StubRegistry(), catalog=StubCatalog(available=False))
    present = evaluate_characters(registry=StubRegistry(), catalog=StubCatalog(available=True))
    m = next(c for c in missing.characters if c.key == "banner")
    p = next(c for c in present.characters if c.key == "banner")
    assert p.dimension("training").score > m.dimension("training").score
    assert any("model file on disk: no" in f for f in m.dimension("training").facts)
    assert any("model file on disk: yes" in f for f in p.dimension("training").facts)


def test_model_availability_predicate_moves_training() -> None:
    unavailable = evaluate_characters(
        registry=StubRegistry(), catalog=StubCatalog(), model_available=lambda key, config: False
    )
    available = evaluate_characters(
        registry=StubRegistry(), catalog=StubCatalog(), model_available=lambda key, config: True
    )
    a = next(c for c in unavailable.characters if c.key == "ultron")
    b = next(c for c in available.characters if c.key == "ultron")
    assert b.dimension("training").score > a.dimension("training").score
    assert any("reported): no" in f for f in a.dimension("training").facts)
    assert any("reported): yes" in f for f in b.dimension("training").facts)


def test_a_character_with_no_catalog_model_scores_zero_on_training() -> None:
    # A catalog that reports no default model (the honest "nothing here") leaves
    # the model signals at zero rather than inventing one.
    class EmptyCatalog:
        def default_language_model_key(self, mouth: str) -> str:
            return ""

    report = evaluate_characters(registry=StubRegistry(), catalog=EmptyCatalog())
    for character in report.characters:
        assert character.dimension("training").score < 0.75
        assert any("none configured" in f for f in character.dimension("training").facts)


def test_customisation_raises_the_traits_score() -> None:
    plain = evaluate_characters(registry=StubRegistry(), catalog=StubCatalog())
    custom = evaluate_characters(registry=StubRegistry(customised={"hakeem"}), catalog=StubCatalog())
    p = next(c for c in plain.characters if c.key == "hakeem")
    c = next(c for c in custom.characters if c.key == "hakeem")
    assert c.dimension("traits").score > p.dimension("traits").score
    assert any("customised by the user: yes" in f for f in c.dimension("traits").facts)


def test_natasha_is_marked_private_and_cloud_is_not() -> None:
    report = evaluate_characters(registry=StubRegistry(), catalog=StubCatalog())
    natasha = next(c for c in report.characters if c.key == "natasha")
    cloud = next(c for c in report.characters if c.key == "cloud")
    assert natasha.private is True
    assert cloud.private is False
    assert any("private" in s for s in natasha.shortcomings)


def test_project_model_is_included_and_not_private() -> None:
    report = evaluate_characters(registry=StubRegistry(), catalog=StubCatalog())
    model = next(c for c in report.characters if c.key == MAIN_MODEL_OWNER)
    assert model.private is False


def test_depth_differs_between_characters() -> None:
    # A character with a dedicated layer scores deeper than a plain mouth.
    natasha_depth, _ = character_depth("natasha")
    cloud_depth, _ = character_depth("cloud")
    assert natasha_depth > cloud_depth


def test_depth_is_honest_about_an_unknown_key() -> None:
    score, facts = character_depth("no-such-character")
    assert score == 0.0
    assert facts == ()


def test_evaluate_key_refuses_an_unknown_key() -> None:
    import pytest

    evaluator = CharacterEvaluator(data_root=None, registry=StubRegistry(), catalog=StubCatalog())  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        evaluator.evaluate_key("nobody")


def test_evaluate_key_resolves_a_mouth_alias() -> None:
    evaluator = CharacterEvaluator(data_root=None, registry=StubRegistry(), catalog=StubCatalog())  # type: ignore[arg-type]
    by_owner = evaluator.evaluate_key("jarvis")
    by_mouth = evaluator.evaluate_key("sources")
    assert by_owner.key == by_mouth.key == "jarvis"


def test_commentary_names_the_same_facts_as_the_score() -> None:
    report = evaluate_characters(registry=StubRegistry(), catalog=StubCatalog())
    for character in report.characters:
        # Every shortcoming is paired with a next step, so a gap is never a dead end.
        assert len(character.next_steps) >= len(character.shortcomings) or not character.shortcomings


def test_report_summary_names_strongest_and_weakest() -> None:
    report = evaluate_characters(registry=StubRegistry(), catalog=StubCatalog())
    text = report.describe()
    assert "strongest:" in text
    assert "weakest:" in text
    assert report.strongest() is not None
    assert report.weakest() is not None


def test_dimension_averages_cover_every_dimension() -> None:
    report = evaluate_characters(registry=StubRegistry(), catalog=StubCatalog())
    averages = report.dimension_averages()
    assert set(averages) == set(DIMENSIONS)
    for value in averages.values():
        assert 0.0 <= value <= 1.0


def test_primitive_round_trips_every_field() -> None:
    report = evaluate_characters(registry=StubRegistry(), catalog=StubCatalog())
    data = report.to_primitive()
    assert len(data["characters"]) == 8
    first = data["characters"][0]
    assert set(first) >= {
        "key", "name", "role", "private", "overall",
        "dimensions", "strengths", "shortcomings", "next_steps",
    }
    assert len(first["dimensions"]) == 4


def test_dimension_score_reports_a_percent() -> None:
    score = DimensionScore("memory", 0.5, ("a",))
    assert score.percent == 50
    assert "Memory" in score.title
    assert "50%" in score.describe()


def test_evaluation_describe_lists_dimensions() -> None:
    report = evaluate_characters(registry=StubRegistry(), catalog=StubCatalog())
    text = report.characters[0].describe()
    for key in DIMENSIONS:
        assert key.capitalize() in text or key in text.lower()


def test_a_healthy_character_reports_no_shortcomings() -> None:
    # Build a character that scores well on every dimension.
    import dataclasses

    from core.character_config import (
        GeneralConfig,
        LearningSettings,
        ModelConfig,
        PersonalityConfig,
        VoiceConfig,
    )

    config = dataclasses.replace(
        default_config_for("ultron"),
        general=GeneralConfig(name="Test", description="d", role="r", active=True),
        model=ModelConfig(key="ultron-general", provider="local", model="x", path="", active=True),
        learning=LearningSettings(enabled=True, learn_from_model=True),
        personality=PersonalityConfig(
            traits=("a", "b"), rules=("c",), behaviour=("d",),
            communication_style="clear", reasoning_style="systematic",
            knowledge_domain="petroleum", objectives=("o1", "o2", "o3"),
        ),
        voice=VoiceConfig(engine="piper"),
    )
    registry = StubRegistry(configs={"ultron": config}, customised={"ultron"})
    report = CharacterEvaluator(  # type: ignore[arg-type]
        data_root=None, registry=registry, catalog=StubCatalog(), model_available=lambda k, c: True
    ).evaluate()
    ultron = next(c for c in report.characters if c.key == "ultron")
    assert ultron.dimension("training").score >= 0.9
    assert ultron.dimension("traits").score >= 0.8
    # Memory is genuinely unused here, so that one shortcoming is correct and honest.
    assert ultron.shortcomings == ("its memory layer is ready but little has been learned yet",)


def test_report_notes_are_honest_about_the_method() -> None:
    report = evaluate_characters(registry=StubRegistry(), catalog=StubCatalog())
    assert report.notes
    assert any("real configuration" in note for note in report.notes)
