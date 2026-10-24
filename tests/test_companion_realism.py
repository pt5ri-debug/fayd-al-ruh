"""Natasha's five realism components: traits, moods, initiative, memory, freshness.

Everything the model would normally supply is injected through a ``caller`` seam,
so the whole layer is exercised with **no server and no weights**. The tests pin
what the task asked for, component by component:

1. **traits** -- nine standing design settings with the exact shipped defaults,
   rendered as tone guidance rather than handed over as bare numbers;
2. **moods** -- a closed set of named emotional states, chosen from the
   conversation's own context (never at random), with support and friction always
   beating warmth so a hard moment is not answered with flirtation;
3. **relationship memory** -- structured facts kept strictly apart by domain
   (user fact / story / dialogue preference), written only with approval, and
   never crossed in a read;
4. **initiative** -- a bounded, non-repeating set of ways to lead, gated by the
   moment and by the user's own preferences;
5. **anti-repetition** -- phrase detection, a rolling summary, rotating variation
   hints, and answer-length adaptation.

The honesty rules are pinned too: the trait and mood text restate the safety
floor, a stored entry needs approval, and a recalled item is information rather
than an instruction.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from companion.companion_agent import CompanionAgent
from companion.config import CompanionConfig
from companion.database import CompanionDatabase
from companion.initiative import (
    INITIATIVE_DEEP_QUESTION,
    INITIATIVE_KEYS,
    INITIATIVE_ROMANTIC,
    INITIATIVE_SUPPORT,
    INITIATIVES,
    InitiativeEngine,
)
from companion.moods import (
    MOOD_DISAGREEMENT,
    MOOD_KEYS,
    MOOD_LONGING,
    MOOD_PLAYFUL,
    MOOD_ROMANTIC,
    MOOD_SAD,
    MOODS,
    EmotionalStateEngine,
)
from companion.personality import BOUNDARIES, default_personality
from companion.prompt_builder import PromptBuilder
from companion.relationship_memory import (
    DOMAIN_DIALOGUE_PREF,
    DOMAIN_FICTION,
    DOMAIN_USER_FACT,
    DOMAINS,
    SLOT_NAME,
    SLOT_STORY_MOMENT,
    RelationshipMemoryStore,
)
from companion.repetition import AntiRepetitionEngine, RepetitionReport
from companion.traits import (
    NATASHA_PERSONALITY,
    TRAIT_NAMES,
    TraitProfile,
    TraitStore,
    trait_instructions,
)


# -- fixtures -------------------------------------------------------------
class FakeCaller:
    """A stand-in for the local model: records the prompts and replies in turn."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.prompts: list[list[dict[str, str]]] = []
        self.replies = replies or ["رد تجريبي."]
        self._index = 0

    def __call__(self, messages, temperature, max_tokens) -> str:
        self.prompts.append(list(messages))
        reply = self.replies[min(self._index, len(self.replies) - 1)]
        self._index += 1
        return reply

    @property
    def system(self) -> str:
        return self.prompts[-1][0]["content"]


@pytest.fixture()
def config(tmp_path: Path) -> CompanionConfig:
    return CompanionConfig(db_path=tmp_path / "companion.db")


@pytest.fixture()
def caller() -> FakeCaller:
    return FakeCaller()


@pytest.fixture()
def agent(config: CompanionConfig, caller: FakeCaller) -> CompanionAgent:
    return CompanionAgent(config, caller=caller)


@pytest.fixture()
def database(tmp_path: Path) -> Iterator[CompanionDatabase]:
    db = CompanionDatabase(tmp_path / "relmem.db")
    yield db
    db.close()


# =========================================================================
# Component 1: personality traits
# =========================================================================
def test_the_design_defaults_are_exactly_the_ones_the_task_named() -> None:
    """The nine traits ship with the exact values the character was designed to."""
    assert NATASHA_PERSONALITY == {
        "romantic_intensity": 0.98,
        "emotional_expressiveness": 0.98,
        "affection": 0.95,
        "playfulness": 0.90,
        "spontaneity": 0.95,
        "conversational_initiative": 0.90,
        "emotional_sensitivity": 0.95,
        "formal_tone": 0.10,
        "repetitiveness": 0.05,
    }


def test_the_profile_fields_match_the_design_dictionary() -> None:
    """The dataclass defaults and the design dict cannot drift apart."""
    assert TraitProfile().to_primitive() == NATASHA_PERSONALITY
    assert TraitProfile().is_design_default()


def test_a_trait_must_be_a_unit_value() -> None:
    with pytest.raises(ValueError):
        TraitProfile(romantic_intensity=1.5)
    with pytest.raises(ValueError):
        TraitProfile(formal_tone=-0.1)


def test_traits_are_rendered_as_guidance_not_as_bare_numbers() -> None:
    """A high trait becomes an instruction to adopt, never a float to interpret."""
    lines = trait_instructions()
    assert len(lines) == len(TRAIT_NAMES)
    joined = " ".join(lines)
    # No bare number is ever sent as the whole instruction.
    assert "0.98" not in joined
    assert "romantic feeling run high" in joined
    # The low-formality design setting reads as an instruction to be informal.
    assert "informally" in joined


def test_a_high_trait_restates_the_safety_floor_in_its_own_text() -> None:
    """An instruction that could be misread as licence still carries the floor."""
    lines = trait_instructions()
    joined = " ".join(lines)
    assert "expression only" in joined
    assert "consent" in joined


def test_the_personality_generator_renders_the_traits_itself() -> None:
    """The standing instruction carries the traits, so no caller has to add them."""
    prompt = default_personality().system_prompt()
    assert "How your standing traits shape your voice" in prompt
    assert "romantic feeling run high" in prompt
    # The unconditional honesty rules are still present.
    for rule in BOUNDARIES:
        assert rule in prompt


def test_the_personality_generator_accepts_an_override_profile() -> None:
    strict = default_personality().system_prompt(traits=TraitProfile(romantic_intensity=0.0))
    assert "Keep romance quiet and understated" in strict


def test_trait_overrides_persist_and_reset(database: CompanionDatabase) -> None:
    store = TraitStore(database)
    assert store.load() == TraitProfile()
    store.save(TraitProfile(formal_tone=0.9, playfulness=0.1))
    assert store.load().formal_tone == 0.9
    # A reset returns to the shipped design, not to a neutral zero.
    assert store.reset().is_design_default()


def test_a_corrupt_saved_profile_falls_back_to_the_design(database: CompanionDatabase) -> None:
    store = TraitStore(database)
    database.execute(
        "INSERT INTO companion_settings(key, value, updated_at) VALUES(?, ?, ?)",
        ("trait_profile", "{not json", "2020-01-01T00:00:00+00:00"),
    )
    assert store.load().is_design_default()


def test_agent_trait_round_trip(agent: CompanionAgent) -> None:
    assert agent.traits().is_design_default()
    updated = agent.update_traits(repetitiveness=0.8)
    assert updated.repetitiveness == 0.8
    assert agent.traits().repetitiveness == 0.8
    assert agent.reset_traits().is_design_default()


# =========================================================================
# Component 2: emotional state engine
# =========================================================================
def test_the_mood_catalogue_is_closed_and_named() -> None:
    assert set(MOODS) == set(MOOD_KEYS)
    for key, mood in MOODS.items():
        assert mood.key == key
        assert mood.arabic and mood.english
        assert mood.style


def test_a_hard_moment_is_not_answered_with_romance() -> None:
    """A support or friction cue wins over a warm one, and romance is switched off."""
    engine = EmotionalStateEngine()
    assert engine.observe("أنا حزين جداً اليوم").key == MOOD_SAD
    assert engine.observe("أنا غاضب منك").key == MOOD_DISAGREEMENT
    for key in (MOOD_SAD, MOOD_DISAGREEMENT):
        assert MOODS[key].allows_romance is False
        assert "romance" in MOODS[key].style.casefold()


def test_a_warm_cue_selects_the_matching_state() -> None:
    engine = EmotionalStateEngine()
    assert engine.observe("أشتاق إليك").key == MOOD_LONGING
    assert engine.observe("هههههه").key == MOOD_PLAYFUL
    assert engine.observe("أحبك يا قلبي").key == MOOD_ROMANTIC


def test_the_state_is_stable_when_nothing_new_fits() -> None:
    """An unrelated message keeps the current state rather than jumping to default."""
    engine = EmotionalStateEngine()
    engine.observe("أشتاق إليك")
    assert engine.observe("ما هي عاصمة مصر؟").key == MOOD_LONGING


def test_the_state_can_be_disabled_and_reset() -> None:
    engine = EmotionalStateEngine(enabled=False)
    assert engine.observe("أنا حزين").key == MOOD_ROMANTIC
    engine = EmotionalStateEngine()
    engine.observe("أنا حزين")
    assert engine.reset().key == MOOD_ROMANTIC


def test_an_unknown_state_is_refused() -> None:
    engine = EmotionalStateEngine()
    with pytest.raises(KeyError):
        engine.set_mood("euphoric")
    # An explicit choice is honoured and labelled as such.
    assert engine.set_mood(MOOD_SAD).key == MOOD_SAD


def test_the_mood_reaches_the_prompt_with_its_style(agent: CompanionAgent, caller: FakeCaller) -> None:
    session = agent.new_session("mood")
    reply = agent.ask(session.id, "أشتاق إليك")
    assert reply.mood == MOOD_LONGING
    assert "Your emotional state right now" in caller.system
    assert "longing" in caller.system


# =========================================================================
# Component 3: relationship memory
# =========================================================================
def test_the_three_domains_are_declared(database: CompanionDatabase) -> None:
    assert set(DOMAINS) == {DOMAIN_USER_FACT, DOMAIN_FICTION, DOMAIN_DIALOGUE_PREF}


def test_a_relationship_memory_needs_approval(database: CompanionDatabase) -> None:
    store = RelationshipMemoryStore(database)
    with pytest.raises(ValueError):
        store.remember(DOMAIN_DIALOGUE_PREF, SLOT_NAME, "سارة")
    entry = store.remember(DOMAIN_DIALOGUE_PREF, SLOT_NAME, "سارة", approved=True)
    assert entry.value == "سارة"
    assert store.value_of(DOMAIN_DIALOGUE_PREF, SLOT_NAME) == "سارة"


def test_an_unknown_domain_is_refused(database: CompanionDatabase) -> None:
    store = RelationshipMemoryStore(database)
    with pytest.raises(ValueError):
        store.remember("secret_third_thing", SLOT_NAME, "x", approved=True)


def test_a_repeated_slot_replaces_rather_than_piles_up(database: CompanionDatabase) -> None:
    store = RelationshipMemoryStore(database)
    store.remember(DOMAIN_DIALOGUE_PREF, SLOT_NAME, "سارة", approved=True)
    store.remember(DOMAIN_DIALOGUE_PREF, SLOT_NAME, "سلمى", approved=True)
    active = store.entries(domain=DOMAIN_DIALOGUE_PREF)
    assert len(active) == 1
    assert active[0].value == "سلمى"
    # The older row is kept, deactivated, so the change is recorded.
    assert len(store.entries(domain=DOMAIN_DIALOGUE_PREF, include_inactive=True)) == 2


def test_a_read_never_crosses_the_domain_it_was_asked_for(database: CompanionDatabase) -> None:
    """A factual question can never be answered from a story entry."""
    store = RelationshipMemoryStore(database)
    store.remember(DOMAIN_FICTION, SLOT_STORY_MOMENT, "التقينا في مكتبة", approved=True)
    store.remember(DOMAIN_USER_FACT, "home", "القاهرة", approved=True)
    facts = store.recall("مكتبة القاهرة", domain=DOMAIN_USER_FACT)
    assert facts
    assert all(entry.domain == DOMAIN_USER_FACT for entry in facts)
    story = store.recall("مكتبة", domain=DOMAIN_FICTION)
    assert all(entry.domain == DOMAIN_FICTION for entry in story)
    # Unscoped, it returns everything -- which is what the prompt builder needs, so
    # it can render each domain in its own block and never merge them.
    everything = store.recall("مكتبة القاهرة")
    assert {entry.domain for entry in everything} == {DOMAIN_USER_FACT, DOMAIN_FICTION}


def test_a_fiction_entry_says_it_is_a_story(database: CompanionDatabase) -> None:
    store = RelationshipMemoryStore(database)
    entry = store.remember(DOMAIN_FICTION, SLOT_STORY_MOMENT, "التقينا في مكتبة", approved=True)
    assert entry.fictional
    assert "story" in entry.label
    assert "not real" in entry.label


def test_forget_and_reset(database: CompanionDatabase) -> None:
    store = RelationshipMemoryStore(database)
    entry = store.remember(DOMAIN_DIALOGUE_PREF, SLOT_NAME, "سارة", approved=True)
    assert store.forget(entry.id)
    assert store.value_of(DOMAIN_DIALOGUE_PREF, SLOT_NAME) is None
    store.remember(DOMAIN_DIALOGUE_PREF, SLOT_NAME, "سارة", approved=True)
    store.remember(DOMAIN_USER_FACT, "home", "القاهرة", approved=True)
    assert store.reset(domain=DOMAIN_USER_FACT) == 1
    assert store.entries(domain=DOMAIN_DIALOGUE_PREF)


def test_the_summary_is_plain_words_and_names_each_part(database: CompanionDatabase) -> None:
    store = RelationshipMemoryStore(database)
    assert "nothing saved" in store.summary().describe()
    store.remember(DOMAIN_DIALOGUE_PREF, SLOT_NAME, "سارة", approved=True)
    store.remember(DOMAIN_FICTION, SLOT_STORY_MOMENT, "مكتبة المطر", approved=True)
    summary = store.summary()
    assert summary.total == 2
    text = summary.describe()
    assert "سارة" in text
    assert "مكتبة المطر" in text


def test_a_proposal_is_never_stored_on_its_own(database: CompanionDatabase) -> None:
    store = RelationshipMemoryStore(database)
    proposals = store.propose("من فضلك احفظ أنني أحب القهوة")
    assert proposals
    # Proposing stores nothing: the user still has to approve.
    assert store.entries() == []


def test_a_story_wording_lands_in_the_fiction_domain(database: CompanionDatabase) -> None:
    store = RelationshipMemoryStore(database)
    proposals = store.propose("تذكر في قصتنا أننا التقينا في مكتبة")
    assert proposals
    assert proposals[0].domain == DOMAIN_FICTION


def test_agent_relationship_memory_round_trip(agent: CompanionAgent) -> None:
    entry = agent.remember_relationship(
        DOMAIN_DIALOGUE_PREF, SLOT_NAME, "سارة", approved=True
    )
    assert agent.relationship_memory_summary().name == "سارة"
    assert agent.forget_relationship_memory(entry.id)
    assert agent.relationship_memory_summary().name is None


def test_the_three_domains_reach_the_prompt_in_separate_blocks(
    agent: CompanionAgent, caller: FakeCaller
) -> None:
    agent.remember_relationship(DOMAIN_USER_FACT, "home", "القاهرة", approved=True)
    agent.remember_relationship(DOMAIN_FICTION, SLOT_STORY_MOMENT, "مكتبة المطر", approved=True)
    agent.remember_relationship(DOMAIN_DIALOGUE_PREF, SLOT_NAME, "سارة", approved=True)
    session = agent.new_session("relmem")
    agent.ask(session.id, "أين أسكن وما اسمك؟")
    system = caller.system
    assert "What you know about the user (real facts)" in system
    assert "What happened in our story (role-play -- never real)" in system
    assert "How the two of you have agreed to talk" in system


def test_the_export_carries_the_traits_and_the_relationship_memory(agent: CompanionAgent) -> None:
    agent.remember_relationship(DOMAIN_DIALOGUE_PREF, SLOT_NAME, "سارة", approved=True)
    data = agent.export_data()
    assert data["traits"] == NATASHA_PERSONALITY
    assert data["relationship_memory"]
    # The export is plain JSON, so a backup is portable.
    json.dumps(data, ensure_ascii=False)


# =========================================================================
# Component 4: conversational initiative
# =========================================================================
def test_the_initiative_catalogue_is_closed() -> None:
    assert len(INITIATIVE_KEYS) == 6
    for key in INITIATIVE_KEYS:
        assert INITIATIVES[key].key == key
        assert INITIATIVES[key].instruction


def test_a_supportive_moment_offers_support_not_romance() -> None:
    engine = InitiativeEngine()
    choice = engine.choose(mood_key=MOOD_SAD)
    assert choice is not None
    assert choice.key in {INITIATIVE_SUPPORT, INITIATIVE_DEEP_QUESTION}
    assert not INITIATIVES[choice.key].romantic


def test_romance_is_filtered_out_when_the_moment_forbids_it() -> None:
    engine = InitiativeEngine()
    available = engine.available(allow_romance=False)
    assert INITIATIVE_ROMANTIC not in available
    assert all(not INITIATIVES[key].romantic for key in available)


def test_an_initiative_is_never_offered_twice() -> None:
    engine = InitiativeEngine()
    first = engine.choose(mood_key=MOOD_LONGING)
    assert first is not None
    engine.mark_used(first.key)
    second = engine.choose(mood_key=MOOD_LONGING)
    assert second is not None
    assert second.key != first.key


def test_a_low_initiative_trait_means_she_only_follows() -> None:
    engine = InitiativeEngine()
    assert engine.choose(initiative_level=0.2) is None
    assert engine.choose(initiative_level=0.9) is not None


def test_the_engine_can_be_disabled() -> None:
    assert InitiativeEngine(enabled=False).choose(mood_key=MOOD_LONGING) is None


def test_a_used_initiative_can_be_seeded() -> None:
    engine = InitiativeEngine()
    engine.load_used([INITIATIVE_ROMANTIC])
    assert INITIATIVE_ROMANTIC not in engine.available()


def test_the_initiative_reaches_the_prompt(agent: CompanionAgent, caller: FakeCaller) -> None:
    session = agent.new_session("initiative")
    reply = agent.ask(session.id, "أشتاق إليك")
    assert reply.initiative is not None
    assert "How you may take the lead this turn" in caller.system


def test_a_supportive_mood_offers_a_non_romantic_initiative_end_to_end(
    agent: CompanionAgent, caller: FakeCaller
) -> None:
    session = agent.new_session("support")
    reply = agent.ask(session.id, "أنا حزين جداً اليوم")
    assert reply.mood == MOOD_SAD
    assert reply.initiative is not None
    assert not INITIATIVES[reply.initiative].romantic


# =========================================================================
# Component 5: anti-repetition
# =========================================================================
def test_a_repeated_phrase_is_detected_longest_first() -> None:
    engine = AntiRepetitionEngine()
    engine.observe("يا قلبي أنت أغلى إنسان في الدنيا")
    report = engine.check("يا قلبي أنت أغلى إنسان في الدنيا واكثر")
    assert report.is_repetitive
    assert report.phrases
    assert report.score > 0


def test_a_fresh_reply_is_not_flagged() -> None:
    engine = AntiRepetitionEngine()
    engine.observe("يا قلبي أنت أغلى إنسان")
    report = engine.check("ما رأيك أن نتحدث عن يومك؟")
    assert not report.is_repetitive
    assert report.score == 0.0


def test_a_repeated_opener_is_flagged_on_its_own() -> None:
    engine = AntiRepetitionEngine()
    engine.observe("صباح الخير يا حبيبي كيف حالك اليوم")
    report = engine.check("صباح الخير يا عزيزي")
    assert report.opener_repeated


def test_the_window_is_bounded() -> None:
    engine = AntiRepetitionEngine(window=3)
    for index in range(6):
        engine.observe(f"رسالة رقم {index} مختلفة تماما")
    assert len(engine.replies) == 3


def test_the_rolling_summary_is_bounded_and_deduplicated() -> None:
    engine = AntiRepetitionEngine(window=3)
    for index in range(5):
        engine.observe_user(f"سؤال رقم {index}")
    assert len(engine.summary.split("سؤال")) - 1 == 3


def test_the_variation_hint_rotates() -> None:
    engine = AntiRepetitionEngine()
    hints = {engine.variation_hint() for _ in range(6)}
    assert len(hints) > 1


def test_the_guidance_tightens_with_a_low_repetitiveness_trait() -> None:
    engine = AntiRepetitionEngine()
    strict = engine.guidance(repetitiveness=0.05)
    relaxed = engine.guidance(repetitiveness=0.9)
    assert "Never reuse" in strict
    assert "familiar refrain" in relaxed


def test_the_guidance_names_what_was_repeated() -> None:
    engine = AntiRepetitionEngine()
    engine.observe("يا قلبي أنت أغلى إنسان في الدنيا")
    report = engine.check("يا قلبي أنت أغلى إنسان في الدنيا")
    guidance = engine.guidance(report=report)
    assert "repeats earlier wording" in guidance


def test_the_length_hint_adapts_to_the_question() -> None:
    engine = AntiRepetitionEngine()
    assert "short" in engine.length_hint("صباح الخير")
    assert "full answer" in engine.length_hint("اشرح لي بالتفصيل " + "كلمة " * 40)


def test_the_repetition_engine_can_be_disabled() -> None:
    engine = AntiRepetitionEngine(enabled=False)
    engine.observe("يا قلبي")
    assert not engine.check("يا قلبي").is_repetitive
    assert engine.guidance() == ""


def test_the_engine_can_be_seeded_from_a_resumed_session() -> None:
    engine = AntiRepetitionEngine()
    engine.load_replies(["يا قلبي أنت أغلى إنسان في الدنيا"])
    assert engine.check("يا قلبي أنت أغلى إنسان في الدنيا").is_repetitive


def test_repetition_is_measured_on_the_agent_reply(agent: CompanionAgent) -> None:
    caller = agent._caller
    assert isinstance(caller, FakeCaller)
    caller.replies = ["يا قلبي أنت أغلى إنسان في الدنيا"]
    session = agent.new_session("repeat")
    first = agent.ask(session.id, "مرحبا")
    assert first.repetition is not None
    second = agent.ask(session.id, "ومرة أخرى")
    assert second.repetition is not None
    assert second.repetition.is_repetitive


# =========================================================================
# wiring: config switches and the assembled prompt
# =========================================================================
def test_each_layer_can_be_switched_off(tmp_path: Path) -> None:
    config = CompanionConfig(
        db_path=tmp_path / "off.db",
        mood_enabled=False,
        initiative_enabled=False,
        anti_repetition_enabled=False,
    )
    caller = FakeCaller()
    agent = CompanionAgent(config, caller=caller)
    session = agent.new_session("off")
    reply = agent.ask(session.id, "أشتاق إليك")
    assert reply.mood is None
    assert reply.initiative is None
    assert reply.repetition is None
    assert "Your emotional state right now" not in caller.system
    assert "How you may take the lead this turn" not in caller.system
    assert "How to keep this turn fresh and well-sized" not in caller.system


def test_a_too_small_repetition_window_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CompanionConfig(db_path=tmp_path / "x.db", repetition_window=0)


def test_an_older_database_is_migrated_in_place(tmp_path: Path) -> None:
    """A v1 database gains the relationship-memory table without losing its rows."""
    import sqlite3

    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    # A minimal v1 database: the sessions table, one row, and version 1.
    connection.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT NOT NULL, "
        "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO sessions(id, title, created_at, updated_at) VALUES('s1', 'قديم', 'x', 'x')"
    )
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    database = CompanionDatabase(path)
    try:
        assert database.version == 2
        # The old row survived, and the new table exists and works.
        assert database.query_one("SELECT * FROM sessions WHERE id = 's1'") is not None
        store = RelationshipMemoryStore(database)
        entry = store.remember(DOMAIN_DIALOGUE_PREF, SLOT_NAME, "سارة", approved=True)
        assert store.value_of(DOMAIN_DIALOGUE_PREF, SLOT_NAME) == "سارة"
        assert entry.id
    finally:
        database.close()


def test_a_database_from_a_newer_build_is_refused(tmp_path: Path) -> None:
    """A version this build cannot migrate is refused rather than corrupted."""
    import sqlite3

    from companion.database import CompanionDatabaseError

    path = tmp_path / "newer.db"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version = 99")
    connection.commit()
    connection.close()

    with pytest.raises(CompanionDatabaseError):
        CompanionDatabase(path)


def test_the_prompt_orders_the_layers_as_documented(agent: CompanionAgent, caller: FakeCaller) -> None:
    """The standing rules come first and the newest question last."""
    agent.remember_relationship(DOMAIN_DIALOGUE_PREF, SLOT_NAME, "سارة", approved=True)
    session = agent.new_session("order")
    agent.ask(session.id, "أشتاق إليك")
    system = caller.system
    positions = [
        system.index("How your standing traits shape your voice"),
        system.index("Dialogue settings the user chose"),
        system.index("What has happened in this conversation so far"),
        system.index("Your emotional state right now"),
        system.index("How you may take the lead this turn"),
        system.index("How to keep this turn fresh and well-sized"),
    ]
    assert positions == sorted(positions)
    # The unconditional honesty rules sit inside the standing character block, so
    # they come *before* every per-turn layer that could otherwise be read as an
    # override.
    assert system.index("Honesty, consent, and content rules") < positions[1]
    # The newest user turn is the final message, after the system block.
    assert caller.prompts[-1][-1]["role"] == "user"
    assert caller.prompts[-1][-1]["content"] == "أشتاق إليك"


def test_the_prompt_builder_labels_every_layer_it_added(agent: CompanionAgent, caller: FakeCaller) -> None:
    session = agent.new_session("notes")
    reply = agent.ask(session.id, "أشتاق إليك")
    notes = " ".join(reply.prompt_notes)
    assert "character traits" in notes
    assert "emotional state" in notes
    assert "initiative" in notes
    assert "anti-repetition" in notes


def test_a_prompt_can_be_built_with_no_character_layers() -> None:
    """The builder still works with the base arguments alone (backward compatible)."""
    built = PromptBuilder().build(
        personality=default_personality(),
        question="مرحبا",
    )
    assert built.messages[-1]["content"] == "مرحبا"
    assert "Your emotional state right now" not in built.system


def test_a_disabled_emotion_layer_is_honoured_by_the_builder() -> None:
    from companion.emotion_state import DialogueEmotion

    built = PromptBuilder().build(
        personality=default_personality(),
        question="مرحبا",
        emotion=DialogueEmotion(warmth=1.0),
        emotion_enabled=False,
    )
    assert "Your current tone in this conversation" not in built.system


def test_the_system_prompt_forbids_confusing_the_three_relationship_blocks() -> None:
    built = PromptBuilder().build(personality=default_personality(), question="مرحبا")
    assert "never confuse them" in built.system


def test_the_repetition_report_serialises_for_json() -> None:
    report = RepetitionReport()
    json.dumps(report.to_primitive())


def test_the_trait_and_mood_text_is_bilingual_where_it_names_a_state() -> None:
    for mood in MOODS.values():
        assert mood.arabic and mood.english
        assert mood.label == f"{mood.english} ({mood.arabic})"


# =========================================================================
# CLI: the four new subcommands
# =========================================================================
def test_cli_traits_shows_the_design_defaults(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import natasha as natasha_cli

    monkeypatch.setenv("FAYD_NATASHA_DB", str(tmp_path / "cli.db"))
    assert natasha_cli.main(["traits"]) == 0
    out = capsys.readouterr().out
    assert "Character traits" in out
    assert "romantic intensity" in out
    assert "repetitiveness" in out


def test_cli_traits_sets_and_resets(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import natasha as natasha_cli

    monkeypatch.setenv("FAYD_NATASHA_DB", str(tmp_path / "cli.db"))
    assert natasha_cli.main(["traits", "--formal-tone", "0.9"]) == 0
    assert "formal tone=high" in capsys.readouterr().out
    assert natasha_cli.main(["traits", "--reset"]) == 0
    assert "shipped design" in capsys.readouterr().out


def test_cli_traits_refuses_an_out_of_range_value(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import natasha as natasha_cli

    monkeypatch.setenv("FAYD_NATASHA_DB", str(tmp_path / "cli.db"))
    assert natasha_cli.main(["traits", "--formal-tone", "2.0"]) == 2
    assert "Not changed" in capsys.readouterr().err


def test_cli_traits_json(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import natasha as natasha_cli

    monkeypatch.setenv("FAYD_NATASHA_DB", str(tmp_path / "cli.db"))
    assert natasha_cli.main(["traits", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == NATASHA_PERSONALITY


def test_cli_mood_shows_sets_and_resets(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import natasha as natasha_cli

    monkeypatch.setenv("FAYD_NATASHA_DB", str(tmp_path / "cli.db"))
    assert natasha_cli.main(["mood"]) == 0
    assert "Emotional state" in capsys.readouterr().out
    assert natasha_cli.main(["mood", "--set", "sad"]) == 0
    assert "sad" in capsys.readouterr().out
    assert natasha_cli.main(["mood", "--reset"]) == 0


def test_cli_mood_json(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import natasha as natasha_cli

    monkeypatch.setenv("FAYD_NATASHA_DB", str(tmp_path / "cli.db"))
    assert natasha_cli.main(["mood", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["key"] in MOOD_KEYS


def test_cli_initiative_lists_all_of_them(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import natasha as natasha_cli

    monkeypatch.setenv("FAYD_NATASHA_DB", str(tmp_path / "cli.db"))
    assert natasha_cli.main(["initiative"]) == 0
    out = capsys.readouterr().out
    for key in INITIATIVE_KEYS:
        assert key in out


def test_cli_relationship_memory_round_trip(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import natasha as natasha_cli

    monkeypatch.setenv("FAYD_NATASHA_DB", str(tmp_path / "cli.db"))
    code = natasha_cli.main(
        [
            "relationship-memory",
            "--remember", "سارة",
            "--domain", "dialogue_pref",
            "--slot", "name",
            "--approve",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Remembered" in out
    # Format: "Remembered [domain/slot] RMEM-xxxx: value".
    entry_id = out.split()[2].rstrip(":")

    assert natasha_cli.main(["relationship-memory"]) == 0
    assert "سارة" in capsys.readouterr().out

    assert natasha_cli.main(["relationship-memory", "--forget", entry_id]) == 0
    assert "Forgotten" in capsys.readouterr().out


def test_cli_relationship_memory_needs_approval(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import natasha as natasha_cli

    monkeypatch.setenv("FAYD_NATASHA_DB", str(tmp_path / "cli.db"))
    code = natasha_cli.main(
        ["relationship-memory", "--remember", "سارة", "--domain", "dialogue_pref", "--slot", "name"]
    )
    assert code == 2
    assert "Not stored" in capsys.readouterr().err


def test_cli_relationship_memory_requires_domain_and_slot(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import natasha as natasha_cli

    monkeypatch.setenv("FAYD_NATASHA_DB", str(tmp_path / "cli.db"))
    code = natasha_cli.main(["relationship-memory", "--remember", "x", "--approve"])
    assert code == 2
    assert "needs --domain and --slot" in capsys.readouterr().err


def test_cli_relationship_memory_json(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import natasha as natasha_cli

    monkeypatch.setenv("FAYD_NATASHA_DB", str(tmp_path / "cli.db"))
    natasha_cli.main(
        ["relationship-memory", "--remember", "سارة", "--domain", "dialogue_pref",
         "--slot", "name", "--approve"]
    )
    capsys.readouterr()
    assert natasha_cli.main(["relationship-memory", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["entries"][0]["value"] == "سارة"
    assert "summary" in payload


def test_the_new_subcommands_are_reachable_from_the_cli() -> None:
    from ai import natasha as natasha_cli

    for command in ("traits", "mood", "initiative", "relationship-memory"):
        assert command in natasha_cli._COMMANDS


def test_the_new_menu_entries_point_at_the_real_subcommands() -> None:
    from cli import menu

    by_label = {label: argv for label, _action, argv in menu.MENU}
    assert by_label["Natasha: her standing character traits (the realism settings)"] == ("traits",)
    assert by_label["Natasha: her emotional state (named mood for the moment)"] == ("mood",)
    assert by_label["Natasha: the initiatives she may take (never repeated)"] == ("initiative",)
    assert by_label[
        "Natasha: the structured relationship memory (facts, story, preferences)"
    ] == ("relationship-memory",)


# =========================================================================
# GUI: the sidebar actions and the mouth
# =========================================================================
def test_the_new_sidebar_actions_run_the_real_subcommands() -> None:
    """The four components are reachable from the window's own sidebar."""
    from cli.menu import run_natasha
    from gui import actions as gui_actions

    by_key = {action.key: action for action in gui_actions.ACTIONS}
    for key, argv in (
        ("natasha-traits", ("traits",)),
        ("natasha-mood", ("mood",)),
        ("natasha-initiative", ("initiative",)),
        ("natasha-relationship-memory", ("relationship-memory",)),
    ):
        action = by_key[key]
        assert action.command is run_natasha
        assert action.argv == argv
        # They are real buttons, not hidden ones: the window draws them.
        assert not action.hidden
        assert action in gui_actions.visible_actions()
        # And they stay resolvable by key, like every other action.
        assert gui_actions.action_by_key(key).key == key


def test_the_mouth_reports_the_mood_and_initiative_in_its_reply() -> None:
    """The chat mouth's agent carries the new state through the same exchange."""
    from companion.moods import MOOD_SAD

    config = CompanionConfig(db_path=None, persistent=False)
    agent = CompanionAgent(config, caller=FakeCaller(["أنا هنا معك."]))
    session = agent.new_session("mouth")
    reply = agent.ask(session.id, "أنا حزين جداً اليوم")
    assert reply.mood == MOOD_SAD
    assert reply.repetition is not None
    assert reply.initiative is None or reply.initiative in INITIATIVE_KEYS


def test_the_mouth_answer_is_unchanged_by_the_new_layers(monkeypatch) -> None:
    """The visible reply is still the model's own words (plus the header)."""
    from ai import natasha_mouth

    config = CompanionConfig(db_path=None, persistent=False)
    agent = CompanionAgent(config, caller=FakeCaller(["كلامي أنا فقط."]))
    assistant = natasha_mouth.NatashaAssistant(agent)
    answer = assistant.ask("كيف حالك؟")
    assert "كلامي أنا فقط." in answer.text
    assert answer.text.startswith(natasha_mouth.HEADER)
