"""The universal linguistic polish and the Arabic vocalizer.

The project answers through seven characters and its own model; every answer now
passes through one shared polisher (:mod:`core.text_polish`, wired in
``core.jarvis``), and an Arabic answer can be **fully vocalized** (tashkeel)
through :mod:`core.arabic_tashkeel`. These tests pin the behaviour the task asked
for:

* the polish is applied to *any* character's answer, not only the companion's;
* it is deterministic, presentation-only (never adds/removes/reorders a word), and
  idempotent;
* it is explainable -- the report names what was changed;
* the vocalization has two forms (formal with إعراب, colloquial without), never
  touches a Latin identifier/number/formula, and is idempotent;
* the choice is a saved preference, flipped by the ``tashkeel`` conversation
  control, and off by default.
"""

from __future__ import annotations

import pytest
from core.arabic_tashkeel import (
    MODE_COLLOQUIAL,
    MODE_FORMAL,
    diacritize,
    diacritize_report,
    has_arabic,
    strip_diacritics,
)
from core.text_polish import (
    TASHKEEL_COLLOQUIAL,
    TASHKEEL_FORMAL,
    polish_answer,
    polish_text,
    resolve_tashkeel,
    tashkeel_text,
)

# --------------------------------------------------------------------------- #
# The Arabic vocalizer
# --------------------------------------------------------------------------- #


def test_the_vocalizer_marks_a_simple_sentence() -> None:
    text = diacritize("ضغط المكمن", mode=MODE_FORMAL)
    assert text == "ضَغْطُ الْمَكْمَنِ"


def test_the_vocalizer_handles_the_article() -> None:
    """A moon letter keeps the ``ل`` (sukun); a sun letter assimilates it (a shadda)."""
    assert diacritize("المكمن").startswith("الْمَكْمَن")  # moon: the ``ل`` carries a sukun
    assert diacritize("الشمس") == "الشّمس"  # sun: the ``ل`` is assimilated into a shadda


def test_the_vocalizer_is_idempotent() -> None:
    for source in [
        "ضغط المكمن",
        "المسامية هي نسبة الفراغات في الصخر.",
        "كيف أقيس النفاذية؟",
        "من ضغط المكمن",
        "الحفر في الطبقة",
    ]:
        once = diacritize(source)
        twice = diacritize(once)
        assert twice == once, source


def test_the_vocalizer_never_touches_a_latin_identifier_or_number() -> None:
    text = diacritize("القيمة 3.14 والوحدة SPE ومدى 10--20", mode=MODE_FORMAL)
    assert "3.14" in text
    assert "SPE" in text
    assert "10--20" in text


def test_an_english_reply_is_returned_byte_identical() -> None:
    source = "The porosity is 0.2 and SPE 12345."
    assert diacritize(source) == source
    assert not has_arabic(source)


def test_the_colloquial_mode_drops_the_case_endings() -> None:
    formal = diacritize("ضغط المكمن", mode=MODE_FORMAL)
    colloquial = diacritize("ضغط المكمن", mode=MODE_COLLOQUIAL)
    assert formal == "ضَغْطُ الْمَكْمَنِ"
    assert colloquial == "ضَغْط الْمَكْمَن"
    # The case endings (the final short vowels) are what the spoken form drops.
    assert strip_diacritics(formal) == strip_diacritics(colloquial)


def test_an_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        diacritize("ضغط", mode="loud")


def test_the_vocalizer_reports_its_coverage() -> None:
    report = diacritize_report("المسامية هي نسبة الفراغات في الصخر.")
    assert report.words == 6
    assert report.covered == 6
    assert report.coverage == 1.0
    assert report.changed


def test_strip_diacritics_round_trips_the_bare_letters() -> None:
    assert strip_diacritics("ضَغْطُ") == "ضغط"


# --------------------------------------------------------------------------- #
# The universal polish
# --------------------------------------------------------------------------- #


def test_the_polish_repairs_arabic_typography() -> None:
    report = polish_answer("المسامية في الصخر , وهي مهمة ؟", language="ar")
    assert report.text == "المسامية في الصخر، وهي مهمة؟"
    assert report.language == "ar"
    assert any("space" in change for change in report.changes)
    assert any("Arabic punctuation" in change for change in report.changes)


def test_the_polish_repairs_english_typography() -> None:
    report = polish_answer("hello there ,how are you ?  i am fine .", language="en")
    assert report.text == "Hello there, how are you? I am fine."
    assert report.language == "en"


def test_the_polish_is_idempotent() -> None:
    for case in [
        "المسامية في الصخر , وهي مهمة ؟",
        "hello there ,how are you ?",
        "القيمة 3.14 والمدى 10--20",
    ]:
        once = polish_answer(case)
        twice = polish_answer(once.text)
        assert twice.text == once.text, case


def test_the_polish_never_adds_or_removes_a_word() -> None:
    source = "the reservoir pressure is 3000 psi , and the porosity is 0.2"
    polished = polish_text(source, language="en")

    def words(text: str) -> list[str]:
        stripped = [word.strip(".,!?؛،").casefold() for word in text.split()]
        return [word for word in stripped if word]

    assert words(polished) == words(source)


def test_tashkeel_is_off_unless_asked_for() -> None:
    """The default output is the project's long-standing bare Arabic."""
    report = polish_answer("المسامية في الصخر", language="ar", tashkeel=False)
    assert not report.diacritized
    assert report.tashkeel_mode == ""
    assert strip_diacritics(report.text) == report.text


def test_tashkeel_can_be_asked_for_explicitly() -> None:
    report = polish_answer(
        "المسامية في الصخر , وهي مهمة", language="ar", tashkeel=True, tashkeel_mode=TASHKEEL_FORMAL
    )
    assert report.diacritized
    assert report.tashkeel_mode == TASHKEEL_FORMAL
    # The typography was repaired *and* the text vocalized.
    assert "،" in report.text
    assert strip_diacritics(report.text) != report.text


def test_tashkeel_never_touches_an_english_reply() -> None:
    report = polish_answer("hello there ,friend", language="en", tashkeel=True)
    assert report.text == "Hello there, friend."
    assert not report.diacritized
    assert not has_arabic(report.text)


def test_the_polish_report_is_explainable() -> None:
    report = polish_answer("المسامية في الصخر , وهي مهمة", language="ar", tashkeel=True)
    assert report.changed
    assert "tashkeel" in report.describe()
    clean = polish_answer("هذه جملة سليمة.", language="ar")
    assert clean.describe().startswith("polish")


def test_tashkeel_text_is_the_direct_doorway() -> None:
    assert tashkeel_text("ضغط المكمن") == "ضَغْطُ الْمَكْمَنِ"
    assert tashkeel_text("ضغط المكمن", mode=TASHKEEL_COLLOQUIAL) == "ضَغْط الْمَكْمَن"


# --------------------------------------------------------------------------- #
# The saved preference
# --------------------------------------------------------------------------- #


def test_the_preference_defaults_to_tashkeel_off() -> None:
    from core.preferences import PreferenceRecord

    record = PreferenceRecord()
    assert record.tashkeel is False
    assert record.tashkeel_mode == TASHKEEL_FORMAL


def test_the_preference_round_trips() -> None:
    from core.preferences import PreferenceRecord

    record = PreferenceRecord(tashkeel=True, tashkeel_mode=TASHKEEL_COLLOQUIAL)
    assert PreferenceRecord.from_primitive(record.to_primitive()) == record


def test_a_pre_existing_preference_file_reads_as_off() -> None:
    """An upgrade must never start vocalizing behind the user's back."""
    from core.preferences import PreferenceRecord

    record = PreferenceRecord.from_primitive({"user_name": "Sam", "language": "ar"})
    assert record.tashkeel is False
    assert record.tashkeel_mode == TASHKEEL_FORMAL


def test_an_unknown_tashkeel_mode_is_rejected() -> None:
    from core.preferences import PreferenceRecord

    with pytest.raises(ValueError):
        PreferenceRecord(tashkeel_mode="loud")


def test_resolve_tashkeel_honours_an_explicit_argument(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert resolve_tashkeel(enabled=True, mode=TASHKEEL_COLLOQUIAL) == (True, TASHKEEL_COLLOQUIAL)
    with pytest.raises(ValueError):
        resolve_tashkeel(enabled=True, mode="loud")


def test_resolve_tashkeel_follows_the_saved_preference(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from core.paths import data_dir
    from core.preferences import UserPreferences

    UserPreferences(data_dir()).update(tashkeel=True, tashkeel_mode=TASHKEEL_COLLOQUIAL)
    assert resolve_tashkeel() == (True, TASHKEEL_COLLOQUIAL)


# --------------------------------------------------------------------------- #
# The conversation control and the Jarvis wiring
# --------------------------------------------------------------------------- #


def _jarvis(tmp_path, preferences=None):
    from core.assistant import build_default_assistant
    from core.command_router import NaturalLanguageRouter
    from core.jarvis import JarvisMode
    from core.tools import build_sandbox

    store: dict[str, object] = {}

    def remember(name: str, value: object) -> None:
        store[name] = value
        if preferences is not None:
            preferences.update(**{name: value})

    mode = JarvisMode(
        build_default_assistant(),
        router=NaturalLanguageRouter(build_sandbox(tmp_path)),
        on_preference=remember,
    )
    return mode, store


def test_the_tashkeel_control_turns_it_on_and_persists(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from core.paths import data_dir
    from core.preferences import UserPreferences

    preferences = UserPreferences(data_dir())
    mode, store = _jarvis(tmp_path, preferences)
    turn = mode.handle("tashkeel on", speak=False)
    assert turn.intent == "tashkeel"
    assert "on" in turn.response.lower()
    assert preferences.record.tashkeel is True
    assert preferences.record.tashkeel_mode == TASHKEEL_FORMAL
    assert store["tashkeel"] is True


def test_the_tashkeel_control_chooses_the_colloquial_form(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from core.paths import data_dir
    from core.preferences import UserPreferences

    preferences = UserPreferences(data_dir())
    mode, _ = _jarvis(tmp_path, preferences)
    mode.handle("tashkeel colloquial", speak=False)
    assert preferences.record.tashkeel is True
    assert preferences.record.tashkeel_mode == TASHKEEL_COLLOQUIAL


def test_the_tashkeel_control_turns_it_off(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from core.paths import data_dir
    from core.preferences import UserPreferences

    preferences = UserPreferences(data_dir())
    mode, _ = _jarvis(tmp_path, preferences)
    mode.handle("tashkeel on", speak=False)
    mode.handle("tashkeel off", speak=False)
    assert preferences.record.tashkeel is False


def test_a_bare_tashkeel_toggles(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from core.paths import data_dir
    from core.preferences import UserPreferences

    preferences = UserPreferences(data_dir())
    mode, _ = _jarvis(tmp_path, preferences)
    mode.handle("tashkeel", speak=False)
    assert preferences.record.tashkeel is True
    mode.handle("tashkeel", speak=False)
    assert preferences.record.tashkeel is False


def test_an_arabic_answer_is_polished_and_vocalized_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from core.paths import data_dir
    from core.preferences import UserPreferences

    UserPreferences(data_dir()).update(tashkeel=True, tashkeel_mode=TASHKEEL_FORMAL)
    mode, _ = _jarvis(tmp_path)
    turn = mode.handle("ما هي المسامية؟", speak=False)
    assert turn.routed == "answer"
    # The answer's Arabic carries vowel marks, and the graph's snake_case labels
    # never leaked as code tokens.
    assert strip_diacritics(turn.response) != turn.response
    assert "_" not in turn.response


def test_an_english_answer_is_polished_but_never_vocalized(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from core.paths import data_dir
    from core.preferences import UserPreferences

    UserPreferences(data_dir()).update(tashkeel=True, tashkeel_mode=TASHKEEL_FORMAL)
    mode, _ = _jarvis(tmp_path)
    turn = mode.handle("how do I measure porosity on a core sample?", speak=False)
    assert turn.routed == "answer"
    # The answer is English (it may carry an Arabic alias from the graph), but the
    # English rules added no vowel marks: it was not vocalized.
    assert strip_diacritics(turn.response) == turn.response


class _StubAssistant:
    """A minimal assistant returning a fixed messy answer, for the polish gate."""

    def __init__(self, text: str) -> None:
        self._text = text

    def ask(self, question: str, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(text=self._text, citations=(), confidence=0.0)

    def describe(self) -> str:
        return "the stub mouth"


def test_the_polish_can_be_switched_off(tmp_path) -> None:
    from core.command_router import NaturalLanguageRouter
    from core.jarvis import JarvisMode
    from core.tools import build_sandbox

    question = "hello there ,how are you ?"
    messy = "hello there ,friend"

    def _mode(polish: bool) -> JarvisMode:
        mode = JarvisMode(
            _StubAssistant(messy),
            router=NaturalLanguageRouter(build_sandbox(tmp_path)),
            polish=polish,
        )
        return mode

    raw = _mode(polish=False).handle(question, speak=False)
    polished = _mode(polish=True).handle(question, speak=False)
    assert raw.response == messy  # the raw mouth output, untouched
    assert polished.response == "Hello there, friend."  # the shared polish applied


def test_the_lead_is_polished_in_step_with_the_body(tmp_path) -> None:
    """The lead opens the body, so it must be polished the same way."""
    from core.assistant import build_default_assistant

    answer = build_default_assistant().ask("what should I do to increase permeability?")
    if answer.lead is None:
        pytest.skip("no lead for this question in this build")
    mode, _ = _jarvis(tmp_path)
    turn = mode.handle("what should I do to increase permeability?", speak=False)
    assert turn.lead
    assert turn.response.startswith(turn.lead)


# --------------------------------------------------------------------------- #
# The companion (Natasha) and the scholar (Hakeem) share the same polish
# --------------------------------------------------------------------------- #


class _FakeCompanionCaller:
    """A model seam returning one fixed reply, so the agent runs with no weights."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def __call__(self, messages, temperature, max_tokens) -> str:
        return self.reply


def _natasha_agent(tmp_path, reply: str):
    from companion import CompanionAgent, CompanionConfig

    config = CompanionConfig(db_path=tmp_path / "natasha.db")
    return CompanionAgent(config, caller=_FakeCompanionCaller(reply))


def test_natasha_honours_the_saved_tashkeel_preference(tmp_path, monkeypatch) -> None:
    """A companion reply is polished by the *universal* entry point, so the same
    preference every other character honours reaches Natasha too."""
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from core.paths import data_dir
    from core.preferences import UserPreferences

    messy = "المسامية هي نسبة الفراغات في الصخر , وهي مهمة ؟"
    agent = _natasha_agent(tmp_path, messy)
    session = agent.new_session("a")
    bare = agent.ask(session.id, "ما هي المسامية؟").text
    assert " ," not in bare  # typography repaired, no vocalization yet
    assert "َ" not in bare

    UserPreferences(data_dir()).update(tashkeel=True, tashkeel_mode=TASHKEEL_FORMAL)
    agent2 = _natasha_agent(tmp_path, messy)
    session2 = agent2.new_session("b")
    vocalized = agent2.ask(session2.id, "ما هي المسامية؟").text
    assert "َ" in vocalized  # now fully vocalized
    # Vocalization is presentation-only: the bare letters are unchanged.
    assert strip_diacritics(vocalized) == bare


def test_natasha_leaves_english_untouched_by_tashkeel(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from core.paths import data_dir
    from core.preferences import UserPreferences

    UserPreferences(data_dir()).update(tashkeel=True, tashkeel_mode=TASHKEEL_FORMAL)
    agent = _natasha_agent(tmp_path, "hello there ,friend")
    session = agent.new_session("en")
    reply = agent.ask(session.id, "hello").text
    assert reply == "Hello there, friend."
    assert not has_arabic(reply)


def test_hakeem_honours_the_saved_tashkeel_preference(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from core.paths import data_dir
    from core.preferences import UserPreferences

    messy = "ضغط المكمن هو الضغط داخل المسام , وهو مهم ؟"

    def _hakeem():
        from hakeem import HakeemAgent, HakeemConfig

        config = HakeemConfig(db_path=tmp_path / "hakeem.db")
        return HakeemAgent(config, caller=_FakeCompanionCaller(messy))

    first = _hakeem()
    bare = first.ask(first.new_session("a").id, "ما هو ضغط المكمن؟").text
    assert "َ" not in bare
    UserPreferences(data_dir()).update(tashkeel=True, tashkeel_mode=TASHKEEL_FORMAL)
    second = _hakeem()
    vocalized = second.ask(second.new_session("b").id, "ما هو ضغط المكمن؟").text
    assert "َ" in vocalized


# --------------------------------------------------------------------------- #
# The environment overrides (a headless run's one-process choice)
# --------------------------------------------------------------------------- #


def test_the_environment_can_force_tashkeel_on(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FAYD_TASHKEEL", "on")
    assert resolve_tashkeel() == (True, TASHKEEL_FORMAL)


def test_the_environment_can_force_the_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FAYD_TASHKEEL", "1")
    monkeypatch.setenv("FAYD_TASHKEEL_MODE", "colloquial")
    assert resolve_tashkeel() == (True, TASHKEEL_COLLOQUIAL)


def test_an_explicit_argument_beats_the_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FAYD_TASHKEEL", "on")
    monkeypatch.setenv("FAYD_TASHKEEL_MODE", "formal")
    assert resolve_tashkeel(enabled=False, mode=TASHKEEL_COLLOQUIAL) == (
        False,
        TASHKEEL_COLLOQUIAL,
    )


def test_a_typo_in_the_environment_is_ignored(tmp_path, monkeypatch) -> None:
    """An unrecognised value never silently turns vocalization on or off."""
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FAYD_TASHKEEL", "maybe")
    monkeypatch.setenv("FAYD_TASHKEEL_MODE", "loud")
    assert resolve_tashkeel() == (False, TASHKEEL_FORMAL)


def test_a_homograph_keeps_its_first_curated_form() -> None:
    """``مِنْ`` (the preposition) must not be clobbered by ``مَنْ`` (the interrogative).

    The two share a skeleton; a plain dict comprehension over the lexicon let the
    later entry silently win, so the preposition came out as ``مَنْ``. The lexicon
    is now first-wins, and this pins the fix.
    """
    assert diacritize("من") == "مِنْ"
    assert diacritize("من فضلك").startswith("مِنْ")
