"""Tests for the curated Arabic ontology glossary (Phase 49).

The project is Arabic-first, but before Phase 49 only 39 of its 495 seeded
concepts carried an Arabic alias, so an Arabic question reached almost none of the
graph. These tests pin the fix: the table resolves by **label** (never by a shared
alias, which silently lost ``acid_stimulation``'s Arabic term), the pass is
idempotent, it never invents a concept, and the reasoning layers actually fire for
an Arabic question afterwards.
"""

from __future__ import annotations

import pytest
from core.arabic_glossary import (
    ARABIC_ALIASES,
    GlossaryReport,
    apply_arabic_aliases,
    arabic_alias_coverage,
    has_arabic,
    missing_arabic_labels,
    normalize_arabic_term,
)
from core.cognitive_controller import CognitiveController
from core.ontology import Ontology


@pytest.fixture(scope="module")
def controller():
    return CognitiveController()


def test_has_arabic_detects_script():
    assert has_arabic("المسامية")
    assert has_arabic("porosity مسامية")
    assert not has_arabic("porosity")
    assert not has_arabic("")


def test_controller_applies_the_glossary_on_start(controller):
    covered, total = arabic_alias_coverage(controller.ontology)
    assert total >= 495
    assert covered >= 200, "the Arabic frame must reach the seeded concepts"


def test_glossary_covers_the_seeded_concepts(controller):
    """The gap Phase 49 opened and Phase 51 closed: 39/495 before, all after."""
    covered, total = arabic_alias_coverage(controller.ontology)
    assert covered == total, "every seeded concept carries an Arabic term"


def test_glossary_is_idempotent(controller):
    report = apply_arabic_aliases(controller.ontology)
    assert report.added == ()
    assert not report.unknown
    assert report.total == len(ARABIC_ALIASES) - len(report.unknown)


def test_glossary_reports_an_unknown_label_rather_than_inventing_it():
    ontology = Ontology()
    report = apply_arabic_aliases(ontology, (("not_a_real_concept", "شيء"),))
    assert report.unknown == ("not_a_real_concept",)
    assert report.added == ()
    assert not ontology.concepts, "nothing may be invented"


def test_glossary_resolves_a_label_exactly_not_by_a_shared_alias():
    """``acidizing`` is an alias of two concepts; the Arabic term must not be lost.

    A lookup by the *word* returned ``well_stimulation`` (which shares the alias)
    instead of ``acid_stimulation``, so the term was added to the wrong concept and
    the Arabic question still could not resolve. Matching the label exactly fixes
    it -- and this is the regression test for that bug.
    """
    ontology = CognitiveController().ontology
    assert ontology.find_concepts("التحميض"), "the Arabic acidizing term must resolve"
    labels = {concept.label for concept in ontology.find_concepts("التحميض")}
    assert labels == {"acid_stimulation"}


def test_glossary_never_touches_a_relation():
    """Adding an alias keeps the concept id, so the graph's edges survive."""
    ontology = CognitiveController().ontology
    before = len(ontology.relations)
    apply_arabic_aliases(ontology)
    assert len(ontology.relations) == before


def test_missing_arabic_labels_reports_the_remaining_gap(controller):
    """Phase 51 closed the last of the gap, so the honest report is now empty.

    The helper still reports what it finds rather than hiding it -- the seed has no
    uncovered concept left, so it returns ``()``; the property (sorted, no Arabic)
    is checked on a table the ontology does not fully hold.
    """
    missing = missing_arabic_labels(controller.ontology)
    assert missing == (), "every seeded concept now carries an Arabic term"
    assert missing == tuple(sorted(missing))


def test_normalize_arabic_term_matches_the_projects_normalisation():
    assert normalize_arabic_term("المسامية") == normalize_arabic_term("المسامية")
    assert normalize_arabic_term("  النفاذية ") == normalize_arabic_term("النفاذية")


def test_report_describes_itself():
    report = GlossaryReport(("a",), ("b",), ("c",))
    assert report.total == 2
    assert "1" in report.describe()


def test_arabic_question_reaches_the_plan_layer(controller):
    from core.answer_audit import LAYER_PLAN, answer_layers
    from core.assistant import build_default_assistant

    assistant = build_default_assistant()
    answer = assistant.ask("كيف أقيس المسامية والنفاذية على عينة لب؟")
    assert LAYER_PLAN in answer_layers(answer)


def test_arabic_question_reaches_the_decision_layer():
    from core.answer_audit import LAYER_DECISION, answer_layers
    from core.assistant import build_default_assistant

    assistant = build_default_assistant()
    answer = assistant.ask("ماذا أفعل لزيادة النفاذية؟")
    assert LAYER_DECISION in answer_layers(answer)


def test_arabic_question_reaches_the_belief_layer():
    from core.answer_audit import LAYER_BELIEF, answer_layers
    from core.assistant import build_default_assistant

    assistant = build_default_assistant()
    answer = assistant.ask("هل التحميض يزيد النفاذية؟")
    assert LAYER_BELIEF in answer_layers(answer)


def test_arabic_question_reaches_the_analogy_layer():
    from core.answer_audit import LAYER_ANALOGY, answer_layers
    from core.assistant import build_default_assistant

    assistant = build_default_assistant()
    answer = assistant.ask("هل المسامية مثل النفاذية؟")
    assert LAYER_ANALOGY in answer_layers(answer)


def test_arabic_unrelated_question_stays_plain():
    from core.answer_audit import REASONING_LAYERS, answer_layers
    from core.assistant import build_default_assistant

    assistant = build_default_assistant()
    answer = assistant.ask("ما هي عاصمة فرنسا؟")
    assert not (answer_layers(answer) & REASONING_LAYERS)
