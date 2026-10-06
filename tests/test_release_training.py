"""Tests for the release training pipeline and the curriculum-fed corpus."""

from __future__ import annotations

import json

from core.curriculum_knowledge import course_keys, courses
from core.domain_knowledge import discipline_keys, disciplines
from ml.llm.corpus import build_corpus, curriculum_sentences
from ml.release import (
    RELEASE_MANIFEST_NAME,
    TrainingConfig,
    corpus_summary,
    format_manifest,
    knowledge_summary,
    train_release,
)


def test_curriculum_sentences_cover_every_discipline_and_course():
    sentences = curriculum_sentences()
    text = " ".join(sentences)

    for discipline in disciplines():
        assert discipline.label in text
        for label, _aliases in discipline.concepts:
            assert label in text
    for course in courses():
        assert course.label in text
        for label, _aliases in course.concepts:
            assert label in text


def test_curriculum_sentences_include_relations():
    text = " ".join(curriculum_sentences())
    # A relation sentence has the shape "<source> <relation> <target>."
    for source, target, relation, _confidence in courses()[0].relations:
        assert f"{source} {relation} {target}." in text


def test_build_corpus_contains_the_curriculum_block():
    corpus = build_corpus(repeats=1, structured_repeats=1)
    assert "is a concept from" in corpus
    # The corpus is larger than the hand-written prose alone.
    assert len(corpus) > len(" ".join(curriculum_sentences()))


def test_knowledge_summary_reports_full_coverage():
    summary = knowledge_summary()
    assert summary["disciplines"] == len(discipline_keys())
    assert summary["courses"] == len(course_keys())
    assert summary["curriculum_coverage"] == 1.0
    assert summary["ontology_concepts"] > 0
    assert summary["ontology_relations"] > 0


def test_corpus_summary_counts_match_the_generator():
    config = TrainingConfig(llm_repeats=2, llm_structured_repeats=2)
    summary = corpus_summary(config)
    assert summary["curriculum_sentences"] == len(curriculum_sentences())
    assert summary["characters"] > 0


def test_train_release_writes_every_artifact_and_a_manifest(tmp_path):
    config = TrainingConfig(
        episodes=3,
        llm_steps=5,
        llm_repeats=1,
        llm_structured_repeats=1,
        include_base=False,
    )
    checkpoint = tmp_path / "llm" / "fayd-llm.npz"
    manifest = train_release(config, save_dir=tmp_path, checkpoint=checkpoint)

    assert (tmp_path / RELEASE_MANIFEST_NAME).exists()
    assert checkpoint.exists()
    assert (tmp_path / "petroleum").is_dir()
    assert manifest["language_model"]["steps"] == 5
    assert manifest["petroleum"]["overall_success"] >= 0.0
    assert manifest["knowledge"]["curriculum_coverage"] == 1.0

    stored = json.loads((tmp_path / RELEASE_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert stored["version"] == manifest["version"]
    assert stored["knowledge"] == manifest["knowledge"]


def test_format_manifest_handles_a_manifest_without_training_results():
    from ml.release import TrainingConfig as Config

    config = Config(episodes=1, llm_steps=1)
    summary = format_manifest(
        {
            "version": "9.9.9",
            "trained_at": "2026-01-01T00:00:00+00:00",
            "git_revision": "",
            "config": {
                "episodes": config.episodes,
                "llm_steps": config.llm_steps,
            },
            "knowledge": knowledge_summary(),
        }
    )
    assert "9.9.9" in summary
    assert "Curriculum coverage" in summary
