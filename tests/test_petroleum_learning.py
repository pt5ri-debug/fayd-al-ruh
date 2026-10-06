"""Tests for the petroleum-engineering task catalog and its learning run."""

from core.domain_knowledge import discipline_keys
from core.petroleum_tasks import (
    PETROLEUM_HELD_OUT_TASKS,
    PETROLEUM_TASKS,
    petroleum_task_by_name,
    petroleum_tasks,
    success_fact_for,
    tasks_by_discipline,
)
from core.sandbox_environment import SandboxEnvironment
from ml.petroleum import format_report, petroleum_success, train_petroleum


def test_petroleum_task_names_are_unique():
    names = [task.name for task in petroleum_tasks()]
    assert len(names) == len(set(names))


def test_every_task_declares_a_known_success_fact():
    for task in petroleum_tasks():
        assert success_fact_for(task.name)


def test_held_out_tasks_are_not_trained_on():
    trained = {task.name for task in PETROLEUM_TASKS}
    held_out = {task.name for task in PETROLEUM_HELD_OUT_TASKS}
    assert trained.isdisjoint(held_out)
    assert len(held_out) >= 1


def test_each_task_is_solvable_by_an_explicit_plan():
    """Every task must reach its milestone with a legal, strictly-progressing plan."""
    for task in petroleum_tasks():
        goal = petroleum_success(task.name)
        environment = SandboxEnvironment(task)
        visited = {environment.state.facts}
        while not goal(environment.state) and environment.step_count < task.max_steps:
            progressed = False
            for spec in task.allowed_actions:
                if not environment.state.satisfies(spec.preconditions):
                    continue
                next_facts = (environment.state.facts - spec.remove_effects) | spec.add_effects
                if next_facts == environment.state.facts or next_facts in visited:
                    continue
                environment.step(spec.to_action())
                visited.add(environment.state.facts)
                progressed = True
                break
            if not progressed:
                break
        assert goal(environment.state), task.name


def test_tasks_cover_every_discipline_in_the_ontology():
    covered = set(tasks_by_discipline())
    assert set(discipline_keys()).issubset(covered)


def test_task_lookup_by_name():
    task = petroleum_task_by_name("drill-well")
    assert task.discipline == "drilling"


def test_train_petroleum_learns_every_trained_discipline():
    report = train_petroleum(episodes=120)
    assert report.overall_success == 1.0
    assert set(report.discipline_success) == set(discipline_keys())
    assert all(value == 1.0 for value in report.discipline_success.values())


def test_train_petroleum_generalises_to_unseen_disciplines():
    report = train_petroleum(episodes=120)
    assert report.generalisation_success
    assert set(report.generalisation_success) == {"geoscience", "flow_assurance"}
    assert all(value == 1.0 for value in report.generalisation_success.values())


def test_include_held_out_false_skips_generalisation():
    report = train_petroleum(episodes=120, include_held_out=False)
    assert report.generalisation_success == {}
    assert all(outcome.name in {t.name for t in PETROLEUM_TASKS} for outcome in report.outcomes)


def test_petroleum_report_serialises_cleanly():
    report = train_petroleum(episodes=120)
    data = report.as_dict()
    assert data["overall_success"] == 1.0
    assert len(data["tasks"]) == len(report.outcomes)
    assert "discipline_success" in data


def test_format_report_names_tasks_and_disciplines():
    report = train_petroleum(episodes=120)
    text = format_report(report)
    assert "drill-well" in text
    assert "Overall held-out success" in text
    assert "هندسة الحفر" in text
