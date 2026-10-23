"""The best-effort helper keeps a deliberate swallow non-fatal *and* visible."""

from __future__ import annotations

import pytest
from core.best_effort import (
    MAX_RECORDED,
    best_effort,
    clear_swallowed,
    swallowed_errors,
)


@pytest.fixture(autouse=True)
def _clean():
    clear_swallowed()
    yield
    clear_swallowed()


def test_a_successful_operation_records_nothing():
    with best_effort("do the thing"):
        result = 1 + 1
    assert result == 2
    assert swallowed_errors() == ()


def test_a_failure_is_swallowed_but_recorded():
    with best_effort("save the preference"):
        raise OSError("read-only folder")
    recorded = swallowed_errors()
    assert len(recorded) == 1
    assert recorded[0].what == "save the preference"
    assert isinstance(recorded[0].error, OSError)
    assert "read-only folder" in recorded[0].describe()


def test_execution_continues_after_the_swallowed_error():
    reached = False
    with best_effort("probe an optional engine"):
        raise RuntimeError("absent")
    reached = True
    assert reached


def test_on_error_observes_without_failing():
    seen: list[BaseException] = []
    with best_effort("try a sibling engine", on_error=seen.append):
        raise ValueError("no voice")
    assert isinstance(seen[0], ValueError)
    assert len(swallowed_errors()) == 1


def test_the_record_is_bounded():
    for index in range(MAX_RECORDED + 10):
        with best_effort(f"attempt {index}"):
            raise RuntimeError(str(index))
    recorded = swallowed_errors()
    assert len(recorded) == MAX_RECORDED
    # The newest survive; the oldest are dropped.
    assert recorded[-1].what == f"attempt {MAX_RECORDED + 9}"
