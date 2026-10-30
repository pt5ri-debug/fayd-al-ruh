"""The operation log: bounded, redacted, and forgiving of a corrupt line.

The log is the answer to "why did this model fail?" -- a question the project could
not answer before, because nothing recorded a model operation. These tests pin the
three properties that make it safe to keep: it never grows without limit, it never
stores a credential, and a corrupt row never breaks a reader.
"""

from __future__ import annotations

import json

import pytest
from core.model_log import (
    KEY,
    KNOWN_OPERATIONS,
    MAX_EVENTS,
    NAMESPACE,
    OP_FAILURE,
    OP_LOAD,
    OP_REGISTER,
    ModelEvent,
    ModelLog,
    model_log,
    redact,
)


@pytest.fixture()
def store(tmp_path):
    return tmp_path / "log-root"


# ------------------------------------------------------------------ redaction


def test_redact_removes_a_bearer_token():
    cleaned = redact("Authorization: Bearer abcdefghijklmnop")
    assert "abcdefghijklmnop" not in cleaned
    assert "***" in cleaned


def test_redact_removes_an_api_key_assignment():
    for text in (
        "api_key=sk-abcdefghijklmnopqrst",
        "token: hf_abcdefghijklmn",
        "password = hunter2hunter2",
        "secret=ghp_abcdefghijklmnop",
    ):
        cleaned = redact(text)
        assert "abcdefghijklmn" not in cleaned
        assert "hunter2hunter2" not in cleaned


def test_redact_removes_a_credential_inside_a_url():
    cleaned = redact("failed to reach https://user:supersecret@example.com/v1/models")
    assert "supersecret" not in cleaned
    assert "example.com" in cleaned


def test_redact_leaves_ordinary_text_alone():
    text = "The file is not a GGUF: /home/user/models/thing.gguf"
    assert redact(text) == text


def test_redact_caps_a_runaway_message():
    cleaned = redact("x" * 50_000)
    assert len(cleaned) <= 1000


def test_redact_handles_none():
    assert redact(None) == ""  # type: ignore[arg-type]


# ------------------------------------------------------------------- events


def test_an_event_round_trips_through_a_mapping():
    event = ModelEvent(
        operation=OP_LOAD, model_id="m", display_name="Model", runtime="llama.cpp",
        ok=True, message="loaded", at="2026-01-01T00:00:00+00:00",
    )
    restored = ModelEvent.from_primitive(event.to_primitive())
    assert restored == event


def test_an_event_with_no_operation_is_unreadable():
    assert ModelEvent.from_primitive({"model_id": "m"}) is None
    assert ModelEvent.from_primitive("not a mapping") is None
    assert ModelEvent.from_primitive(None) is None


def test_an_event_describes_itself_readably():
    ok = ModelEvent(operation=OP_LOAD, display_name="Ultron", ok=True, at="now")
    assert "Ultron" in ok.describe()
    assert "ok" in ok.describe()
    bad = ModelEvent(operation=OP_FAILURE, display_name="Broken", ok=False, message="it broke")
    assert "failed" in bad.describe()
    assert "it broke" in bad.describe()


def test_an_event_with_no_model_still_describes_itself():
    assert "the manager" in ModelEvent(operation=OP_REGISTER).describe()


# --------------------------------------------------------------- appending


def test_appending_and_reading_back(store):
    log = ModelLog(store)
    log.append(OP_REGISTER, model_id="m", display_name="Model", ok=True, message="hello")
    events = log.entries()
    assert len(events) == 1
    assert events[0].model_id == "m"
    assert events[0].message == "hello"
    assert events[0].at


def test_entries_are_oldest_first_and_tail_is_newest_first(store):
    log = ModelLog(store)
    for index in range(5):
        log.append(OP_REGISTER, model_id=f"m{index}")
    assert [event.model_id for event in log.entries()] == ["m0", "m1", "m2", "m3", "m4"]
    assert [event.model_id for event in log.tail(2)] == ["m4", "m3"]


def test_a_zero_or_negative_tail_is_empty(store):
    log = ModelLog(store)
    log.append(OP_REGISTER, model_id="m")
    assert log.tail(0) == ()
    assert log.tail(-3) == ()


def test_for_model_filters_to_one_model(store):
    log = ModelLog(store)
    log.append(OP_LOAD, model_id="a")
    log.append(OP_LOAD, model_id="b")
    log.append(OP_LOAD, model_id="a")
    assert len(log.for_model("a")) == 2
    assert log.for_model("") == ()


def test_failures_returns_only_the_failed_events(store):
    log = ModelLog(store)
    log.append(OP_LOAD, model_id="a", ok=True)
    log.append(OP_FAILURE, model_id="a", ok=False, message="broke")
    assert [event.ok for event in log.failures()] == [False]


def test_a_message_is_redacted_before_it_is_stored(store):
    log = ModelLog(store)
    log.append(OP_FAILURE, ok=False, message="api_key=sk-abcdefghijklmnopqrst")
    assert "abcdefghijklmn" not in log.entries()[0].message


def test_an_unknown_operation_is_still_recorded(store):
    """Refusing to log something that happened would be worse than a new label."""
    log = ModelLog(store)
    event = log.append("something-new", model_id="m")
    assert event.operation == "something-new"
    assert log.entries()[0].operation == "something-new"


def test_an_empty_operation_becomes_a_failure(store):
    log = ModelLog(store)
    assert log.append("", model_id="m").operation == OP_FAILURE


def test_the_log_is_bounded_to_the_most_recent_events(store):
    log = ModelLog(store)
    for index in range(MAX_EVENTS + 25):
        log.append(OP_REGISTER, model_id=f"m{index}")
    events = log.entries()
    assert len(events) == MAX_EVENTS
    # The oldest entries are the ones dropped.
    assert events[0].model_id == "m25"
    assert events[-1].model_id == f"m{MAX_EVENTS + 24}"


def test_the_log_survives_a_new_instance(store):
    ModelLog(store).append(OP_REGISTER, model_id="m", message="persisted")
    assert ModelLog(store).entries()[0].message == "persisted"


def test_the_log_stores_under_its_own_namespace_and_key(store):
    ModelLog(store).append(OP_REGISTER, model_id="m")
    path = store / NAMESPACE / f"{KEY}.json"
    assert path.is_file()
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["version"] == 1
    assert len(document["events"]) == 1


# ------------------------------------------------------------------ corrupt


def test_a_corrupt_document_reads_as_empty(store):
    path = store / NAMESPACE
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{KEY}.json").write_text("{not json", encoding="utf-8")
    assert ModelLog(store).entries() == ()


def test_a_document_of_the_wrong_shape_reads_as_empty(store):
    path = store / NAMESPACE
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{KEY}.json").write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")
    assert ModelLog(store).entries() == ()


def test_a_corrupt_row_is_skipped_and_counted(store):
    path = store / NAMESPACE
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{KEY}.json").write_text(
        json.dumps({"events": [
            {"operation": OP_LOAD, "model_id": "good"},
            {"no_operation": True},
            "not a mapping",
        ]}),
        encoding="utf-8",
    )
    log = ModelLog(store)
    assert len(log.entries()) == 1
    assert log.corrupt_rows() == 2


def test_corrupt_rows_is_zero_for_a_healthy_log(store):
    log = ModelLog(store)
    log.append(OP_LOAD, model_id="m")
    assert log.corrupt_rows() == 0


# ------------------------------------------------------------------- clear


def test_clearing_requires_confirmation(store):
    from core.model_registry import ConfirmationRequired

    log = ModelLog(store)
    log.append(OP_LOAD, model_id="m")
    with pytest.raises(ConfirmationRequired):
        log.clear()
    assert len(log.entries()) == 1


def test_clearing_with_confirmation_empties_the_log(store):
    log = ModelLog(store)
    log.append(OP_LOAD, model_id="m")
    assert log.clear(confirm=True) == 1
    assert log.entries() == ()


# ------------------------------------------------------------------ describe


def test_describe_an_empty_log_says_so(store):
    assert "No model operations" in ModelLog(store).describe()


def test_describe_lists_the_tail(store):
    log = ModelLog(store)
    for index in range(3):
        log.append(OP_REGISTER, model_id=f"m{index}", display_name=f"Model {index}")
    text = log.describe()
    assert "Model 0" in text
    assert "Model 2" in text


def test_the_helper_builds_a_log(store):
    model_log(store).append(OP_LOAD, model_id="m")
    assert len(model_log(store).entries()) == 1


def test_known_operations_is_a_closed_set():
    assert OP_LOAD in KNOWN_OPERATIONS
    assert OP_REGISTER in KNOWN_OPERATIONS
    assert OP_FAILURE in KNOWN_OPERATIONS
