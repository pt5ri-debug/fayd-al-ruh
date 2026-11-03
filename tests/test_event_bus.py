"""The event bus is a notification channel: ordered, isolated, and bounded."""

from __future__ import annotations

import pytest
from core.best_effort import clear_swallowed, swallowed_errors
from core.event_bus import (
    EVENT_CYCLE_FINISHED,
    EVENT_CYCLE_STARTED,
    EVENT_NAMES,
    Event,
    EventBus,
    Subscription,
)


@pytest.fixture(autouse=True)
def _clean():
    clear_swallowed()
    yield
    clear_swallowed()


# -- construction ---------------------------------------------------------


def test_a_negative_history_size_is_refused():
    with pytest.raises(ValueError):
        EventBus(history=-1)


def test_a_non_positive_queue_limit_is_refused():
    with pytest.raises(ValueError):
        EventBus(queue=True, queue_limit=0)


def test_an_event_needs_a_name():
    with pytest.raises(ValueError):
        Event(name="")


def test_a_subscriber_must_be_callable():
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.subscribe("not callable")  # type: ignore[arg-type]


# -- delivery -------------------------------------------------------------


def test_a_subscriber_receives_a_published_event_in_order():
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda event: seen.append(event.name))
    bus.publish(EVENT_CYCLE_STARTED)
    bus.publish(EVENT_CYCLE_FINISHED)
    assert seen == [EVENT_CYCLE_STARTED, EVENT_CYCLE_FINISHED]


def test_a_prefix_subscribes_to_a_family():
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda event: seen.append(event.name), prefix="cycle")
    bus.publish(EVENT_CYCLE_STARTED)
    bus.publish("belief.revised")
    bus.publish("cycle")  # the exact prefix also matches
    assert seen == [EVENT_CYCLE_STARTED, "cycle"]


def test_a_prefix_does_not_match_a_longer_word():
    """``cycle`` must not match ``cyclone`` -- a boundary check, not ``startswith``."""
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda event: seen.append(event.name), prefix="cycle")
    bus.publish("cyclone.started")
    assert seen == []


def test_the_empty_prefix_subscribes_to_everything():
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda event: seen.append(event.name), prefix="")
    bus.publish("anything")
    bus.publish("else.at.all")
    assert seen == ["anything", "else.at.all"]


def test_a_raising_subscriber_is_isolated_and_recorded():
    bus = EventBus()
    seen: list[str] = []

    def boom(_event: Event) -> None:
        raise RuntimeError("observer failed")

    bus.subscribe(boom)
    bus.subscribe(lambda event: seen.append(event.name))
    event = bus.publish("thing.happened")
    # The publisher is unaffected and the other subscriber still ran.
    assert event.name == "thing.happened"
    assert seen == ["thing.happened"]
    assert len(swallowed_errors()) == 1


def test_unsubscribing_stops_delivery():
    bus = EventBus()
    seen: list[str] = []
    subscription = bus.subscribe(lambda event: seen.append(event.name))
    bus.publish("one")
    assert bus.unsubscribe(subscription) is True
    bus.publish("two")
    assert seen == ["one"]
    assert bus.unsubscribe(subscription) is False


def test_subscriber_count_and_history():
    bus = EventBus()
    first = bus.subscribe(lambda event: None)
    bus.subscribe(lambda event: None)
    assert bus.subscriber_count == 2
    bus.publish("a")
    bus.publish("b")
    assert bus.count() == 2
    assert bus.count("a") == 1
    bus.unsubscribe(first)
    assert bus.subscriber_count == 1


# -- queue mode -----------------------------------------------------------


def test_queue_mode_defers_delivery_until_drain():
    bus = EventBus(queue=True)
    seen: list[str] = []
    bus.subscribe(lambda event: seen.append(event.name))
    bus.publish("one")
    bus.publish("two")
    assert seen == []
    assert bus.pending == 2
    delivered = bus.drain()
    assert delivered == 2
    assert seen == ["one", "two"]
    assert bus.pending == 0


def test_a_bounded_queue_drops_oldest_and_counts_it():
    bus = EventBus(queue=True, queue_limit=2)
    seen: list[str] = []
    bus.subscribe(lambda event: seen.append(event.name))
    for index in range(5):
        bus.publish(f"e{index}")
    assert bus.pending == 2
    assert bus.dropped == 3
    bus.drain()
    assert seen == ["e3", "e4"]


def test_drain_with_a_limit_spreads_the_work():
    bus = EventBus(queue=True)
    seen: list[str] = []
    bus.subscribe(lambda event: seen.append(event.name))
    for index in range(4):
        bus.publish(f"e{index}")
    assert bus.drain(limit=2) == 2
    assert seen == ["e0", "e1"]
    assert bus.drain(limit=2) == 2
    assert seen == ["e0", "e1", "e2", "e3"]


def test_drain_on_a_synchronous_bus_does_nothing():
    bus = EventBus()
    assert bus.drain() == 0


# -- reading --------------------------------------------------------------


def test_recent_returns_newest_first_and_honours_a_prefix():
    bus = EventBus()
    bus.publish("cycle.started")
    bus.publish("belief.revised")
    bus.publish("cycle.finished")
    assert [e.name for e in bus.recent(limit=2)] == ["cycle.finished", "belief.revised"]
    assert [e.name for e in bus.recent(limit=5, prefix="cycle")] == [
        "cycle.finished",
        "cycle.started",
    ]
    assert bus.recent(limit=0) == ()


def test_history_is_bounded():
    bus = EventBus(history=2)
    for index in range(5):
        bus.publish(f"e{index}")
    assert [e.name for e in bus.history] == ["e3", "e4"]


def test_clear_forgets_everything():
    bus = EventBus(queue=True)
    bus.publish("one")
    bus.clear()
    assert bus.count() == 0
    assert bus.pending == 0
    assert bus.dropped == 0


def test_iteration_and_length_read_the_history():
    bus = EventBus()
    bus.publish("one")
    bus.publish("two")
    assert len(bus) == 2
    assert [event.name for event in bus] == ["one", "two"]


def test_describe_names_the_mode():
    assert "synchronous" in EventBus().describe()
    assert "queued" in EventBus(queue=True).describe()


def test_to_dict_round_trips_the_event_fields():
    event = Event(name="a.b", data={"x": 1}, source="test")
    payload = event.to_dict()
    assert payload["name"] == "a.b"
    assert payload["data"] == {"x": 1}
    assert payload["source"] == "test"
    assert "at" in payload


def test_the_declared_event_names_are_dotted_and_unique():
    assert len(set(EVENT_NAMES)) == len(EVENT_NAMES)
    assert all("." in name for name in EVENT_NAMES)


def test_subscription_matches_is_a_boundary_check():
    subscription = Subscription(prefix="cycle", callback=lambda event: None, id=0)
    assert subscription.matches("cycle.started") is True
    assert subscription.matches("cycle") is True
    assert subscription.matches("cyclone") is False
