"""The notification feed and its dashboard section (Phase 5).

The feed is Tk-free, so it is exercised with no display; the dashboard section it
drives is checked here too, so the two can never disagree about what arrived.
"""

from __future__ import annotations

import pytest
from gui.dashboard import DEFAULT_RECENT_NOTIFICATIONS, build_dashboard
from gui.notifications import (
    DEFAULT_LIMIT,
    LEVELS,
    Notification,
    NotificationFeed,
)

# --------------------------------------------------------------- the feed


def test_a_fresh_feed_is_empty() -> None:
    feed = NotificationFeed()
    assert feed.items == ()
    assert len(feed) == 0
    assert feed.unread == 0


def test_posting_records_newest_first() -> None:
    feed = NotificationFeed()
    feed.post("first")
    feed.post("second")
    assert [item.title for item in feed.items] == ["second", "first"]


def test_post_result_records_the_outcome_as_the_level() -> None:
    feed = NotificationFeed()
    ok = feed.post_result("Training finished", ok=True)
    bad = feed.post_result("Download failed", ok=False, detail="404")
    assert ok.level == "success"
    assert bad.level == "error"
    assert bad.detail == "404"


def test_an_unknown_level_is_refused() -> None:
    feed = NotificationFeed()
    with pytest.raises(ValueError):
        feed.post("x", level="catastrophe")


def test_the_feed_is_bounded() -> None:
    feed = NotificationFeed(limit=3)
    for index in range(10):
        feed.post(f"n{index}")
    assert len(feed) == 3
    assert [item.title for item in feed.items] == ["n9", "n8", "n7"]


def test_unread_counts_arrivals_until_marked_seen() -> None:
    feed = NotificationFeed()
    feed.post("a")
    feed.post("b")
    assert feed.unread == 2
    feed.mark_seen()
    assert feed.unread == 0
    feed.post("c")
    assert feed.unread == 1


def test_recent_returns_the_newest_bounded_slice() -> None:
    feed = NotificationFeed()
    for index in range(5):
        feed.post(f"n{index}")
    assert [item.title for item in feed.recent(2)] == ["n4", "n3"]
    assert feed.recent(0) == ()


def test_clear_empties_the_feed_and_the_badge() -> None:
    feed = NotificationFeed()
    feed.post("a")
    feed.clear()
    assert feed.items == ()
    assert feed.unread == 0


def test_describe_reads_an_empty_feed_honestly() -> None:
    assert NotificationFeed().describe() == "No notifications yet."


def test_describe_lists_each_notification() -> None:
    feed = NotificationFeed()
    feed.post("Training finished", detail="12 steps", level="success")
    text = feed.describe()
    assert "Training finished" in text
    assert "12 steps" in text
    assert "success" in text


def test_notification_describe_is_plain_text() -> None:
    item = Notification(title="Done", detail="ok", level="info", at=0.0)
    assert item.describe() == "[info] Done\n  ok"


def test_levels_are_the_closed_vocabulary() -> None:
    assert LEVELS == ("info", "success", "warning", "error")


def test_the_default_limit_is_positive() -> None:
    assert DEFAULT_LIMIT >= 1


# ------------------------------------------------------- dashboard section


def test_dashboard_draws_the_notifications_section() -> None:
    feed = NotificationFeed()
    feed.post("Training finished", detail="done", level="success")
    dashboard = build_dashboard(notifications=feed)
    keys = [section.key for section in dashboard.sections]
    assert "home:notifications" in keys
    section = next(s for s in dashboard.sections if s.key == "home:notifications")
    assert section.rows[0].label == "[success] Training finished"


def test_dashboard_drops_the_section_when_the_feed_is_empty() -> None:
    dashboard = build_dashboard(notifications=NotificationFeed())
    assert "home:notifications" not in {s.key for s in dashboard.sections}


def test_dashboard_accepts_no_feed() -> None:
    dashboard = build_dashboard(notifications=None)
    assert "home:notifications" not in {s.key for s in dashboard.sections}


def test_dashboard_reads_only_the_newest_notifications() -> None:
    feed = NotificationFeed()
    for index in range(DEFAULT_RECENT_NOTIFICATIONS + 3):
        feed.post(f"n{index}")
    dashboard = build_dashboard(notifications=feed)
    section = next(s for s in dashboard.sections if s.key == "home:notifications")
    assert len(section.rows) == DEFAULT_RECENT_NOTIFICATIONS


def test_dashboard_tolerates_a_broken_feed() -> None:
    class Broken:
        def recent(self, count: int = 5):
            raise RuntimeError("boom")

    dashboard = build_dashboard(notifications=Broken())
    assert "home:notifications" not in {s.key for s in dashboard.sections}
