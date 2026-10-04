from datetime import datetime, timedelta, timezone

import pytest

from core.memory import CognitiveMemory
from core.retention_policy import RetentionPolicy, prune


def _aged_item(memory: CognitiveMemory, item_id: str, importance: float, age_days: float, access_count: int = 0):
    item = memory.store(item_id, "observation", f"content-{item_id}", importance)
    item.created_at = datetime.now(timezone.utc) - timedelta(days=age_days)
    item.access_count = access_count
    return item


def test_policy_validates_bounds():
    with pytest.raises(ValueError):
        RetentionPolicy(max_age_days=-1)
    with pytest.raises(ValueError):
        RetentionPolicy(minimum_importance=1.5)
    with pytest.raises(ValueError):
        RetentionPolicy(decay_rate=-0.1)
    with pytest.raises(ValueError):
        RetentionPolicy(max_items=0)


def test_expired_requires_both_age_and_low_importance():
    # decay_rate=0 keeps retained importance equal to stored importance, so the
    # age gate and the strength gate can be tested independently.
    policy = RetentionPolicy(max_age_days=10, minimum_importance=0.1, decay_rate=0.0)
    memory = CognitiveMemory()
    weak_old = _aged_item(memory, "OLD-WEAK", 0.05, 30)
    strong_old = _aged_item(memory, "OLD-STRONG", 0.9, 30)
    weak_new = _aged_item(memory, "NEW-WEAK", 0.05, 1)

    assert policy.expired(weak_old) is True
    assert policy.expired(strong_old) is False
    assert policy.expired(weak_new) is False


def test_frequently_recalled_items_are_protected():
    policy = RetentionPolicy(max_age_days=10, minimum_importance=0.1, decay_rate=0.001)
    memory = CognitiveMemory()
    forgotten = _aged_item(memory, "FORGOTTEN", 0.2, 30)
    recalled = _aged_item(memory, "RECALLED", 0.2, 30, access_count=50)

    assert policy.retained_importance(recalled) > policy.retained_importance(forgotten)
    assert policy.expired(forgotten) is True
    assert policy.expired(recalled) is False


def test_protected_item_types_are_never_evicted():
    policy = RetentionPolicy(max_age_days=10, minimum_importance=0.1, protect_item_types=frozenset({"belief_state"}))
    memory = CognitiveMemory()
    item = memory.store("BELIEF-1", "belief_state", {"belief_id": "BL-1"}, 0.01)
    item.created_at = datetime.now(timezone.utc) - timedelta(days=365)

    assert policy.expired(item) is False
    assert prune(memory, policy).removed == []


def test_prune_removes_only_expired_items():
    policy = RetentionPolicy(max_age_days=10, minimum_importance=0.1, decay_rate=0.0)
    memory = CognitiveMemory()
    weak_old = _aged_item(memory, "OLD-WEAK", 0.05, 30)
    _aged_item(memory, "OLD-STRONG", 0.9, 30)

    report = prune(memory, policy)

    assert report.removed == [weak_old.item_id]
    assert report.removed_count == 1
    assert memory.get("OLD-WEAK") is None
    assert memory.get("OLD-STRONG") is not None


def test_max_items_evicts_weakest_survivors():
    policy = RetentionPolicy(max_age_days=365, minimum_importance=0.0, max_items=2)
    memory = CognitiveMemory()
    for index, importance in enumerate([0.2, 0.4, 0.6, 0.8]):
        memory.store(f"ITEM-{index}", "observation", f"content-{index}", importance)

    report = prune(memory, policy)

    assert memory.size == 2
    assert set(report.removed) == {"ITEM-0", "ITEM-1"}


def test_prune_eviction_order_is_deterministic():
    policy = RetentionPolicy(max_age_days=1, minimum_importance=0.5)
    memory = CognitiveMemory()
    for item_id in ("A", "B", "C"):
        item = memory.store(item_id, "observation", item_id, 0.1)
        item.created_at = datetime.now(timezone.utc) - timedelta(days=5)

    first = prune(memory, policy).removed
    assert first == ["A", "B", "C"]


def test_report_keeps_remaining_count():
    policy = RetentionPolicy(max_age_days=10, minimum_importance=0.1, decay_rate=0.0)
    memory = CognitiveMemory()
    _aged_item(memory, "OLD-WEAK", 0.05, 30)
    memory.store("KEEP", "observation", "keep", 0.9)

    report = prune(memory, policy)

    assert report.kept == 1
    assert report.policy is policy
