from __future__ import annotations

from enum import Enum


class TemporalRelation(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    CONTAINS = "contains"
    OVERLAPS = "overlaps"
    SIMULTANEOUS = "simultaneous"
