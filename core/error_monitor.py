from __future__ import annotations

from dataclasses import dataclass

from core.prediction_error import PredictionError


@dataclass(frozen=True)
class ErrorSummary:
    count: int
    average_error: float
    highest_error_prediction_id: str | None


class ErrorMonitor:
    def inspect(self, errors: list[PredictionError]) -> ErrorSummary:
        if not errors:
            return ErrorSummary(0, 0.0, None)
        highest = max(errors, key=lambda error: error.error)
        return ErrorSummary(len(errors), round(sum(error.error for error in errors) / len(errors), 6), highest.prediction_id)
