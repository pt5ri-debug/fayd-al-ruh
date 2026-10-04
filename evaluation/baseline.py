from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.benchmark import compare_metric_dicts, report_to_dict
from evaluation.runner import EvaluationReport, run_suite

BASELINE_PATH = Path(__file__).with_name("baseline.json")


def default_report() -> EvaluationReport:
    return run_suite()


def save_baseline(report: EvaluationReport, path: Path = BASELINE_PATH) -> Path:
    path.write_text(
        json.dumps(report_to_dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No baseline file at {path}. Run with --update-baseline first.")
    return json.loads(path.read_text(encoding="utf-8"))


def check_against_baseline(
    report: EvaluationReport,
    path: Path = BASELINE_PATH,
    tolerance: float = 0.0,
):
    """Compare a fresh report against the stored baseline metrics.

    ``tolerance`` widens every metric's allowed slack, which is useful when a
    metric is measured through a non-deterministic component.
    """
    baseline = load_baseline(path)
    specs = _specs_with_tolerance(tolerance)
    return compare_metric_dicts(baseline["metrics"], report.metrics.__dict__, specs)


def _specs_with_tolerance(tolerance: float):
    from evaluation.benchmark import METRIC_SPECS, MetricSpec

    if tolerance <= 0.0:
        return METRIC_SPECS
    return tuple(
        MetricSpec(spec.name, spec.higher_is_better, spec.tolerance + tolerance, spec.label)
        for spec in METRIC_SPECS
    )
