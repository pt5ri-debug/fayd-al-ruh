import pytest

from evaluation.metrics import (
    brier_summary,
    calibration_bins,
    expected_calibration_error,
)


def test_brier_summary_matches_murphy_decomposition():
    forecasts = [(0.9, 1.0), (0.8, 1.0), (0.2, 0.0), (0.1, 0.0), (0.5, 1.0)]
    summary = brier_summary(forecasts)

    assert summary.count == 5
    # brier = reliability - resolution + uncertainty
    assert summary.brier == pytest.approx(
        summary.reliability - summary.resolution + summary.uncertainty, abs=1e-6
    )


def test_brier_summary_empty_is_zero():
    summary = brier_summary([])
    assert summary.count == 0
    assert summary.brier == 0.0


def test_brier_score_rewards_accurate_confident_forecasts():
    good = brier_summary([(0.9, 1.0), (0.1, 0.0)])
    bad = brier_summary([(0.1, 1.0), (0.9, 0.0)])
    assert good.brier < bad.brier


def test_calibration_bins_group_by_confidence():
    bins = calibration_bins([(0.15, 0.0), (0.25, 0.0), (0.85, 1.0)], bins=10)
    assert [bucket.count for bucket in bins] == [1, 1, 1]
    assert bins[0].mean_confidence == pytest.approx(0.15)
    assert bins[-1].empirical_accuracy == 1.0


def test_expected_calibration_error_is_zero_for_perfect_calibration():
    forecasts = [(0.0, 0.0), (1.0, 1.0)]
    assert expected_calibration_error(forecasts) == 0.0


def test_expected_calibration_error_is_positive_when_overconfident():
    forecasts = [(0.9, 0.0), (0.9, 0.0)]
    assert expected_calibration_error(forecasts) == pytest.approx(0.9)


def test_calibration_bins_rejects_invalid_bin_count():
    with pytest.raises(ValueError):
        calibration_bins([(0.5, 1.0)], bins=0)
