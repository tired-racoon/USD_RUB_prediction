"""Проверка формул и границ эталонных прогнозов."""

import numpy as np
import pytest
from scipy.stats import norm

from usd_rub.models import (
    MODEL_FACTORIES,
    AutoRegression,
    RandomWalk,
    SeasonalNaive,
)


def test_random_walk_and_drift():
    y = np.array([70.0, 72.0, 71.0, 75.0])
    result = RandomWalk().fit(y).predict(3)
    np.testing.assert_equal(result.mean, [75, 75, 75])
    variance = np.mean(np.diff(y) ** 2)
    np.testing.assert_allclose(
        result.upper - result.mean, norm.ppf(0.975) * np.sqrt(variance * np.arange(1, 4))
    )
    drift = RandomWalk(drift=True).fit(y).predict(3)
    np.testing.assert_allclose(drift.mean, 75 + np.arange(1, 4) * (75 - 70) / 3)


def test_ar1_recovers_known_equation():
    y = [80.0]
    for _ in range(25):
        y.append(10 + 0.8 * y[-1])
    model = AutoRegression().fit(y)
    assert model.phi == pytest.approx(0.8)
    assert model.intercept == pytest.approx(10)
    forecast = model.predict(3)
    expected = [y[-1]]
    for _ in range(3):
        expected.append(10 + 0.8 * expected[-1])
    np.testing.assert_allclose(forecast.mean, expected[1:])


def test_return_ar_variance_accumulates_covariance():
    rng = np.random.default_rng(42)
    returns = [0.001]
    for shock in rng.normal(0, 0.005, 100):
        returns.append(0.0002 + 0.6 * returns[-1] + shock)
    y = 80 * np.exp(np.cumsum(returns))
    model = AutoRegression(returns=True).fit(y)
    result = model.predict(3)
    log_var = ((np.log(result.upper) - np.log(result.lower)) / (2 * norm.ppf(0.975))) ** 2
    expected = model.sigma2 * np.array(
        [
            1,
            1 + (1 + model.phi) ** 2,
            1 + (1 + model.phi) ** 2 + (1 + model.phi + model.phi**2) ** 2,
        ]
    )
    np.testing.assert_allclose(log_var, expected)


@pytest.mark.parametrize("name", MODEL_FACTORIES)
def test_constant_series_has_exact_forecast(name):
    result = MODEL_FACTORIES[name]().fit(np.full(100, 90.0)).predict(7)
    np.testing.assert_allclose(result.mean, 90)
    np.testing.assert_allclose(result.lower, 90)
    np.testing.assert_allclose(result.upper, 90)


def test_weekly_naive_repeats_last_week():
    model = SeasonalNaive().fit(np.arange(70.0, 91.0))
    np.testing.assert_equal(model.predict(9).mean, [84, 85, 86, 87, 88, 89, 90, 84, 85])


@pytest.mark.parametrize("values", [[1, np.nan, 2], [1, 0, 2], [1, -2, 3], [1, np.inf, 2]])
def test_invalid_history_rejected(values):
    with pytest.raises(ValueError):
        RandomWalk().fit(values)


def test_prediction_before_fit_and_invalid_horizon():
    with pytest.raises(ValueError):
        RandomWalk().predict()
    with pytest.raises(ValueError):
        RandomWalk().fit([70, 71, 72]).predict(0)
