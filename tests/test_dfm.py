"""Состояния Калмана, доходности и доступность данных DFM."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag, solve_discrete_lyapunov
from scipy.stats import norm

from usd_rub.dfm import TwoStepFactorModel, kalman_smoother
from usd_rub.factor_backtest import factor_walk_forward
from usd_rub.factor_data import VintagePanel


def batch_states(data, loading, transition, process, noise):
    count, factors = len(data), transition.shape[0]
    stationary = solve_discrete_lyapunov(transition, process)
    prior = np.zeros((count * factors, count * factors))
    for i in range(count):
        for j in range(i + 1):
            covariance = np.linalg.matrix_power(transition, i - j) @ stationary
            prior[i * factors : (i + 1) * factors, j * factors : (j + 1) * factors] = covariance
            prior[j * factors : (j + 1) * factors, i * factors : (i + 1) * factors] = covariance.T
    design = block_diag(*([loading] * count))
    observed = np.isfinite(data.ravel())
    design = design[observed]
    values = data.ravel()[observed]
    variance = np.tile(noise, count)[observed]
    covariance = np.linalg.inv(np.linalg.inv(prior) + (design.T / variance) @ design)
    mean = covariance @ ((design.T / variance) @ values)
    return mean.reshape(count, factors), covariance


def test_kalman_matches_joint_gaussian_conditioning_with_missing_observations():
    loading = np.array([[1.0, 0.3], [0.4, 1.0], [0.6, -0.2]])
    transition = np.array([[0.7, 0.1], [-0.1, 0.5]])
    process, noise = np.diag([0.2, 0.3]), np.array([0.1, 0.2, 0.3])
    data = np.array([[1, np.nan, 0.5], [0.7, -0.4, np.nan], [np.nan] * 3, [0.1, 0.4, np.nan]])
    states = kalman_smoother(data, loading, transition, process, noise)
    means, covariance = batch_states(data, loading, transition, process, noise)
    np.testing.assert_allclose(states.smoothed_mean, means, atol=1e-12)
    for t in range(len(data)):
        np.testing.assert_allclose(
            states.smoothed_covariance[t],
            covariance[2 * t : 2 * t + 2, 2 * t : 2 * t + 2],
            atol=1e-12,
        )
        prefix, prefix_covariance = batch_states(data[: t + 1], loading, transition, process, noise)
        np.testing.assert_allclose(states.filtered_mean[t], prefix[-1], atol=1e-12)
        np.testing.assert_allclose(
            states.filtered_covariance[t], prefix_covariance[-2:, -2:], atol=1e-12
        )


def test_missing_tail_predicts_states_and_new_measurement_reduces_uncertainty():
    transition, process = np.array([[0.8]]), np.array([[0.2]])
    missing = kalman_smoother(
        np.array([[1.0], [np.nan]]), np.ones((1, 1)), transition, process, np.array([0.1])
    )
    observed = kalman_smoother(
        np.array([[1.0], [0.5]]), np.ones((1, 1)), transition, process, np.array([0.1])
    )
    assert missing.filtered_mean[-1, 0] == pytest.approx(0.8 * missing.filtered_mean[0, 0])
    assert observed.filtered_covariance[-1, 0, 0] < missing.filtered_covariance[-1, 0, 0]


def test_return_forecast_variance_includes_common_shock_cross_covariances():
    model = TwoStepFactorModel(factors=1, shocks=1, min_balanced=10)
    model.transition_ = np.array([[0.8]])
    model.process_covariance_ = np.array([[0.1]])
    model.bridge_, model.intercept_, model.bridge_variance_, model.last_ = (
        np.array([0.02]),
        0.001,
        0.0004,
        90.0,
    )
    model.states_ = SimpleNamespace(
        filtered_mean=np.array([[0.5]]), filtered_covariance=np.array([[[0.3]]])
    )
    forecast = model.predict(7)
    for h in range(1, 8):
        weights = 0.02 * np.cumsum(0.8 ** np.arange(h))
        variance = (
            (0.02 * sum(0.8**k for k in range(1, h + 1))) ** 2 * 0.3
            + sum(weights**2) * 0.1
            + h * 0.0004
        )
        mean = np.log(90) + h * 0.001 + 0.5 * 0.02 * sum(0.8**k for k in range(1, h + 1))
        assert forecast.mean[h - 1] == pytest.approx(np.exp(mean + variance / 2))
        assert forecast.lower[h - 1] == pytest.approx(
            np.exp(mean - norm.ppf(0.975) * np.sqrt(variance))
        )


def source(dates, values, available=None, basis="recorded_publication"):
    dates = pd.DatetimeIndex(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "value": values,
            "available_at": available
            if available is not None
            else dates.tz_localize("Europe/Moscow") + pd.Timedelta(days=1),
            "availability_basis": basis,
            "ingested_at": "2026-11-18T10:00:00Z",
        }
    )


def test_vintage_panel_applies_revision_only_after_release_and_keeps_ragged_tail():
    frame = source(
        ["2020-01-01", "2020-01-03", "2020-01-03"],
        [10, 12, 13],
        ["2020-01-02T00:00:00+03:00", "2020-01-04T00:00:00+03:00", "2020-01-06T00:00:00+03:00"],
    )
    panel = VintagePanel({"x": frame}, {"x": {"transform": "diff"}}, "2020-01-01", "2020-01-07")
    early = panel.snapshot("2020-01-04T00:00:00+03:00")
    assert early.loc["2020-01-02", "x"] == 0
    assert early.loc["2020-01-03", "x"] == 2
    assert pd.isna(early.loc["2020-01-04", "x"])
    late = panel.snapshot("2020-01-06T00:00:00+03:00")
    assert late.loc["2020-01-03", "x"] == 3
    with pytest.raises(ValueError, match="возрастанию"):
        panel.snapshot("2020-01-04T00:00:00+03:00")


def test_unknown_release_is_not_retroactive_in_strict_panel():
    frame = source(["2020-01-01", "2020-01-02"], [10, 12], ["2026-11-18T10:00:00Z"] * 2, "unknown")
    specs = {"x": {"transform": "diff", "assumed_lag_days": 2}}
    strict = VintagePanel({"x": frame}, specs, "2020-01-01", "2020-01-07")
    assumed = VintagePanel({"x": frame}, specs, "2020-01-01", "2020-01-07", mode="assumed")
    cutoff = "2020-01-04T00:00:00+03:00"
    assert strict.snapshot(cutoff).isna().all().all()
    assert assumed.snapshot(cutoff).loc["2020-01-02", "x"] == 2


def synthetic_data(length=75):
    rng = np.random.default_rng(91)
    factors = np.zeros((length, 2))
    for t in range(1, length):
        factors[t] = factors[t - 1] * [0.7, 0.4] + rng.normal(size=2)
    data = factors @ np.array([[1.0, 0.2, 0.8, -0.3], [0.1, 0.8, -0.2, 1.0]]) + rng.normal(
        scale=0.3, size=(length, 4)
    )
    target = pd.Series(
        90 * np.exp(np.cumsum(factors @ [0.001, 0.0005] + rng.normal(scale=0.001, size=length))),
        index=pd.date_range("2020-01-01", periods=length),
    )
    return data, target


def test_two_step_model_uses_balanced_part_and_accepts_partially_missing_edge():
    data, y = synthetic_data()
    data[-3:, 0] = np.nan
    data[-1:, 1:] = np.nan
    model = TwoStepFactorModel(min_balanced=20).fit(data, y)
    assert model.balanced_size_ == len(data) - 3
    assert max(abs(np.linalg.eigvals(model.transition_))) < 1
    prediction = model.predict(7)
    assert np.isfinite(prediction.mean).all()
    assert (prediction.lower <= prediction.upper).all()


def test_internal_market_closure_remains_missing_in_kalman_step():
    data, y = synthetic_data()
    data[30:40, 0] = np.nan
    model = TwoStepFactorModel(min_balanced=20).fit(data, y)
    assert model.balanced_first_ == 40
    assert model.balanced_size_ == 35
    assert len(model.states_.filtered_mean) == len(data)
    assert np.isfinite(model.predict().mean).all()


def test_ragged_edge_updates_state_without_changing_balanced_pca_parameters():
    data, y = synthetic_data()
    data[-5:, 0] = np.nan
    original = TwoStepFactorModel(min_balanced=20).fit(data, y)
    changed = data.copy()
    changed[-5:, 1:] += 10
    updated = TwoStepFactorModel(min_balanced=20).fit(changed, y)
    for name in [
        "mean_",
        "scale_",
        "loadings_",
        "transition_",
        "process_covariance_",
        "observation_variance_",
    ]:
        np.testing.assert_allclose(getattr(original, name), getattr(updated, name))
    assert not np.allclose(original.states_.filtered_mean[-1], updated.states_.filtered_mean[-1])


def test_rank_deficient_common_shocks_can_be_smoothed():
    transition = np.eye(2) * 0.5
    process = np.diag([1.0, 0.0])
    states = kalman_smoother(
        np.array([[1.0], [2.0], [np.nan]]),
        np.array([[1.0, 0.0]]),
        transition,
        process,
        np.array([0.1]),
    )
    assert np.isfinite(states.smoothed_mean).all()
    np.testing.assert_allclose(states.smoothed_mean[:, 1], 0)


def test_future_target_and_releases_do_not_change_earlier_factor_forecasts():
    data, y = synthetic_data(55)
    frames = {f"x{i}": source(y.index, data[:, i]) for i in range(4)}
    specs = {name: {"transform": "diff"} for name in frames}
    settings = {"min_balanced": 15}

    def run(target, observations):
        panel = VintagePanel(observations, specs, y.index.min(), y.index.max())
        return factor_walk_forward(
            target, panel, settings, min_train=30, window=30, horizon=3, test_start="2020-01-30"
        )[0]

    original = run(y, frames)
    changed = y.copy()
    cutoff = y.index[43]
    changed.loc[cutoff:] *= 2
    alternatives = {name: frame.copy() for name, frame in frames.items()}
    for frame in alternatives.values():
        frame.loc[frame.date >= cutoff, "value"] += 100
    result = run(changed, alternatives)
    columns = ["model", "origin", "horizon", "prediction", "lower", "upper"]
    pd.testing.assert_frame_equal(
        original.loc[original.origin < cutoff, columns], result.loc[result.origin < cutoff, columns]
    )
