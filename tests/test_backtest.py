"""Проверка последовательной валидации и метрик."""

import numpy as np
import pandas as pd
import pytest

from usd_rub.backtest import daily_target, dm_test, score, walk_forward


def history():
    return pd.Series(
        80 + np.cumsum(np.random.default_rng(3).normal(0, 0.3, 80)),
        index=pd.date_range("2020-01-01", periods=80),
    )


def test_future_changes_do_not_affect_earlier_forecasts():
    y = history()
    original = walk_forward(y, min_train=20)
    changed = y.copy()
    changed.iloc[50:] += 3
    alternative = walk_forward(changed, min_train=20)
    cutoff = y.index[50]
    pd.testing.assert_frame_equal(
        original.loc[original.origin < cutoff, ["model", "origin", "horizon", "prediction"]],
        alternative.loc[alternative.origin < cutoff, ["model", "origin", "horizon", "prediction"]],
    )


def test_horizon_and_training_window():
    y = history()
    result = walk_forward(y, models=["rw"], min_train=20, window=25)
    assert set(result.horizon) == set(range(1, 8))
    assert (result.target_date - result.origin).dt.days.equals(result.horizon)
    assert result.train_size.max() == 25
    assert result.origin.max() == y.index[-8]


def test_rw_skill_is_zero_and_mase_has_train_only_scale():
    y = history()
    result = walk_forward(y, models=["rw", "ar1"], min_train=20)
    metrics = score(result)
    np.testing.assert_allclose(metrics.loc[metrics.model == "rw", "rmse_skill_vs_rw"], 0)
    first = result.iloc[0]
    assert first.mase_scale == pytest.approx(np.abs(np.diff(y.iloc[:20])).mean())


def test_weekends_carried_only_until_last_observation():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-09"]),
            "value": [90.0, 91.0],
            "ingested_at": ["2024-01-10"] * 2,
        }
    )
    result = daily_target(frame)
    assert result.loc["2024-01-07"] == 90
    with pytest.raises(ValueError):
        daily_target(frame, end="2024-01-10")


def test_irregular_history_rejected():
    with pytest.raises(ValueError):
        walk_forward(history().iloc[::2], min_train=10)


def test_constant_differential_has_no_spurious_dm_significance():
    stat, p = dm_test(np.ones(100), np.ones(100), 6)
    assert np.isnan(stat) and np.isnan(p)


def percentage_fixture():
    origins = pd.date_range("2026-01-01", periods=3)
    frame = pd.DataFrame(
        {
            "model": "ar1",
            "origin": origins,
            "horizon": 1,
            "actual": [100.0, 200.0, 200.0],
            "prediction": [110.0, 160.0, 200.0],
            "last": [90.0, 210.0, 200.0],
            "lower": [80.0, 150.0, 190.0],
            "upper": [120.0, 220.0, 210.0],
            "mase_scale": 5.0,
        }
    )
    return pd.concat([frame, frame.assign(model="rw", prediction=frame["last"])], ignore_index=True)


def test_percentage_metrics_use_actual_level_and_explicit_percent_units():
    metrics = score(percentage_fixture()).set_index("model").loc["ar1"]
    assert metrics.mape_pct == pytest.approx(10)
    assert metrics.rmspe_pct == pytest.approx(np.sqrt((10**2 + 20**2) / 3))
    assert metrics.smape_pct == pytest.approx(200 * (10 / 210 + 40 / 360) / 3)
    assert metrics.mape_changed_pct == pytest.approx(15)
    assert metrics.n_changed == 2


def test_percentage_metrics_are_invariant_to_currency_unit():
    frame = percentage_fixture()
    original = score(frame).set_index("model")
    scaled = frame.copy()
    columns = ["actual", "prediction", "last", "lower", "upper", "mase_scale"]
    scaled[columns] *= 1000
    alternative = score(scaled).set_index("model")
    percentage_columns = ["mape_pct", "smape_pct", "rmspe_pct", "mape_changed_pct"]
    np.testing.assert_allclose(original[percentage_columns], alternative[percentage_columns])
    np.testing.assert_allclose(original.rmse * 1000, alternative.rmse)


def test_percentage_metrics_handle_nonpositive_forecasts_without_dropping_errors():
    frame = percentage_fixture()
    frame.loc[frame.model == "ar1", "prediction"] = [0, -200, 200]
    metrics = score(frame).set_index("model").loc["ar1"]
    assert metrics.mape_pct == pytest.approx(100)
    assert metrics.smape_pct == pytest.approx(400 / 3)
    assert metrics.nonpositive_predictions == 2


@pytest.mark.parametrize("actual", [0.0, -1.0, np.nan, np.inf])
def test_percentage_metrics_reject_invalid_actuals(actual):
    frame = percentage_fixture()
    frame.loc[0, "actual"] = actual
    with pytest.raises(ValueError, match="положительных actual"):
        score(frame)


def test_changed_only_percentage_metric_is_missing_on_constant_target():
    frame = percentage_fixture()
    frame["last"] = frame["actual"]
    metrics = score(frame)
    assert metrics.mape_changed_pct.isna().all()
    assert (metrics.n_changed == 0).all()
