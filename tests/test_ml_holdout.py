"""Граница обучения и текущие данные в хронологическом тесте."""

import numpy as np
import pandas as pd
import pytest

from usd_rub.ml_dataset import clean_dataset, feature_hit_rates, feature_panels
from usd_rub.ml_experiment import baseline_predictions, eligible_rows
from usd_rub.ml_features import rvi_features
from usd_rub.models import AutoRegression
from usd_rub.storage import Payload, observations


def test_training_labels_mature_before_the_split_for_each_horizon():
    for horizon in range(1, 8):
        rows = eligible_rows(100, horizon, window=30)
        assert rows[0] == 70
        assert rows[-1] + horizon == 99
        assert len(rows) == 30 - horizon


def test_dataset_filter_uses_train_dates_and_keeps_labels():
    data = pd.DataFrame(
        {
            "split": ["train", "train", "test"],
            "usd_rub": [90.0, 91.0, 92.0],
            "target_return_1d": [0.01, np.nan, np.nan],
            "own__at_threshold": [1.0, np.nan, np.nan],
            "extended__news__test_only": [np.nan, np.nan, 1.0],
        }
    )
    clean = clean_dataset(data, 0.5)
    assert feature_hit_rates(clean).to_dict() == {"own__at_threshold": 0.5}
    pd.testing.assert_frame_equal(clean, data.drop(columns="extended__news__test_only"))


def test_extended_column_replaces_strict_column_including_missing_dates():
    data = pd.DataFrame(
        {
            "fx__rate": [80.0, 81.0],
            "extended__fx__rate": [np.nan, 82.0],
            "extended__macro__rate": [1.0, 2.0],
        }
    )
    panels = feature_panels(data)
    assert panels["strict"].columns.tolist() == ["fx__rate"]
    assert panels["extended"].columns.tolist() == ["fx__rate", "macro__rate"]
    assert pd.isna(panels["extended"]["fx__rate"].iloc[0])


def test_rvi_closing_price_lags_one_day_and_expires_after_a_week():
    payload = Payload(b"", "https://example.com/", "hash", "2025-02-01T00:00:00Z")
    close = observations(
        ["2014-04-15", "2014-04-16", "2014-04-22"],
        [20, 30, 45],
        "rvi",
        "index points",
        payload,
        "conservative_daily_close",
        lag_days=1,
    )
    features = rvi_features(close, pd.date_range("2014-04-15", "2014-04-30"))
    assert pd.isna(features.loc["2014-04-16", "markets__rvi_level"])
    assert features.loc["2014-04-22", "markets__rvi_level"] == 30
    assert features.loc["2014-04-23", "markets__rvi_change_5"] == pytest.approx(np.log(1.5))
    assert features.loc["2014-04-29", "markets__rvi_age"] == 7
    assert features.loc["2014-04-30"].isna().all()


def test_baseline_updates_observed_level_while_training_parameters_stay_frozen():
    history = 70 + np.sin(np.arange(32) * 0.6)
    values = np.r_[history, 80.0, 85.0, 90.0]
    target = pd.Series(values, index=pd.date_range("2024-12-01", periods=len(values)))
    split = len(history)
    result = baseline_predictions(target, split, np.arange(split, len(values)), np.arange(1, 3), 30)
    beta = AutoRegression().fit(history[-30:])
    for i, origin in enumerate(range(split, len(values))):
        last = values[origin]
        assert result["rw"][i].mean[0] == pytest.approx(last)
        assert result["historical_mean"][i].mean[0] == pytest.approx(history[-30:].mean())
        assert result["ar1"][i].mean[0] == pytest.approx(beta.intercept + beta.phi * last)
