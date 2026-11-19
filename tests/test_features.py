"""Публикации, ревизии и единицы расчётных признаков."""

import pandas as pd
import pytest

from usd_rub.features import asof_series, build_features


def frame(dates, values, publications):
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "value": values,
            "available_at": publications,
            "availability_basis": "recorded_publication",
            "ingested_at": "2026-11-17",
        }
    )


def test_delayed_release_and_revision_never_appear_early():
    data = frame(
        ["2020-01-01", "2020-01-01", "2020-02-01"],
        [10, 11, 12],
        ["2020-02-10T10:00:00Z", "2020-03-10T10:00:00Z", "2020-03-01T10:00:00Z"],
    )
    cutoffs = pd.to_datetime(
        ["2020-02-09T15:00:00Z", "2020-02-11T15:00:00Z", "2020-03-11T15:00:00Z"]
    )
    result = asof_series(data, cutoffs)
    assert pd.isna(result.value.iloc[0])
    assert result.value.iloc[1] == 10
    assert result.value.iloc[2] == 12


def test_no_backward_fill():
    data = frame(["2020-01-02"], [90], ["2020-01-02T00:00:00Z"])
    result = asof_series(data, pd.date_range("2020-01-01", periods=3, tz="UTC"))
    assert pd.isna(result.value.iloc[0])


def test_cross_direction_is_rub_per_cny_times_cny_per_usd():
    data = {
        name: frame(["2020-01-01"], [value], ["2020-01-01T00:00:00Z"])
        for name, value in [("cny_tom_close", 10), ("usd_cny", 7), ("usd_rub", 71)]
    }
    result = build_features(data, "2020-01-01", "2020-01-02")
    assert result.usd_rub_cross.iloc[0] == 70
    assert result.cross_spread.iloc[0] == 1
    assert result.cross_spread_pct.iloc[0] == pytest.approx(100 / 70)


def test_brent_lag_uses_old_cutoff_and_its_available_observation():
    data = frame(["2026-01-01", "2026-01-03"], [70, 80], ["2026-01-01T00:00Z", "2026-01-04T00:00Z"])
    result = build_features({"brent": data}, "2026-01-01", "2026-03-05")
    assert result.brent_lag_30d.loc["2026-02-02"].iloc[0] == 70
    assert result.brent_lag_30d.loc["2026-02-03"].iloc[0] == 80
    assert result.brent_lag_60d.loc["2026-03-02"].iloc[0] == 70


def test_naive_cutoff_rejected():
    with pytest.raises(ValueError):
        asof_series(
            frame(["2020-01-01"], [1], ["2020-01-02T00:00:00Z"]),
            pd.date_range("2020-01-01", periods=2),
        )


def test_zero_fx_quote_does_not_become_a_zero_cross():
    data = {
        name: frame(["2020-01-01"], [value], ["2020-01-01T00:00:00Z"])
        for name, value in [("cny_tom_close", 0), ("usd_cny", 7), ("usd_rub", 71)]
    }
    result = build_features(data, "2020-01-01", "2020-01-01")
    assert pd.isna(result.usd_rub_cross.iloc[0])
    assert pd.isna(result.cross_log_basis.iloc[0])
