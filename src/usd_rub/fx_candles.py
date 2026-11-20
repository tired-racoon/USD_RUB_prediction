"""Признаки свечей ISS на срезе 00:00 МСК."""

import numpy as np
import pandas as pd

PUBLICATION_LAG = pd.Timedelta(minutes=15)
MAX_AGE_DAYS = 4


def session_prices(candles, instrument, dates):
    """Собрать дневной профиль по свечам, доступным до полуночи."""
    bars = candles.loc[candles.instrument.eq(instrument)].sort_values("begin").copy()
    bars["day"] = bars.begin.dt.normalize()
    end = bars.day + pd.DateOffset(days=1)
    bars = bars.loc[(bars.end + PUBLICATION_LAG <= end) & bars.close.gt(0)]
    group = bars.groupby("day")
    daily = group.agg(close=("close", "last"), high=("high", "max"), low=("low", "min"))
    daily["morning"] = bars.loc[bars.end.dt.hour < 13].groupby("day").close.last()
    daily["afternoon"] = bars.loc[bars.end.dt.hour < 16].groupby("day").close.last()
    daily["early_evening"] = bars.loc[bars.end.dt.hour < 19].groupby("day").close.last()
    volume = group.volume.sum(min_count=1).where(lambda values: values > 0)
    signed_volume = bars.volume * np.sign(bars.close - bars.open)
    late = bars.end.dt.hour.ge(16)
    daily["signed_volume"] = signed_volume.groupby(bars.day).sum(min_count=1) / volume
    daily["late_volume"] = bars.loc[late].groupby("day").volume.sum(min_count=1) / volume
    previous_sessions = np.log(daily.close).diff()
    daily["session_return"] = previous_sessions
    daily["return_3_sessions"] = np.log(daily.close).diff(3)
    daily["vol_20_sessions"] = previous_sessions.rolling(20, min_periods=10).std()
    daily["obs_date"] = daily.index
    daily.index += pd.DateOffset(days=1)
    result = daily.reindex(dates, method="ffill", limit=MAX_AGE_DAYS - 1)
    result["age"] = ((dates - result.obs_date) / pd.Timedelta(days=1)).to_numpy()
    return result


def candle_features(candles, official, yuan):
    """Посчитать спреды к курсам ЦБ, диапазоны и движения внутри сессии."""
    bars = (
        pd.read_csv(candles, parse_dates=["begin", "end"])
        if not isinstance(candles, pd.DataFrame)
        else candles
    )
    instruments = {
        "cny_spot": "CNYRUB_TOM",
        "usd_spot": "USD000UTSTOM",
        "usd_perp": "USDRUBF",
        "cny_perp": "CNYRUBF",
    }
    sessions = {
        key: session_prices(bars, ticker, official.index) for key, ticker in instruments.items()
    }
    result = {}
    for key, session in sessions.items():
        bank = official if key.startswith("usd") else yuan
        log_close = np.log(session.close)
        basis = log_close - np.log(bank)
        result[f"{key}__basis"] = basis
        result[f"{key}__basis_centered"] = basis - basis.ewm(span=21, min_periods=7).mean()
        result[f"{key}__age"] = session.age
        result[f"{key}__return_1d"] = log_close.diff()
        for field in ("morning", "afternoon", "early_evening"):
            result[f"{key}__since_{field}"] = log_close - np.log(session[field])
        result[f"{key}__range"] = np.log(session.high / session.low)
        for field in ("session_return", "return_3_sessions", "vol_20_sessions"):
            result[f"{key}__{field}"] = session[field]
        if key.endswith("perp"):
            for field in ("signed_volume", "late_volume"):
                result[f"{key}__{field}"] = session[field]
    cny, usd = sessions["cny_perp"], sessions["usd_perp"]
    ratio = np.log((usd.close / cny.close).where(usd.obs_date.eq(cny.obs_date)))
    result["cross__perp_basis"] = ratio - np.log(official / yuan)
    result["cross__perp_change_1d"] = ratio.diff()
    return pd.DataFrame(result, index=official.index).replace([np.inf, -np.inf], np.nan)
