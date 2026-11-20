"""Фундаментальные и валютные признаки по мотивам Lin, Liu и Vincent."""

import numpy as np
import pandas as pd

from usd_rub.features import asof_series
from usd_rub.market_correction import MARKET_FEATURES
from usd_rub.sources.markets import fetch_fred
from usd_rub.storage import Client, read_series, save_series

FRED = {
    "us_m2": ("M2SL", "USD bn"),
    "us_industrial": ("INDPRO", "index 2017=100"),
    "us_exports": ("BOPGEXP", "USD mn"),
    "us_imports": ("BOPGIMP", "USD mn"),
    "us_epu": ("USEPUINDXD", "index"),
    "us_cp_3m": ("DCPF3M", "% p.a."),
    "us_bill_3m": ("DTB3", "% p.a."),
}
PEERS = ("usd_cny", "usd_brl", "usd_mxn", "usd_inr", "usd_zar")
LOCAL = (
    "cpi_ru_mom_index",
    "cpi_us",
    "m2",
    "industrial_production_mom_sa",
    "exports",
    "imports",
    "vix",
    *PEERS,
)
FUNDAMENTAL_SIGNS = {
    "macro__theory_inflation_gap_mom": 1,
    "macro__theory_inflation_gap_yoy": 1,
    "macro__theory_m2_growth_gap": 1,
    "macro__theory_trade_balance_gap": -1,
    "macro__theory_production_growth_gap": -1,
    "rates__theory_policy_gap": -1,
}
FUNDING = "global__theory_cp_bill_spread_3m"
GLOBAL_FX_FEATURES = (
    FUNDING,
    "global__theory_us_epu_7d",
    "global__theory_vix_level",
    "global__theory_em_usd_return_21d",
    "global__theory_em_fx_abs_return_21d",
    "fx__theory_rub_relative_momentum_21d",
    "fx__theory_rub_momentum_63d",
    "fx__theory_ppp_value_gap",
    "news__gpr_daily_shock_30d",
)
MACRO_CORRECTION_FEATURES = (*MARKET_FEATURES, *GLOBAL_FX_FEATURES, *FUNDAMENTAL_SIGNS)
LAGS = {"cpi": 55, "money": 65, "trade": 95, "production": 75, "epu": 7, "funding": 4}


def read_theory_sources(data_dir, fred_root, dates, *, frozen):
    """Взять локальные ряды и зафиксированный или текущий ответ FRED."""
    paths = [data_dir / "normalized" / f"{name}.csv" for name in LOCAL]
    frames = {name: read_series(data_dir, name) for name in LOCAL}
    start = (dates[0] - pd.Timedelta(days=730)).date()
    end = dates[-1].date()
    with Client(fred_root) as client:
        for name, (code, unit) in FRED.items():
            path = fred_root / "normalized" / f"{name}.csv"
            if frozen and path.is_file():
                frames[name] = read_series(fred_root, name)
            else:
                frames[name] = fetch_fred(client, code, name, unit, start, end)
                if frames[name].empty:
                    raise ValueError(f"FRED не вернул наблюдения {code}")
                if frozen:
                    save_series(frames[name], fred_root)
                else:
                    path = fred_root / "raw" / "objects" / frames[name].raw_sha256.iloc[0]
            paths.append(path)
    return frames, paths


def latest_values(frame):
    """Восстановить последнее сохранённое значение каждого периода."""
    return (
        frame.sort_values(["date", "ingested_at"])
        .drop_duplicates("date", keep="last")
        .set_index("date")
        .value.sort_index()
    )


def delayed(values, dates, lag, maximum_age):
    """Сдвинуть ряд на допущенный лаг выпуска и ограничить перенос."""
    values = values.dropna()
    if values.empty:
        return pd.Series(np.nan, index=dates)
    values = values.set_axis(values.index + pd.Timedelta(days=lag))
    calendar = pd.date_range(min(dates[0], values.index[0]), dates[-1])
    return values.reindex(calendar).ffill(limit=maximum_age).reindex(dates)


def fundamentals(panel, target, sources):
    """Сопоставить месяцы двух стран до расчёта разностей."""
    keys = (
        "cpi_ru_mom_index",
        "cpi_us",
        "m2",
        "us_m2",
        "exports",
        "imports",
        "us_exports",
        "us_imports",
        "industrial_production_mom_sa",
        "us_industrial",
    )
    levels = pd.DataFrame({key: latest_values(sources[key]) for key in keys}).asfreq("MS")
    ru_inflation = np.log(levels.cpi_ru_mom_index / 100)
    us_prices = np.log(levels.cpi_us)
    ru_prices = ru_inflation.cumsum()
    ru_trade = (levels.exports - levels.imports) / (levels.exports + levels.imports)
    us_trade = (levels.us_exports - levels.us_imports) / (levels.us_exports + levels.us_imports)
    ru_growth = np.log(levels.industrial_production_mom_sa / 100).rolling(12).sum()
    dates = target.index
    macro = {
        "inflation_gap_mom": (ru_inflation - us_prices.diff(), "cpi"),
        "inflation_gap_yoy": (ru_inflation.rolling(12).sum() - us_prices.diff(12), "cpi"),
        "m2_growth_gap": (np.log(levels.m2).diff(12) - np.log(levels.us_m2).diff(12), "money"),
        "trade_balance_gap": (ru_trade - us_trade, "trade"),
        "production_growth_gap": (ru_growth - np.log(levels.us_industrial).diff(12), "production"),
    }
    features = {
        f"macro__theory_{name}": delayed(values, dates, LAGS[lag_key], 100)
        for name, (values, lag_key) in macro.items()
    }
    features["rates__theory_policy_gap"] = (
        panel["rates__key_rate_level"] - panel["global__fed_funds_level"]
    )
    relative_prices = delayed(ru_prices - us_prices, dates, LAGS["cpi"], 100)
    real_rate = np.log(target) - relative_prices
    value_gap = real_rate - real_rate.shift().rolling(1095, min_periods=730).mean()
    return pd.DataFrame(features, index=dates), value_gap


def global_and_fx(target, sources, value_gap, peer_lags):
    """Собрать риск, общий долларовый фактор, momentum и отклонение от ППС."""
    dates = target.index
    cutoffs = dates.tz_localize("Europe/Moscow")
    vix = asof_series(sources["vix"], cutoffs, max_age_days=15).value.set_axis(dates)
    funding = latest_values(sources["us_cp_3m"]) - latest_values(sources["us_bill_3m"])
    epu = np.log1p(latest_values(sources["us_epu"]).asfreq("D")).rolling(7, min_periods=5).mean()
    peers = pd.DataFrame(
        {
            name: np.log(delayed(latest_values(sources[name]), dates, peer_lags[name], 15))
            for name in PEERS
        },
        index=dates,
    )
    peer_moves = peers.diff(21)
    dollar_strength = peer_moves.mean(axis=1).where(peer_moves.count(axis=1) >= 4)
    daily_peer_moves = peers.diff()
    fx_vol = daily_peer_moves.abs().mean(axis=1).where(daily_peer_moves.count(axis=1) >= 4)
    own_move = np.log(target).diff(21)
    return pd.DataFrame(
        {
            FUNDING: delayed(funding, dates, LAGS["funding"], 10),
            "global__theory_us_epu_7d": delayed(epu, dates, LAGS["epu"], 10),
            "global__theory_vix_level": np.log(vix),
            "global__theory_em_usd_return_21d": dollar_strength,
            "global__theory_em_fx_abs_return_21d": fx_vol.rolling(21, min_periods=15).mean(),
            "fx__theory_rub_relative_momentum_21d": own_move - dollar_strength,
            "fx__theory_rub_momentum_63d": np.log(target).diff(63),
            "fx__theory_ppp_value_gap": value_gap,
        },
        index=dates,
    )


def theory_features(panel, target, sources, peer_lags):
    """Выпустить два блока признаков без будущих значений."""
    fundamental, value_gap = fundamentals(panel, target, sources)
    market = global_and_fx(target, sources, value_gap, peer_lags)
    return fundamental, market
