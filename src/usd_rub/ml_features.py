"""Признаки для прямых прогнозов ошибки случайного блуждания."""

from pathlib import Path

import numpy as np
import pandas as pd

from usd_rub.features import asof_series, build_features
from usd_rub.freshness import ensure_allowed
from usd_rub.storage import read_series


def load_panel(root, config, target):
    root = Path(root)
    series = {
        name for block in config["blocks"].values() for names in block.values() for name in names
    }
    paths = [root / "normalized" / f"{name}.csv" for name in sorted(series)]
    frames = {path.stem: read_series(root, path.stem) for path in paths}
    for name in frames:
        ensure_allowed(name)
    event_paths = sorted((root / "calendars").glob("*_events.csv"))
    events = pd.concat([pd.read_csv(path) for path in event_paths], ignore_index=True)
    if events.event_id.duplicated().any():
        raise ValueError("Повтор event_id в календарях")
    panels = {}
    for mode, lags in [("strict", None), ("extended", config["assumed_lags"])]:
        aligned = build_features(
            frames,
            target.index.min(),
            target.index.max(),
            cutoff="00:00",
            assumed_lags=lags,
            events=events,
        )
        if not aligned.index.tz_localize(None).equals(target.index):
            raise ValueError("Даты признаков и цели не совпадают")
        if mode == "strict":
            contract_frame = frames["si_front_settle"].dropna(subset=["contract_expiry"]).copy()
            contract_frame["value"] = (
                pd.to_datetime(contract_frame.contract_expiry) - contract_frame.date
            ).dt.days
            days = asof_series(contract_frame, aligned.index, max_age_days=15)
            contract_days = days.value - days.age_days
        aligned.index = target.index
        aligned["si_days_to_expiry"] = contract_days.to_numpy()
        panels[mode] = engineer(aligned, target, config["blocks"])
    return panels, paths + event_paths


def engineer(panel, target, blocks):
    dates = target.index
    logs = np.log(target)
    features = {
        "own__return_1": logs.diff(),
        "own__return_2": logs.diff(2),
        "own__return_5": logs.diff(5),
        "own__return_10": logs.diff(10),
        "own__return_21": logs.diff(21),
        "own__vol_7": logs.diff().rolling(7).std(),
        "own__vol_21": logs.diff().rolling(21).std(),
        "own__vol_60": logs.diff().rolling(60).std(),
    }
    for day in range(7):
        features[f"calendar__weekday_{day}"] = (dates.dayofweek == day).astype(float)
    for name in [
        "days_to_month_end",
        "days_to_25_nominal",
        "days_to_28_nominal",
        "days_to_cbr_meeting",
        "cbr_unscheduled_meeting_flag",
        "days_since_cbr_unscheduled_meeting",
    ]:
        features[f"calendar__{name}"] = panel[name].astype(float).clip(upper=60)
    for column in panel:
        if column.startswith(("event_tax_", "days_to_tax_")):
            features[f"calendar__{column}"] = panel[column]
        if column.startswith(("event_ofac_", "event_uk_")):
            features[f"news__{column}_7d"] = panel[column].rolling(7, min_periods=7).sum()
            features[f"news__{column}_30d"] = panel[column].rolling(30, min_periods=30).sum()

    for block, transforms in blocks.items():
        for transform, names in transforms.items():
            short, long = (30, 90) if block == "macro" else (1, 5)
            for name in names:
                value = panel[name]
                if transform == "log":
                    scaled = np.log(value.where(value > 0))
                elif transform == "log1p":
                    scaled = np.log1p(value.where(value >= 0))
                else:
                    scaled = value
                key = f"{block}__{name}"
                features[f"{key}_change_{short}"] = scaled.diff(short)
                features[f"{key}_change_{long}"] = scaled.diff(long)
                if transform != "log" or block == "news":
                    features[f"{key}_level"] = scaled
                if block in {"fx", "markets", "global"} and transform != "log1p":
                    features[f"{key}_age"] = panel[f"{name}__age_days"].where(value.notna())
                if block == "news":
                    features[f"{key}_shock_30d"] = (
                        scaled - scaled.rolling(30, min_periods=15).mean()
                    )

    for name in ["si_front_settle", "usdrubf_close", "moex_usd_fixing"]:
        features[f"fx__{name}_basis"] = np.log(panel[name].where(panel[name] > 0)) - logs
    features["fx__cny_basis"] = np.log(panel.cny_tom_close.where(panel.cny_tom_close > 0)) - np.log(
        panel.cny_rub
    )
    features["fx__cny_perpetual_basis"] = np.log(
        panel.cnyrubf_close.where(panel.cnyrubf_close > 0)
    ) - np.log(panel.cny_tom_close.where(panel.cny_tom_close > 0))
    features["fx__si_days_to_expiry"] = panel.si_days_to_expiry
    features["fx__si_roll_or_gap"] = panel.si_days_to_expiry.diff().gt(10).astype(float)
    features["global__synthetic_basis"] = np.log(panel.usd_rub_cross) - logs
    features["rates__ofz_slope"] = panel.ofz_10y_2y_spread
    features["rates__ruonia_spread"] = panel.ruonia_key_spread
    features["global__ust_slope"] = panel.ust_slope
    return pd.DataFrame(features, index=dates).replace([np.inf, -np.inf], np.nan)
