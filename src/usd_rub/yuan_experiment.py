"""Прогноз официального USD/RUB через две юаневые пары и базис."""

import argparse
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from usd_rub.backtest import daily_target, score
from usd_rub.features import asof_series
from usd_rub.ml_experiment import (
    Candidate,
    actual_at,
    bootstrap_gain,
    choose_shrink,
    fit_predict,
    interval_from_validation,
    prediction_rows,
    relative_rmse,
    to_levels,
)
from usd_rub.storage import Client, input_fingerprints, observations, read_series

ECB_ARCHIVE = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
SERIES = ("usd_rub", "cny_tom_vwap", "cny_tom_close", "rusfar_cny")
RIDGE_MODELS = ("ridge_direct", "ridge_basis", "ridge_yuan", "ridge_yuan_funding")


def ecb_usd_cny(payload):
    """Курсы ЕЦБ за день доступны со следующей полуночи МСК."""
    with ZipFile(BytesIO(payload.content)) as archive:
        with archive.open("eurofxref-hist.csv") as file:
            rates = pd.read_csv(file, usecols=["Date", "USD", "CNY"]).dropna()
    if (
        rates.empty
        or not (np.isfinite(rates[["CNY", "USD"]]) & rates[["CNY", "USD"]].gt(0)).all().all()
    ):
        raise ValueError("Некорректные курсы ЕЦБ")
    return observations(
        rates["Date"],
        rates["CNY"] / rates["USD"],
        "ecb_usd_cny",
        "CNY/USD",
        payload,
        availability="ecb_daily_next_moscow_midnight",
        lag_days=1,
    )


def align_inputs(frames, ecb, target):
    cutoffs = target.index.tz_localize("Europe/Moscow")
    aligned = {"usd_rub": target}
    for name, frame in {**frames, "ecb_usd_cny": ecb}.items():
        snapshot = asof_series(frame, cutoffs, max_age_days=15)
        for key in ("value", "observation_date", "available_at", "age_days"):
            column = name if key == "value" else f"{name}__{key}"
            aligned[column] = snapshot[key].array
    panel = pd.DataFrame(aligned, index=target.index)
    panel["cross_vwap"] = panel.cny_tom_vwap * panel.ecb_usd_cny
    panel["cross_close"] = panel.cny_tom_close * panel.ecb_usd_cny
    panel["basis_factor"] = panel.usd_rub / panel.cross_vwap
    return panel


def feature_sets(panel):
    groups = {}
    for name in ("usd_rub", "cny_tom_vwap", "ecb_usd_cny", "basis_factor"):
        log_rate = np.log(panel[name])
        groups[name] = pd.DataFrame(
            {f"{name}_return_{lag}": log_rate.diff(lag) for lag in (1, 2, 5, 10, 21)},
            index=panel.index,
        )
        groups[name][f"{name}_vol_21"] = log_rate.diff().rolling(21).std()
    levels = np.log(panel.basis_factor)
    local = pd.DataFrame(
        {
            "basis": levels,
            "basis_vs_21d": levels - levels.rolling(21).median(),
            "close_vwap_spread": np.log(panel.cny_tom_close / panel.cny_tom_vwap),
            "moex_age": panel.cny_tom_vwap__age_days,
            "ecb_age": panel.ecb_usd_cny__age_days,
        }
    )
    weekdays = pd.get_dummies(panel.index.dayofweek, prefix="weekday", drop_first=True, dtype=float)
    weekdays.index = panel.index
    base = pd.concat([weekdays, local, *groups.values()], axis=1)
    legs = {
        "cny_tom_vwap": pd.concat(
            [weekdays, local, groups["cny_tom_vwap"], groups["usd_rub"], groups["basis_factor"]],
            axis=1,
        ),
        "ecb_usd_cny": pd.concat([weekdays, groups["ecb_usd_cny"], local[["ecb_age"]]], axis=1),
        "basis_factor": base,
    }
    rate = panel.rusfar_cny
    funding = {
        name: features.assign(
            rusfar_cny=rate,
            rusfar_cny_vs_21d=rate - rate.rolling(21, min_periods=10).median(),
            rusfar_cny_age=panel.rusfar_cny__age_days.where(rate.notna()),
        )
        for name, features in legs.items()
        if name != "ecb_usd_cny"
    }
    return {
        key: {name: features.to_numpy(dtype=float) for name, features in columns.items()}
        for key, columns in {"base": legs, "funding": {**legs, **funding}}.items()
    }


def forecasts(panel, matrices, split, origins, horizons, ridge):
    last = panel.usd_rub.to_numpy()[origins, None]
    direct, _ = fit_predict(
        matrices["base"]["basis_factor"], panel.usd_rub.to_numpy(), ridge, split, origins, horizons
    )
    predicted = {
        "ridge_direct": last * (1 + direct),
        "cross_vwap": np.repeat(panel.cross_vwap.to_numpy()[origins, None], len(horizons), axis=1),
        "cross_close": np.repeat(
            panel.cross_close.to_numpy()[origins, None], len(horizons), axis=1
        ),
        "rw": np.repeat(last, len(horizons), axis=1),
    }
    for version, name in (("base", "ridge_yuan"), ("funding", "ridge_yuan_funding")):
        factors = {}
        for leg, features in matrices[version].items():
            change, _ = fit_predict(
                features, panel[leg].to_numpy(), ridge, split, origins, horizons
            )
            factors[leg] = 1 + change
        cross_change = factors["cny_tom_vwap"] * factors["ecb_usd_cny"]
        if name == "ridge_yuan":
            predicted["ridge_basis"] = last * factors["basis_factor"]
            predicted["ridge_cross"] = panel.cross_vwap.to_numpy()[origins, None] * cross_change
        predicted[name] = last * cross_change * factors["basis_factor"]
    return predicted


def archived_rows(path, evaluated):
    archive = pd.read_csv(path, parse_dates=["origin", "target_date"], float_precision="round_trip")
    keys = ["origin", "horizon"]
    baseline = evaluated.loc[evaluated.model.eq("rw")].set_index(keys).sort_index()
    models = {
        "ensemble": "archive_ensemble",
        "histboost_extended_full_4_pooled": "archive_histboost",
    }
    comparison = []
    for name, label in models.items():
        rows = archive.loc[archive.model.eq(name)].set_index(keys, verify_integrity=True)
        rows = rows.reindex(baseline.index)
        if not rows[["target_date", "last", "actual"]].equals(
            baseline[["target_date", "last", "actual"]]
        ):
            raise ValueError(f"Архив {name} не совпадает с теми же датами и фактическими курсами")
        if not np.isfinite(rows.prediction).all():
            raise ValueError(f"В архиве {name} есть пропуски прогнозов")
        comparison.append(rows.assign(model=label).reset_index())
    return pd.concat(comparison, ignore_index=True)


def run(root, output, config_path, old_predictions=None):
    config = json.loads(config_path.read_text())
    if output.exists():
        raise FileExistsError(f"Каталог эксперимента уже существует: {output}")
    target = daily_target(read_series(root, "usd_rub"), config["start"], config["end"])
    local = {name: read_series(root, name) for name in SERIES[1:]}
    with Client(root) as client:
        payload = client.get(ECB_ARCHIVE)
    panel = align_inputs(local, ecb_usd_cny(payload), target)
    levels = panel[["cross_vwap", "cross_close"]]
    if not (np.isfinite(levels) & levels.gt(0)).all().all():
        raise ValueError("В выбранном периоде есть даты без свежих положительных котировок кросса")
    matrices = feature_sets(panel)
    values = target.to_numpy()
    horizons = np.arange(1, config["horizon"] + 1)
    split = target.index.searchsorted(config["test_start"])
    origins = np.arange(split, len(target) - horizons[-1])
    if not len(origins):
        raise ValueError("Тестовый период не содержит завершённых прогнозов")
    splits, validation = [], []
    ridge = Candidate("ridge", "yuan_experiment", window=1460, complexity=0.15)
    for year in config["validation_years"]:
        boundary = target.index.searchsorted(f"{year}-01-01")
        next_boundary = target.index.searchsorted(f"{year + 1}-01-01")
        if next_boundary > split:
            raise ValueError("Валидация пересекается с тестом")
        valid = np.arange(boundary, next_boundary - horizons[-1])
        splits.append((valid, actual_at(values, valid, horizons), values[valid]))
        validation.append(forecasts(panel, matrices, boundary, valid, horizons, ridge))
    last = values[origins]
    actual = actual_at(values, origins, horizons)
    test = forecasts(panel, matrices, split, origins, horizons, ridge)
    lower_split = max(0, split - config["baseline_window"])
    scale = np.mean(np.abs(np.diff(values[lower_split:split])))
    all_rows, validation_rows = [], []
    gains = {}
    for name, estimate in test.items():
        valid_estimates = [fold[name] for fold in validation]
        shrink = 1
        if name in RIDGE_MODELS:
            relative = [a / s[2][:, None] - 1 for a, s in zip(valid_estimates, splits, strict=True)]
            _, shrink = choose_shrink(relative, splits)
            valid_estimates = [
                to_levels(r, s[2], shrink) for r, s in zip(relative, splits, strict=True)
            ]
            if shrink != 1:
                estimate = to_levels(estimate / last[:, None] - 1, last, shrink)
        for (valid, known, anchor), predicted in zip(splits, valid_estimates, strict=True):
            validation_rows.append(
                {
                    "model": name,
                    "year": target.index[valid[0]].year,
                    "n_origins": len(valid),
                    "shrink": shrink,
                    "mean_rmse_ratio": relative_rmse(predicted, known, anchor).mean(),
                }
            )
        lower, upper = interval_from_validation(estimate, last, valid_estimates, splits)
        gains[name] = bootstrap_gain(estimate, actual, last)
        all_rows.append(
            prediction_rows(name, target, origins, horizons, estimate, lower, upper, scale, np.nan)
        )
    rows = pd.concat(all_rows, ignore_index=True)
    if old_predictions is not None:
        rows = pd.concat([rows, archived_rows(old_predictions, rows)], ignore_index=True)

    metrics = score(rows)
    metrics_by_year = pd.concat(
        [score(group).assign(year=year) for year, group in rows.groupby(rows.origin.dt.year)],
        ignore_index=True,
    )
    output.mkdir(parents=True)
    for name, table in {
        "panel": panel.rename_axis("date").reset_index(),
        "predictions": rows,
        "metrics": metrics,
        "metrics_by_year": metrics_by_year,
        "validation": pd.DataFrame(validation_rows),
    }.items():
        table.to_csv(output / f"{name}.csv", index=False)
    inputs = [
        Path(__file__),
        config_path,
        *(root / "normalized" / f"{name}.csv" for name in SERIES),
    ]
    if old_predictions is not None:
        inputs.append(old_predictions)
    metadata = {
        "ecb": {"url": payload.url, "sha256": payload.sha256},
        "inputs": input_fingerprints(inputs),
        "ridge_window": ridge.window,
        "ridge_penalty_per_row": ridge.complexity,
        "gain_vs_rw_ci95_pct": gains,
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("config/yuan_holdout.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-predictions", type=Path)
    args = parser.parse_args()
    run(args.data, args.output, args.config, args.old_predictions)


if __name__ == "__main__":
    main()
