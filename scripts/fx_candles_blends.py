"""Проверить долю рыночных моделей по числу предстоящих фиксингов ЦБ."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from forecast_stacking import panel_index
from fx_candles_holdout import CHANGE, INCUMBENT, metrics
from llm_news_holdout import bootstrap

from usd_rub.fixing_calendar import HORIZONS, fixing_mask
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints

GROUPS = {
    "candles": (
        "candles_compact/ridge/direct/decay_365/0.035",
        "candles_compact/huber/residual/regime/0.03",
    ),
    "daily": (
        "daily_compact/ridge/direct/decay_365/0.035",
        "daily_compact/huber/residual/regime/0.03",
    ),
    "basis": (
        "basis/ridge/direct/decay_365/0.035",
        "basis/ridge/direct/rolling_365/0.045",
        "basis/pool/residual/regime/0.05",
    ),
    "nbrb": (
        "basis_nbrb/ridge/direct/decay_365/0.035",
        "basis_nbrb/ridge/residual/decay_365/0.035",
        "candles_nbrb/ridge/direct/decay_365/0.035",
    ),
    "trees": (
        "candles/xgb/direct/annual_730/3",
        "daily/xgb/direct/annual_730/3",
    ),
}
GROUPS["all"] = tuple(name for family in ("candles", "daily", "nbrb") for name in GROUPS[family])
WINDOWS = {"online_180": 180, "online_365": 365, "annual_730": 730}


def learn_blend(dates, fix_count, baseline, candidate, actual, last, policy, limit):
    """Оценить доли смеси по известным ошибкам прогнозов."""
    delta = candidate - baseline
    error = actual - baseline
    result = baseline.copy()
    months = dates.to_period("Y" if policy.startswith("annual") else "M")
    for month in months.unique():
        query = months == month
        cutoff = dates[query][0]
        age = (cutoff - dates).days.to_numpy()
        history = (age > HORIZONS[-1]) & (age <= WINDOWS[policy])
        if history.sum() < 90:
            continue
        normalizer = np.maximum(np.mean((actual[history] - last[history, None]) ** 2, axis=0), 0.01)
        fit_weight = np.exp2(-age[history, None] / (WINDOWS[policy] / 2)) / normalizer
        changes = delta[history]
        miss = error[history]
        for step in range(1, limit + 1):
            mask = fix_count[history] == step
            products = fit_weight * mask
            cross = np.sum(changes * miss * products)
            variance = np.sum(changes**2 * products)
            alpha = np.clip(cross / (1.25 * variance + 1e-10), 0, 1)
            result[query] += alpha * delta[query] * (fix_count[query] == step)
    return result


def predictions(frame, dates, fix_count, actual, last):
    """Собрать фиксированные и адаптивные смеси на общих прогнозах."""
    shape = actual.shape
    baseline = frame[INCUMBENT].to_numpy().reshape(shape)
    result = {INCUMBENT: baseline}
    for group, names in GROUPS.items():
        candidate = frame[list(names)].mean(axis=1).to_numpy().reshape(shape)
        for limit, policy in product((1, 2, 3), ("quarter", "half", "full", *WINDOWS)):
            key = f"{group}/{limit}/{policy}"
            if policy in ("quarter", "half", "full"):
                alpha = {"quarter": 0.25, "half": 0.5, "full": 1}[policy]
                mask = (fix_count > 0) & (fix_count <= limit)
                result[key] = baseline + alpha * (candidate - baseline) * mask
            else:
                result[key] = learn_blend(
                    dates, fix_count, baseline, candidate, actual, last, policy, limit
                )
    return result


def run(args):
    """Сохранить ошибки каждой смеси, прогнозы и парные сравнения."""
    parent = args.predictions.parent / "run.json"
    stage = json.loads(parent.read_text())
    dates_test_start = pd.Timestamp(json.loads(args.config.read_text())["test_start"])
    frame = pd.read_csv(args.predictions, parse_dates=["origin"], float_precision="round_trip")
    frame = frame.set_index(["origin", "horizon"])
    dates = pd.DatetimeIndex(frame.index.get_level_values("origin").unique())
    pd.testing.assert_index_equal(frame.index, panel_index(dates))
    holdout = dates >= dates_test_start
    if int(holdout.sum()) != stage["test_origins"]:
        raise ValueError("Даты бэктеста расходятся с архивом")
    actual = frame.actual.to_numpy().reshape(len(dates), len(HORIZONS))
    last = frame["last"].to_numpy().reshape(actual.shape)[:, 0]
    calendar = json.loads(args.calendar.read_text())
    fixes = fixing_mask(dates, calendar, True)
    np.testing.assert_array_equal(fixes.ravel(), frame.fixing.to_numpy())
    fix_count = fixes.cumsum(axis=1)
    result = predictions(frame, dates, fix_count, actual, last)
    years = {year: dates.year == year for year in dates.year.unique()}
    validation_years = [year for year in years if year < dates_test_start.year]
    post_otc = (dates >= CHANGE) & ~holdout
    if not post_otc.any():
        raise ValueError("Нет валидационных дат после перехода на внебиржевой фиксинг")

    summaries, yearly = [], []
    for name, point in result.items():
        year_scores = {}
        for year, rows in years.items():
            values = metrics(point[rows], actual[rows], last[rows])
            year_scores[year] = values["mean_rmse_to_rw"]
            yearly.append({"experiment": name, "year": year, **values})
        summaries.append(
            {
                "experiment": name,
                "validation": np.mean([year_scores[year] for year in validation_years]),
                "validation_post_otc": metrics(
                    point[post_otc],
                    actual[post_otc],
                    last[post_otc],
                )["mean_rmse_to_rw"],
                **metrics(point[holdout], actual[holdout], last[holdout]),
            }
        )
    scores = pd.DataFrame(summaries).sort_values("validation")
    selected = scores.iloc[0].experiment
    recent = scores.nsmallest(1, "validation_post_otc").iloc[0].experiment
    best = scores.nsmallest(1, "mean_rmse_to_rw").iloc[0].experiment
    comparisons = [
        {
            "experiment": name,
            "gain_ci_pp": bootstrap(
                result[INCUMBENT][holdout],
                result[name][holdout],
                actual[holdout],
                last[holdout],
            ),
        }
        for name in dict.fromkeys((selected, recent, best))
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(scores, args.output / "summary.csv")
    write_csv(pd.DataFrame(yearly), args.output / "by_year.csv")
    predictions_frame = pd.DataFrame(
        {name: values.ravel() for name, values in result.items()}, index=frame.index
    )
    write_csv(predictions_frame.reset_index(), args.output / "predictions.csv.gz")
    run_report = {
        "inputs": input_fingerprints(
            [args.predictions, parent, args.calendar, args.config, Path(__file__)]
        ),
        "groups": GROUPS,
        "experiments": len(result) - 1,
        "chosen_on_validation": selected,
        "chosen_on_validation_post_otc": recent,
        "lowest_test_rmse": best,
        "test_origins": int(holdout.sum()),
        "validation_post_otc_origins": int(post_otc.sum()),
        "paired_vs_incumbent": comparisons,
    }
    (args.output / "run.json").write_text(json.dumps(run_report, indent=2) + "\n")
    columns = [
        "experiment",
        "validation",
        "validation_post_otc",
        "mean_rmse_to_rw",
        "rmse_1d",
        "rmse_7d",
    ]
    for metric in ("validation", "validation_post_otc", "mean_rmse_to_rw"):
        print(scores.nsmallest(8, metric)[columns].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument("--calendar", type=Path, default=Path("config/cbr_fixing_calendar.json"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "data/reports/fx_candles_holdout/predictions.csv.gz",
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("data/reports/fx_candles_blends"))
    run(parser.parse_args())
