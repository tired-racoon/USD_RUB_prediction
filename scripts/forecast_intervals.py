"""Построить прогнозные интервалы основного ансамбля USD/RUB."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from tsururu_diagnostics import CALIBRATION, conformal_radius

from usd_rub.fixing_calendar import HORIZONS
from usd_rub.sources.news_archive import write_csv

LEVELS = (99, 95, 90, 75, 50)
ARCHIVE_COLUMNS = ["fold", "origin", "target_date", "horizon", "actual", "published_ensemble"]


def forecast_intervals(point, archive):
    """Откалибровать пять интервалов на дозревших ошибках каждого горизонта."""
    if (
        point.origin.nunique() != 1
        or not np.array_equal(point.horizon, HORIZONS)
        or not np.array_equal(point.target_date, point.origin + pd.to_timedelta(HORIZONS, unit="D"))
        or not np.isfinite(point.prediction).all()
    ):
        raise ValueError("Нужен один прогноз по всем семи календарным горизонтам")
    origin = point.origin.iat[0]
    past = archive.loc[archive.fold.eq("holdout") & archive.target_date.lt(origin)]
    if past.duplicated(["origin", "horizon"]).any():
        raise ValueError("Дубли дат и горизонтов в архиве")
    rows = []
    for forecast in point.itertuples(index=False):
        matching = past.loc[past.horizon.eq(forecast.horizon)].sort_values("origin")
        sample = matching.tail(CALIBRATION)
        errors = (sample.actual - sample.published_ensemble).abs().to_numpy()
        if len(errors) != CALIBRATION or not np.isfinite(errors).all():
            raise ValueError(f"Недостаточно известных ошибок на горизонте {forecast.horizon}")
        row = dict(
            origin=origin,
            target_date=forecast.target_date,
            horizon=forecast.horizon,
            prediction=forecast.prediction,
            calibration_n=len(sample),
            calibration_last_target=sample.target_date.max(),
        )
        for level in LEVELS:
            radius = conformal_radius(errors, level / 100)
            row[f"lower_{level}"] = forecast.prediction - radius
            row[f"upper_{level}"] = forecast.prediction + radius
        rows.append(row)
    return pd.DataFrame(rows)


def plot_intervals(table, output, anchor):
    """Нарисовать точечный прогноз и вложенные прогнозные интервалы."""
    matplotlib.use("Agg")
    from matplotlib import dates as mdates
    from matplotlib import pyplot as plt

    origin = table.origin.iat[0]
    dates = table.target_date
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    shades = ("#e4ebf4", "#c6d9ee", "#9cbce1", "#6e9bd0", "#447cb9")
    for level, color in zip(LEVELS, shades, strict=True):
        ax.fill_between(
            dates, table[f"lower_{level}"], table[f"upper_{level}"], color=color, label=f"{level}%"
        )
    ax.plot(
        dates, table.prediction, color="#102e49", lw=2, marker="o", ms=4, label="Прогноз ансамбля"
    )
    if anchor is not None:
        ax.scatter(
            origin,
            anchor,
            facecolor="white",
            edgecolor="#102e49",
            zorder=3,
            label="Курс ЦБ на дату выпуска",
        )
    sunday = origin + pd.Timedelta(days=6 - origin.dayofweek)
    if dates.min() <= sunday < dates.max():
        ax.axvline(sunday + pd.Timedelta(hours=12), color="#687889", ls=":", lw=1)
    ax.set_title(f"Прогноз USD/RUB от {origin:%d.%m.%Y}", loc="left", weight="bold")
    ax.set_ylabel("₽ за доллар США")
    ax.set_xticks(pd.DatetimeIndex([origin]).append(pd.DatetimeIndex(dates)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    """Сохранить таблицу и график рядом с текущим прогнозом."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument(
        "--archive", type=Path, default=Path("data/reports/tsururu_holdout/predictions.csv.gz")
    )
    parser.add_argument("--anchor", type=float)
    args = parser.parse_args()
    dates = ["origin", "target_date"]
    point = pd.read_csv(args.forecast, parse_dates=dates).sort_values("horizon")
    archive = pd.read_csv(args.archive, parse_dates=dates, usecols=ARCHIVE_COLUMNS)
    table = forecast_intervals(point, archive)
    write_csv(table, args.forecast.parent / "forecast_intervals.csv")
    plot_intervals(table, args.forecast.parent / "forecast_intervals.png", args.anchor)


if __name__ == "__main__":
    main()
