"""Корреляции индексов GPR с USD/RUB и график по датам."""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from usd_rub.sources.gpr_ru_press import DAILY, MONTHLY
from usd_rub.storage import read_series

ORIGINAL = "gpr_russia_monthly"


def rates(root):
    values = read_series(root, "usd_rub").sort_values("ingested_at")
    daily = (
        values.drop_duplicates("date", keep="last")
        .set_index("date")["value"]
        .sort_index()
        .asfreq("D", method="ffill")
    )
    monthly = daily.resample("MS").mean()
    complete = daily.resample("MS").count().eq(monthly.index.days_in_month)
    return daily, monthly.loc[complete]


def summarize(series, fx):
    paired = pd.concat(
        [series.rename("gpr"), fx.rename("fx"), fx.pct_change(fill_method=None).rename("change")],
        axis=1,
    ).dropna(subset=["gpr", "fx"])
    return {
        "series": series.name,
        "first": paired.index.min().date(),
        "last": paired.index.max().date(),
        "n_level": len(paired),
        "pearson_level": paired.gpr.corr(paired.fx),
        "n_change": paired.change.count(),
        "pearson_change": paired.gpr.corr(paired.change),
    }


def plot(daily, original, fx_daily, fx_monthly, output):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#374151",
            "text.color": "#172033",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
        }
    )
    navy, teal, rust = "#25466b", "#08776d", "#a1452b"
    figure = plt.figure(figsize=(12, 10), layout="constrained")
    grid = figure.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.45])
    rate_ax = figure.add_subplot(grid[0, :])
    daily_ax = figure.add_subplot(grid[1, :], sharex=rate_ax)
    month_ax = figure.add_subplot(grid[2, :], sharex=rate_ax)
    daily_scatter = figure.add_subplot(grid[3, 0])
    month_scatter = figure.add_subplot(grid[3, 1])

    figure.suptitle("Геополитический риск и курс доллара", fontweight="bold", fontsize=19)
    start, end = daily.index.min(), daily.index.max()
    fx_on_dates = fx_daily.loc[start:end]
    rate_ax.plot(fx_on_dates.index, fx_on_dates, color=navy, linewidth=1)
    rate_ax.set(title="Официальный курс Банка России", ylabel="₽ за $1")

    weekly = daily.rolling(7).mean()
    daily_ax.plot(daily.index, daily, color=teal, alpha=0.25, linewidth=0.6, label="Дни")
    daily_ax.plot(weekly.index, weekly, color=teal, linewidth=1.05, label="Среднее за 7 дней")
    daily_ax.legend(loc="upper left", frameon=False, fontsize=9)
    daily_ax.set(title="Индекс по заголовкам МК и «Интерфакса»", ylabel="% заголовков")

    edges = original.index.append(pd.DatetimeIndex([original.index[-1] + pd.offsets.MonthBegin()]))
    month_ax.stairs(original, edges, baseline=None, color=rust, linewidth=1.2)
    month_ax.set(title="GPRC_RUS Caldara–Iacoviello по отчётным месяцам", ylabel="Пункты")
    month_ax.set_xlim(start, end)
    month_ax.xaxis.set_major_locator(mdates.YearLocator())
    month_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in (rate_ax, daily_ax):
        ax.tick_params(labelbottom=False)

    for ax, series, fx, color, title, unit, size, alpha in (
        (daily_scatter, daily, fx_daily, teal, "Российские СМИ по дням", "% заголовков", 8, 0.2),
        (month_scatter, original, fx_monthly, rust, "GPRC_RUS по месяцам", "Пункты", 18, 0.55),
    ):
        paired = pd.concat([series.rename("gpr"), fx.rename("fx")], axis=1).dropna()
        ax.scatter(paired.gpr, paired.fx, s=size, color=color, alpha=alpha, linewidths=0)
        correlation = paired.gpr.corr(paired.fx)
        label = f"r = {correlation:.3f}".replace(".", ",")
        ax.set(
            title=f"{title}  ·  {label}",
            xlabel=unit,
            ylabel="₽ за $1" if ax == daily_scatter else "Средний курс, ₽ за $1",
        )

    for ax in figure.axes:
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.title.set_position((0, 1))
        ax.title.set_horizontalalignment("left")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--gpr-dir", type=Path, default=Path("docs"))
    parser.add_argument("--output", type=Path, default=Path("docs/gpr_russia_usd_rub.png"))
    args = parser.parse_args()
    daily = pd.read_csv(args.gpr_dir / "gpr_russia_daily.csv", index_col="date", parse_dates=True)
    month = pd.read_csv(args.gpr_dir / "gpr_russia_monthly.csv", index_col="date", parse_dates=True)
    fx_daily, fx_monthly = rates(args.data_dir)
    comparisons = [
        (daily[DAILY], fx_daily),
        (month[MONTHLY], fx_monthly),
        (month[ORIGINAL], fx_monthly),
    ]
    print(pd.DataFrame(summarize(series, fx) for series, fx in comparisons).to_string(index=False))
    plot(daily[DAILY], month[ORIGINAL], fx_daily, fx_monthly, args.output)


if __name__ == "__main__":
    main()
