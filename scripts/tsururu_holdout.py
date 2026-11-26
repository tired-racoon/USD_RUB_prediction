"""Прогнать основной ансамбль и эксперименты из лекции на сплитах Tsururu."""

import argparse
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from tsururu_diagnostics import compare_points, intervals, residual_tests
from tsururu_hybrids import BASE, MAIN, RW, corrections, econometric

from usd_rub.ensemble_forecast import CLASSICAL, classical_predictions
from usd_rub.fixing_calendar import HORIZONS, apply_calendar, fixing_mask
from usd_rub.market_correction import (
    COMPACT_CORRECTION_FRACTION,
    COMPACT_MACRO_BUNDLE,
    corrected_forecast,
    load_correction_bundle,
)
from usd_rub.ml_dataset import feature_panels, read_dataset
from usd_rub.ml_experiment import actual_at
from usd_rub.models import validate_history
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints
from usd_rub.theory_features import (
    COMPACT_FUNDAMENTAL_SIGNS,
    COMPACT_MACRO_FEATURES,
    MACRO_CORRECTION_FEATURES,
)
from usd_rub.tsururu_splits import Fold, temporal_folds

PUBLISHED = "published_ensemble"
PAIRS = (
    (RW, BASE),
    (RW, MAIN),
    (BASE, MAIN),
    (BASE, "xgb_fold8/0.5"),
    (BASE, "ridge_lag/0.5"),
    (MAIN, "xgb_fixed8_gated/0.5"),
    ("xgb_direct", "xgb_level/1"),
    (PUBLISHED, MAIN),
)


@dataclass
class Result:
    fold: Fold
    actual: np.ndarray
    last: np.ndarray
    points: dict

    def metrics(self, scope, rows=slice(None)):
        actual, last = self.actual[rows], self.last[rows]
        scale = np.sqrt(np.mean((actual - last[:, None]) ** 2, axis=0))
        for name, point in self.points.items():
            errors = point[rows] - actual
            rmse = np.sqrt(np.mean(errors**2, axis=0))
            for h, r, a, naive in zip(
                HORIZONS, rmse, np.abs(errors).mean(axis=0), scale, strict=True
            ):
                yield (scope, name, h, len(actual), r, a, r / naive)

    def forecasts(self, dates):
        n = len(HORIZONS)
        origins = dates[self.fold.origins]
        return pd.DataFrame(
            {
                "fold": self.fold.name,
                "origin": np.repeat(origins, n),
                "target_date": np.repeat(origins, n)
                + pd.to_timedelta(np.tile(HORIZONS, len(origins)), unit="D"),
                "horizon": np.tile(HORIZONS, len(origins)),
                "actual": self.actual.ravel(),
                "last": np.repeat(self.last, n),
                **{name: point.ravel() for name, point in self.points.items()},
            }
        )


def plot_timeline(path, dates, splits, start):
    """Показать обучение, оценку и защитный зазор на общей оси лет."""
    matplotlib.use("Agg")
    from matplotlib import dates as mdates
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch

    colors = dict(
        history="#b1bdc9",
        train="#526c84",
        warmup="#b68d57",
        cv="#58a096",
        test="#365fc6",
        purge="#d4515d",
    )
    fig, ax = plt.subplots(figsize=(13, 5), layout="constrained")
    labels = []
    for i, fold in enumerate(splits):
        first = dates[fold.cutoff]
        labels.append(
            {"holdout": "Тест, одна модель", "annual_2026": "Тест, обновление в 2026"}.get(
                fold.name, f"{first.year} · {'архив ошибок' if fold.warmup else 'валидация'}"
            )
        )
        color = colors[
            "warmup" if fold.warmup else "cv" if fold.name.startswith("tsururu") else "test"
        ]
        for left, right, segment in (
            (dates[0], dates[start], colors["history"]),
            (dates[start], first - pd.Timedelta(days=7), colors["train"]),
            (first - pd.Timedelta(days=7), first, colors["purge"]),
            (first, dates[fold.origins[-1]] + pd.Timedelta(days=1), color),
        ):
            ax.barh(i, (right - left).days, left=mdates.date2num(left), height=0.56, color=segment)
    ax.axvline(dates[start], color="#434955", lw=0.8, ls="--", zorder=0)
    ax.axvline(dates[splits[-2].cutoff], color="#434955", lw=0.8, ls="--", zorder=0)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(dates[0].replace(month=1, day=1), dates[-1] + pd.Timedelta(days=90))
    ax.grid(axis="x", alpha=0.14)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_title("USD/RUB · даты выпуска прогнозов на 7 дней", loc="left", weight="bold", pad=14)
    legend = [
        ("history", "История до Tsururu"),
        ("train", "Обучение"),
        ("purge", "7 дней до границы"),
        ("warmup", "Прогнозы для поправки"),
        ("cv", "Валидация"),
        ("test", "Тест"),
    ]
    ax.legend(
        handles=[Patch(color=colors[key], label=label) for key, label in legend],
        loc="upper left",
        bbox_to_anchor=(0, -0.08),
        ncol=6,
        frameon=False,
    )
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def run(args):
    """Переобучить модели, выбрать варианты до теста и записать прогнозы."""
    config = json.loads(args.config.read_text())
    calendar = json.loads(args.calendar.read_text())
    dataset = read_dataset(args.dataset)
    dates, levels = dataset.index, validate_history(dataset.usd_rub.to_numpy(dtype=float))
    panels = feature_panels(dataset)
    panel = panels["extended"][list(MACRO_CORRECTION_FEATURES)].replace([np.inf, -np.inf], np.nan)
    folds, start = temporal_folds(dates, config["test_start"])
    holdout = Fold("holdout", dates.get_loc(pd.Timestamp(config["test_start"])), len(dates))
    annual = Fold("annual_2026", dates.get_loc(pd.Timestamp("2026-01-01")), len(dates))
    archive = np.full((len(dates), len(HORIZONS)), np.nan)
    vol = pd.Series(np.log(levels)).diff().rolling(21).std().clip(lower=0.001).to_numpy()
    frozen, meta = load_correction_bundle(
        COMPACT_MACRO_BUNDLE, COMPACT_MACRO_FEATURES, COMPACT_FUNDAMENTAL_SIGNS
    )
    if pd.Timestamp(meta["trained_before"]) != dates[holdout.cutoff]:
        raise ValueError("Дата обучения опубликованной поправки расходится с границей теста")

    splits, results, selections = [], [], []
    for i, fold in enumerate([*folds, holdout, annual]):
        cutoff, origins = fold.cutoff, fold.origins
        last = levels[origins]
        mask = fixing_mask(dates[origins], calendar, True)
        classical = classical_predictions(panels, levels, cutoff, origins)
        archive[origins] = apply_calendar(classical, last, mask, pool=True)
        points = econometric(levels, cutoff, origins, mask)
        points[BASE] = archive[origins]
        n_calibration, threshold = 0, None
        if i:
            adjusted, selected, n_calibration, threshold = corrections(
                panel, levels, archive, cutoff, origins, vol, mask
            )
            points.update(adjusted)
            selections.extend((fold.name, rank, key) for rank, key in enumerate(selected, 1))
        if fold == holdout:
            features = panel.iloc[origins][list(COMPACT_MACRO_FEATURES)]
            points[PUBLISHED] = corrected_forecast(
                frozen, features, points[BASE], last, mask, fraction=COMPACT_CORRECTION_FRACTION
            )
        if any(not (np.isfinite(point).all() and (point > 0).all()) for point in points.values()):
            raise ValueError(f"{fold.name}: прогноз должен быть конечным и положительным")
        splits.append(
            {
                "fold": fold.name,
                "role": "warmup" if fold.warmup else "validation" if fold in folds else "test",
                "tsururu_train_start": dates[start] if fold in folds else pd.NaT,
                "expanding_train_start": dates[0],
                "tree_train_start": dates[max(0, cutoff - CLASSICAL[1][0].window)],
                "train_last_origin_h1": dates[cutoff - 2],
                "train_last_origin_h7": dates[cutoff - 8],
                "train_last_target": dates[cutoff - 1],
                "block_start": dates[cutoff],
                "block_end": dates[fold.stop - 1],
                "last_scored_origin": dates[origins[-1]],
                "scored_origins": len(origins),
                "correction_train_origins": n_calibration,
                "gate_threshold_daily_log_vol": threshold,
            }
        )
        results.append(Result(fold, actual_at(levels, origins, HORIZONS), last, points))
        print(
            f"{fold.name}: {dates[cutoff].date()} — {dates[origins[-1]].date()}, {len(origins)} дат",
            flush=True,
        )

    validation = [result for result in results if result.fold in folds and not result.fold.warmup]
    test = results[-2]
    metrics = [row for result in results for row in result.metrics(result.fold.name)]
    for year in dates[holdout.origins].year.unique():
        metrics.extend(test.metrics(f"holdout_{year}", dates[holdout.origins].year == year))
    table = pd.DataFrame(
        metrics, columns=["scope", "model", "horizon", "n", "rmse_rub", "mae_rub", "rmse_to_rw"]
    )
    cv = table.loc[table.scope.isin(result.fold.name for result in validation)]
    h7 = cv.loc[cv.horizon.eq(7)].groupby("model").rmse_to_rw.mean().rename("cv_rmse7_to_rw")
    avg = cv.groupby("model").rmse_to_rw.mean().rename("cv_rmse_to_rw")
    ranking = pd.concat([h7, avg], axis=1).sort_values([h7.name, avg.name])
    hold = table.loc[table.scope.eq("holdout")]
    for h in (1, 7):
        values = hold.loc[hold.horizon.eq(h)].set_index("model")
        ranking = ranking.join(values[["rmse_rub", "mae_rub"]].add_suffix(f"_{h}d"), how="outer")
    ranking["holdout_rmse_to_rw"] = hold.groupby("model").rmse_to_rw.mean()
    ranking = ranking.sort_values([h7.name, avg.name]).reset_index()

    history = [result for result in results[:-1] if MAIN in result.points]
    origins = np.concatenate([result.fold.origins for result in history])
    point = np.vstack([result.points[MAIN] for result in history])
    actual = np.vstack([result.actual for result in history])
    interval_metrics, bounds = intervals(
        levels, origins, point, actual, np.flatnonzero(origins >= holdout.cutoff), vol
    )
    bounds["origin"] = dates[bounds.pop("position")]
    bounds["target_date"] = bounds.origin + pd.to_timedelta(bounds.horizon, unit="D")
    stats = compare_points(test.points, test.actual, PAIRS)
    checks = residual_tests(levels[: holdout.cutoff], test.actual, test.points[MAIN])
    write_csv(pd.DataFrame(splits), args.output / "splits.csv")
    write_csv(table, args.output / "metrics.csv")
    write_csv(ranking, args.output / "summary.csv")
    write_csv(
        pd.DataFrame(selections, columns=["fold", "rank", "feature"]),
        args.output / "correction_features.csv",
    )
    write_csv(
        pd.concat([r.forecasts(dates) for r in results], ignore_index=True),
        args.output / "predictions.csv.gz",
    )
    write_csv(interval_metrics, args.output / "interval_metrics.csv")
    write_csv(bounds, args.output / "intervals.csv.gz")
    write_csv(stats, args.output / "paired_tests.csv")
    write_csv(checks, args.output / "diagnostics.csv")
    plot_timeline(args.output / "timeline.png", dates, [*folds, holdout, annual], start)
    metadata = {
        "inputs": input_fingerprints(
            [
                args.dataset,
                args.config,
                args.calendar,
                COMPACT_MACRO_BUNDLE,
                Path(__file__),
                Path(temporal_folds.__code__.co_filename),
                Path(corrections.__code__.co_filename),
                Path(compare_points.__code__.co_filename),
            ]
        ),
        "packages": {
            name: version(name) for name in ("tsururu", "scikit-learn", "xgboost", "statsmodels")
        },
        "selected_by_cv": ranking.iloc[0].model,
        "selected_correction_by_cv": ranking.loc[
            ranking.model.str.contains("/") | ranking.model.eq(BASE)
        ]
        .iloc[0]
        .model,
        "interval_model": MAIN,
        "primary_horizon": int(HORIZONS[-1]),
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/reports/theory_holdout/dataset.csv.gz")
    )
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument("--calendar", type=Path, default=Path("config/cbr_fixing_calendar.json"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/tsururu_holdout"))
    with threadpool_limits(limits=2):
        run(parser.parse_args())
