"""Sequential, point-in-time Ridge target/shrinkage/recency experiments.

This is deliberately separate from the frozen ml_experiment.py baseline. The
2025-26 holdout is evaluated only after validation choices are fixed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from usd_rub.backtest import dm_test
from usd_rub.ml_dataset import EXTENDED_PREFIX, FEATURE_GROUPS, feature_panels, read_dataset
from usd_rub.ml_experiment import specific_features


HORIZONS = np.arange(1, 8)
TARGETS = ("level", "difference", "log_return")
HALF_LIVES = (30, 90, 180, 365, 730)
WINDOWS = (None, 730, 1460)
ALPHAS = np.linspace(0, 1, 21)


def validation_safe_dataset(path):
    dataset = read_dataset(path)
    feature_columns = [name for name in dataset if
                       name.removeprefix(EXTENDED_PREFIX).split("__", 1)[0] in FEATURE_GROUPS]
    early = dataset.loc[dataset.index < "2021-01-01", feature_columns]
    keep = early.notna().mean().ge(0.5)
    return dataset.drop(columns=keep.index[~keep])


def matrix(dataset):
    panel = specific_features(feature_panels(dataset), "pilot")
    # The current level is known at the midnight forecast origin. Include it
    # for all target formulations so the comparison uses identical features.
    panel = panel.assign(current_usd_rub=dataset["usd_rub"])
    return panel.to_numpy(dtype=float), panel.columns.tolist()


def make_folds(dates, test_start):
    train_end = dates.searchsorted(pd.Timestamp(test_start))
    folds = []
    for year in (2021, 2022, 2023, 2024):
        start = dates.searchsorted(pd.Timestamp(f"{year}-01-01"))
        stop = min(dates.searchsorted(pd.Timestamp(f"{year + 1}-01-01")), train_end)
        folds.append((start, np.arange(start, stop - HORIZONS[-1])))
    test = np.arange(train_end, len(dates) - HORIZONS[-1])
    return folds, (train_end, test)


def fitted_features(x, rows):
    history = x[rows]
    finite = np.isfinite(history)
    keep = finite.sum(axis=0) >= 60
    if not keep.any():
        raise ValueError("No features with 60 historical observations")
    history = np.where(finite[:, keep], history[:, keep], np.nan)
    center = np.nanmean(history, axis=0)
    scale = np.nanstd(history, axis=0)
    scale = np.where(scale < 1e-10, 1, scale)

    def transform(values):
        z = (values[:, keep] - center) / scale
        return np.clip(np.nan_to_num(z, nan=0, posinf=8, neginf=-8), -8, 8)

    return transform, keep


def train_predict(x, y, boundary, origins, target, window=None, half_life=None):
    first = max(0, boundary - window) if window else 0
    # All labels are strictly before the refit boundary, including h=7.
    transform, _ = fitted_features(x, np.arange(first, boundary - HORIZONS[-1]))
    query = transform(x[origins])
    result = np.empty((len(origins), len(HORIZONS)))
    for j, h in enumerate(HORIZONS):
        rows = np.arange(first, boundary - h)
        assert len(rows) and rows[-1] + h < boundary
        if target == "level":
            label = y[rows + h]
        elif target == "difference":
            label = y[rows + h] - y[rows]
        else:
            label = np.log(y[rows + h] / y[rows])
        weight = (
            np.ones(len(rows))
            if half_life is None
            else 2.0 ** (-(boundary - rows) / half_life)
        )
        # Keep effective regularization comparable as sample weights change.
        weight /= weight.mean()
        model = Ridge(alpha=0.15 * len(rows), fit_intercept=True)
        model.fit(transform(x[rows]), label, sample_weight=weight)
        prediction = model.predict(query)
        if target == "level":
            result[:, j] = prediction
        elif target == "difference":
            result[:, j] = y[origins] + prediction
        else:
            result[:, j] = y[origins] * np.exp(prediction)
    return result


def fold_score(predictions, folds, y, alpha=1.0):
    scores = []
    for prediction, (_, origins) in zip(predictions, folds, strict=True):
        actual = y[origins[:, None] + HORIZONS]
        rw = y[origins, None]
        estimate = rw + alpha * (prediction - rw)
        ratio = np.sqrt(np.mean((estimate - actual) ** 2, axis=0)) / np.sqrt(
            np.mean((rw - actual) ** 2, axis=0)
        )
        scores.append(ratio.mean())
    return float(np.mean(scores))


def evaluate(name, prediction, origins, y):
    actual = y[origins[:, None] + HORIZONS]
    last = y[origins, None]
    rows = []
    for j, h in enumerate(HORIZONS):
        error = prediction[:, j] - actual[:, j]
        rw_error = last[:, 0] - actual[:, j]
        rmse = np.sqrt(np.mean(error**2))
        rw_rmse = np.sqrt(np.mean(rw_error**2))
        moving = actual[:, j] != last[:, 0]
        dm, p = dm_test(error**2, rw_error**2, lag=h - 1)
        rows.append(
            {
                "model": name,
                "horizon": h,
                "n": len(origins),
                "mae": np.mean(np.abs(error)),
                "rmse": rmse,
                "relative_rmse": rmse / rw_rmse,
                "rmse_reduction_pct": 100 * (1 - rmse / rw_rmse),
                "directional_accuracy": np.mean(
                    np.sign(prediction[:, j] - last[:, 0])
                    == np.sign(actual[:, j] - last[:, 0])
                ),
                "directional_accuracy_changed": np.mean(
                    np.sign(prediction[moving, j] - last[moving, 0])
                    == np.sign(actual[moving, j] - last[moving, 0])
                ),
                "dm_stat_hac": dm,
                "dm_pvalue_hac": p,
            }
        )
    return rows


def save_plots(output, dates, origins, y, predictions):
    for h in (1, 7):
        j = h - 1
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(dates[origins], y[origins + h], label="Actual", color="black", lw=1)
        for name, prediction in predictions.items():
            ax.plot(dates[origins], prediction[:, j], label=name, lw=0.9, alpha=0.8)
        ax.set(title=f"USD/RUB forecast, {h} calendar day(s)", ylabel="RUB per USD")
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(output / f"forecast_h{h}.png", dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, h in zip(axes, (1, 7), strict=True):
        j = h - 1
        actual = y[origins + h]
        for name, prediction in predictions.items():
            ax.plot(
                dates[origins],
                np.cumsum((prediction[:, j] - actual) ** 2),
                label=name,
                lw=1,
            )
        ax.set(ylabel=f"Cumulative squared error, h={h}")
        ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output / "cumulative_squared_error.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/usd_rub_2011_2026.csv.gz")
    parser.add_argument("--output", default="experiments/staged_ridge_h1_h3_20261119")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = validation_safe_dataset(args.dataset)
    dates = dataset.index
    y = dataset.usd_rub.to_numpy(dtype=float)
    x, columns = matrix(dataset)
    folds, (test_boundary, test_origins) = make_folds(dates, "2025-01-01")
    assert dates[test_origins[0]] == pd.Timestamp("2025-01-01")
    assert dates[test_origins[-1]] == pd.Timestamp("2026-11-11")
    validation = []

    # H1: same feature panel, Ridge penalty, folds, no shrinkage.
    h1 = {}
    for target in TARGETS:
        forecasts = [train_predict(x, y, split, origins, target) for split, origins in folds]
        h1[target] = forecasts
        validation.append({"stage": "H1", "target": target, "alpha": 1.0,
                           "window": "expanding", "half_life": "none",
                           "cv_relative_rmse": fold_score(forecasts, folds, y)})
    h1_winner = min(TARGETS, key=lambda name: fold_score(h1[name], folds, y))

    # H2: scalar shrinkage is selected solely on the same historical folds.
    alpha_scores = [(fold_score(h1[h1_winner], folds, y, a), a) for a in ALPHAS]
    h2_score, h2_alpha = min(alpha_scores)
    for score, alpha in alpha_scores:
        validation.append({"stage": "H2", "target": h1_winner, "alpha": alpha,
                           "window": "expanding", "half_life": "none",
                           "cv_relative_rmse": score})

    # H3: isolate the impact of recency; use the H1 target and H2 alpha.
    h3 = []
    for window in WINDOWS:
        for half_life in (None, *HALF_LIVES):
            forecasts = [train_predict(x, y, split, origins, h1_winner,
                                       window=window, half_life=half_life)
                         for split, origins in folds]
            score = fold_score(forecasts, folds, y, h2_alpha)
            row = {"stage": "H3", "target": h1_winner, "alpha": h2_alpha,
                   "window": window or "expanding", "half_life": half_life or "none",
                   "cv_relative_rmse": score}
            h3.append(row)
            validation.append(row)
    winner = min(h3, key=lambda row: row["cv_relative_rmse"])
    pd.DataFrame(validation).to_csv(output / "validation.csv", index=False)
    validation_years = []
    for target, forecasts in h1.items():
        for year, prediction, (_, origins) in zip((2021, 2022, 2023, 2024), forecasts, folds,
                                                  strict=True):
            actual = y[origins[:, None] + HORIZONS]
            rw = y[origins, None]
            ratio = np.sqrt(np.mean((prediction - actual) ** 2, axis=0)) / np.sqrt(
                np.mean((rw - actual) ** 2, axis=0)
            )
            validation_years.append({"year": year, "model": f"h1_{target}",
                                     "relative_rmse": float(ratio.mean())})
    for year, prediction, (_, origins) in zip((2021, 2022, 2023, 2024), h1[h1_winner], folds,
                                              strict=True):
        actual = y[origins[:, None] + HORIZONS]
        rw = y[origins, None]
        shrunk = rw + h2_alpha * (prediction - rw)
        ratio = np.sqrt(np.mean((shrunk - actual) ** 2, axis=0)) / np.sqrt(
            np.mean((rw - actual) ** 2, axis=0)
        )
        validation_years.append({"year": year, "model": "h2_shrink",
                                 "relative_rmse": float(ratio.mean())})
    pd.DataFrame(validation_years).to_csv(output / "validation_by_year.csv", index=False)
    selection = {"h1_target": h1_winner, "h1_cv_relative_rmse": fold_score(h1[h1_winner], folds, y),
                 "h2_alpha": float(h2_alpha), "h2_cv_relative_rmse": h2_score,
                 "h3_selected": winner, "features": columns,
                 "feature_screening_end_exclusive": "2021-01-01",
                 "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                 "test_first": str(dates[test_origins[0]].date()),
                 "test_last": str(dates[test_origins[-1]].date()),
                 "test_count": len(test_origins)}
    (output / "selection.json").write_text(json.dumps(selection, indent=2))

    # The holdout is accessed only after writing the validation selection.
    raw = {target: train_predict(x, y, test_boundary, test_origins, target)
           for target in TARGETS}
    rw = np.repeat(y[test_origins, None], len(HORIZONS), axis=1)
    predictions = {"rw": rw, **{f"h1_{target}": forecast for target, forecast in raw.items()}}
    predictions["h2_shrink"] = rw + h2_alpha * (raw[h1_winner] - rw)
    win_window = winner["window"] if isinstance(winner["window"], int) else None
    win_half_life = winner["half_life"] if isinstance(winner["half_life"], int) else None
    h3_raw = train_predict(x, y, test_boundary, test_origins, h1_winner,
                           window=win_window, half_life=win_half_life)
    predictions["h3_recency"] = rw + h2_alpha * (h3_raw - rw)
    assert all(np.isfinite(prediction).all() for prediction in predictions.values())
    metrics = [row for name, forecast in predictions.items()
               for row in evaluate(name, forecast, test_origins, y)]
    pd.DataFrame(metrics).to_csv(output / "test_metrics.csv", index=False)
    records = []
    for name, forecast in predictions.items():
        for j, h in enumerate(HORIZONS):
            records.append(pd.DataFrame({"model": name, "origin": dates[test_origins],
                                         "horizon": h, "last": y[test_origins],
                                         "actual": y[test_origins + h],
                                         "prediction": forecast[:, j]}))
    combined = pd.concat(records, ignore_index=True)
    combined.to_csv(output / "test_predictions.csv", index=False)
    years = []
    for (year, name, h), part in combined.groupby([combined.origin.dt.year, "model", "horizon"]):
        rw_part = combined.loc[(combined.origin.dt.year == year)
                               & combined.model.eq("rw") & combined.horizon.eq(h)]
        rmse = np.sqrt(np.mean((part.prediction - part.actual) ** 2))
        rw_rmse = np.sqrt(np.mean((rw_part.prediction - rw_part.actual) ** 2))
        years.append({"year": year, "model": name, "horizon": h,
                      "relative_rmse": rmse / rw_rmse})
    pd.DataFrame(years).to_csv(output / "test_by_year.csv", index=False)
    save_plots(output, dates, test_origins, y,
               {name: predictions[name] for name in ("rw", "h1_log_return", "h2_shrink", "h3_recency")})
    print(json.dumps(selection, indent=2))
    print(pd.DataFrame(metrics).groupby("model").relative_rmse.mean().sort_values())


if __name__ == "__main__":
    main()
