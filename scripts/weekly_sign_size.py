"""Раздельно спрогнозировать знак и размер изменения USD/RUB за семь дней."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from forecast_stacking import ARCHIVE_FILES, read_panel
from foundation_ensemble import CLASSICAL, RW
from si_options_holdout import magnitude_forecast, option_blocks, point_forecast
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from weekly_literature_holdout import HORIZON, gamma_frame, residual_features, summarize

from usd_rub.ml_dataset import feature_panels, read_dataset
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints


def direction(factors, train, query):
    """Оценить вероятность роста при условии, что курс изменится."""
    moves = train.loc[train.actual.ne(train["last"])]
    y = moves.actual.gt(moves["last"])
    options = {"frequency": pd.Series((y.sum() + 1) / (len(y) + 2), index=query.index)}
    for name, features in factors.items():
        x = features.loc[moves.index]
        x = x.loc[:, x.notna().mean().ge(0.5)]
        mean, scale = x.mean(), x.std(ddof=0).where(lambda s: s.gt(1e-12), 1)

        def matrix(data):
            return ((data[x.columns] - mean) / scale).fillna(0).clip(-6, 6)

        model = LogisticRegression(C=10 / len(y))
        model.fit(matrix(x), y)
        options[name] = pd.Series(
            model.predict_proba(matrix(features.loc[query.index]))[:, 1], index=query.index
        )
    return pd.DataFrame(options)


def predict(train, query, factors, sizes):
    """Оценить обе составляющие по уже известным недельным исходам."""
    cutoff = query.index[0]
    history = train.loc[train.index + HORIZON < cutoff]
    delta = history.actual - history["last"]
    result = direction(factors, history, query)
    result["flat_probability"] = delta.loc[history.new_fixing].eq(0).mean()
    result.loc[~query.new_fixing, "flat_probability"] = 1
    for name, cols in sizes.items():
        result[name] = magnitude_forecast(cols, history, query, cutoff)
    result["average_size"] = (delta / history["last"]).abs().mean() * query["last"]
    for label, group in {"up": history.loc[delta.gt(0)], "down": history.loc[delta.lt(0)]}.items():
        result[f"size_if_{label}"] = magnitude_forecast(sizes["realized"], group, query, cutoff)
    return result


def direction_metrics(frame, truth, names):
    """Посчитать точность знака и качество вероятностей на днях с движением."""
    moves = truth.ne(0)
    y = truth.loc[moves].gt(0).to_numpy()
    rows = []
    for name in names:
        p = frame.loc[moves, name].to_numpy()
        rows.append(
            {
                "model": name,
                "dates": len(y),
                "accuracy": ((p >= 0.5) == y).mean(),
                "brier": np.mean((p - y) ** 2),
            }
        )
    return pd.DataFrame(rows)


def run(args):
    dataset = read_dataset(args.dataset)
    panel = feature_panels(dataset)["extended"]
    manifest = args.archive / "archives.json"
    groups = json.loads(manifest.read_text())["groups"]
    archives = [args.archive / filename for filename in ARCHIVE_FILES]
    names = [CLASSICAL, RW, *sum(groups.values(), [])]
    train, test = [gamma_frame(read_panel(path, names)) for path in archives]
    for pack in (train, test):
        np.testing.assert_allclose(pack["last"], dataset.usd_rub.loc[pack.index], rtol=1e-12)
        np.testing.assert_allclose(
            pack.actual, dataset.usd_rub.loc[pack.index + HORIZON], rtol=1e-12
        )
    if train.index.max() + HORIZON >= test.index.min():
        raise ValueError("Архив обучения пересекает тест")

    blocks = option_blocks(panel)
    inputs = residual_features(panel)["spread_dynamics"]
    pooled = pd.concat([train, test])
    trend = ((pooled.ensemble - pooled["last"]) / pooled["last"]).rename("gamma_return").to_frame()
    market = trend.join(inputs)
    factors = {"gamma": trend, "market": market, "market_rr": market.join(blocks["rr"])}
    vol = panel[
        ["own__vol_7", "own__vol_21", "own__vol_60", "markets__rvi_level", "markets__rvi_change_5"]
    ]
    sizes = {"realized": vol, "with_iv": vol.join(blocks["iv"]), "with_rr": vol.join(blocks["rr"])}
    dates = train.loc["2022":].index
    validation = pd.concat(
        [predict(train, fold, factors, sizes) for _, fold in train.loc[dates].groupby(dates.year)]
    )
    forecasts = predict(train, test, factors, sizes)
    truth = test.actual - test["last"]
    valid_delta = (train.actual - train["last"]).loc[dates]
    validation_sign = direction_metrics(validation, valid_delta, ("frequency", *factors))
    best_sign = validation_sign.loc[validation_sign.brier.idxmin(), "model"]
    sign_metrics = direction_metrics(forecasts, truth, ("frequency", *factors))
    baseline = point_forecast(inputs, train, test, test.index[0])
    sign_metrics.loc[len(sign_metrics)] = [
        "point_forecast",
        int(truth.ne(0).sum()),
        np.sign(baseline - test["last"])
        .loc[truth.ne(0)]
        .eq(np.sign(truth.loc[truth.ne(0)]))
        .mean(),
        np.nan,
    ]

    size_names = ["average_size", *sizes]
    size_validation = pd.DataFrame(
        {
            "model": size_names,
            "rmse_rub": (
                validation[size_names].sub(valid_delta.abs(), axis=0).pow(2).mean() ** 0.5
            ).to_numpy(),
        }
    )
    best_size = size_validation.loc[size_validation.rmse_rub.idxmin(), "model"]
    size_reference = test.assign(actual=truth.abs(), ensemble=forecasts.average_size)
    size_metrics = summarize(size_reference, dict(forecasts[size_names].items()))
    p = forecasts[best_sign]
    point = {
        "direct": baseline,
        "same_size_both_directions": test["last"]
        + (forecasts.realized * (2 * p - 1)).where(test.new_fixing, 0),
        "selected_size_both_directions": test["last"]
        + (forecasts[best_size] * (2 * p - 1)).where(test.new_fixing, 0),
        "size_depends_on_direction": test["last"]
        + (1 - forecasts.flat_probability)
        * (p * forecasts.size_if_up - (1 - p) * forecasts.size_if_down),
    }
    point_metrics = summarize(test.assign(ensemble=baseline), point)
    forecast_table = forecasts.join(pd.DataFrame(point)).join(test[["last", "actual"]])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "sign_validation": validation_sign,
        "sign_test": sign_metrics,
        "size_validation": size_validation,
        "size_test": size_metrics,
        "course_test": point_metrics,
    }.items():
        write_csv(frame, args.output / f"{name}.csv")
        print(name, frame.round(5).to_string(index=False), sep="\n", flush=True)
    write_csv(
        forecast_table.rename_axis("origin").reset_index(), args.output / "predictions.csv.gz"
    )
    meta = {
        "sign_model": best_sign,
        "size_model": best_size,
        "validation_dates": len(validation),
        "test_dates": len(test),
        "unchanged_test_dates": int(truth.eq(0).sum()),
        "inputs": input_fingerprints([args.dataset, manifest, *archives]),
    }
    (args.output / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/datasets/usd_rub_2011_2026_full_si_options.csv.gz"),
    )
    parser.add_argument("--archive", type=Path, default=Path("data/reports/forecast_stacking"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/weekly_sign_size"))
    with threadpool_limits(limits=2):
        run(parser.parse_args())
