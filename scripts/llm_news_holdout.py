"""Проверить, улучшают ли новостные события прогноз действующего ансамбля."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from label_llm_news import CHANNELS, STAGES, save
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from usd_rub.backtest import score
from usd_rub.ml_dataset import clean_dataset, feature_panels, read_dataset
from usd_rub.ml_experiment import (
    actual_at,
    candidates,
    fit_predict,
    interval_from_validation,
    prediction_rows,
    relative_rmse,
    selection_loss,
    specific_features,
    to_levels,
)
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints

FIRST_OOF = 2017
SHRINKS = (0, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1)
GROUPS = ("lexical", "llm", "combined")


def reference(args, dataset, config):
    """Сверить ансамбль, даты и курсы с предыдущим завершённым экспериментом."""
    report_path = args.reference.parent / "run.json"
    report = json.loads(report_path.read_text())
    fingerprints = input_fingerprints([args.dataset, args.config])
    if not {entry["sha256"] for entry in fingerprints} <= {
        entry["sha256"] for entry in report["inputs"]
    }:
        raise ValueError("Ансамбль рассчитан на другом снимке или конфигурации")
    dates = pd.date_range(config["start"], config["end"])
    if not dataset.index.equals(dates):
        raise ValueError("Снимок и конфигурация задают разные календарные даты")
    split = dates.searchsorted(config["test_start"])
    horizons = np.arange(1, config["horizon"] + 1)
    origins = np.arange(split, len(dates) - horizons[-1])
    old = pd.read_csv(args.reference, parse_dates=["origin", "target_date"])
    ensemble = old.loc[old.model.eq("ensemble")].sort_values(["origin", "horizon"])
    values = dataset.usd_rub.to_numpy()
    query = origins[:, None] + horizons[None, :]
    np.testing.assert_array_equal(ensemble.origin, np.repeat(dates[origins], len(horizons)))
    np.testing.assert_array_equal(ensemble.horizon, np.tile(horizons, len(origins)))
    np.testing.assert_array_equal(ensemble.target_date, dates.to_numpy()[query].ravel())
    np.testing.assert_allclose(ensemble.actual, values[query].ravel(), rtol=1e-12)
    np.testing.assert_allclose(
        ensemble["last"], np.repeat(values[origins], len(horizons)), rtol=1e-12
    )
    if not np.isfinite(ensemble.prediction).all():
        raise ValueError("В ансамбле есть неконечные прогнозы")
    return report, ensemble, horizons, origins


def build_oof(args, dataset, config, previous, horizons):
    """Переобучить прежние компоненты на каждом годовом срезе."""
    csv = args.output / "baseline_oof.csv.gz"
    manifest = args.output / "baseline_oof.json"
    inputs = input_fingerprints(
        [
            args.dataset,
            args.config,
            args.reference.parent / "run.json",
            Path(fit_predict.__code__.co_filename),
        ]
    )
    if csv.exists() and manifest.exists():
        if json.loads(manifest.read_text())["inputs"] == inputs:
            return pd.read_csv(csv, parse_dates=["origin", "target_date"])
    target = dataset.usd_rub
    levels = target.to_numpy()
    dates = target.index
    test_year = pd.Timestamp(config["test_start"]).year
    coverage = config.get("min_hit_rate", 0.5)
    panels = feature_panels(clean_dataset(dataset, coverage))
    lookup = {entry.name: entry for entry in candidates()}
    val_settings = pd.read_csv(args.reference.parent / "validation_candidates.csv").set_index(
        "name"
    )
    weights = {name: w for name, w in previous["ensemble_weights"].items() if w > 1e-12}
    models = {name: lookup[name] for name in weights if name != "rw"}
    matrices = {
        name: specific_features(panels, model.setup).to_numpy() for name, model in models.items()
    }
    rows = []
    for year in range(FIRST_OOF, test_year):
        start = dates.searchsorted(f"{year}-01-01")
        end = dates.searchsorted(f"{year + 1}-01-01")
        origins = np.arange(start, end - horizons[-1])
        last = levels[origins]
        prediction = np.repeat(last[:, None], len(horizons), axis=1) * weights.get("rw", 0)
        for name, model in models.items():
            correction, _ = fit_predict(
                matrices[name], levels, model, start, origins, horizons, min_hit_rate=coverage
            )
            prediction += weights[name] * to_levels(
                correction, last, val_settings.loc[name, "shrink"]
            )
        nan_interval = np.full_like(prediction, np.nan)
        window = config["baseline_window"]
        scale = float(np.abs(np.diff(levels[start - window : start])).mean())
        rows.append(
            prediction_rows(
                "ensemble_oof",
                target,
                origins,
                horizons,
                prediction,
                nan_interval,
                nan_interval,
                scale,
                start,
            )
        )
        print(f"{year}: {len(origins)} прогнозов вне обучающего окна", flush=True)
    result = pd.concat(rows, ignore_index=True)
    write_csv(result, csv)
    save(
        manifest,
        {
            "inputs": inputs,
            "weights": weights,
            "first_year": FIRST_OOF,
            "last_year": test_year - 1,
            "rows": len(result),
        },
    )
    return result


def event_features(root):
    """Свести события завершённых московских суток в признаки для следующей даты."""
    run = json.loads((root / "label_run.json").read_text())
    days = pd.date_range(run["start"], run["end"])
    blocks = {}
    counts = []
    for day in days:
        path = root / "days" / f"{day.date()}.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        item = {
            "publication_date": day,
            "status": record.get("status", "missing"),
            "titles": len(record.get("selection", [])),
            "events": len(record.get("events", [])),
        }
        counts.append(item)
        if item["status"] != "ok":
            continue
        for group in ("lexical", "llm"):
            events = (
                record["events"]
                if group == "llm"
                else [
                    {"channel": channel, "stage": title["stage"]}
                    for title in record["selection"]
                    for channel in title["channels"]
                ]
            )
            row = {
                f"{channel}_{stage}": 0 for channel, stage in itertools.product(CHANNELS, STAGES)
            }
            for event in events:
                row[f"{event['channel']}_{event['stage']}"] += 1
            blocks.setdefault(group, {})[day] = row
    columns = {}
    for group, rows in blocks.items():
        frame = pd.DataFrame.from_dict(rows, orient="index").reindex(days)
        for window in (1, 3, 7):
            for name, values in frame.items():
                columns[f"news__{group}_event_{name}_{window}d"] = (
                    values if window == 1 else values.rolling(window).sum()
                )
    features = pd.DataFrame(columns, index=days).rename_axis("date")
    features.index = features.index + pd.Timedelta(days=1)
    write_csv(pd.DataFrame(counts), root / "event_coverage.csv")
    write_csv(features, root / "daily_event_features.csv.gz", index=True)
    return features


def correction(frame, train, dates, baseline, subset, method, regularization):
    """Обучить поправку к сохранённым вневыборочным ошибкам ансамбля."""
    methods = {
        "ridge": lambda: Ridge(alpha=regularization * len(train), fit_intercept=False),
        "histboost": lambda: HistGradientBoostingRegressor(
            max_iter=65,
            learning_rate=0.04,
            min_samples_leaf=45,
            max_leaf_nodes=7,
            l2_regularization=regularization,
            early_stopping=False,
            random_state=191,
        ),
    }
    result = np.zeros_like(baseline, dtype=float)
    is_known = frame.loc[dates, subset].notna().all(axis=1)
    query = dates[is_known]
    if query.empty:
        return result
    for horizon, part in train.groupby("horizon", sort=True):
        available = frame.reindex(part.origin)[subset].notna().all(axis=1).to_numpy()
        part = part.loc[available]
        if len(part) < 180:
            raise ValueError("Мало вневыборочной истории для обучения поправки")
        weights = (part["last"].to_numpy() / part["last"].median()) ** 2
        x = np.log1p(frame.loc[part.origin, subset].to_numpy())
        q = np.log1p(frame.loc[query, subset].to_numpy())
        center, spread = x.mean(axis=0), x.std(axis=0)
        spread[spread < 1e-10] = 1
        y = ((part.actual - part.prediction) / part["last"]).to_numpy()
        model = methods[method]()
        model.fit(((x - center) / spread).clip(-8, 8), y, sample_weight=weights)
        result[is_known.to_numpy(), int(horizon) - 1] = model.predict(
            ((q - center) / spread).clip(-8, 8)
        )
    return result


def bootstrap(baseline, challenger, actual, last):
    """Оценить разницу нормированного RMSE на общих календарных блоках."""
    rng = np.random.default_rng(191)
    n = len(last)
    start = rng.integers(n, size=(2000, (n + 27) // 28))
    positions = ((start[:, :, None] + np.arange(28)) % n).reshape(2000, -1)[:, :n]
    delta = np.sqrt(((baseline - actual) ** 2)[positions].mean(axis=1)) - np.sqrt(
        ((challenger - actual) ** 2)[positions].mean(axis=1)
    )
    normalizer = np.sqrt(((last[:, None] - actual) ** 2)[positions].mean(axis=1))
    return np.quantile(100 * (delta / normalizer).mean(axis=1), [0.025, 0.975]).tolist()


def compare(args):
    """Выбрать поправки на валидации и измерить эффект на holdout."""
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    dataset = read_dataset(args.dataset)
    previous, incumbent, horizons, origins = reference(args, dataset, config)
    oof = build_oof(args, dataset, config, previous, horizons)
    if args.prepare_baseline:
        return

    if args.features:
        features = read_dataset(args.features)
        feature_inputs = [args.features]
    else:
        features = event_features(args.output)
        feature_inputs = [
            args.output / "label_run.json",
            args.output / "daily_event_features.csv.gz",
        ]
    expected = {
        f"news__{group}_event_{channel}_{stage}_{window}d"
        for group, channel, stage, window in itertools.product(
            ("lexical", "llm"), CHANNELS, STAGES, (1, 3, 7)
        )
    }
    if features.empty or set(features.columns) != expected or not features.index.is_unique:
        raise ValueError("Ожидаются уникальные дневные даты и полный набор счётчиков событий")
    features = features.reindex(dataset.index)
    subset = {
        group: [col for col in features if group == "combined" or col.startswith(f"news__{group}_")]
        for group in GROUPS
    }
    levels = dataset.usd_rub.to_numpy()
    splits, folds = [], []
    for year in config["validation_years"]:
        valid = oof.loc[oof.origin.dt.year.eq(year)]
        cutoff = pd.Timestamp(year, 1, 1)
        dates = pd.DatetimeIndex(valid.origin.unique())
        base = valid.prediction.to_numpy().reshape(-1, len(horizons))
        train = oof.loc[oof.target_date.lt(cutoff)]
        origin_positions = dataset.index.get_indexer(dates)
        last = levels[origin_positions]
        splits.append((origin_positions, actual_at(levels, origin_positions, horizons), last))
        folds.append((train, dates, base, last))
    incumbent_cv = selection_loss([fold[2] for fold in folds], splits)
    validation, ranking = {}, []
    for group, method, parameter in itertools.product(GROUPS, ("ridge", "histboost"), (0.05, 0.5)):
        raw = [
            base
            + last[:, None]
            * correction(features, train, dates, base, subset[group], method, parameter)
            for train, dates, base, last in folds
        ]
        key = f"{group}_{method}_{parameter:g}"
        loss, shrink = min(
            (
                selection_loss(
                    [fold[2] + k * (pred - fold[2]) for fold, pred in zip(folds, raw, strict=True)],
                    splits,
                ),
                k,
            )
            for k in SHRINKS
        )
        validation[key] = [
            fold[2] + shrink * (pred - fold[2]) for fold, pred in zip(folds, raw, strict=True)
        ]
        ranking.append(
            {
                "name": key,
                "features": group,
                "method": method,
                "regularization": parameter,
                "shrink": shrink,
                "cv_rmse_to_rw": loss,
                "cv_rmse_to_rw_raw": selection_loss(raw, splits),
            }
        )
        print(f"{key}: CV {loss:.6f}, усадка {shrink:g}", flush=True)
    table = pd.DataFrame(ranking).sort_values(["cv_rmse_to_rw", "cv_rmse_to_rw_raw", "name"])
    write_csv(table, args.output / "validation_candidates.csv")
    selected = table.drop_duplicates("features")
    winner = (
        selected.iloc[0]["name"] if selected.iloc[0]["cv_rmse_to_rw"] < incumbent_cv else "ensemble"
    )
    save(
        args.output / "selection.json",
        {
            "incumbent_cv": incumbent_cv,
            "champion": winner,
            "selected": selected.to_dict("records"),
        },
    )

    base = incumbent.prediction.to_numpy().reshape(-1, len(horizons))
    dates = dataset.index[origins]
    actual, last = actual_at(levels, origins, horizons), levels[origins]
    train = oof.loc[oof.target_date.lt(config["test_start"])]
    train_size = train.origin.nunique()
    scale = incumbent.mase_scale.iloc[0]
    bounds = np.full_like(base, np.nan)
    records = [
        incumbent,
        prediction_rows(
            "rw",
            dataset.usd_rub,
            origins,
            horizons,
            np.repeat(last[:, None], len(horizons), axis=1),
            bounds,
            bounds,
            scale,
            train_size,
        ),
    ]
    comparisons = []
    for row in selected.itertuples(index=False):
        raw = correction(
            features, train, dates, base, subset[row.features], row.method, row.regularization
        )
        prediction = base + row.shrink * raw * last[:, None]
        lower, upper = interval_from_validation(prediction, last, validation[row.name], splits)
        records.append(
            prediction_rows(
                row.name,
                dataset.usd_rub,
                origins,
                horizons,
                prediction,
                lower,
                upper,
                scale,
                train_size,
            )
        )
        gain = 100 * np.mean(
            relative_rmse(base, actual, last) - relative_rmse(prediction, actual, last)
        )
        lo, hi = bootstrap(base, prediction, actual, last)
        raw_gain = 100 * np.mean(
            relative_rmse(base, actual, last)
            - relative_rmse(base + raw * last[:, None], actual, last)
        )
        comparisons.append(
            {
                "model": row.name,
                "gain_pp": gain,
                "ci_low_pp": lo,
                "ci_high_pp": hi,
                "gain_unshrunk_pp": raw_gain,
            }
        )
    predictions = pd.concat(records, ignore_index=True)
    metrics = score(predictions)
    summary = (
        metrics.assign(rmse_to_rw=1 - metrics.rmse_skill_vs_rw)
        .groupby("model")[["rmse_to_rw", "mape_pct"]]
        .mean()
    )
    for h in (1, horizons[-1]):
        summary[f"rmse_{h}d"] = metrics.loc[metrics.horizon.eq(h)].set_index("model").rmse
    summary = summary.join(pd.DataFrame(comparisons).set_index("model")).sort_values("rmse_to_rw")
    write_csv(predictions, args.output / "predictions.csv.gz")
    write_csv(metrics, args.output / "metrics.csv")
    write_csv(summary, args.output / "summary.csv", index=True)
    save(
        args.output / "run.json",
        {
            "champion": winner,
            "incumbent_cv": incumbent_cv,
            "shrink_grid": SHRINKS,
            "test_origins": len(origins),
            "test_usable_days": int(features.loc[dates].notna().all(axis=1).sum()),
            "inputs": input_fingerprints(
                [
                    args.config,
                    args.dataset,
                    args.reference,
                    *feature_inputs,
                ]
            ),
        },
    )
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/usd_rub_2011_2026_full.csv.gz")
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/reports/ml_holdout_minfin_rosstat/with_file/predictions.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/reports/llm_news_holdout"))
    parser.add_argument("--features", type=Path, help="Готовый дневной снимок признаков")
    parser.add_argument("--prepare-baseline", action="store_true")
    with threadpool_limits(limits=2):
        compare(parser.parse_args())
