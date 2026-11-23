"""Сократить набор признаков по TreeSHAP и gain, заново обучить модели USD/RUB."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from export_ml_dataset import write_dataset
from forecast_extensions import load_inputs
from forecast_stacking import panel_index
from foundation_ensemble import CLASSICAL as ARCHIVED_CLASSICAL
from threadpoolctl import threadpool_limits
from xgboost import DMatrix, XGBRegressor

from usd_rub.ensemble_forecast import CLASSICAL as PRODUCTION
from usd_rub.fixing_calendar import HORIZONS
from usd_rub.ml_dataset import EXTENDED_PREFIX, FEATURE_GROUPS, feature_panels, read_dataset
from usd_rub.ml_experiment import (
    Candidate,
    fit_predict,
    relative_rmse,
    specific_features,
    to_levels,
)
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints

SELECTIONS = {
    "full": ("combined", None),
    "active": ("combined", None),
    "shap32": ("shap", 32),
    "gain32": ("gain", 32),
    "combined16": ("combined", 16),
    "combined32": ("combined", 32),
    "combined64": ("combined", 64),
    "combined128": ("combined", 128),
}
MODELS = {
    "ridge": Candidate("ridge", "extended", 1460, 0.15),
    "histboost": Candidate("histboost", "extended", 1460, 3, pooled=True),
    "extra_trees": Candidate("extra_trees", "extended", 1460, 3, pooled=True),
    "catboost": Candidate("catboost", "extended", 1460, 3, pooled=True),
}
MARKET_BUDGETS = {"small": (8, 16), "medium": (12, 32), "large": (16, 48)}
WINDOW = 1460
MIN_COVERAGE = 0.5
RECENT_WINDOW = 120
MIN_RECENT_COVERAGE = 0.2


def score_row(prediction, actual, last):
    """Посчитать ошибки в рублях и нормированный RMSE."""
    error = prediction - actual
    rmse = np.sqrt(np.square(error).mean(axis=0))
    mae = np.abs(error).mean(axis=0)
    ratios = relative_rmse(prediction, actual, last)
    return {
        "rmse_to_rw": float(ratios.mean()),
        "rmse7_to_rw": float(ratios[-1]),
        **{f"rmse_{h}d": rmse[h - 1] for h in HORIZONS},
        **{f"mae_{h}d": mae[h - 1] for h in HORIZONS},
    }


def tree_importance(train, target, probe, weights):
    """Вычислить средний модуль TreeSHAP и полный gain XGBoost."""
    features = list(train)
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=18,
        max_bin=64,
        subsample=0.9,
        colsample_bytree=0.7,
        reg_lambda=12,
        tree_method="hist",
        n_jobs=2,
        random_state=191,
    )
    model.fit(train, target, sample_weight=weights)
    booster = model.get_booster()
    contrib = booster.predict(DMatrix(probe, nthread=2), pred_contribs=True)
    np.testing.assert_allclose(contrib.sum(axis=1), model.predict(probe), atol=2e-5, rtol=1e-5)
    return pd.DataFrame(
        {
            "shap": np.abs(contrib[:, :-1]).mean(axis=0),
            "gain": pd.Series(booster.get_score(importance_type="total_gain"), dtype=float).reindex(
                features, fill_value=0
            ),
        },
        index=features,
    )


def fuse_importance(scores):
    """Объединить относительную важность двух горизонтов и двух методов."""
    merged = sum(
        weight * score.div(score.sum().replace(0, 1))
        for score, weight in zip(scores, (0.25, 0.75), strict=True)
    )
    merged["combined"] = merged[["shap", "gain"]].mean(axis=1)
    return merged.sort_values("combined", ascending=False, kind="stable").rename_axis("feature")


def recently_observed(panel, cutoff):
    """Проверить, что признак продолжал поступать перед фолдом."""
    return (
        panel.iloc[max(0, cutoff - RECENT_WINDOW) : cutoff].notna().mean().ge(MIN_RECENT_COVERAGE)
    )


def rank(panel, levels, cutoff):
    """Обучить ранжирование до внутреннего 180-дневного окна."""
    first = max(cutoff - WINDOW, 0)
    inner = cutoff - 180
    train = np.arange(first, inner - 7)
    probe = np.arange(inner, cutoff - 7)
    if len(train) < 180:
        raise ValueError("Для ранжирования не хватает обучающей истории")
    sample = panel.iloc[train]
    eligible = sample.notna().mean().ge(MIN_COVERAGE) & sample.nunique().gt(1)
    eligible &= recently_observed(panel, cutoff)
    sample = sample.loc[:, eligible]
    sample = sample.loc[
        :, panel.iloc[first : cutoff - 1][sample.columns].notna().mean().ge(MIN_COVERAGE)
    ]
    probe = panel.iloc[probe][sample.columns]
    weights = (levels[train] / np.median(levels[train])) ** 2
    scores = [
        tree_importance(sample, 100 * (levels[train + h] / levels[train] - 1), probe, weights)
        for h in (1, 7)
    ]
    return fuse_importance(scores)


def dataset_inputs(args):
    """Выделить расширенную панель и привязать внутридневной снимок по датам."""
    dataset, calibration, test, _, paths = load_inputs(args)
    base = feature_panels(dataset)
    micro = read_dataset(args.intraday)
    pd.testing.assert_index_equal(dataset.index, micro.index)
    micro = micro.rename(columns=lambda name: "fx__intraday_" + name.replace("__", "_"))
    panel = base["extended"].join(micro).replace([np.inf, -np.inf], np.nan)
    return dataset, panel, base, calibration, test, [*paths, args.intraday]


def production_predictions(base, levels, cutoff, origins, archive, rows):
    """Переоценить оригинальные компоненты и их фиксированную смесь."""
    components = {}
    totals = {name: np.zeros_like(archive.actual[rows]) for name in ("full", *MARKET_BUDGETS)}
    selections = {name: {} for name in MARKET_BUDGETS}
    for candidate, weight, shrink in PRODUCTION:
        features = specific_features(base, candidate.setup)
        features = features.loc[:, ~features.columns.str.contains("__theory_", regex=False)]
        supported = features.iloc[:cutoff].notna().mean().ge(MIN_COVERAGE)
        features = features.loc[:, supported]
        ranking = rank(features, levels, cutoff).combined
        panels = {"full": features.columns}
        for label, budgets in MARKET_BUDGETS.items():
            columns = ranking[ranking.gt(0)].head(budgets[int(candidate.model != "ridge")]).index
            panels[label] = columns
            selections[label][candidate.model] = columns.tolist()
        for label, columns in panels.items():
            raw, indices = fit_predict(
                features[columns].to_numpy(dtype=float),
                levels,
                candidate,
                cutoff,
                origins,
                HORIZONS,
                min_hit_rate=MIN_COVERAGE,
            )
            value = to_levels(raw, levels[origins], shrink)
            suffix = "" if label == "full" else f"/{label}"
            components[f"production_{candidate.model}{suffix}"] = (
                archive.project(value, rows),
                len(indices),
            )
            totals[label] += weight * value
    combined = archive.project(totals.pop("full"), rows)
    np.testing.assert_allclose(
        combined, archive.projected[ARCHIVED_CLASSICAL][rows], atol=2e-6, rtol=1e-8
    )
    components["production_classical"] = (combined, None)
    for label, values in totals.items():
        count = len(set().union(*map(set, selections[label].values())))
        components[f"production_classical/{label}"] = (archive.project(values, rows), count)
    return components, selections


def fit_fold(dataset, panel, base, archive, rows, ranking):
    """Переобучить сопоставимые полные и сокращённые модели."""
    dates = archive.dates[rows]
    origins = dataset.index.get_indexer(dates)
    cutoff = origins[0]
    levels = dataset.usd_rub.to_numpy()
    fitted, production_columns = production_predictions(
        base, levels, cutoff, origins, archive, rows
    )
    for selection, (method, limit) in SELECTIONS.items():
        if limit is None:
            sample = panel.iloc[max(0, cutoff - WINDOW) : cutoff - 1]
            keep = sample.notna().mean().ge(MIN_COVERAGE) & sample.nunique().gt(1)
            if selection == "active":
                keep &= recently_observed(panel, cutoff)
            features = panel.columns[keep]
        else:
            scores = ranking[method].sort_values(ascending=False, kind="stable")
            features = scores[scores.gt(0)].head(limit).index
        x = panel.loc[:, features].to_numpy(dtype=float)
        for label, candidate in MODELS.items():
            raw, used = fit_predict(
                x,
                levels,
                candidate,
                cutoff,
                origins,
                HORIZONS,
                min_hit_rate=MIN_COVERAGE,
            )
            name = f"{label}/{selection}"
            fitted[name] = (archive.project(to_levels(raw, levels[origins]), rows), len(used))
        print(dates[0].date(), selection, len(features), flush=True)
    return fitted, production_columns


def export_panel(args, dataset, panel, ranking, selected, market):
    """Записать признаки, выбранные по валидации, и исходные цели."""
    selected_columns = {}
    for model, selection in selected.items():
        method, limit = SELECTIONS[selection]
        scores = ranking[method].sort_values(ascending=False, kind="stable")
        selected_columns[model] = scores[scores.gt(0)].head(limit).index.tolist()
    features = sorted(set().union(*map(set, [*selected_columns.values(), *market.values()])))
    context = [
        name
        for name in dataset
        if name.removeprefix(EXTENDED_PREFIX).split("__", 1)[0] not in FEATURE_GROUPS
    ]
    values = {}
    for name in features:
        if name in dataset:
            values[name] = dataset[name]
        key = f"{EXTENDED_PREFIX}{name}" if f"{EXTENDED_PREFIX}{name}" in dataset else name
        values[key] = panel[name]
    write_dataset(dataset[context].join(pd.DataFrame(values, index=dataset.index)), args.reduced)
    return features, selected_columns


def run(args):
    """Сохранить рейтинг, сокращённый датасет и новые вневыборочные метрики."""
    dataset, panel, base, calibration, test, paths = dataset_inputs(args)
    results, importance, history = [], [], {}
    folds = [
        (year, calibration, calibration.dates.year == year)
        for year in sorted(set(calibration.dates.year))
    ] + [("test", test, np.ones(len(test.dates), dtype=bool))]
    for label, archive, rows in folds:
        cutoff = dataset.index.get_loc(archive.dates[rows][0])
        ranking = rank(panel, dataset.usd_rub.to_numpy(), cutoff)
        importance.append(ranking.reset_index().assign(fold=label))
        fitted, production_columns = fit_fold(dataset, panel, base, archive, rows, ranking)
        fitted["random_walk"] = (
            np.broadcast_to(archive.last[rows, None], archive.actual[rows].shape),
            0,
        )
        for name, (prediction, count) in fitted.items():
            results.append(
                {
                    "fold": label,
                    "model": name,
                    "features_used": count,
                    **score_row(prediction, archive.actual[rows], archive.last[rows]),
                }
            )
            history.setdefault(name, []).append(prediction.ravel())
    metrics = pd.DataFrame(results)
    cv = (
        metrics.loc[metrics.fold.ne("test")]
        .groupby("model")[["rmse_to_rw", "rmse7_to_rw"]]
        .mean()
        .rename(columns=lambda name: "cv_" + name)
    )
    summary = metrics.loc[metrics.fold.eq("test")].drop(columns="fold").join(cv, on="model")
    summary = summary.sort_values(["cv_rmse7_to_rw", "cv_rmse_to_rw"])
    selected = {
        model: summary.loc[
            summary.model.str.startswith(f"{model}/")
            & ~summary.model.isin([f"{model}/full", f"{model}/active"])
        ]
        .iloc[0]
        .model.split("/")[1]
        for model in MODELS
    }
    args.output.mkdir(parents=True, exist_ok=True)
    selected_market = (
        summary.loc[summary.model.str.startswith("production_classical/")].iloc[0].model
    )
    market = production_columns[selected_market.split("/")[1]]
    columns, fit_columns = export_panel(args, dataset, panel, ranking, selected, market)
    write_csv(metrics, args.output / "metrics_by_fold.csv")
    write_csv(summary, args.output / "summary.csv")
    write_csv(pd.concat(importance, ignore_index=True), args.output / "importance.csv")
    index = panel_index(calibration.dates.append(test.dates))
    predictions = pd.DataFrame(
        {name: np.concatenate(values) for name, values in history.items()}, index=index
    )
    write_csv(predictions.reset_index(), args.output / "predictions.csv.gz")
    manifest = {
        "inputs": input_fingerprints([*paths, Path(__file__)]),
        "available_features": panel.shape[1],
        "eligible_final_ranking": len(ranking),
        "selected_features": columns,
        "selected_by_model": selected,
        "selected_features_by_model": fit_columns,
        "selected_market_mixture": selected_market,
        "selected_market_features": market,
        "threshold_coverage": MIN_COVERAGE,
        "recent_coverage_threshold": MIN_RECENT_COVERAGE,
        "recent_days": RECENT_WINDOW,
        "loss_for_selection": "mean validation RMSE at 7 days divided by the random walk RMSE",
        "reduced_snapshot": input_fingerprints([args.reduced])[0],
    }
    (args.output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        summary[
            ["model", "features_used", "cv_rmse7_to_rw", "rmse_1d", "rmse_7d", "rmse_to_rw"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/reports/theory_holdout/dataset.csv.gz")
    )
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument("--archive", type=Path, default=Path("data/reports/forecast_stacking"))
    parser.add_argument(
        "--intraday", type=Path, default=Path("datasets/usd_rub_2011_2026_fx_candles.csv.gz")
    )
    parser.add_argument("--output", type=Path, default=Path("data/reports/shap_holdout"))
    parser.add_argument(
        "--reduced", type=Path, default=Path("datasets/usd_rub_2011_2026_shap.csv.gz")
    )
    with threadpool_limits(limits=2):
        run(parser.parse_args())
