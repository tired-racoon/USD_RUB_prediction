"""Проверить макродифференциалы и монотонный XGBoost на USD/RUB."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from export_ml_dataset import write_dataset
from forecast_extensions import annual_folds, load_inputs
from forecast_stacking import HORIZONS, panel_index
from foundation_ensemble import CLASSICAL, RW
from llm_news_holdout import bootstrap

from usd_rub.ensemble_forecast import blend_predictions
from usd_rub.market_correction import MARKET_FEATURES, XGB_PARAMS, fit_corrections, predict_levels
from usd_rub.ml_dataset import feature_panels
from usd_rub.ml_experiment import relative_rmse, selection_loss
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints
from usd_rub.theory_features import (
    FRED,
    FUNDAMENTAL_SIGNS,
    FUNDING,
    LAGS,
    MACRO_CORRECTION_FEATURES,
    read_theory_sources,
    theory_features,
)

PRIMARY = "ensemble_t0_gamma"
GAMMA = "t0_gamma/market_calendar"
FRACTIONS = (0.0, 0.25, 0.5, 1.0)


def read_sources(args, dates):
    """Кэшировать FRED отдельно от рабочего набора источников."""
    return read_theory_sources(args.data_dir, args.output / "fred", dates, frozen=True)


def predictor_sets(panel, fundamental, market):
    """Оставить одинаковые наборы столбцов в паре с ограничениями и без них."""
    control = panel[list(MARKET_FEATURES)]
    global_fx = control.join(market).join(panel[["news__gpr_daily_shock_30d"]])
    all_features = global_fx.join(fundamental)[list(MACRO_CORRECTION_FEATURES)]
    return {
        "market": (control, {}),
        "fundamentals": (control.join(fundamental), {}),
        "fundamentals_monotone": (control.join(fundamental), FUNDAMENTAL_SIGNS),
        "global_fx": (global_fx, {}),
        "all": (all_features, {}),
        "all_monotone": (all_features, FUNDAMENTAL_SIGNS),
        "all_monotone_funding": (all_features, FUNDAMENTAL_SIGNS | {FUNDING: 1}),
    }


def references(archive):
    """Восстановить основной ансамбль и случайное блуждание на общих датах."""
    return {
        PRIMARY: blend_predictions(
            archive.raw[CLASSICAL], archive.raw[GAMMA], archive.last, archive.fixes
        ),
        RW: archive.projected[RW],
    }


def fit_fold(panel, signs, calibration, train, train_ref, query, rows, query_ref):
    """Предсказать логарифмическое изменение к заданному якорю по каждому горизонту."""
    x = panel.loc[calibration.dates[train]]
    q = panel.loc[query.dates[rows]]
    models = fit_corrections(
        x, calibration.actual[train], train_ref[train], calibration.last[train], signs
    )
    return predict_levels(models, q, query_ref[rows], query.last[rows])


def period_metrics(prediction, actual, last):
    """Посчитать общую ошибку и RMSE в рублях на 1 и 7 дней."""
    errors = np.sqrt(np.mean((prediction - actual) ** 2, axis=0))
    return {
        "rmse_to_rw": relative_rmse(prediction, actual, last).mean(),
        "rmse_1d": errors[0],
        "rmse_7d": errors[-1],
    }


def evaluate(calibration, test, designs):
    """Настроить доли на 2022–2024 и сравнить их на 680 общих датах."""
    _, folds = annual_folds(calibration)
    splits = [calibration.split(check) for _, check in folds]
    cal_ref, test_ref = references(calibration), references(test)
    scores, by_year = [], []
    holdout = {**test_ref}
    selected = []
    for label, (panel, signs) in designs.items():
        for mode, reference in (("direct", RW), ("residual", PRIMARY)):
            name = f"{mode}_{label}"
            anchor, baseline = cal_ref[reference], test_ref[reference]
            estimates = [
                fit_fold(panel, signs, calibration, train, anchor, calibration, check, anchor)
                for train, check in folds
            ]
            validation = {
                fraction: [
                    calibration.project(
                        anchor[check] + fraction * (prediction - anchor[check]), check
                    )
                    for (_, check), prediction in zip(folds, estimates, strict=True)
                ]
                for fraction in FRACTIONS
            }
            losses = {
                fraction: selection_loss(predictions, splits)
                for fraction, predictions in validation.items()
            }
            fraction = min(FRACTIONS, key=losses.get)
            scores.extend(
                dict(
                    model=name,
                    fraction=value,
                    validation_rmse_to_rw=loss,
                    selected=value == fraction,
                )
                for value, loss in losses.items()
            )
            by_year.extend(
                dict(
                    model=name,
                    fraction=value,
                    year=dates[0].year,
                    **period_metrics(prediction, actual, last),
                )
                for value, predictions in validation.items()
                for prediction, (dates, actual, last) in zip(predictions, splits, strict=True)
            )
            raw = fit_fold(
                panel,
                signs,
                calibration,
                np.ones(len(calibration.dates), dtype=bool),
                anchor,
                test,
                slice(None),
                baseline,
            )
            predicted = test.project(baseline + fraction * (raw - baseline))
            holdout[name] = predicted
            holdout[f"{name}_fixed25"] = test.project(baseline + 0.25 * (raw - baseline))
            holdout[f"{name}_raw"] = test.project(raw)
            selected.append(
                dict(
                    model=name,
                    reference=reference,
                    fraction=fraction,
                    validation_rmse_to_rw=losses[fraction],
                    validation_raw_rmse_to_rw=losses[1.0],
                    test_raw_rmse_to_rw=relative_rmse(
                        holdout[f"{name}_raw"], test.actual, test.last
                    ).mean(),
                    test_fixed25_rmse_to_rw=relative_rmse(
                        holdout[f"{name}_fixed25"], test.actual, test.last
                    ).mean(),
                )
            )
        print(f"XGBoost: {label}", flush=True)

    control = [
        dict(
            model=key,
            reference=key,
            fraction=0.0,
            validation_rmse_to_rw=selection_loss(
                [cal_ref[key][check] for _, check in folds], splits
            ),
        )
        for key in (PRIMARY, RW)
    ]
    for row in [*control, *selected]:
        result = period_metrics(holdout[row["model"]], test.actual, test.last)
        row.update(test_rmse_to_rw=result.pop("rmse_to_rw"), **result)
    return pd.DataFrame(scores), pd.DataFrame(by_year), pd.DataFrame([*control, *selected]), holdout


def paired(holdout, test):
    """Сравнить группы входов и ограничения при одинаковых долях поправки."""
    contrasts = (
        ("fundamentals", "market"),
        ("global_fx", "market"),
        ("all", "global_fx"),
        ("fundamentals_monotone", "fundamentals"),
        ("all_monotone", "all"),
        ("all_monotone", "fundamentals_monotone"),
        ("all_monotone_funding", "all_monotone"),
    )
    cases = [
        (f"{mode}_{name}_{amount}", f"{mode}_{reference}_{amount}")
        for mode, amount in (("direct", "raw"), ("residual", "raw"), ("residual", "fixed25"))
        for name, reference in contrasts
    ]
    cases += [(name, PRIMARY) for name in holdout if name.startswith("residual_")]
    rows = []
    for candidate, reference in cases:
        low, high = bootstrap(holdout[reference], holdout[candidate], test.actual, test.last)
        gain = (
            relative_rmse(holdout[reference], test.actual, test.last).mean()
            - relative_rmse(holdout[candidate], test.actual, test.last).mean()
        )
        rows.append(
            dict(
                candidate=candidate,
                reference=reference,
                gain_pp=100 * gain,
                gain_ci_low_pp=low,
                gain_ci_high_pp=high,
            )
        )
    return pd.DataFrame(rows)


def run(args):
    """Выпустить дополненный снимок и парное сравнение прогнозов."""
    dataset, calibration, test, _, paths = load_inputs(args)
    config = json.loads(args.config.read_text())
    sources, source_paths = read_sources(args, dataset.index)
    panel = feature_panels(dataset)["extended"]
    fundamental, market = theory_features(panel, dataset.usd_rub, sources, config["assumed_lags"])
    new_features = fundamental.join(market)
    output_dataset = args.output / "dataset.csv.gz"
    write_dataset(dataset.join(new_features), output_dataset)
    coverage = pd.DataFrame(
        {
            "calibration": new_features.loc[calibration.dates].notna().mean(),
            "test": new_features.loc[test.dates].notna().mean(),
        }
    ).rename_axis("feature")
    designs = predictor_sets(panel, fundamental, market)
    validation, validation_by_year, summary, holdout = evaluate(calibration, test, designs)
    comparisons = paired(holdout, test)
    index = panel_index(test.dates)
    forecasts = pd.DataFrame(
        {name: prediction.ravel() for name, prediction in holdout.items()}, index=index
    ).assign(actual=test.actual.ravel())
    realized_7d = test.actual[:, -1]
    base_error = (holdout[PRIMARY][:, -1] - realized_7d) ** 2
    cumulative = pd.DataFrame(
        {
            name: (base_error - (prediction[:, -1] - realized_7d) ** 2).cumsum()
            for name, prediction in holdout.items()
        },
        index=test.dates.rename("origin"),
    )
    metrics = []
    for name, prediction in holdout.items():
        ratios = relative_rmse(prediction, test.actual, test.last)
        error = np.sqrt(((prediction - test.actual) ** 2).mean(axis=0))
        metrics.append(
            pd.DataFrame(dict(model=name, horizon=HORIZONS, rmse=error, rmse_to_rw=ratios))
        )
    test_by_year = []
    for year in test.dates.year.unique():
        rows = test.dates.year == year
        test_by_year.extend(
            dict(
                model=name,
                year=year,
                **period_metrics(prediction[rows], test.actual[rows], test.last[rows]),
            )
            for name, prediction in holdout.items()
        )
    for frame, filename in (
        (coverage.reset_index(), "coverage.csv"),
        (validation, "validation.csv"),
        (validation_by_year, "validation_by_year.csv"),
        (summary, "summary.csv"),
        (pd.concat(metrics, ignore_index=True), "metrics.csv"),
        (pd.DataFrame(test_by_year), "test_by_year.csv"),
        (comparisons, "paired_comparisons.csv"),
        (forecasts.reset_index(), "predictions.csv.gz"),
        (cumulative.reset_index(), "cumulative_sse_7d.csv"),
    ):
        write_csv(frame, args.output / filename)
    run = {
        "xgboost": xgboost.__version__,
        "xgb_params": XGB_PARAMS,
        "source_series": FRED,
        "assumed_lags_days": LAGS,
        "validation_years": [int(year) for year in calibration.dates.year.unique()[1:]],
        "calibration_origins": len(calibration.dates),
        "test_origins": len(test.dates),
        "features": {
            name: {"columns": panel.columns.tolist(), "monotone_constraints": signs}
            for name, (panel, signs) in designs.items()
        },
        "inputs": input_fingerprints(
            [*paths, *source_paths, Path(__file__), Path(theory_features.__code__.co_filename)]
        ),
        "dataset": input_fingerprints([output_dataset])[0],
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(summary.round(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/usd_rub_2011_2026_full.csv.gz")
    )
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument("--archive", type=Path, default=Path("data/reports/forecast_stacking"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/theory_holdout"))
    run(parser.parse_args())
