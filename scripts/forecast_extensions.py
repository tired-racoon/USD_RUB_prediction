"""Сравнить Prophet, позднее добавление признаков и смеси прогнозов USD/RUB."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from forecast_stacking import (
    ARCHIVE_FILES,
    HORIZONS,
    KEYS,
    convex_blend,
    panel_index,
    read_panel,
    ridge_deltas,
)
from foundation_ensemble import CLASSICAL, RW, summarize
from late_fusion_holdout import MODES, feature_layout, standardize
from llm_news_holdout import bootstrap
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from usd_rub.ml_dataset import read_dataset
from usd_rub.ml_experiment import choose_shrink, relative_rmse, selection_loss, to_levels
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints


def load_inputs(args):
    """Проверить архив и сравнить даты и цели со снимком."""
    manifest = args.archive / "archives.json"
    archived = json.loads(manifest.read_text())
    groups = archived["groups"]
    archives = [args.archive / name for name in ARCHIVE_FILES]
    paths = [args.dataset, args.config, manifest, *archives]
    hashes = {Path(item["path"]).name: item["sha256"] for item in input_fingerprints(archives)}
    if hashes != {Path(item["path"]).name: item["sha256"] for item in archived["archives"]}:
        raise ValueError("Архив прогнозов изменился после сборки")
    names = [CLASSICAL, RW, *sum(groups.values(), [])]
    calibration, test = [read_panel(path, names) for path in archives]
    dataset = read_dataset(args.dataset)
    config = json.loads(args.config.read_text())
    dates = pd.date_range(config["start"], config["end"])
    cutoff = pd.Timestamp(config["test_start"])
    horizon = pd.Timedelta(days=int(HORIZONS[-1]))
    if (
        config["horizon"] != HORIZONS[-1]
        or not dates.equals(dataset.index)
        or not np.array_equal(dataset.split, np.where(dates < cutoff, "train", "test"))
    ):
        raise ValueError("Снимок не соответствует разбиению")
    expected = pd.date_range(f"{config['validation_years'][0]}-01-01", cutoff - horizon)
    expected = expected[
        expected.year.isin(config["validation_years"])
        & (expected.year == (expected + horizon).year)
    ]
    np.testing.assert_array_equal(calibration.dates, expected)
    np.testing.assert_array_equal(
        test.dates, dates[(dates >= cutoff) & (dates + horizon <= dates[-1])]
    )
    for split in (calibration, test):
        positions = dates.get_indexer(split.dates)
        levels = dataset.usd_rub.to_numpy(dtype=float)
        np.testing.assert_allclose(split.last, levels[positions], rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            split.actual, levels[positions[:, None] + HORIZONS], rtol=1e-12, atol=0
        )
    return dataset, calibration, test, groups, paths


def save_candidates(output, family, dates, predictions, paths, settings):
    """Выгрузить прогнозы до применения календаря фиксинга."""
    output.mkdir(parents=True, exist_ok=True)
    index = panel_index(dates)
    frame = pd.DataFrame({name: v.ravel() for name, v in predictions.items()}, index=index)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Неконечные прогнозы {family}")
    archive = output / f"{family}_raw.csv.gz"
    write_csv(frame.reset_index(), archive)
    metadata = {
        **settings,
        "inputs": input_fingerprints(paths),
        "output": input_fingerprints([archive])[0],
    }
    (output / f"{family}_run.json").write_text(json.dumps(metadata, indent=2) + "\n")


def run_prophet(args, dataset, dates, paths):
    """Рассчитать самостоятельные прогнозы Prophet."""
    import prophet
    from prophet_holdout import REFIT_DAYS, SEED, WINDOWS, prophet_predictions

    predictions = prophet_predictions(dataset.usd_rub, dates, HORIZONS, args.jobs)
    save_candidates(
        args.output,
        "prophet",
        dates,
        predictions,
        [*paths, Path(prophet_predictions.__code__.co_filename)],
        dict(prophet=prophet.__version__, refit_days=REFIT_DAYS, windows=WINDOWS, seed=SEED),
    )


def run_sequence(args, dataset, calibration, test, paths):
    """Обучить нейросети и сохранить прогнозы на всех эпохах."""
    import torch
    from late_fusion_holdout import sequence_predictions
    from sequence_holdout import CHECKPOINTS, LOOKBACK, SEED, TRAIN_WINDOW, train_sequence

    torch.set_num_threads(2)
    predictions, coverage = sequence_predictions(
        dataset, calibration.dates, test.dates, HORIZONS, args.device
    )
    save_candidates(
        args.output,
        "sequence",
        calibration.dates.append(test.dates),
        predictions,
        [
            *paths,
            Path(sequence_predictions.__code__.co_filename),
            Path(train_sequence.__code__.co_filename),
        ],
        dict(
            torch=torch.__version__,
            device=args.device,
            lookback=LOOKBACK,
            train_window=TRAIN_WINDOW,
            epochs=CHECKPOINTS,
            seed=SEED,
        ),
    )
    write_csv(coverage, args.output / "feature_coverage.csv")


def load_candidates(output, family, calibration, test, inputs):
    """Проверить происхождение и порядок новых прогнозов перед агрегацией."""
    path = output / f"{family}_raw.csv.gz"
    metadata = json.loads((output / f"{family}_run.json").read_text())
    hashes = {item["sha256"] for item in metadata["inputs"]}
    if (
        any(item["sha256"] not in hashes for item in inputs)
        or metadata["output"]["sha256"] != input_fingerprints([path])[0]["sha256"]
    ):
        raise ValueError(f"Изменились входы или прогнозы {family}")
    table = pd.read_csv(path, parse_dates=["origin"], float_precision="round_trip").set_index(KEYS)
    pd.testing.assert_index_equal(table.index, panel_index(calibration.dates.append(test.dates)))
    for split in (calibration, test):
        frame = table.loc[panel_index(split.dates)]
        for name, column in frame.items():
            prediction = column.to_numpy().reshape(split.actual.shape)
            split.raw[name] = prediction
            split.projected[name] = split.project(prediction)
    return table.columns.tolist()


def annual_folds(calibration):
    """Взять годовые фолды и убрать из обучения цели с датами после разделения."""
    dates = calibration.dates
    ends = dates + pd.Timedelta(days=int(HORIZONS[-1]))
    annual = [dates.year == year for year in dates.year.unique()]
    return annual, [(ends < dates[check][0], check) for check in annual[1:]]


def select_sequence(calibration, test, names, annual):
    """Выбрать эпоху и усадку отдельно для каждого способа подачи признаков."""
    splits = [calibration.split(rows) for rows in annual]
    candidates = []
    for name in names:
        if name == "ridge_snapshot":
            group, epoch = name, 0
        else:
            group, checkpoint = name.rsplit("_e", maxsplit=1)
            epoch = int(checkpoint)
        prediction = calibration.projected[name]
        relative = prediction / calibration.last[:, None] - 1
        loss, shrink = choose_shrink([relative[rows] for rows in annual], splits)
        candidates.append(
            dict(
                model=name,
                group=group,
                epoch=epoch,
                shrink=shrink,
                cv_rmse_to_rw=loss,
                cv_rmse_to_rw_raw=selection_loss([prediction[rows] for rows in annual], splits),
            )
        )
    ranking = pd.DataFrame(candidates).sort_values(["cv_rmse_to_rw", "epoch", "model"])
    ranking["selected"] = ~ranking.group.duplicated()
    choices = {}
    for row in ranking.loc[ranking.selected].itertuples(index=False):
        relative = test.raw[row.model] / test.last[:, None] - 1
        prediction = test.project(to_levels(relative, test.last, row.shrink))
        choices[row.group] = (row.cv_rmse_to_rw, prediction, {}, {})
    return ranking, choices


def frozen_heads(dataset, calibration, test, names, folds):
    """Обучить линейную голову на выходах замороженных моделей и табличном срезе."""
    values, _, _ = feature_layout(dataset)
    calibration_x = values[dataset.index.get_indexer(calibration.dates)]
    test_x = values[dataset.index.get_indexer(test.dates)]
    designs = {"frozen": (names, False), "frozen_late": (names, True), "snapshot_only": ([], True)}
    alphas = (0.01, 0.1, 1.0)
    losses = {name: {alpha: [] for alpha in alphas} for name in designs}
    predictions = {}
    for train, check in [*folds, (np.ones(len(calibration.dates), dtype=bool), None)]:
        source = calibration if check is not None else test
        rows = check if check is not None else slice(None)
        features = np.concatenate((calibration_x, test_x)) if check is None else calibration_x
        x, _, _ = standardize(features, np.flatnonzero(train))
        x /= np.sqrt(x.shape[1])
        train_x = x[: len(calibration_x)][train]
        query_x = x[check] if check is not None else x[len(calibration_x) :]
        target = calibration.actual[train] - calibration.last[train, None]
        scale = np.sqrt(np.mean(target**2, axis=0))
        for name, (backbones, context) in designs.items():
            train_inputs, query_inputs = [], []
            for h, factor in enumerate(scale):
                history = [
                    (calibration.projected[key][train, h] - calibration.last[train]) / factor
                    for key in backbones
                ]
                query = [
                    (source.projected[key][rows, h] - source.last[rows]) / factor
                    for key in backbones
                ]
                if context:
                    history.append(train_x)
                    query.append(query_x)
                train_inputs.append(np.column_stack(history))
                query_inputs.append(np.column_stack(query))
            val = (
                {
                    alpha: selection_loss(
                        losses[name][alpha], [calibration.split(c) for _, c in folds]
                    )
                    for alpha in alphas
                }
                if check is None
                else {}
            )
            for alpha in [min(val, key=val.get)] if check is None else alphas:
                horizon_predictions = []
                for h, (features_train, features_query) in enumerate(
                    zip(train_inputs, query_inputs, strict=True)
                ):
                    model = Ridge(alpha=alpha * len(target), fit_intercept=False)
                    model.fit(features_train, target[:, h] / scale[h])
                    horizon_predictions.append(model.predict(features_query) * scale[h])
                forecast = source.project(
                    source.last[rows, None] + np.column_stack(horizon_predictions), rows
                )
                if check is None:
                    predictions[name] = val[alpha], forecast, alpha
                else:
                    losses[name][alpha].append(forecast)
    return predictions


def evaluate(args, dataset, calibration, test, groups, paths):
    """Сравнить одиночные прогнозы и добавки к тем же ансамблям."""
    inputs = input_fingerprints(paths)
    prophet, sequence = [
        load_candidates(args.output, family, calibration, test, inputs)
        for family in ("prophet", "sequence")
    ]
    annual, folds = annual_folds(calibration)
    scores, choices = select_sequence(calibration, test, sequence, annual)
    for name in [CLASSICAL, RW, *prophet, *groups["sequence"]]:
        loss = selection_loss(
            [calibration.projected[name][rows] for rows in annual],
            [calibration.split(rows) for rows in annual],
        )
        choices[name] = (loss, test.projected[name], {}, {})

    by_mode = {mode: [name for name in sequence if name.endswith(f"_{mode}_e04")] for mode in MODES}
    archived = [*sum(groups.values(), []), RW]
    market = by_mode["market_history"]
    late = by_mode["late"]
    mixtures = {
        "convex_rw": (convex_blend, [RW]),
        "convex_prophet": (convex_blend, prophet),
        "convex_prophet_rw": (convex_blend, [RW, *prophet]),
        "ridge_archived": (ridge_deltas, archived),
        "ridge_archived_prophet": (ridge_deltas, [*archived, *prophet]),
        "ridge_archived_market": (ridge_deltas, [*archived, *market]),
        "ridge_archived_all_history": (ridge_deltas, [*archived, *by_mode["all_history"]]),
        "ridge_archived_late": (ridge_deltas, [*archived, *late]),
        "ridge_archived_market_late": (ridge_deltas, [*archived, *market, *late]),
        "ridge_archived_prophet_late": (ridge_deltas, [*archived, *prophet, *late]),
    }
    settings, coefficients = {}, []
    for name, (fit, components) in mixtures.items():
        loss, forecast, weights, alpha = fit(calibration, test, components, folds)
        choices[name] = loss, forecast, {}, {}
        settings[name] = dict(alpha=alpha, components=[CLASSICAL, *components])
        coefficients.extend(
            dict(candidate=name, horizon=int(h), model=key, weight=weight)
            for h, row in zip(HORIZONS, weights, strict=True)
            for key, weight in zip(settings[name]["components"], row, strict=True)
        )
    for name, (loss, forecast, alpha) in frozen_heads(
        dataset, calibration, test, groups["foundation"], folds
    ).items():
        choices[name] = loss, forecast, {}, {}
        settings[name] = dict(alpha=alpha)

    summary, metrics, predictions, _ = summarize(
        choices, (test.dates, test.actual, test.last, test.raw[CLASSICAL], test.fixes)
    )
    test_losses = summary.set_index("candidate").test_rmse_to_rw
    architectures = sorted({name.split("_", 1)[0] for name in late})
    pairs = [
        ("convex_prophet", CLASSICAL),
        ("convex_prophet_rw", "convex_rw"),
        ("ridge_archived_prophet", "ridge_archived"),
        ("ridge_archived_market", "ridge_archived"),
        ("ridge_archived_late", "ridge_archived"),
        ("ridge_archived_late", "ridge_archived_market"),
        ("ridge_archived_market_late", "ridge_archived_market"),
        ("ridge_archived_prophet_late", "ridge_archived_late"),
        ("frozen_late", "frozen"),
        ("frozen_late", "snapshot_only"),
        *[
            (f"{name}_late", f"{name}_{mode}")
            for name in architectures
            for mode in ("market_history", "all_history")
        ],
    ]
    comparisons = []
    for candidate, reference in pairs:
        low, high = bootstrap(choices[reference][1], choices[candidate][1], test.actual, test.last)
        comparisons.append(
            dict(
                candidate=candidate,
                reference=reference,
                gain_pp=100 * (test_losses[reference] - test_losses[candidate]),
                gain_ci_low_pp=low,
                gain_ci_high_pp=high,
            )
        )
    summary["validation_period"] = np.where(
        summary.candidate.isin(settings), "2022–2024", "2021–2024"
    )
    for frame, filename in (
        (scores, "sequence_validation.csv"),
        (summary, "summary.csv"),
        (metrics, "metrics.csv"),
        (predictions, "predictions.csv.gz"),
        (pd.DataFrame(coefficients), "coefficients.csv"),
        (pd.DataFrame(comparisons), "paired_comparisons.csv"),
    ):
        write_csv(frame, args.output / filename)
    run = {
        "prophet_selected": min(prophet, key=lambda name: choices[name][0]),
        "sequence_selected": scores.loc[scores.selected].iloc[0].model,
        "calibration_dates": len(calibration.dates),
        "test_dates": len(test.dates),
        "settings": settings,
        "inputs": input_fingerprints(
            [
                *paths,
                Path(__file__),
                *[
                    args.output / f"{key}_{suffix}"
                    for key in ("prophet", "sequence")
                    for suffix in ("raw.csv.gz", "run.json")
                ],
            ]
        ),
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(summary.round(5).to_string(index=False))


def evaluate_rbc(args, dataset, calibration, test, paths):
    """Оценить РБК на датах после появления обучающей истории."""
    import torch
    from late_fusion_holdout import rbc_predictions

    torch.set_num_threads(2)
    names = load_candidates(args.output, "sequence", calibration, test, input_fingerprints(paths))
    annual, _ = annual_folds(calibration)
    ranking, _ = select_sequence(calibration, test, names, annual)
    settings = ranking.loc[ranking.selected & ranking.group.str.endswith("_late")]
    checkpoints = tuple(sorted(settings.epoch.unique()))
    dates, forecasts = rbc_predictions(dataset, test.dates, HORIZONS, args.device, checkpoints)
    rows = test.dates.get_indexer(dates)
    actual, last = test.actual[rows], test.last[rows]
    results = []
    prediction_rows = {"actual": actual.ravel()}
    for config in settings.itertuples(index=False):
        control = config.model.replace("_late_", "_late_without_rbc_")
        predictions = [
            test.project(to_levels(forecasts[name] / last[:, None] - 1, last, config.shrink), rows)
            for name in (control, config.model)
        ]
        plain, added = [
            relative_rmse(prediction, actual, last).mean() for prediction in predictions
        ]
        low, high = bootstrap(*predictions, actual, last)
        results.append(
            dict(
                model=config.group,
                origins=len(rows),
                epoch=config.epoch,
                shrink=config.shrink,
                rmse_to_rw_without_rbc=plain,
                rmse_to_rw_with_rbc=added,
                gain_pp=100 * (plain - added),
                gain_ci_low_pp=low,
                gain_ci_high_pp=high,
            )
        )
        for name, prediction in zip((control, config.model), predictions, strict=True):
            prediction_rows[name] = prediction.ravel()
    metrics = pd.DataFrame(results)
    write_csv(metrics, args.output / "rbc_metrics.csv")
    write_csv(
        pd.DataFrame(prediction_rows, index=panel_index(dates)).reset_index(),
        args.output / "rbc_predictions.csv.gz",
    )
    metadata = dict(
        dates=len(dates),
        first_origin=str(dates[0].date()),
        last_origin=str(dates[-1].date()),
        grid="quarterly",
        torch=torch.__version__,
        device=args.device,
        inputs=input_fingerprints(
            [
                *paths,
                Path(__file__),
                Path(rbc_predictions.__code__.co_filename),
                args.output / "sequence_raw.csv.gz",
                args.output / "sequence_run.json",
            ]
        ),
    )
    (args.output / "rbc_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metrics.round(5).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prophet", "sequence", "evaluate", "rbc"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/datasets/usd_rub_2011_2026_full_si_options.csv.gz"),
    )
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument("--archive", type=Path, default=Path("data/reports/forecast_stacking"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/forecast_extensions"))
    parser.add_argument("--jobs", type=int, default=6, help="Число параллельных подгонок Prophet")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        dataset, calibration, test, groups, paths = load_inputs(args)
        if args.action == "prophet":
            run_prophet(args, dataset, calibration.dates.append(test.dates), paths)
        elif args.action == "sequence":
            run_sequence(args, dataset, calibration, test, paths)
        elif args.action == "evaluate":
            evaluate(args, dataset, calibration, test, groups, paths)
        else:
            evaluate_rbc(args, dataset, calibration, test, paths)


if __name__ == "__main__":
    main()
