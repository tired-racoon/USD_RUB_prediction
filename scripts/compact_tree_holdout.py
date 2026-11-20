"""Переобучить древесные модели с отбором признаков внутри каждого фолда."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from export_ml_dataset import write_dataset
from sequence_holdout import prepare_fold
from threadpoolctl import threadpool_limits

from usd_rub.ml_dataset import EXTENDED_PREFIX, FEATURE_GROUPS, feature_panels, read_dataset
from usd_rub.ml_experiment import (
    actual_at,
    candidates,
    fit_predict,
    interval_from_validation,
    prediction_rows,
    relative_rmse,
    selection_loss,
    to_levels,
)
from usd_rub.storage import input_fingerprints

BUDGETS = (32, 64)


def compare(args):
    """Проверить размеры панелей с гиперпараметрами прошлого эксперимента."""
    config = json.loads(args.config.read_text())
    dataset = read_dataset(args.dataset)
    benchmark = pd.read_csv(args.benchmark)
    run_path = args.benchmark.parent / "with_file" / "run.json"
    previous = json.loads(run_path.read_text())
    expected = {item["sha256"] for item in previous["inputs"]}
    if not {item["sha256"] for item in input_fingerprints([args.dataset, args.config])} <= expected:
        raise ValueError("Контрольные метрики рассчитаны для другого снимка или конфигурации")

    levels = dataset.usd_rub.to_numpy()
    index = dataset.index
    panel = feature_panels(dataset)["extended"]
    horizons = np.arange(1, config["horizon"] + 1)
    test_start = index.searchsorted(config["test_start"])
    min_hit_rate = config.get("min_hit_rate", 0.5)
    lookup = {candidate.name: candidate for candidate in candidates()}
    models = benchmark.loc[benchmark.model.isin(["catboost", "histboost", "extra_trees"])]
    model_settings = {
        row.model: (lookup[row.configuration], row.shrink, row.with_file_rmse_to_rw_pct)
        for row in models.itertuples(index=False)
    }
    if len(model_settings) != 3:
        raise ValueError("Нет трёх контрольных конфигураций")

    def estimate(cutoff, origins, schedule):
        _, _, _, _, _, _, ranking = prepare_fold(panel, levels, cutoff, horizons, min_hit_rate)
        result = {}
        matrices = {
            k: panel[ranking.feature.head(k)].to_numpy(dtype=float)
            for k in sorted({budget for _, budget in schedule})
        }
        for name, budget in schedule:
            candidate, shrink, _ = model_settings[name]
            prediction, kept = fit_predict(
                matrices[budget],
                levels,
                candidate,
                cutoff,
                origins,
                horizons,
                min_hit_rate=min_hit_rate,
            )
            result[(name, budget)] = (to_levels(prediction, levels[origins], shrink), len(kept))
        return result, ranking

    validation = {(name, size): [] for name in model_settings for size in BUDGETS}
    splits = []
    for year in config["validation_years"]:
        cutoff = index.searchsorted(f"{year}-01-01")
        end = index.searchsorted(f"{year + 1}-01-01")
        if end > test_start:
            raise ValueError("Валидация пересекает holdout")
        origins = np.arange(cutoff, end - horizons[-1])
        splits.append((origins, actual_at(levels, origins, horizons), levels[origins]))
        result, _ = estimate(cutoff, origins, validation)
        for key, (prediction, _) in result.items():
            validation[key].append(prediction)
        print(f"Проверены панели за {year}", flush=True)

    ranking = pd.DataFrame(
        [
            {"model": key[0], "budget": key[1], "cv_rmse_to_rw": selection_loss(vals, splits)}
            for key, vals in validation.items()
        ]
    ).sort_values(["cv_rmse_to_rw", "budget"])
    args.output.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.output / "tree_validation_candidates.csv", index=False)
    selected = ranking.drop_duplicates("model").set_index("model").budget.to_dict()
    origins = np.arange(test_start, len(index) - horizons[-1])
    actual, last = actual_at(levels, origins, horizons), levels[origins]
    test, test_rank = estimate(test_start, origins, selected.items())
    window = config["baseline_window"]
    scale = float(np.abs(np.diff(levels[test_start - window : test_start])).mean())
    rows, predictions = [], []
    for (name, budget), (prediction, kept) in test.items():
        candidate, shrink, full = model_settings[name]
        ratio = relative_rmse(prediction, actual, last)
        lower, upper = interval_from_validation(
            prediction, last, validation[(name, budget)], splits
        )
        predictions.append(
            prediction_rows(
                f"{name}_top{budget}",
                dataset.usd_rub,
                origins,
                horizons,
                prediction,
                lower,
                upper,
                scale,
                min(test_start, candidate.window or test_start),
            )
        )
        rows.append(
            {
                "model": name,
                "configuration": candidate.name,
                "feature_budget": budget,
                "features_used": kept,
                "shrink": shrink,
                "origins": len(origins),
                "full_rmse_to_rw_pct": full,
                "compact_rmse_to_rw_pct": 100 * ratio.mean(),
                "gain_pp": full - 100 * ratio.mean(),
                "rmse_1d": np.sqrt(((actual[:, 0] - prediction[:, 0]) ** 2).mean()),
                "rmse_7d": np.sqrt(((actual[:, -1] - prediction[:, -1]) ** 2).mean()),
            }
        )
    pd.concat(predictions, ignore_index=True).to_csv(
        args.output / "tree_predictions.csv", index=False
    )
    for budget in BUDGETS:
        if budget in selected.values() and budget != 64:
            kept = set(test_rank.feature.head(budget))
            columns = [
                column
                for column in dataset
                if (feature := column.removeprefix(EXTENDED_PREFIX)) in kept
                or feature.split("__", 1)[0] not in FEATURE_GROUPS
            ]
            write_dataset(dataset[columns], args.output / f"dataset_top{budget}.csv.gz")
    results = pd.DataFrame(rows).sort_values("compact_rmse_to_rw_pct")
    results.to_csv(args.output / "tree_comparison.csv", index=False)
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/usd_rub_2011_2026_full.csv.gz")
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/reports/ml_holdout_minfin_rosstat/paired_extended.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/reports/sequence_holdout"))
    with threadpool_limits(limits=2):
        compare(parser.parse_args())
