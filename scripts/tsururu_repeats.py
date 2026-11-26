"""Повторить отбор признаков, древесные модели и нейросети на фолдах Tsururu."""

import argparse
import json
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from usd_rub.fixing_calendar import HORIZONS, apply_calendar, fixing_mask
from usd_rub.ml_dataset import feature_panels, read_dataset
from usd_rub.ml_experiment import Candidate, actual_at, fit_predict, relative_rmse, to_levels
from usd_rub.sequence_features import prepare_fold
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints
from usd_rub.tsururu_splits import Fold, temporal_folds

TREES = {
    "histboost": (Candidate("histboost", "extended", None, 4, pooled=True), 0.75),
    "extra_trees": (Candidate("extra_trees", "extended", None, 4, pooled=True), 0.5),
    "catboost": (Candidate("catboost", "extended", 1460, 3, pooled=True), 0.5),
}
BUDGETS = (32, 64)
SHRINKS = (0, 0.25, 0.5, 0.75, 1.0)


def tree_estimate(panel, levels, fold, grid, rank):
    """Переобучить деревья на полной панели и по ранжированию внутри фолда."""
    _, _, _, _, _, _, ranked = prepare_fold(panel, levels, fold.cutoff, HORIZONS, 0.5)
    shap = rank(panel, levels, fold.cutoff)
    priorities = {"ridge": ranked.feature, "shap_gain": shap.index[shap.combined.gt(0)]}
    inputs = {("full", 0): panel.to_numpy(dtype=float)}
    for method, order in priorities.items():
        for budget in BUDGETS:
            inputs[(method, budget)] = panel[list(order[:budget])].to_numpy(dtype=float)
    output = {}
    for name, method, budget in grid:
        candidate, shrink = TREES[name]
        raw, _ = fit_predict(
            inputs[(method, budget)],
            levels,
            candidate,
            fold.cutoff,
            fold.origins,
            HORIZONS,
            min_hit_rate=0.5,
        )
        output[(name, method, budget)] = raw * shrink
    return output


def sequence_estimate(panel, levels, fold, grid):
    """Передать выбранные эпохи и панели существующим нейросетям."""
    from sequence_holdout import fit_fold

    schedule = {"all": {}, "top64": {}}
    for model, budget, epoch in grid:
        schedule[budget][model] = (*schedule[budget].get(model, ()), epoch)
    output, _, _ = fit_fold(
        panel, levels, fold.cutoff, fold.origins, HORIZONS, 0.5, "cpu", schedule
    )
    return output


def run(args):
    """Выбрать эпохи, усадку и размер панели по недельному RMSE до теста."""
    dataset = read_dataset(args.dataset)
    dates, levels = dataset.index, dataset.usd_rub.to_numpy(dtype=float)
    config = json.loads(args.config.read_text())
    calendar = json.loads(args.calendar.read_text())
    panel = feature_panels(dataset)["extended"].replace([np.inf, -np.inf], np.nan)
    if args.experiment == "sequence":
        import torch
        from sequence_holdout import ARCHITECTURES, CHECKPOINTS

        torch.set_num_threads(2)
        fit, shrinks, packages = sequence_estimate, SHRINKS, ("torch", "scikit-learn")
        grid = [
            (name, budget, epoch)
            for budget in ("all", "top64")
            for name in ("ridge", *ARCHITECTURES)
            for epoch in ((0,) if name == "ridge" else CHECKPOINTS)
        ]
    else:
        from shap_holdout import rank

        fit, shrinks = partial(tree_estimate, rank=rank), (1.0,)
        packages = ("xgboost", "catboost", "scikit-learn")
        grid = [(model, "full", 0) for model in TREES]
        grid += [(model, "ridge", budget) for model in TREES for budget in BUDGETS]
        grid += [("histboost", "shap_gain", budget) for budget in BUDGETS]
    folds, _ = temporal_folds(dates, config["test_start"])
    scores = []
    for fold in folds:
        if fold.warmup:
            continue
        prediction = fit(panel, levels, fold, grid)
        last = levels[fold.origins]
        actual = actual_at(levels, fold.origins, HORIZONS)
        mask = fixing_mask(dates[fold.origins], calendar, True)
        for (model, panel_key, parameter), relative in prediction.items():
            for shrink in shrinks:
                point = apply_calendar(to_levels(relative, last, shrink), last, mask, pool=True)
                ratios = relative_rmse(point, actual, last)
                scores.append(
                    (fold.name, model, panel_key, parameter, shrink, ratios[-1], ratios.mean())
                )
        print(f"{fold.name}: {args.experiment} проверен", flush=True)

    keys = ["model", "panel", "parameter", "shrink"]
    by_fold = pd.DataFrame(scores, columns=["fold", *keys, "cv_rmse7_to_rw", "cv_rmse_to_rw"])
    ranking = by_fold.groupby(keys, as_index=False).mean(numeric_only=True)
    ranking = ranking.sort_values(["cv_rmse7_to_rw", "cv_rmse_to_rw", "parameter", "shrink"])
    ranking["selected_by_panel"] = ~ranking.duplicated(["model", "panel"])
    ranking["selected_by_model"] = ~ranking.model.duplicated()
    picked = ranking.loc[ranking.selected_by_panel]
    final = list(picked[["model", "panel", "parameter"]].itertuples(index=False, name=None))
    test = Fold("holdout", dates.get_loc(pd.Timestamp(config["test_start"])), len(dates))
    prediction = fit(panel, levels, test, final)
    origins = test.origins
    actual, last = actual_at(levels, origins, HORIZONS), levels[origins]
    mask = fixing_mask(dates[origins], calendar, True)
    rows = []
    forecasts = {
        "origin": np.repeat(dates[origins], len(HORIZONS)),
        "horizon": np.tile(HORIZONS, len(origins)),
        "actual": actual.ravel(),
        "last": np.repeat(last, len(HORIZONS)),
    }
    for row in picked.itertuples(index=False):
        raw = prediction[(row.model, row.panel, row.parameter)]
        point = apply_calendar(to_levels(raw, last, row.shrink), last, mask, pool=True)
        forecasts[f"{row.model}/{row.panel}"] = point.ravel()
        for h, errors, naives in zip(
            HORIZONS, (point - actual).T, (last[:, None] - actual).T, strict=True
        ):
            rmse = np.sqrt(np.mean(errors**2))
            rows.append(
                {
                    **row._asdict(),
                    "horizon": h,
                    "rmse_rub": rmse,
                    "mae_rub": np.abs(errors).mean(),
                    "rmse_to_rw": rmse / np.sqrt(np.mean(naives**2)),
                }
            )
    prefix = args.experiment
    write_csv(by_fold, args.output / f"{prefix}_validation_folds.csv")
    write_csv(ranking, args.output / f"{prefix}_selection.csv")
    write_csv(pd.DataFrame(rows), args.output / f"{prefix}_metrics.csv")
    write_csv(pd.DataFrame(forecasts), args.output / f"{prefix}_predictions.csv.gz")
    paths = [args.dataset, args.config, args.calendar, Path(__file__)]
    (args.output / f"{prefix}_run.json").write_text(
        json.dumps(
            {
                "inputs": input_fingerprints(paths),
                "packages": {name: version(name) for name in packages},
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("sequence", "trees"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/usd_rub_2011_2026_full.csv.gz")
    )
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument("--calendar", type=Path, default=Path("config/cbr_fixing_calendar.json"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/tsururu_holdout"))
    with threadpool_limits(limits=2):
        run(parser.parse_args())
