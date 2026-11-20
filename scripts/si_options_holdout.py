"""Проверка волатильности и перекоса опционов Si на горизонте семь дней."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from forecast_stacking import ARCHIVE_FILES, read_panel
from foundation_ensemble import CLASSICAL, RW
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from weekly_literature_holdout import (
    HORIZON,
    annual_score,
    gamma_frame,
    residual_features,
    residual_fit,
    summarize,
)

from usd_rub.ml_dataset import feature_panels, read_dataset
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints


def option_blocks(panel):
    """Разделить уровень и динамику IV и risk reversal."""
    iv, rr = [], []
    for days in (7, 30):
        iv.extend(f"fx__si_iv_{days}d_{key}" for key in ("level", "change_5"))
        rr.extend(f"fx__si_rr25_{days}d_{key}" for key in ("level", "change_5"))
    blocks = {"base": panel.iloc[:, :0], "iv": panel[iv].copy(), "rr": panel[rr]}
    for name in iv:
        if name.endswith("_level"):
            blocks["iv"][name] = np.log(blocks["iv"][name].where(lambda v: v > 0))
    blocks["iv_and_rr"] = blocks["iv"].join(blocks["rr"])
    return blocks


def point_forecast(features, calibration, query, cutoff):
    """Применить регуляризацию и долю поправки прежнего гибрида."""
    return query.ensemble + 0.25 * residual_fit(features, calibration, query, cutoff, (0.5,))[0.5]


def magnitude_forecast(features, calibration, query, cutoff):
    """Предсказать недельное изменение курса по абсолютной величине."""
    train = calibration.loc[calibration.index + HORIZON < cutoff]
    history = features.loc[train.index]
    selected = history.columns[history.notna().mean().ge(0.5)]
    center = history[selected].mean()
    stdev = history[selected].std(ddof=0).where(lambda s: s.gt(1e-12), 1)

    def matrix(dates):
        return ((features.loc[dates, selected] - center) / stdev).fillna(0).clip(-6, 6)

    target = (train.actual / train["last"] - 1).abs()
    weights = (train["last"] / train["last"].median()).pow(2)
    model = Ridge(alpha=0.15 * len(train))
    model.fit(matrix(train.index), target, sample_weight=weights)
    return pd.Series(
        model.predict(matrix(query.index)).clip(min=0) * query["last"], index=query.index
    )


def run(args):
    dataset = read_dataset(args.dataset)
    panel = feature_panels(dataset)["extended"]
    groups = json.loads((args.archive / "archives.json").read_text())["groups"]
    names = [CLASSICAL, RW, *sum(groups.values(), [])]
    archives = [args.archive / filename for filename in ARCHIVE_FILES]
    calibration, test = [gamma_frame(read_panel(path, names)) for path in archives]
    for split in (calibration, test):
        np.testing.assert_allclose(split["last"], dataset.usd_rub.loc[split.index], rtol=1e-12)
        np.testing.assert_allclose(
            split.actual, dataset.usd_rub.loc[split.index + HORIZON], rtol=1e-12
        )
    if calibration.index.max() + HORIZON >= test.index.min():
        raise ValueError("Архив обучения пересекает тест")
    dates = calibration.index[calibration.index.year >= 2022]
    blocks = option_blocks(panel)
    realized = panel[
        ["own__vol_7", "own__vol_21", "own__vol_60", "markets__rvi_level", "markets__rvi_change_5"]
    ]
    scoring = []
    tests = []
    predictions = {}
    for target, forecast, features in (
        ("point", point_forecast, residual_features(panel)["spread_dynamics"]),
        ("magnitude", magnitude_forecast, realized),
    ):
        valid_ref, test_ref = calibration.loc[dates].copy(), test.copy()
        if target == "magnitude":
            for frame in (valid_ref, test_ref):
                frame["actual"] = (frame.actual - frame["last"]).abs()
                frame["last"] = 0
        candidates = {}
        test_values = {}
        for name, block in blocks.items():
            design = features.join(block)
            validation = pd.concat(
                [
                    forecast(design, calibration, fold, fold.index[0])
                    for _, fold in calibration.loc[dates].groupby(dates.year)
                ]
            )
            candidates[name] = annual_score(validation, valid_ref)
            test_values[name] = forecast(design, calibration, test, test.index[0])
        best = min(candidates, key=candidates.get)
        test_ref["ensemble"] = test_values["base"]
        results = summarize(test_ref, test_values).assign(target=target)
        scoring.extend(
            {"target": target, "model": name, "rmse_to_rw": loss, "selected": name == best}
            for name, loss in candidates.items()
        )
        tests.append(results)
        predictions.update({f"{target}_{name}": values for name, values in test_values.items()})

    args.output.mkdir(parents=True, exist_ok=True)
    validation = pd.DataFrame(scoring)
    metrics = pd.concat(tests, ignore_index=True)
    write_csv(validation, args.output / "validation.csv")
    write_csv(metrics, args.output / "metrics.csv")
    write_csv(
        pd.DataFrame(predictions)
        .join(test[["actual", "last"]])
        .rename_axis("origin")
        .reset_index(),
        args.output / "predictions.csv.gz",
    )
    metadata = {
        "calibration_dates": len(dates),
        "test_dates": len(test),
        "inputs": input_fingerprints([args.dataset, *archives, args.archive / "archives.json"]),
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(validation.round(5).to_string(index=False))
    print(metrics.round(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/datasets/usd_rub_2011_2026_full_si_options.csv.gz"),
    )
    parser.add_argument("--archive", type=Path, default=Path("data/reports/forecast_stacking"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/si_options_holdout"))
    with threadpool_limits(limits=2):
        run(parser.parse_args())
