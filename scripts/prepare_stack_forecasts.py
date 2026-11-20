"""Сохранить вневыборочные прогнозы ранее проверенных моделей USD/RUB."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from correlation_holdout import HORIZONS, forecasts
from threadpoolctl import threadpool_limits

from usd_rub.ml_dataset import clean_dataset, feature_panels, read_dataset
from usd_rub.ml_experiment import (
    candidates,
    fit_predict,
    selection_loss,
    specific_features,
    to_levels,
)
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints

CALIBRATION_START = 2021


def load_archive(args, target):
    """Взять общие даты калибровки из исторического архива ансамбля."""
    dates, actual, last, ensemble = forecasts(args.oof, "ensemble_oof", target)
    cutoff = pd.Timestamp(args.test_start)
    eligible = (dates.year >= CALIBRATION_START) & (
        dates + pd.Timedelta(days=int(HORIZONS[-1])) < cutoff
    )
    return dates[eligible], actual[eligible], last[eligible], ensemble[eligible]


def forecast_rows(name, dates, origins, forecast):
    """Сохранить прогноз с ключами даты и горизонта."""
    if not np.isfinite(forecast).all():
        raise ValueError(f"Неконечные прогнозы {name}")
    return pd.DataFrame(
        {
            "model": name,
            "origin": np.repeat(dates[origins], len(HORIZONS)),
            "horizon": np.tile(HORIZONS, len(origins)),
            "prediction": forecast.ravel(),
        }
    )


def sequence_oof(args, dataset, calendar, coverage):
    """Повторить годовые прогоны с зафиксированными эпохами и усадкой."""
    import torch
    from sequence_holdout import fit_fold

    torch.set_num_threads(2)
    previous = json.loads(args.sequence_run.read_text())
    if previous["torch"] != torch.__version__ or previous["device"] != args.device:
        raise ValueError(f"Ожидается PyTorch {previous['torch']} на {previous['device']}")
    dates, actual, last, _ = calendar
    source = dataset.index
    levels = dataset.usd_rub.to_numpy()
    panel = feature_panels(dataset)["extended"]
    specs = {
        (row["architecture"], row["features"], row["epoch"]): row for row in previous["selected"]
    }
    schedule = {"all": {}, "top64": {}}
    for name, budget, epoch in specs:
        schedule[budget][name] = (epoch,)

    rows, splits = [], []
    estimates = {key: [] for key in specs}
    for year in sorted(set(dates.year)):
        mask = dates.year == year
        origins = source.get_indexer(dates[mask])
        cutoff = source.searchsorted(f"{year}-01-01")
        results, _, _ = fit_fold(
            panel, levels, cutoff, origins, HORIZONS, coverage, args.device, schedule
        )
        splits.append((dates[mask], actual[mask], last[mask]))
        for key, prediction in results.items():
            model = specs[key]
            adjusted = to_levels(prediction, last[mask], model["shrink"])
            estimates[key].append(adjusted)
            rows.append(forecast_rows(model["model"], source, origins, adjusted))
    for key, values in estimates.items():
        np.testing.assert_allclose(
            selection_loss(values, splits),
            specs[key]["cv_rmse_to_rw"],
            atol=2e-6,
            rtol=0,
            err_msg=f"Пересчёт калибровки {specs[key]['model']}",
        )
    return (
        rows,
        [args.sequence_run, Path(fit_fold.__code__.co_filename)],
        {"torch": torch.__version__},
    )


def classical_oof(args, dataset, calendar, coverage):
    """Восстановить отдельные прогнозы классических компонентов."""
    reference_run = args.reference.parent / "run.json"
    reference_settings = args.reference.parent / "validation_candidates.csv"
    run = json.loads(reference_run.read_text())
    weights = {name: w for name, w in run["ensemble_weights"].items() if w > 1e-12}
    classifier = {item.name: item for item in candidates()}
    settings = pd.read_csv(reference_settings).set_index("name")
    source = dataset.index
    levels = dataset.usd_rub.to_numpy()
    panels = feature_panels(clean_dataset(dataset, coverage))
    matrices = {
        name: specific_features(panels, classifier[name].setup).to_numpy()
        for name in weights
        if name != "rw"
    }
    dates, _, last, ensemble = calendar
    rows = []
    for year in sorted(set(dates.year)):
        mask = dates.year == year
        origins = source.get_indexer(dates[mask])
        cutoff = source.searchsorted(f"{year}-01-01")
        total = np.broadcast_to(last[mask, None], ensemble[mask].shape) * weights.get("rw", 0)
        for name, x in matrices.items():
            raw, _ = fit_predict(
                x, levels, classifier[name], cutoff, origins, HORIZONS, min_hit_rate=coverage
            )
            forecast = to_levels(raw, last[mask], settings.at[name, "shrink"])
            total += weights[name] * forecast
            rows.append(forecast_rows(name, source, origins, forecast))
        np.testing.assert_allclose(total, ensemble[mask], rtol=1e-10, atol=1e-10)
        print(f"Классические модели {year}", flush=True)
    return (
        rows,
        [reference_run, reference_settings, Path(fit_predict.__code__.co_filename)],
        {"weights": weights},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("data/reports")
    parser.add_argument("family", choices=("sequence", "classical"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/usd_rub_2011_2026_full.csv.gz")
    )
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument("--oof", type=Path, default=root / "llm_news_holdout/baseline_oof.csv.gz")
    parser.add_argument("--sequence-run", type=Path, default=root / "sequence_holdout/run.json")
    parser.add_argument(
        "--reference",
        type=Path,
        default=root / "ml_holdout_minfin_rosstat/with_file/predictions.csv",
    )
    parser.add_argument("--output", type=Path, default=root / "forecast_stacking")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.test_start = config["test_start"]
    dataset = read_dataset(args.dataset)
    calendar = load_archive(args, dataset.usd_rub)
    if not len(calendar[0]):
        raise ValueError("Архив калибровки пуст")
    function = sequence_oof if args.family == "sequence" else classical_oof
    with threadpool_limits(limits=2):
        rows, paths, metadata = function(args, dataset, calendar, config.get("min_hit_rate", 0.5))
    output = args.output / f"{args.family}_oof.csv.gz"
    frame = pd.concat(rows, ignore_index=True).sort_values(["model", "origin", "horizon"])
    write_csv(frame, output)
    metadata.update(
        models=sorted(frame.model.unique()),
        calibration_origins=len(calendar[0]),
        input_files=input_fingerprints(
            [args.dataset, args.config, args.oof, Path(__file__), *paths]
        ),
    )
    (args.output / f"{args.family}_oof.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{output}: {len(frame):,} прогнозов", flush=True)


if __name__ == "__main__":
    main()
