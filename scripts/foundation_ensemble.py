"""Добавить предобученные прогнозы к существующему ансамблю USD/RUB."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from correlation_holdout import HORIZONS, apply_calendar
from foundation_holdout import MODELS, MODES, experiment_inputs

from usd_rub.ml_experiment import blend_weights, relative_rmse, selection_loss
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints

GRID = (0, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6)
FIXED_WEIGHT = 0.1
LAST_SHORT = 3
CLASSICAL = "ensemble"
RW = "random_walk"


def archived_forecasts(directory, names, inputs):
    """Сопоставить даты, цели и календарную обработку с исходным экспериментом."""
    dates, actual, last, base, fixes, *_ = inputs
    raw = {CLASSICAL: base, RW: np.broadcast_to(last[:, None], actual.shape)}
    projected = {
        name: apply_calendar(values, last, fixes, pool=True) for name, values in raw.items()
    }
    files = []
    for name in names:
        path = directory / f"{name}.csv.gz"
        frame = pd.read_csv(path, parse_dates=["origin"], float_precision="round_trip")
        expected = MODES[:1] if name == "exaone_finance" else MODES
        if set(frame["mode"]) != set(expected) or not frame["model"].eq(name).all():
            raise ValueError(f"Неполный архив: {path}")
        files.extend([path, path.with_suffix("").with_suffix(".json")])
        for mode, group in frame.groupby("mode"):
            np.testing.assert_array_equal(group.origin, np.repeat(dates, len(HORIZONS)))
            np.testing.assert_array_equal(group.horizon, np.tile(HORIZONS, len(dates)))
            np.testing.assert_allclose(group.actual, actual.ravel(), rtol=1e-12)
            key = f"{name}/{mode}"
            raw[key] = group.prediction_raw.to_numpy().reshape(actual.shape)
            projected[key] = group.prediction.to_numpy().reshape(actual.shape)
            np.testing.assert_allclose(
                projected[key], apply_calendar(raw[key], last, fixes, pool=True), rtol=1e-12
            )
    return raw, projected, files


def evaluate_weights(calibration, test, names):
    """Выбрать спецификации и веса по четырём годовым фолдам."""
    inputs, raw, projected = calibration
    test_inputs, test_raw = test
    dates, actual, last, _, changes, *_ = inputs
    _, _, test_last, _, test_changes, *_ = test_inputs
    folds = [dates.year == year for year in sorted(set(dates.year))]
    splits = [(dates[mask], actual[mask], last[mask]) for mask in folds]

    def score(values):
        return selection_loss([values[mask] for mask in folds], splits)

    def add(key, weights):
        """Сохранить прогноз по весам, определённым на калибровке."""
        cal_blend = sum(projected[name] * w for name, w in weights.items())
        test_raw_blend = sum(test_raw[name] * w for name, w in weights.items())
        test_blend = apply_calendar(test_raw_blend, test_last, test_changes, pool=True)
        selected[key] = (score(cal_blend), test_blend, weights, weights)

    def choose_weight(name):
        """Подобрать долю отдельного прогноза по фиксированной сетке."""
        delta = projected[name] - projected[CLASSICAL]
        return min((score(projected[CLASSICAL] + w * delta), w) for w in GRID)[1]

    selected = {}
    add(CLASSICAL, {CLASSICAL: 1.0})
    rw_weight = choose_weight(RW)
    add("ensemble_plus_rw", {CLASSICAL: 1 - rw_weight, RW: rw_weight})
    variants = [key for key in raw if key not in (CLASSICAL, RW)]
    fixed = [f"{model}/{MODES[0] if model == 'exaone_finance' else MODES[1]}" for model in names]
    add(
        "fixed_nn_10",
        {CLASSICAL: 1 - FIXED_WEIGHT, **dict.fromkeys(fixed, FIXED_WEIGHT / len(fixed))},
    )
    for model in names:
        options = []
        for name in variants:
            if name.split("/", 1)[0] != model:
                continue
            weight = choose_weight(name)
            weights = {CLASSICAL: 1 - weight, name: weight}
            options.append(
                (score(sum(projected[key] * w for key, w in weights.items())), name, weights)
            )
        _, _, weights = min(options)
        add(f"pair_{model}", weights)

    for key, components in (
        ("joint_nn", [CLASSICAL, *variants]),
        ("joint_nn_rw", [CLASSICAL, *variants, RW]),
    ):
        inputs_per_fold = [[projected[name][mask] for mask in folds] for name in components]
        weights = blend_weights(inputs_per_fold, splits)
        add(key, dict(zip(components, weights, strict=True)))

    short = HORIZONS <= LAST_SHORT
    options = {model: [] for model in names}
    for name in variants:
        delta = raw[name] - raw[CLASSICAL]
        short_delta = apply_calendar(delta * short, np.zeros_like(last), changes, pool=True)
        long_delta = apply_calendar(delta * ~short, np.zeros_like(last), changes, pool=True)
        candidates = (
            (score(projected[CLASSICAL] + a * short_delta + b * long_delta), a, b)
            for a, b in product(GRID, repeat=2)
        )
        loss, shorter, longer = min(candidates)
        options[name.split("/", 1)[0]].append((loss, name, shorter, longer))
    by_model = {f"split_{model}": min(candidates) for model, candidates in options.items()}
    for key, (loss, name, shorter, longer) in {
        "split_nn": min(by_model.values()),
        **by_model,
    }.items():
        mix = np.where(short, shorter, longer)
        raw_test = test_raw[CLASSICAL] + mix * (test_raw[name] - test_raw[CLASSICAL])
        selected[key] = (
            loss,
            apply_calendar(raw_test, test_last, test_changes, pool=True),
            {CLASSICAL: 1 - shorter, name: shorter},
            {CLASSICAL: 1 - longer, name: longer},
        )
    return selected


def summarize(choices, test_inputs):
    """Сохранить общие метрики и парный интервал относительно классического ансамбля."""
    dates, actual, last, _, changes, *_ = test_inputs
    rng = np.random.default_rng(191)
    indices = rng.integers(len(dates), size=(2000, (len(dates) + 27) // 28))
    indices = ((indices[:, :, None] + np.arange(28)) % len(dates)).reshape(2000, -1)
    indices = indices[:, : len(dates)]
    rw = np.sqrt(((last[:, None] - actual) ** 2)[indices].mean(axis=1))
    old = choices[CLASSICAL][1]
    old_boot = np.sqrt(((old - actual) ** 2)[indices].mean(axis=1))
    fixing = changes[:, 0]
    valid = (rw > 0).all(axis=1)
    summary, detailed, predictions, weights = [], [], [], []
    for name, (loss, prediction, shorter, longer) in choices.items():
        ratios = relative_rmse(prediction, actual, last)
        errors = prediction - actual
        rmse = np.sqrt((errors**2).mean(axis=0))
        fitted_boot = np.sqrt((errors**2)[indices].mean(axis=1))
        low, high = np.quantile(
            ((fitted_boot - old_boot)[valid] / rw[valid]).mean(axis=1) * 100, [0.025, 0.975]
        )
        changes_test = changes.cumsum(axis=1) > 0
        np.testing.assert_array_equal(
            prediction[~changes_test], np.broadcast_to(last[:, None], actual.shape)[~changes_test]
        )
        summary.append(
            {
                "candidate": name,
                "validation_rmse_to_rw": loss,
                "test_rmse_to_rw": ratios.mean(),
                "rmse_1d": rmse[0],
                "rmse_3d": rmse[2],
                "rmse_7d": rmse[6],
                "rmse_1d_fixing": np.sqrt((errors[fixing, 0] ** 2).mean()),
                "delta_ci_low_pp": low,
                "delta_ci_high_pp": high,
            }
        )
        detailed.extend(
            {"candidate": name, "horizon": h, "rmse": error, "rmse_to_rw": ratio}
            for h, error, ratio in zip(HORIZONS, rmse, ratios, strict=True)
        )
        predictions.append(
            pd.DataFrame(
                {
                    "candidate": name,
                    "origin": np.repeat(dates, len(HORIZONS)),
                    "horizon": np.tile(HORIZONS, len(dates)),
                    "actual": actual.ravel(),
                    "prediction": prediction.ravel(),
                }
            )
        )
        for interval, values in (("1–3", shorter), ("4–7", longer)):
            weights.extend(
                {"candidate": name, "model": key, "horizons": interval, "weight": value}
                for key, value in values.items()
                if value > 1e-9
            )
    return (
        pd.DataFrame(summary).sort_values("validation_rmse_to_rw", kind="stable"),
        pd.DataFrame(detailed),
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(weights),
    )


def run(args):
    """Оценить итоговые кандидаты на тех же 680 датах после подбора весов."""
    calibration = experiment_inputs(args)
    cal_raw, cal_projected, cal_files = archived_forecasts(
        args.calibration, args.models, calibration
    )
    test = experiment_inputs(args, use_oof=False)
    if calibration[0].max() + pd.Timedelta(days=int(HORIZONS[-1])) >= test[0].min():
        raise ValueError("Цели подбора весов пересекают тест")
    test_raw, _, test_files = archived_forecasts(args.neural_test, args.models, test)
    for a, b in zip(cal_files[1::2], test_files[1::2], strict=True):
        cal_info, test_info = json.loads(a.read_text()), json.loads(b.read_text())
        for key in ("repo", "revision", "weights_sha256"):
            if cal_info[key] != test_info[key]:
                raise ValueError(f"Разные чекпойнты: {a}, {b}")

    choices = evaluate_weights((calibration, cal_raw, cal_projected), (test, test_raw), args.models)
    summary, metrics, predictions, weights = summarize(choices, test)
    for frame, name in (
        (summary, "summary.csv"),
        (metrics, "metrics.csv"),
        (weights, "weights.csv"),
        (predictions, "predictions.csv.gz"),
    ):
        write_csv(frame, args.output / name)
    metadata = {
        "selected_on_validation": summary.iloc[0].candidate,
        "calibration_dates": len(calibration[0]),
        "calibration_years": sorted(map(int, set(calibration[0].year))),
        "test_dates": len(test[0]),
        "grid": list(GRID),
        "models": args.models,
        "input_files": input_fingerprints(
            [args.dataset, args.calendar, args.oof, args.holdout, *cal_files, *test_files]
        ),
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(weights[weights.candidate.eq(metadata["selected_on_validation"])].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("data/reports")
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/usd_rub_2011_2026_full.csv.gz")
    )
    parser.add_argument("--oof", type=Path, default=root / "llm_news_holdout/baseline_oof.csv.gz")
    parser.add_argument(
        "--holdout", type=Path, default=root / "ml_holdout_minfin_rosstat/with_file/predictions.csv"
    )
    parser.add_argument("--calendar", type=Path, default=Path("config/cbr_fixing_calendar.json"))
    parser.add_argument(
        "--calibration", type=Path, default=root / "foundation_ensemble/calibration"
    )
    parser.add_argument("--neural-test", type=Path, default=root / "foundation_holdout")
    parser.add_argument("--output", type=Path, default=root / "foundation_ensemble")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error("Модели должны быть заданы по одному разу")
    run(args)


if __name__ == "__main__":
    main()
