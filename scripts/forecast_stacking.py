"""Собрать архив прогнозов и проверить стек моделей USD/RUB."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from correlation_holdout import HORIZONS, apply_calendar
from foundation_ensemble import CLASSICAL, RW, archived_forecasts, summarize
from foundation_holdout import MODELS, experiment_inputs
from llm_news_holdout import bootstrap
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from usd_rub.ml_experiment import blend_weights, selection_loss
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints

ALPHAS = (0.003, 0.02, 0.1, 0.5, 2.0, 10.0)
KEYS = ["origin", "horizon"]
ARCHIVE_FILES = ("calibration_forecasts.csv.gz", "test_forecasts.csv.gz")


@dataclass
class Forecasts:
    """Прогнозы для одних и тех же дат и горизонтов."""

    dates: pd.DatetimeIndex
    actual: np.ndarray
    last: np.ndarray
    fixes: np.ndarray
    raw: dict
    projected: dict

    def project(self, prediction, rows=slice(None)):
        """Усреднить прогнозы одного фиксинга и сохранить известные курсы."""
        return apply_calendar(prediction, self.last[rows], self.fixes[rows], pool=True)

    def split(self, rows):
        """Вернуть цели для сопоставимой метрики."""
        return self.dates[rows], self.actual[rows], self.last[rows]


def panel_index(dates):
    """Порядок дат и горизонтов в каждом архиве."""
    return pd.MultiIndex.from_product([dates, HORIZONS], names=KEYS)


def read_long(path, names, inputs):
    """Прочитать ранее рассчитанные прогнозы и проверить их совмещение."""
    dates, actual, last, *_ = inputs
    expected = panel_index(dates)
    frame = pd.read_csv(path, parse_dates=["origin"], float_precision="round_trip")
    result = {}
    for name in names:
        model = frame.loc[frame.model.eq(name)].set_index(KEYS).sort_index()
        pd.testing.assert_index_equal(model.index, expected, obj=name)
        if "actual" in model:
            np.testing.assert_allclose(model.actual, actual.ravel(), rtol=1e-12)
            np.testing.assert_allclose(model["last"], np.repeat(last, len(HORIZONS)), rtol=1e-12)
        if "target_date" in model:
            targets = expected.get_level_values("origin") + pd.to_timedelta(
                expected.get_level_values("horizon"), unit="D"
            )
            np.testing.assert_array_equal(pd.to_datetime(model.target_date), targets)
        result[name] = model.prediction.to_numpy().reshape(actual.shape)
    return result


def prepare(args):
    """Выпустить независимые от обучения архивы калибровки и теста."""
    args.output.mkdir(parents=True, exist_ok=True)
    sequence_file = args.output / "sequence_oof.csv.gz"
    classical_file = args.output / "classical_oof.csv.gz"
    sequence_meta = sequence_file.with_suffix("").with_suffix(".json")
    classical_meta = classical_file.with_suffix("").with_suffix(".json")
    sequence_names = json.loads(sequence_meta.read_text())["models"]
    classical_names = json.loads(classical_meta.read_text())["models"]
    groups = {
        "classical": classical_names + [key for key in sequence_names if key.startswith("ridge_")],
        "sequence": [key for key in sequence_names if not key.startswith("ridge_")],
    }
    dependencies = [
        args.dataset,
        args.calendar,
        args.oof,
        args.holdout,
        sequence_meta,
        sequence_file,
        classical_meta,
        classical_file,
        args.sequence_test,
    ]
    checkpoint_metadata = {}
    for is_calibration, filename in zip((True, False), ARCHIVE_FILES, strict=True):
        inputs = experiment_inputs(args, use_oof=is_calibration)
        dates, actual, last, _, fixes, *_ = inputs
        directory = args.foundation_calibration if is_calibration else args.foundation_test
        raw, _, sources = archived_forecasts(directory, MODELS, inputs)
        for manifest in sources[1::2]:
            meta = json.loads(manifest.read_text())
            checkpoint = (meta["repo"], meta["revision"], meta["weights_sha256"])
            if checkpoint_metadata.setdefault(manifest.stem, checkpoint) != checkpoint:
                raise ValueError(f"Разные чекпойнты для калибровки и теста: {manifest}")
        dependencies.extend(sources)
        groups["foundation"] = [name for name in raw if name not in (CLASSICAL, RW)]
        raw.update(
            read_long(classical_file if is_calibration else args.holdout, classical_names, inputs)
        )
        raw.update(
            read_long(
                sequence_file if is_calibration else args.sequence_test, sequence_names, inputs
            )
        )
        index = panel_index(dates)
        frame = pd.DataFrame(
            {
                "actual": actual.ravel(),
                "last": np.repeat(last, len(HORIZONS)),
                "fixing": fixes.ravel(),
                **{name: raw[name].ravel() for name in (CLASSICAL, RW, *sum(groups.values(), []))},
            },
            index=index,
        )
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Неконечные значения в архиве")
        write_csv(frame.reset_index(), args.output / filename)
        print(filename, len(dates), len(raw), flush=True)
    metadata = {
        "groups": groups,
        "inputs": input_fingerprints(dependencies),
        "archives": input_fingerprints([args.output / name for name in ARCHIVE_FILES]),
    }
    (args.output / "archives.json").write_text(json.dumps(metadata, indent=2) + "\n")


def read_panel(path, names):
    """Восстановить матрицы из архива без загрузки базовых моделей."""
    columns = [*KEYS, "actual", "last", "fixing", *names]
    frame = pd.read_csv(path, parse_dates=["origin"], float_precision="round_trip")
    if (
        set(frame) != set(columns)
        or not np.isfinite(frame[columns[1:]].to_numpy(dtype=float)).all()
    ):
        raise ValueError(f"Неверный формат архива {path}")
    dates = pd.DatetimeIndex(frame.origin.unique())
    pd.testing.assert_index_equal(frame.set_index(KEYS).index, panel_index(dates))
    if not dates.is_monotonic_increasing or not frame.fixing.isin([0, 1, False, True]).all():
        raise ValueError(f"Неверный календарь {path}")

    def matrix(name):
        return frame[name].to_numpy().reshape(len(dates), len(HORIZONS))

    last = matrix("last")[:, 0]
    fixes = matrix("fixing").astype(bool)
    np.testing.assert_array_equal(matrix("last"), np.broadcast_to(last[:, None], fixes.shape))
    raw = {key: matrix(key) for key in names}
    projected = {key: apply_calendar(value, last, fixes, pool=True) for key, value in raw.items()}
    return Forecasts(dates, matrix("actual"), last, fixes, raw, projected)


def ridge_deltas(calibration, test, names, folds):
    """Выбрать регуляризацию на расширяющихся исторических окнах."""
    anchor = calibration.projected[CLASSICAL]
    features = np.stack([calibration.projected[name] - anchor for name in names], axis=-1)
    target = calibration.actual - anchor
    test_anchor = test.projected[CLASSICAL]
    test_features = np.stack([test.projected[name] - test_anchor for name in names], axis=-1)

    def coefficients(train, alpha):
        """Оценить поправку к классическому прогнозу по каждому горизонту."""
        rw_scale = np.sqrt(
            ((calibration.actual[train] - calibration.last[train, None]) ** 2).mean(axis=0)
        ).clip(min=1e-8)
        weights = []
        for h, scale in enumerate(rw_scale):
            ridge = Ridge(alpha=alpha * np.count_nonzero(train), fit_intercept=False)
            ridge.fit(features[train, h, :] / scale, target[train, h] / scale)
            weights.append(ridge.coef_)
        return np.asarray(weights)

    def predict(base, values, weights):
        return base + np.einsum("nhk,hk->nh", values, weights)

    splits = [calibration.split(check) for _, check in folds]
    scores = []
    for alpha in ALPHAS:
        validation = [
            calibration.project(
                predict(anchor[check], features[check], coefficients(train, alpha)), check
            )
            for train, check in folds
        ]
        scores.append((selection_loss(validation, splits), -alpha))
    loss, negative_alpha = min(scores)
    alpha = -negative_alpha
    weights = coefficients(np.ones(len(calibration.dates), dtype=bool), alpha)
    prediction = test.project(predict(test_anchor, test_features, weights))
    return loss, prediction, np.column_stack((1 - weights.sum(axis=1), weights)), alpha


def convex_blend(calibration, test, names, folds):
    """Обучить неотрицательные веса, общие для всех горизонтов."""
    components = [CLASSICAL, *names]

    def weights_on(train):
        dates = calibration.dates
        groups = [train & (dates.year == year) for year in sorted(set(dates.year[train]))]
        splits = [calibration.split(group) for group in groups]
        predictions = [
            [calibration.projected[name][group] for group in groups] for name in components
        ]
        return blend_weights(predictions, splits)

    validation = []
    for train, check in folds:
        weights = weights_on(train)
        validation.append(
            sum(
                calibration.projected[name][check] * weight
                for name, weight in zip(components, weights, strict=True)
            )
        )
    loss = selection_loss(validation, [calibration.split(check) for _, check in folds])
    weights = weights_on(np.ones(len(calibration.dates), dtype=bool))
    prediction = test.project(
        sum(test.raw[name] * weight for name, weight in zip(components, weights, strict=True))
    )
    return loss, prediction, np.broadcast_to(weights, (len(HORIZONS), len(weights))), None


def fit(args):
    """Настроить стек на калибровке и один раз оценить зафиксированные варианты."""
    metadata = json.loads((args.output / "archives.json").read_text())
    if metadata["archives"] != input_fingerprints([args.output / name for name in ARCHIVE_FILES]):
        raise ValueError("Архивы прогнозов изменились после сборки")
    groups = metadata["groups"]
    names = [CLASSICAL, RW, *sum(groups.values(), [])]
    calibration, test = [read_panel(args.output / name, names) for name in ARCHIVE_FILES]
    last_horizon = pd.Timedelta(days=int(HORIZONS[-1]))
    dates = calibration.dates
    if dates[-1] + last_horizon >= test.dates[0]:
        raise ValueError("Цели калибровки пересекают тест")
    folds = []
    for year in sorted(set(dates.year))[1:]:
        check = dates.year == year
        train = dates + last_horizon < dates[check][0]
        folds.append((train, check))
    baseline_loss = selection_loss(
        [calibration.projected[CLASSICAL][check] for _, check in folds],
        [calibration.split(check) for _, check in folds],
    )
    choices = {CLASSICAL: (baseline_loss, test.projected[CLASSICAL], {}, {})}
    coefficients = [
        {"candidate": CLASSICAL, "model": CLASSICAL, "horizon": h, "weight": 1.0} for h in HORIZONS
    ]
    settings = {CLASSICAL: {"alpha": None, "components": [CLASSICAL]}}
    classical = groups["classical"]
    panels = {
        "classical": classical,
        "sequence": [*classical, *groups["sequence"]],
        "foundation": [*classical, *groups["foundation"]],
        "all": sum(groups.values(), []),
    }
    architectures = sorted({name.split("_", 1)[0] for name in groups["sequence"]})
    architecture_panels = {
        name: [*classical, *(key for key in groups["sequence"] if key.startswith(f"{name}_"))]
        for name in architectures
    }
    controls = {"rescale": [RW], **{f"{key}_rw": [*panel, RW] for key, panel in panels.items()}}
    for method, estimator, inputs in (
        ("ridge", ridge_deltas, panels | architecture_panels | controls),
        ("convex", convex_blend, panels),
    ):
        for group, components in inputs.items():
            name = f"{method}_{group}"
            keys = [CLASSICAL, *components]
            loss, prediction, weights, alpha = estimator(calibration, test, components, folds)
            choices[name] = (loss, prediction, {}, {})
            settings[name] = {"alpha": alpha, "components": keys}
            coefficients.extend(
                {"candidate": name, "model": key, "horizon": h, "weight": weight}
                for h, row in zip(HORIZONS, weights, strict=True)
                for key, weight in zip(keys, row, strict=True)
            )
    earlier = test.project(
        test.raw[CLASSICAL]
        + np.where(HORIZONS > 3, 0.15, 0)
        * (test.raw["t0_gamma/market_calendar"] - test.raw[CLASSICAL])
    )
    choices["previous_t0_15"] = (np.nan, earlier, {}, {})

    summary, metrics, predictions, _ = summarize(
        choices, (test.dates, test.actual, test.last, test.raw[CLASSICAL], test.fixes)
    )
    by_name = summary.set_index("candidate").test_rmse_to_rw
    comparisons = []
    for candidate, reference in (
        ("ridge_sequence", "ridge_classical"),
        ("ridge_all", "ridge_classical"),
        ("ridge_all", "ridge_foundation"),
        ("ridge_sequence_rw", "ridge_classical_rw"),
        ("ridge_all_rw", "ridge_classical_rw"),
        ("ridge_all_rw", "ridge_foundation_rw"),
        ("ridge_all_rw", "ridge_rescale"),
    ):
        low, high = bootstrap(choices[reference][1], choices[candidate][1], test.actual, test.last)
        comparisons.append(
            {
                "candidate": candidate,
                "reference": reference,
                "gain_pp": 100 * (by_name[reference] - by_name[candidate]),
                "gain_ci_low_pp": low,
                "gain_ci_high_pp": high,
            }
        )
    for frame, filename in (
        (summary, "summary.csv"),
        (metrics, "metrics.csv"),
        (predictions, "predictions.csv.gz"),
        (pd.DataFrame(coefficients), "coefficients.csv"),
        (pd.DataFrame(comparisons), "paired_comparisons.csv"),
    ):
        write_csv(frame, args.output / filename)
    control_names = [f"ridge_{key}" for key in controls]
    run = {
        "selected_on_validation": summary.loc[~summary.candidate.isin(control_names)]
        .iloc[0]
        .candidate,
        "amplification_controls": control_names,
        "calibration_dates": len(calibration.dates),
        "test_dates": len(test.dates),
        "validation_years": [int(calibration.dates[check][0].year) for _, check in folds],
        "alphas": ALPHAS,
        "settings": settings,
        "inputs": metadata["archives"],
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("data/reports")
    parser.add_argument("action", choices=("prepare", "fit"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/usd_rub_2011_2026_full.csv.gz")
    )
    parser.add_argument("--oof", type=Path, default=root / "llm_news_holdout/baseline_oof.csv.gz")
    parser.add_argument(
        "--holdout", type=Path, default=root / "ml_holdout_minfin_rosstat/with_file/predictions.csv"
    )
    parser.add_argument("--calendar", type=Path, default=Path("config/cbr_fixing_calendar.json"))
    parser.add_argument(
        "--sequence-test", type=Path, default=root / "sequence_holdout/predictions.csv"
    )
    parser.add_argument(
        "--foundation-calibration", type=Path, default=root / "foundation_ensemble/calibration"
    )
    parser.add_argument("--foundation-test", type=Path, default=root / "foundation_holdout")
    parser.add_argument("--output", type=Path, default=root / "forecast_stacking")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        (prepare if args.action == "prepare" else fit)(args)


if __name__ == "__main__":
    main()
