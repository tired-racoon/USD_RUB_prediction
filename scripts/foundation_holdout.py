"""Сравнить предобученные модели на общем тесте USD/RUB."""

import argparse
import json
import resource
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
from correlation_holdout import HORIZONS, apply_calendar, fixing_mask, forecasts

from usd_rub.ml_dataset import EXTENDED_PREFIX, feature_panels
from usd_rub.ml_experiment import relative_rmse
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints

FEATURES = ("fx__cny_basis", "global__synthetic_basis", "fx__si_front_settle_change_1")
MODELS = (
    "timestral_28m",
    "tirex_2",
    "chronos_2",
    "t0_gamma",
    "ci_proto",
    "timesfm_3",
    "exaone_finance",
)
MODES = ("usd_only", "market_calendar")
HORIZON = len(HORIZONS)
LOOKBACK = 256


def place_ci_blocks_on_mps(backbone, torch):
    """Оставить FP64-нормализацию на CPU и выполнить блоки ci-proto на Metal."""
    device = torch.device("mps")

    def move_input(_, args):
        x, packing, future, reverse = args
        if packing is not None:
            packing = SimpleNamespace(
                **{
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in vars(packing).items()
                }
            )
        return x.to(device), packing, future.to(device), reverse.to(device)

    for block in backbone.stack:
        block.to(device)
        block.register_forward_pre_hook(move_input)
    backbone.stack[-1].register_forward_hook(lambda _block, _args, out: out.cpu())


def load_model(name, path, device, batch_size):
    """Загрузить модель и привести её медианный прогноз к формату [дата, горизонт]."""
    if name == "t0_gamma" and device == "mlx":
        import mlx.core as mx
        from t0.mlx import T0Forecaster, TimeSeries, VariateType

        model = T0Forecaster.from_pretrained(str(path)).compile()

        def predict(target, past, known):
            if past is None:
                return np.array(model.predict(target, HORIZON, [0.5]).median)
            context = np.concatenate((target[:, None], past), axis=1)
            series = TimeSeries.from_array(mx.array(context), mx.array(known), horizon=HORIZON)
            rows = np.arange(series.variates.shape[0])
            historical = (rows < len(target) * context.shape[1]) & (rows % context.shape[1] != 0)
            series.variate_type = mx.where(
                mx.array(historical[:, None]), VariateType.HISTORICAL, series.variate_type
            )
            return np.array(model.predict(series, HORIZON, [0.5]).median)

        return predict

    import torch

    torch.set_num_threads(4)
    if name == "timestral_28m":
        from timestral import TimestralForecaster

        model = TimestralForecaster.from_pretrained(path, device=device)
        return lambda target, past, known: (
            model.forecast(target, HORIZON, past_covariates=past, known_covariates=known)
            .median[:, 0]
            .cpu()
            .numpy()
        )
    if name == "tirex_2":
        from tirex2 import TimeseriesType
        from tirex2 import load_model as load_tirex

        model = load_tirex(path, device=device)
        median = model.quantiles.tolist().index(0.5)

        def predict(target, past, known):
            histories = [
                TimeseriesType(
                    torch.from_numpy(row[None]),
                    None if past is None else torch.from_numpy(past[i]),
                    None if known is None else torch.from_numpy(known[i]),
                )
                for i, row in enumerate(target)
            ]
            out = model.forecast(histories, HORIZON, batch_size=batch_size, output_type="numpy")
            return np.stack([item[0, median] for item in out])

        return predict
    if name == "chronos_2":
        from chronos import Chronos2Pipeline

        model = Chronos2Pipeline.from_pretrained(
            str(path), device_map=device, local_files_only=True
        )

        def predict(target, past, known):
            items = target[:, None]
            if past is not None:
                items = [
                    {
                        "target": row,
                        "past_covariates": dict(zip(FEATURES, past[i], strict=True))
                        | {"fixing": known[i, 0, :-HORIZON]},
                        "future_covariates": {"fixing": known[i, 0, -HORIZON:]},
                    }
                    for i, row in enumerate(target)
                ]
            _, out = model.predict_quantiles(
                items,
                HORIZON,
                quantile_levels=[0.5],
                batch_size=batch_size * (1 if past is None else 5),
                cross_learning=False,
            )
            return torch.cat(out).cpu().numpy()

        return predict
    if name == "ci_proto":
        from ci_mamba_inference import vectorize_recurrence
        from continuousintelligence import load_model as load_ci

        model = load_ci(path, device="cpu")
        model._tta.update(chunk_joint=0, joint_neighbours=0, bfloat16=False)
        vectorize_recurrence(model._model)
        if device == "mps":
            place_ci_blocks_on_mps(model._model, torch)
        median = model.quantiles.tolist().index(0.5)

        def predict(target, past, known):
            return model.forecast(
                target,
                HORIZON,
                past_covariates=past,
                future_covariates=known,
                series_ids=[str(i) for i in range(len(target))],
            )[:, median]

        return predict
    if name == "timesfm_3":
        from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster

        model = TimesFM3Forecaster.from_pretrained(
            str(path), device=device, per_core_batch_size=batch_size
        )
        return lambda target, past, known: np.stack(
            [
                item.forecast.reshape(-1)
                for item in model.predict_batch(
                    list(target),
                    HORIZON,
                    past_only_covariates=None if past is None else list(past),
                    past_future_covariates=None if known is None else list(known),
                )
            ]
        )
    if name == "exaone_finance":
        from exaone_forecast.finance import EXAONEFinanceForecaster

        model = EXAONEFinanceForecaster(path, device=device)
        return lambda target, past, known: model.point(target, HORIZON, batch_size=batch_size)
    raise ValueError(f"Неизвестная модель или устройство: {name}/{device}")


def experiment_inputs(args, *, use_oof=True):
    """Использовать общий набор дат и срезы рынка до 00:00 МСК."""
    columns = {"date", "usd_rub", *FEATURES}
    dataset = pd.read_csv(
        args.dataset,
        index_col="date",
        parse_dates=["date"],
        float_precision="round_trip",
        usecols=lambda name: name.removeprefix(EXTENDED_PREFIX) in columns,
    )
    oof = getattr(args, "oof", None) if use_oof else None
    if oof is None:
        dates, actual, last, base = forecasts(args.holdout, "ensemble", dataset.usd_rub)
    else:
        archive = forecasts(oof, "ensemble_oof", dataset.usd_rub)
        calibration = (archive[0].year >= 2021) & (
            archive[0] + pd.Timedelta(days=HORIZON) < pd.Timestamp("2025-01-01")
        )
        dates, actual, last, base = (values[calibration] for values in archive)
    if not dataset.index.equals(pd.date_range(dataset.index[0], periods=len(dataset), freq="D")):
        raise ValueError("В датасете пропущены календарные дни")
    calendar = json.loads(args.calendar.read_text())
    changes = fixing_mask(dates, calendar, True)
    schedule = fixing_mask(dataset.index - pd.Timedelta(days=1), calendar, True)[:, 0]
    positions = dataset.index.get_indexer(dates)
    history = positions[:, None] - np.arange(LOOKBACK - 1, -1, -1)
    if history.min() < 0:
        raise ValueError("Недостаточно истории USD/RUB")
    features = feature_panels(dataset)["extended"].loc[:, list(FEATURES)].to_numpy(np.float32)
    level = dataset.usd_rub.to_numpy(np.float32)[history]
    historical = features[history].transpose(0, 2, 1).copy()
    known = np.concatenate((schedule[history], changes), axis=1)[:, None].astype(np.float32)
    np.testing.assert_array_equal(level[:, -1], last.astype(np.float32))
    return dates, actual, last, base, changes, level, historical, known


def run_model(args, inputs):
    """Записать прогнозы и время отдельно для каждого набора входов."""
    dates, actual, last, _, changes, target, past, known = inputs
    path = args.checkpoints / args.model
    model = load_model(args.model, path, args.device, args.batch_size)
    modes = MODES[:1] if args.model == "exaone_finance" else MODES
    rows = []
    count = min(len(dates), args.limit) if args.limit else len(dates)
    for mode in modes:
        start_time = perf_counter()
        chunks = []
        for start in range(0, count, args.batch_size):
            batch = slice(start, min(count, start + args.batch_size))
            covariates = (None, None) if mode == "usd_only" else (past[batch], known[batch])
            median = np.asarray(model(target[batch], *covariates), dtype=np.float64)
            expected = (min(args.batch_size, count - start), HORIZON)
            if median.shape != expected or not np.isfinite(median).all():
                raise ValueError(f"{args.model}/{mode}: некорректный прогноз {median.shape}")
            if start == 0 and len(median) > 1:
                single_covariates = tuple(None if cov is None else cov[:1] for cov in covariates)
                single = model(target[:1], *single_covariates)
                np.testing.assert_allclose(median[:1], single, atol=0.02, rtol=1e-5)
            chunks.append(median)
            completed = min(count, start + args.batch_size)
            if start == 0 or completed == count or completed % (args.batch_size * 10) == 0:
                print(
                    args.model,
                    mode,
                    completed,
                    count,
                    round(perf_counter() - start_time, 2),
                    flush=True,
                )
        raw = np.concatenate(chunks)
        adjusted = apply_calendar(raw, last[:count], changes[:count], pool=True)
        rows.append(
            pd.DataFrame(
                {
                    "model": args.model,
                    "mode": mode,
                    "origin": np.repeat(dates[:count], HORIZON),
                    "horizon": np.tile(HORIZONS, count),
                    "actual": actual[:count].ravel(),
                    "prediction_raw": raw.ravel(),
                    "prediction": adjusted.ravel(),
                    "seconds": perf_counter() - start_time,
                }
            )
        )
    if args.limit:
        return
    write_csv(pd.concat(rows, ignore_index=True), args.output / f"{args.model}.csv.gz")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metadata = json.loads((path / "checkpoint.json").read_text()) | {
        "device": args.device,
        "batch_size": args.batch_size,
        "peak_rss_mib": rss * (1 if sys.platform == "darwin" else 1024) / 1024**2,
    }
    (args.output / f"{args.model}.json").write_text(json.dumps(metadata, indent=2) + "\n")


def summarize(args, inputs):
    """Собрать сопоставимые метрики и частные результаты по дням фиксинга."""
    dates, actual, last, base, changes, _, _, _ = inputs
    rw = np.broadcast_to(last[:, None], actual.shape)
    estimate = {
        ("random_walk", "baseline"): (rw, rw, 0),
        ("ensemble", "baseline"): (base, apply_calendar(base, last, changes, pool=True), 0),
    }
    for name in MODELS:
        path = args.output / f"{name}.csv.gz"
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["origin"], float_precision="round_trip")
        expected_modes = MODES[:1] if name == "exaone_finance" else MODES
        if set(frame["mode"]) != set(expected_modes) or not frame.model.eq(name).all():
            raise ValueError(f"Неполный набор прогнозов: {path}")
        for mode, group in frame.groupby("mode"):
            np.testing.assert_array_equal(group.origin, np.repeat(dates, HORIZON))
            np.testing.assert_array_equal(group.horizon, np.tile(HORIZONS, len(dates)))
            np.testing.assert_allclose(group.actual, actual.ravel(), rtol=1e-12)
            raw = group.prediction_raw.to_numpy().reshape(actual.shape)
            forecast = group.prediction.to_numpy().reshape(actual.shape)
            np.testing.assert_allclose(forecast, apply_calendar(raw, last, changes, pool=True))
            estimate[(name, mode)] = (raw, forecast, group.seconds.iloc[0])

    summary, details = [], []
    fixing_day = changes[:, 0]
    new_fixing = changes.cumsum(axis=1) > 0
    rng = np.random.default_rng(191)
    starts = rng.integers(0, len(dates), size=(2000, (len(dates) + 27) // 28))
    positions = ((starts[:, :, None] + np.arange(28)) % len(dates)).reshape(len(starts), -1)
    positions = positions[:, : len(dates)]
    rw_boot = np.sqrt(((rw - actual) ** 2)[positions].mean(axis=1))
    ensemble_boot = np.sqrt(
        ((estimate[("ensemble", "baseline")][1] - actual) ** 2)[positions].mean(axis=1)
    )
    for (name, mode), (raw, forecast, elapsed) in estimate.items():
        rmse = np.sqrt(np.mean((forecast - actual) ** 2, axis=0))
        ratios = relative_rmse(forecast, actual, last)
        forecast_boot = np.sqrt(((forecast - actual) ** 2)[positions].mean(axis=1))
        valid_draws = (rw_boot > 0).all(axis=1)
        differences = 100 * (
            (forecast_boot - ensemble_boot)[valid_draws] / rw_boot[valid_draws]
        ).mean(axis=1)
        lower, upper = np.quantile(differences, [0.025, 0.975])
        fixed_rmse = np.sqrt(
            np.sum(np.where(new_fixing, (forecast - actual) ** 2, 0), axis=0)
            / new_fixing.sum(axis=0)
        )
        details.extend(
            {
                "model": name,
                "mode": mode,
                "horizon": h,
                "rmse": rmse[i],
                "rmse_to_rw": ratios[i],
                "rmse_after_new_fixing": fixed_rmse[i],
                "after_new_fixing_n": new_fixing[:, i].sum(),
            }
            for i, h in enumerate(HORIZONS)
        )
        summary.append(
            {
                "model": name,
                "mode": mode,
                "rmse_to_rw": ratios.mean(),
                "raw_rmse_to_rw": relative_rmse(raw, actual, last).mean(),
                "rmse_1d": rmse[0],
                "rmse_3d": rmse[2],
                "rmse_7d": rmse[6],
                "rmse_1d_fixing": np.sqrt(
                    np.mean((forecast[fixing_day, 0] - actual[fixing_day, 0]) ** 2)
                ),
                "delta_vs_ensemble_ci_low_pp": lower,
                "delta_vs_ensemble_ci_high_pp": upper,
                "seconds": elapsed,
            }
        )
    rows = pd.DataFrame(summary).sort_values("rmse_to_rw")
    write_csv(rows, args.output / "summary.csv")
    write_csv(pd.DataFrame(details), args.output / "metrics.csv")
    metadata = {
        "first_origin": str(dates[0].date()),
        "last_origin": str(dates[-1].date()),
        "test_origins": len(dates),
        "same_day_fixing_origins": int(fixing_day.sum()),
        "lookback": LOOKBACK,
        "historical_covariates": list(FEATURES),
        "input_files": input_fingerprints([args.dataset, args.holdout, args.calendar]),
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(rows.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "summary"), required=True)
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/usd_rub_2011_2026_full.csv.gz")
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        default=Path("data/reports/ml_holdout_minfin_rosstat/with_file/predictions.csv"),
    )
    parser.add_argument("--calendar", type=Path, default=Path("config/cbr_fixing_calendar.json"))
    parser.add_argument("--oof", type=Path, help="Архив прогнозов для калибровки на 2021–2024 гг.")
    parser.add_argument("--checkpoints", type=Path, default=Path("data/model_runtime/checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/foundation_holdout"))
    parser.add_argument("--device", choices=("cpu", "mps", "mlx"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.device is None:
        args.device = "mlx" if args.model == "t0_gamma" else "cpu"
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Размер пакета и число дат должны быть положительными")
    if (
        args.oof is not None
        and args.output.resolve() == Path("data/reports/foundation_holdout").resolve()
    ):
        parser.error("Для калибровочных прогнозов укажите отдельный --output")
    if args.oof is not None and args.model == "summary":
        parser.error("Калибровочные прогнозы используются в foundation_ensemble.py")
    inputs = experiment_inputs(args)
    if args.model == "summary":
        summarize(args, inputs)
    else:
        run_model(args, inputs)


if __name__ == "__main__":
    main()
