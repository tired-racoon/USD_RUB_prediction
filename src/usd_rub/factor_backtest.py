"""Последовательное сравнение DFM со всеми эталонными моделями."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from usd_rub.backtest import daily_target, regime_scores, score, walk_forward
from usd_rub.dfm import TwoStepFactorModel
from usd_rub.factor_data import VintagePanel
from usd_rub.freshness import require_current
from usd_rub.storage import read_series, utc_now


def factor_walk_forward(
    y,
    panel,
    settings,
    min_train=730,
    window=730,
    horizon=7,
    step=1,
    alpha=0.05,
    test_start="2017-01-01",
    progress=False,
):
    if not isinstance(y.index, pd.DatetimeIndex) or not y.index.equals(
        pd.date_range(y.index.min(), y.index.max(), freq="D")
    ):
        raise ValueError("DFM backtest требует непрерывную дневную цель")
    if min_train < 10 or window < min_train or step < 1 or not 1 <= horizon <= 7:
        raise ValueError("Некорректные параметры DFM backtest")
    first = max(min_train - 1, int(y.index.searchsorted(pd.Timestamp(test_start))))
    rows, diagnostics = [], []
    started = time.monotonic()
    for iteration, origin in enumerate(range(first, len(y) - horizon, step), 1):
        start = max(0, origin + 1 - window)
        history = y.iloc[start : origin + 1]
        cutoff = y.index[origin].tz_localize("Europe/Moscow")
        snapshot = panel.snapshot(cutoff).reindex(history.index)
        model = TwoStepFactorModel(**settings).fit(snapshot.to_numpy(), history.to_numpy())
        forecast = model.predict(horizon, alpha)
        if not np.isfinite(forecast.mean).all():
            raise ValueError(f"Неконечный DFM прогноз: {cutoff}")
        scale = float(np.mean(np.abs(np.diff(history))))
        for h in range(1, horizon + 1):
            rows.append(
                {
                    "model": "dfm_giannone_" + panel.mode,
                    "origin": y.index[origin],
                    "horizon": h,
                    "target_date": y.index[origin + h],
                    "last": history.iloc[-1],
                    "actual": y.iloc[origin + h],
                    "prediction": forecast.mean[h - 1],
                    "lower": forecast.lower[h - 1],
                    "upper": forecast.upper[h - 1],
                    "mase_scale": scale,
                    "train_size": len(history),
                }
            )
        diagnostics.append(
            {
                "model": "dfm_giannone_" + panel.mode,
                "origin": y.index[origin],
                "cutoff": str(cutoff),
                "balanced_first": str(history.index[model.balanced_first_].date()),
                "balanced_last": str(history.index[model.balanced_last_].date()),
                "balanced_size": model.balanced_size_,
                "selected_series": ",".join(snapshot.columns[model.columns_]),
                "n_series": len(model.columns_),
                "raw_spectral_radius": model.raw_spectral_radius_,
                "explained_variance_ratio": model.explained_variance_ratio_,
                "last_state_variance_trace": np.trace(model.states_.filtered_covariance[-1]),
                "bridge_variance": model.bridge_variance_,
            }
        )
        if progress and (iteration == 1 or iteration % 100 == 0):
            print(
                f"DFM {panel.mode}: {iteration} дат, {cutoff.date()}, "
                f"{len(model.columns_)} рядов, {time.monotonic() - started:.1f} с",
                flush=True,
            )
    if not rows:
        raise ValueError("Недостаточно данных для DFM backtest")
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def run_comparison(args):
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_dir)
    frames = {name: read_series(root, name) for name in config["series"]}
    target = read_series(root, args.target)
    if args.target in frames:
        raise ValueError("Цель не должна входить в панель индикаторов")
    for frame in [target, *frames.values()]:
        require_current(frame, as_of=args.end)
    y = daily_target(target, args.start, args.end)
    paths = [root / "normalized" / f"{name}.csv" for name in [args.target, *frames]]
    paths += [config_path] + [
        Path(__file__).parent / name
        for name in ["dfm.py", "factor_data.py", "factor_backtest.py", "backtest.py", "models.py"]
    ]

    def fingerprints():
        return [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths
        ]

    inputs = fingerprints()
    settings = {
        key: config[key]
        for key in ["factors", "shocks", "min_balanced", "stability_radius", "variance_floor"]
    }
    modes = ["strict", "assumed"] if args.mode == "both" else [args.mode]
    predictions, diagnostics = [], []
    for mode in modes:
        panel = VintagePanel(
            frames,
            config["series"],
            args.start,
            args.end,
            mode,
            config["interior_fill_limit_days"],
        )
        result, log = factor_walk_forward(
            y,
            panel,
            settings,
            args.min_train,
            args.window,
            args.horizon,
            args.step,
            args.alpha,
            args.test_start,
            progress=True,
        )
        result.to_csv(output / f"predictions_{mode}.csv", index=False)
        log.to_csv(output / f"diagnostics_{mode}.csv", index=False)
        predictions.append(result)
        diagnostics.append(log)
    print("Пересчёт семи бейзлайнов на тех же датах", flush=True)
    baselines = walk_forward(
        y,
        min_train=args.min_train,
        window=args.window,
        horizon=args.horizon,
        step=args.step,
        alpha=args.alpha,
        test_start=args.test_start,
    )
    predictions.append(baselines)
    combined = pd.concat(predictions, ignore_index=True)
    expected = combined.loc[combined.model.eq("rw"), ["origin", "horizon", "actual"]]
    for name, part in combined.groupby("model"):
        check = part.merge(
            expected, on=["origin", "horizon"], validate="one_to_one", suffixes=("", "_rw")
        )
        if (
            len(check) != len(expected)
            or len(part) != len(expected)
            or not np.array_equal(check.actual, check.actual_rw)
        ):
            raise ValueError(f"Несопоставимые даты или фактические значения: {name}")
    if inputs != fingerprints():
        raise ValueError("Входные файлы изменились во время сравнения")
    combined.to_csv(output / "predictions.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(output / "diagnostics.csv", index=False)
    metrics = score(combined, args.step)
    metrics.to_csv(output / "metrics.csv", index=False)
    regime_scores(combined, args.step).to_csv(output / "regime_metrics.csv", index=False)
    record = {
        "created_at": utc_now(),
        "parameters": vars(args),
        "configuration": config,
        "n_target": len(y),
        "target_first": str(y.index.min()),
        "target_last": str(y.index.max()),
        "origin_time": "00:00 Europe/Moscow",
        "horizon_unit": "calendar days",
        "inputs": inputs,
        "availability": {
            "strict": "Stored available_at, unknown publication dates remain at ingestion; exchange close lags are conservative assumptions",
            "assumed": "Additional explicit lags for unknown dates; latest vintages, not fully historical point-in-time data",
        },
        "adaptation": "Daily stationary financial indicators and log USD/RUB returns; no monthly-to-quarterly GDP bridge",
        "selection": "r=2, q=2 and window fixed before evaluating forecasts; no tuning on test data",
    }
    (output / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(
        metrics.loc[
            metrics.horizon.isin([1, args.horizon]),
            [
                "model",
                "horizon",
                "n",
                "rmse",
                "mape_pct",
                "smape_pct",
                "rmspe_pct",
                "rmse_skill_vs_rw",
            ],
        ].to_string(index=False),
        flush=True,
    )
    print(f"Результаты: {output}", flush=True)
    return 0
