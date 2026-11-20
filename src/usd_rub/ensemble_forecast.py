"""Основной прогноз USD/RUB с t0-gamma и рыночной поправкой XGBoost."""

import json
from hashlib import file_digest
from pathlib import Path

import numpy as np
import pandas as pd

from usd_rub.backtest import daily_target
from usd_rub.fixing_calendar import HORIZONS, apply_calendar
from usd_rub.freshness import require_current
from usd_rub.market_correction import (
    MARKET_BUNDLE,
    MARKET_FRACTION,
    load_market_bundle,
    market_forecast,
)
from usd_rub.ml_dataset import feature_panels, read_dataset
from usd_rub.ml_experiment import Candidate, fit_predict, specific_features, to_levels
from usd_rub.ml_features import load_panel
from usd_rub.storage import input_fingerprints, read_series
from usd_rub.t0_gamma import load_t0_gamma, market_inputs

BASE_MODEL = "ensemble_t0_gamma"
MODEL = "ensemble_t0_gamma_xgb_market"
CLASSICAL = (
    (Candidate("ridge", "pilot", None, 0.15), 0.43838276164287915, 1.0),
    (Candidate("histboost", "market", 1460, 3), 0.3844272108930251, 0.75),
    (Candidate("extra_trees", "market", 1460, 3, pooled=True), 0.17719002746409557, 1.0),
)
GAMMA_WEIGHTS = np.where(HORIZONS >= 4, 0.15, 0)
GAMMA_REVISION = "fef67abad3eed4f227c7b536a6e8ee4e9ab1080d"
GAMMA_SHA256 = "1603e255629be22ea45518b502f16c55d06821c7090a1a851673e779e9c245a6"
MIN_HIT_RATE = 0.5


def classical_predictions(panels, levels, split, origins):
    """Посчитать три классических прогноза с параметрами из holdout."""
    prediction = np.zeros((len(origins), len(HORIZONS)))
    for candidate, weight, shrink in CLASSICAL:
        features = specific_features(panels, candidate.setup)
        supported = features.iloc[:split].notna().mean().ge(MIN_HIT_RATE)
        raw, _ = fit_predict(
            features.loc[:, supported].to_numpy(dtype=float),
            levels,
            candidate,
            split,
            origins,
            HORIZONS,
            min_hit_rate=MIN_HIT_RATE,
        )
        prediction += weight * to_levels(raw, levels[origins], shrink)
    return prediction


def blend_predictions(classical, gamma, last, changes):
    """Перейти к смеси 85/15 с четвёртого дня и применить календарь ЦБ."""
    classic, gamma = np.asarray(classical, dtype=float), np.asarray(gamma, dtype=float)
    if gamma.shape != classic.shape or gamma.shape != changes.shape:
        raise ValueError("Прогнозы компонентов и календарь должны быть на общих горизонтах")
    if not (np.isfinite(gamma).all() and np.isfinite(classic).all()):
        raise ValueError("Прогноз компонента содержит пропуски или бесконечные значения")
    return apply_calendar(classic + GAMMA_WEIGHTS * (gamma - classic), last, changes, pool=True)


def forecast_inputs(args, today):
    """Взять срез из локальных рядов или исторического снимка."""
    data_dir = Path(args.data_dir)
    if args.dataset:
        dataset = read_dataset(args.dataset)
        end = pd.Timestamp(args.end) if args.end else min(today, dataset.index.max())
        if end not in dataset.index:
            raise ValueError("Дата прогноза отсутствует в снимке")
        dataset = dataset.loc[:end]
        return dataset.usd_rub, feature_panels(dataset), [Path(args.dataset)]

    frame = read_series(data_dir, "usd_rub")
    end = pd.Timestamp(args.end) if args.end else min(today, frame.date.max())
    require_current(frame, as_of=end if args.end else today)
    config = json.loads(Path(args.config).read_text())
    target = daily_target(frame, config["start"], end)
    panels, paths = load_panel(data_dir, config, target)
    return target, panels, [data_dir / "normalized/usd_rub.csv", Path(args.config), *paths]


def run_forecast(args, today):
    """Вычислить основной прогноз и записать его параметры в метаданные."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise ValueError("Для основного прогноза нужен uv run --extra ensemble") from exc

    target, panels, paths = forecast_inputs(args, pd.Timestamp(today))
    if not target.index.equals(pd.date_range(target.index.min(), target.index.max())):
        raise ValueError("Ожидается непрерывный календарный ряд USD/RUB")
    levels = target.to_numpy(dtype=float)
    if not (np.isfinite(levels).all() and (levels > 0).all()):
        raise ValueError("Ожидается конечный положительный курс USD/RUB")
    origin = target.index[-1]
    cutoff = pd.Timestamp(args.train_before) if args.train_before else origin
    if cutoff > origin:
        raise ValueError("--train-before должен быть не позже даты прогноза")
    split = target.index.searchsorted(cutoff)
    if split <= HORIZONS[-1]:
        raise ValueError("До --train-before нет завершённых семидневных наблюдений")
    name, correction = BASE_MODEL, {}
    if not args.base_only:
        models, metadata = load_market_bundle()
        if cutoff < pd.Timestamp(metadata["trained_before"]):
            raise ValueError(
                f"Поправка XGBoost обучена до {metadata['trained_before']}. "
                "Для более ранней границы используйте --base-only"
            )
        name = MODEL
        correction = {"market_correction": {"fraction": MARKET_FRACTION, **metadata}}
        paths.append(MARKET_BUNDLE)
    query = target.index[-1:]
    calendar = json.loads(Path(args.calendar).read_text())
    level, past, known, changes = market_inputs(target, panels["extended"], query, calendar)

    checkpoint = args.checkpoint or Path(args.data_dir) / "model_runtime/checkpoints/t0_gamma"
    with (checkpoint / "model.safetensors").open("rb") as file:
        if file_digest(file, "sha256").hexdigest() != GAMMA_SHA256:
            raise ValueError("Контрольная сумма весов отличается от проверенного t0-gamma")
    with threadpool_limits(limits=2):
        classical = classical_predictions(panels, levels, split, np.array([len(levels) - 1]))
    gamma = load_t0_gamma(checkpoint)(level, past, known)
    prediction = blend_predictions(classical, gamma, levels[-1:], changes)
    if not args.base_only:
        prediction = market_forecast(
            models, panels["extended"].loc[query], prediction, levels[-1:], changes
        )
    result = pd.DataFrame(
        {
            "model": name,
            "origin": origin,
            "target_date": origin + pd.to_timedelta(HORIZONS, unit="D"),
            "horizon": HORIZONS,
            "prediction": prediction[0],
        }
    )
    run = {
        "model": name,
        "origin": str(origin.date()),
        "train_before": str(cutoff.date()),
        "classical": {cfg.name: {"weight": w, "shrink": s} for cfg, w, s in CLASSICAL},
        "t0_gamma": {
            "revision": GAMMA_REVISION,
            "weights_sha256": GAMMA_SHA256,
            "weights_by_horizon": GAMMA_WEIGHTS.tolist(),
        },
        "inputs": input_fingerprints([*paths, args.calendar, checkpoint / "config.json"]),
        **correction,
    }
    return result, run
