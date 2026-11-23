"""Прогноз USD/RUB с календарём ЦБ и поправкой XGBoost."""

import json
from hashlib import file_digest
from pathlib import Path

import numpy as np
import pandas as pd

from usd_rub.backtest import daily_target
from usd_rub.fixing_calendar import HORIZONS, apply_calendar, fixing_mask
from usd_rub.freshness import require_current
from usd_rub.market_correction import (
    COMPACT_CORRECTION_FRACTION,
    COMPACT_MACRO_BUNDLE,
    CORRECTION_FRACTION,
    MACRO_BUNDLE,
    MARKET_BUNDLE,
    MARKET_FEATURES,
    corrected_forecast,
    load_correction_bundle,
)
from usd_rub.ml_dataset import feature_panels, read_dataset
from usd_rub.ml_experiment import Candidate, fit_predict, specific_features, to_levels
from usd_rub.ml_features import load_panel
from usd_rub.paper_features import MARKET_SERIES, market_features
from usd_rub.storage import Client, input_fingerprints, read_series
from usd_rub.t0_gamma import load_t0_gamma, market_inputs
from usd_rub.theory_features import (
    COMPACT_FUNDAMENTAL_SIGNS,
    COMPACT_MACRO_FEATURES,
    FUNDAMENTAL_SIGNS,
    MACRO_CORRECTION_FEATURES,
    read_theory_sources,
    theory_features,
)
from usd_rub.yuan_experiment import ECB_ARCHIVE

BASE_MODEL = "ensemble_t0_gamma"
MARKET_MODEL = "ensemble_t0_gamma_xgb_market"
LEGACY_MACRO_MODEL = "ensemble_t0_gamma_xgb_macro"
CLASSICAL_MODEL = "ensemble_classical"
MODEL = "ensemble_classical_xgb_shap8"
CLASSICAL = (
    (Candidate("ridge", "pilot", None, 0.15), 0.43838276164287915, 1.0),
    (Candidate("histboost", "market", 1460, 3), 0.3844272108930251, 0.75),
    (Candidate("extra_trees", "market", 1460, 3, pooled=True), 0.17719002746409557, 1.0),
)
GAMMA_WEIGHTS = np.where(HORIZONS >= 4, 0.15, 0)
GAMMA_REVISION = "fef67abad3eed4f227c7b536a6e8ee4e9ab1080d"
GAMMA_SHA256 = "1603e255629be22ea45518b502f16c55d06821c7090a1a851673e779e9c245a6"
MIN_HIT_RATE = 0.5


def classical_features(panels, candidate, split):
    """Выбрать признаки классической модели по заполненности на границе обучения."""
    features = specific_features(panels, candidate.setup)
    features = features.loc[:, ~features.columns.str.contains("__theory_", regex=False)]
    supported = features.iloc[:split].notna().mean().ge(MIN_HIT_RATE)
    return features.loc[:, supported]


def classical_predictions(panels, levels, split, origins):
    """Посчитать три классических прогноза с параметрами из holdout."""
    prediction = np.zeros((len(origins), len(HORIZONS)))
    for candidate, weight, shrink in CLASSICAL:
        features = classical_features(panels, candidate, split)
        raw, _ = fit_predict(
            features.to_numpy(dtype=float),
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


def forecast_inputs(args, today, *, include_ecb=False):
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
    if include_ecb:
        with Client(data_dir) as client:
            ecb = client.get(ECB_ARCHIVE)
            if pd.Timestamp(ecb.ingested_at).tz_convert("Europe/Moscow").date() < end.date():
                client.refresh = True
                ecb = client.get(ECB_ARCHIVE)
        quotes = {name: read_series(data_dir, name) for name in MARKET_SERIES}
        additional = market_features(ecb, quotes, target)
        panels = {name: panel.join(additional) for name, panel in panels.items()}
        paths.append(data_dir / "raw" / "objects" / ecb.sha256)
    return target, panels, [data_dir / "normalized/usd_rub.csv", Path(args.config), *paths]


def macro_panel(args, panel, target, columns=MACRO_CORRECTION_FEATURES):
    """Использовать рассчитанные макропризнаки или собрать их к дате прогноза."""
    if all(name in panel for name in columns):
        return panel, []
    data_dir = Path(args.data_dir)
    fred_root = args.fred_cache or data_dir / "model_runtime" / "theory_fred"
    sources, paths = read_theory_sources(
        data_dir, fred_root, target.index, frozen=args.fred_cache is not None
    )
    lags = json.loads(args.config.read_text())["assumed_lags"]
    fundamental, market = theory_features(panel, target, sources, lags)
    new = fundamental.join(market)
    return panel.drop(columns=panel.columns.intersection(new.columns)).join(new), [
        args.config,
        *paths,
    ]


def run_forecast(args, today):
    """Вычислить основной прогноз и записать его параметры в метаданные."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise ValueError("Для основного прогноза нужен uv run --extra ml") from exc

    legacy = (
        args.base_only or args.market_only or args.legacy_t0_gamma or args.checkpoint is not None
    )
    if args.classical_only and legacy:
        raise ValueError("--checkpoint относится к историческим вариантам с t0-gamma")
    target, panels, paths = forecast_inputs(args, pd.Timestamp(today), include_ecb=not legacy)
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
    name, correction = (BASE_MODEL if legacy else CLASSICAL_MODEL), {}
    panel = panels["extended"]
    query = target.index[-1:]
    models = None
    fraction = CORRECTION_FRACTION if legacy else COMPACT_CORRECTION_FRACTION
    if not (args.base_only or args.classical_only):
        if args.market_only:
            name, bundle, columns = MARKET_MODEL, MARKET_BUNDLE, MARKET_FEATURES
            key, signs = "market_correction", None
        elif legacy:
            name, bundle, columns = LEGACY_MACRO_MODEL, MACRO_BUNDLE, MACRO_CORRECTION_FEATURES
            key, signs = "macro_correction", FUNDAMENTAL_SIGNS
        else:
            name, bundle, columns = MODEL, COMPACT_MACRO_BUNDLE, COMPACT_MACRO_FEATURES
            key, signs = "macro_correction", COMPACT_FUNDAMENTAL_SIGNS
        models, metadata = load_correction_bundle(bundle, columns, signs)
        if cutoff < pd.Timestamp(metadata["trained_before"]):
            option = "--base-only" if legacy else "--classical-only"
            raise ValueError(
                f"Поправка XGBoost обучена до {metadata['trained_before']}. "
                f"Для более ранней границы используйте {option}"
            )
        correction = {key: {"fraction": fraction, **metadata}}
        paths.append(bundle)
        if not args.market_only:
            panel, sources = macro_panel(args, panel, target, columns)
            paths.extend(sources)
        features = panel.loc[query, list(columns)]
    calendar = json.loads(Path(args.calendar).read_text())

    with threadpool_limits(limits=2):
        classical = classical_predictions(panels, levels, split, np.array([len(levels) - 1]))
    last = levels[-1:]
    foundation = {}
    if legacy:
        prediction, changes, path = legacy_prediction(
            args, classical, panels, target, query, calendar
        )
        paths.append(path)
        foundation["t0_gamma"] = {
            "revision": GAMMA_REVISION,
            "weights_sha256": GAMMA_SHA256,
            "weights_by_horizon": GAMMA_WEIGHTS.tolist(),
        }
    else:
        changes = fixing_mask(query, calendar, True)
        prediction = apply_calendar(classical, last, changes, pool=True)
    if models is not None:
        prediction = corrected_forecast(
            models, features, prediction, last, changes, fraction=fraction
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
        "inputs": input_fingerprints([*paths, args.calendar]),
        **foundation,
        **correction,
    }
    return result, run


def legacy_prediction(args, classical, panels, target, query, calendar):
    """Воспроизвести основу с t0-gamma для прежних прогонов."""
    checkpoint = args.checkpoint or Path(args.data_dir) / "model_runtime/checkpoints/t0_gamma"
    with (checkpoint / "model.safetensors").open("rb") as file:
        if file_digest(file, "sha256").hexdigest() != GAMMA_SHA256:
            raise ValueError("Контрольная сумма весов отличается от проверенного t0-gamma")
    level, past, known, changes = market_inputs(target, panels["extended"], query, calendar)
    gamma = load_t0_gamma(checkpoint)(level, past, known)
    prediction = blend_predictions(classical, gamma, target.loc[query].to_numpy(), changes)
    return prediction, changes, checkpoint / "config.json"
