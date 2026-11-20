"""Медианный прогноз t0-gamma с рынком и календарём."""

from pathlib import Path

import numpy as np
import pandas as pd

from usd_rub.fixing_calendar import HORIZONS, fixing_mask

FEATURES = ("fx__cny_basis", "global__synthetic_basis", "fx__si_front_settle_change_1")
LOOKBACK = 256


def market_inputs(target, panel, dates, calendar):
    """Собрать окна курса, рынка и известного календаря на общих датах."""
    positions = target.index.get_indexer(dates)
    history = positions[:, None] - np.arange(LOOKBACK - 1, -1, -1)
    if history.min() < 0:
        raise ValueError("Недостаточно истории USD/RUB для t0-gamma")
    features = panel.loc[target.index, list(FEATURES)].to_numpy(np.float32)
    target_window = target.to_numpy(np.float32)[history]
    past = features[history].transpose(0, 2, 1).copy()
    changes = fixing_mask(dates, calendar, True)
    schedule = fixing_mask(target.index - pd.Timedelta(days=1), calendar, True)[:, 0]
    known = np.concatenate((schedule[history], changes), axis=1)[:, None].astype(np.float32)
    return target_window, past, known, changes


def load_t0_gamma(path):
    """Загрузить локальный чекпойнт и вернуть функцию медианного прогноза."""
    path = Path(path)
    for name in ("config.json", "model.safetensors"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Отсутствует файл t0-gamma: {path / name}")
    try:
        import mlx.core as mx
        from t0.mlx import T0Forecaster, TimeSeries, VariateType
    except ImportError as exc:
        raise ValueError("Для t0-gamma нужен extra ensemble на Mac с Apple Silicon") from exc

    model = T0Forecaster.from_pretrained(str(path)).compile()
    horizon = len(HORIZONS)

    def predict(target, past, known):
        if past is None:
            return np.array(model.predict(target, horizon, [0.5]).median)
        context = np.concatenate((target[:, None], past), axis=1)
        series = TimeSeries.from_array(mx.array(context), mx.array(known), horizon=horizon)
        rows = np.arange(series.variates.shape[0])
        historical = (rows < len(target) * context.shape[1]) & (rows % context.shape[1] != 0)
        series.variate_type = mx.where(
            mx.array(historical[:, None]), VariateType.HISTORICAL, series.variate_type
        )
        return np.array(model.predict(series, horizon, [0.5]).median)

    return predict
