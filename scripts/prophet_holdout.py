"""Прогноз уровня и доходности USD/RUB с Prophet."""

import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pandas as pd

WINDOWS = (180, 730)
REFIT_DAYS = 7
SEED = 191


def fit_block(dates, log_levels, origins, horizons):
    """Переобучить Prophet на первой дате блока и использовать известный курс."""
    from prophet import Prophet

    first = origins[0]
    end = origins[-1] + horizons[-1] + 1
    future = pd.DataFrame({"ds": dates[first:end]})
    offset = origins - first
    targets = offset[:, None] + horizons
    forecasts = {}
    for window, returns in product(WINDOWS, (False, True)):
        start = first - window + 1
        history = log_levels[start : first + 1]
        observed = np.diff(log_levels[start - 1 : first + 1]) if returns else history
        model = Prophet(
            growth="flat" if returns else "linear",
            weekly_seasonality=True,
            yearly_seasonality=False,
            daily_seasonality=False,
            uncertainty_samples=0,
        )
        model.fit(pd.DataFrame({"ds": dates[start : first + 1], "y": observed}), seed=SEED)
        forecast = model.predict(future).yhat.to_numpy()
        if returns:
            forecast = forecast.cumsum()
            change = forecast[targets] - forecast[offset, None]
            forecasts[f"prophet_returns_{window}"] = np.exp(log_levels[origins, None] + change)
        else:
            forecasts[f"prophet_level_{window}"] = np.exp(forecast[targets])
            change = forecast[targets] - forecast[offset, None]
            forecasts[f"prophet_anchored_{window}"] = np.exp(log_levels[origins, None] + change)
    return forecasts


def prophet_predictions(target, dates, horizons, jobs):
    """Получить прогнозы блоками внутри каждого исторического года."""
    logging.getLogger("cmdstanpy").disabled = True
    origins = target.index.get_indexer(dates)
    levels = target.to_numpy(dtype=float)
    if origins.min() < max(WINDOWS) or origins.max() + horizons[-1] >= len(levels):
        raise ValueError("Недостаточно истории или целей для Prophet")
    if not (np.isfinite(levels).all() and (levels > 0).all()):
        raise ValueError("Ожидается конечный положительный курс")
    log_levels = np.log(levels)

    def calculate(block):
        return fit_block(target.index, log_levels, block, horizons)

    predictions = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for year in dates.year.unique():
            period = origins[dates.year == year]
            blocks = [period[i : i + REFIT_DAYS] for i in range(0, len(period), REFIT_DAYS)]
            predictions.extend(pool.map(calculate, blocks))
            print(f"Prophet {year}: {len(period)} дат", flush=True)
    return {name: np.concatenate([fold[name] for fold in predictions]) for name in predictions[0]}
