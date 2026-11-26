"""Проверить дневные версии ARIMA и MLP на общем тесте USD/RUB."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from threadpoolctl import threadpool_limits

from usd_rub.fixing_calendar import HORIZONS, apply_calendar, fixing_mask

TRAIN_DAYS = 1460
TEST_START = "2025-01-01"
LAST_ORIGIN = "2026-11-11"


def arima_forecast(log_levels, cutoff, origins, order):
    """Обновить состояние на каждом срезе, фиксируя параметры до теста."""
    model = ARIMA(log_levels[cutoff - TRAIN_DAYS : cutoff], order=order)
    fitted = model.fit(method_kwargs={"maxiter": 250})
    if not fitted.mle_retvals["converged"]:
        raise ValueError(f"ARIMA{order} не сошлась")
    updated = fitted.extend(log_levels[cutoff : origins[-1] + 1])
    states = updated.predicted_state[:, origins - cutoff + 1].copy()
    system = updated.model.ssm
    prediction = np.empty((len(origins), len(HORIZONS)))
    for step in range(len(HORIZONS)):
        prediction[:, step] = (system["design"] @ states)[0] + system.obs_intercept[0, 0]
        states = system["transition"] @ states + system.state_intercept[:, :1]
    return prediction


def mlp_forecast(log_levels, cutoff, origins):
    """Обучить по одной сети 12–8–1 на изменение каждого горизонта."""
    returns = np.diff(log_levels, prepend=np.nan)
    lags = np.lib.stride_tricks.sliding_window_view(returns, 12)
    prediction = np.empty((len(origins), len(HORIZONS)))
    for step, horizon in enumerate(HORIZONS):
        train = np.arange(cutoff - TRAIN_DAYS, cutoff - horizon)
        change = log_levels[train + horizon] - log_levels[train]
        network = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(8,),
                activation="tanh",
                solver="lbfgs",
                alpha=1,
                max_iter=1000,
                random_state=42,
            ),
        )
        model = TransformedTargetRegressor(regressor=network, transformer=StandardScaler())
        model.fit(lags[train - 11], change)
        prediction[:, step] = log_levels[origins] + model.predict(lags[origins - 11])
    return prediction


def main():
    """Сохранить прогнозы и факты по датам выпуска и горизонтам."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/usd_rub_2011_2026_ensemble.csv.gz")
    )
    parser.add_argument("--calendar", type=Path, default=Path("config/cbr_fixing_calendar.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/paper_baselines_holdout/predictions.csv")
    )
    args = parser.parse_args()

    levels = pd.read_csv(args.dataset, usecols=["date", "usd_rub"], parse_dates=["date"])
    levels = levels.set_index("date")["usd_rub"].sort_index()
    if not levels.index.equals(pd.date_range(levels.index[0], levels.index[-1])):
        raise ValueError("Нужен один курс на каждый календарный день")
    values = levels.to_numpy()
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Курс должен быть конечным и положительным")
    cutoff = levels.index.get_loc(TEST_START)
    origins = np.arange(cutoff, levels.index.get_loc(LAST_ORIGIN) + 1)
    if cutoff < TRAIN_DAYS + 12 or origins[-1] + HORIZONS[-1] >= len(values):
        raise ValueError("В снимке не хватает истории обучения или фактов на семь дней")
    log_levels = 100 * np.log(values / values[cutoff - 1])
    with threadpool_limits(limits=1):
        forecasts = {
            "arima_300": arima_forecast(log_levels, cutoff, origins, (3, 0, 0)),
            "arima_217": arima_forecast(log_levels, cutoff, origins, (2, 1, 7)),
            "mlp_12_8_1": mlp_forecast(log_levels, cutoff, origins),
        }

    dates = levels.index[origins]
    last = values[origins]
    calendar = json.loads(args.calendar.read_text())
    mask = fixing_mask(dates, calendar, official=True)
    frame = pd.DataFrame(
        {
            "origin": dates.repeat(len(HORIZONS)),
            "horizon": np.tile(HORIZONS, len(origins)),
            "actual": values[origins[:, None] + HORIZONS].ravel(),
            "last": np.repeat(last, len(HORIZONS)),
        }
    )
    for name, forecast in forecasts.items():
        points = values[cutoff - 1] * np.exp(forecast / 100)
        frame[name] = apply_calendar(points, last, mask, pool=True).ravel()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
