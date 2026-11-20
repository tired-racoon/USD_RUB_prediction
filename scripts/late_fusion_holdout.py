"""Сравнение способов передачи рыночных и табличных признаков в нейросеть."""

import numpy as np
import pandas as pd

from usd_rub.ml_dataset import feature_panels
from usd_rub.ml_experiment import eligible_rows

MODES = ("market_history", "all_history", "late")


def standardize(values, train):
    """Стандартизовать по обучению и заполнить пропуски нейтральным значением."""
    history = values[train]
    finite = np.isfinite(history)
    counts = finite.sum(axis=0)
    denominator = counts.clip(min=1)
    center = np.where(finite, history, 0).sum(axis=0) / denominator
    errors = np.where(finite, history - center, 0)
    spread = np.sqrt((errors**2).sum(axis=0) / denominator)
    variable = spread > 1e-10
    x = np.zeros(values.shape, dtype=np.float32)
    z = (values[:, variable] - center[variable]) / spread[variable]
    x[:, variable] = np.where(np.isfinite(z), z.clip(-8, 8), 0)
    return x, counts, variable


def feature_layout(dataset):
    """Отдать историю курса и рынков бэкбону, а оставшиеся столбцы голове."""
    panel = feature_panels(dataset)["extended"]
    groups = panel.columns.str.split("__").str[0]
    market = groups.isin(("own", "fx", "markets", "global")) & ~panel.columns.str.endswith("_age")
    if market.all() or not market.any():
        raise ValueError("Нужны рыночные и табличные признаки")
    layout = pd.DataFrame(
        {"feature": panel.columns, "group": groups, "input": np.where(market, "history", "head")}
    )
    return panel.to_numpy(dtype=float), market, layout


def train_fold(
    values,
    market,
    levels,
    cutoff,
    origins,
    horizons,
    device,
    *,
    modes=MODES,
    checkpoints=None,
    hide_context=None,
):
    """Обучить три режима на одинаковых датах, целях и эпохах."""
    import torch
    from sequence_holdout import (
        ARCHITECTURES,
        CHECKPOINTS,
        LOOKBACK,
        TRAIN_WINDOW,
        normalized_targets,
        train_sequence,
    )
    from sklearn.linear_model import Ridge

    if checkpoints is None:
        checkpoints = CHECKPOINTS

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    train = eligible_rows(cutoff, int(horizons[-1]), TRAIN_WINDOW)
    train = train[train >= LOOKBACK - 1]
    x, counts, variable = standardize(values, train)
    targets, weights, scale = normalized_targets(levels, train, horizons)
    offsets = np.arange(1 - LOOKBACK, 1)
    train_windows = train[:, None] + offsets
    query_windows = origins[:, None] + offsets
    y_train, sample_weights = tensor(targets), tensor(weights)
    context_x = x[:, ~market]
    context = (tensor(context_x[train]), tensor(context_x[origins]))
    contexts = {"late": context}
    if hide_context is not None:
        mask = torch.as_tensor(hide_context[~market], dtype=torch.bool, device=device)
        contexts["late_without_rbc"] = tuple(part.masked_fill(mask, 0) for part in context)
    market_x = x[:, market]
    market_windows = tensor(market_x[train_windows]), tensor(market_x[query_windows])
    forecasts = {}
    if "all_history" in modes:
        ridge = Ridge(alpha=0.15 * len(train), fit_intercept=False)
        ridge.fit(x[train], targets, sample_weight=weights)
        forecasts["ridge_snapshot"] = levels[origins, None] * (
            1 + ridge.predict(x[origins]) * scale
        )
    for mode in modes:
        x_train, x_query = (
            (tensor(x[train_windows]), tensor(x[query_windows]))
            if mode == "all_history"
            else market_windows
        )
        for name in ARCHITECTURES:
            predictions = train_sequence(
                name,
                x_train.shape[-1],
                x_train,
                y_train,
                sample_weights,
                x_query,
                checkpoints,
                device,
                context=contexts.get(mode),
            )
            for epoch, predicted in predictions.items():
                forecasts[f"{name}_{mode}_e{epoch:02d}"] = levels[origins, None] * (
                    1 + predicted * scale
                )
        del x_train, x_query
        print(f"{mode}: {len(train)} дат обучения, {len(origins)} прогнозов", flush=True)
    return forecasts, counts, variable, len(train)


def sequence_predictions(dataset, calibration_dates, test_dates, horizons, device):
    """Рассчитать прогнозы фолдов и отчёт о покрытии признаков."""
    values, market, layout = feature_layout(dataset)
    dates = dataset.index
    levels = dataset.usd_rub.to_numpy(dtype=float)
    predictions, coverage = [], []
    folds = [
        (str(year), calibration_dates[calibration_dates.year == year])
        for year in calibration_dates.year.unique()
    ]
    for name, query in [*folds, ("final", test_dates)]:
        origins = dates.get_indexer(query)
        print(f"Нейросети, фолд {name}", flush=True)
        forecasts, counts, varying, n = train_fold(
            values, market, levels, origins[0], origins, horizons, device
        )
        predictions.append(forecasts)
        coverage.append(
            layout.assign(
                fold=name,
                train_rows=n,
                train_observations=counts,
                train_variable=varying,
                query_observations=np.isfinite(values[origins]).sum(axis=0),
            )
        )
    return (
        {name: np.concatenate([fold[name] for fold in predictions]) for name in predictions[0]},
        pd.concat(coverage, ignore_index=True),
    )


def rbc_predictions(dataset, test_dates, horizons, device, checkpoints):
    """Проверить позднюю голову после появления исторических данных РБК."""
    values, market, layout = feature_layout(dataset)
    dates = dataset.index
    levels = dataset.usd_rub.to_numpy(dtype=float)
    rbc = layout.feature.str.startswith("news__rbc_").to_numpy()
    if not rbc.any():
        raise ValueError("В снимке нет признаков РБК")
    eligible = test_dates.to_period("Q")
    last_horizon = pd.Timedelta(days=int(horizons[-1]))
    forecasts, forecast_dates = [], []
    modes = ("late", "late_without_rbc")
    for period in eligible.unique():
        query = test_dates[eligible == period]
        origins = dates.get_indexer(query)
        train_dates = dates + last_horizon < query[0]
        if not np.isfinite(values[np.ix_(train_dates, rbc)]).any():
            continue
        predicted, counts, variable, _ = train_fold(
            values,
            market,
            levels,
            origins[0],
            origins,
            horizons,
            device,
            modes=modes,
            checkpoints=checkpoints,
            hide_context=rbc,
        )
        forecasts.append(predicted)
        forecast_dates.append(query)
        n = np.count_nonzero(variable & rbc)
        print(f"РБК, {period}: {n} меняющихся признаков, {counts[rbc].max()} дат", flush=True)
    if not forecast_dates:
        raise ValueError("На тесте нет квартала с доступной историей РБК")
    return (
        forecast_dates[0].append(forecast_dates[1:]),
        {name: np.concatenate([fold[name] for fold in forecasts]) for name in forecasts[0]},
    )
