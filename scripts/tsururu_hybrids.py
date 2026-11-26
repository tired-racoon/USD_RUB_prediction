"""Точечные прогнозы и поправки на дозревших ошибках вне выборки."""

import numpy as np
import pandas as pd
from shap_holdout import fuse_importance, tree_importance
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from usd_rub.fixing_calendar import HORIZONS, apply_calendar
from usd_rub.market_correction import XGB_PARAMS, fit_corrections, predict_levels
from usd_rub.ml_experiment import actual_at, eligible_rows
from usd_rub.models import AutoRegression, RandomWalk
from usd_rub.theory_features import COMPACT_MACRO_FEATURES, FUNDAMENTAL_SIGNS
from usd_rub.tsururu_splits import matured

FRACTIONS = (0.25, 0.5, 1.0)
MAIN = "xgb_fixed8/0.5"
BASE = "classical"
RW = "random_walk"


def econometric(levels, cutoff, origins, mask):
    """Проверить уровни, логарифмы, доходности и прямые AR-прогнозы."""
    size = len(HORIZONS)
    last = levels[origins]
    result = {RW: np.broadcast_to(last[:, None], (len(origins), size))}
    factories = {
        "rw_drift_730": RandomWalk(drift=True),
        "ar1_level_730": AutoRegression(),
        "ar1_log_730": AutoRegression(log=True),
        "ar1_return_730": AutoRegression(returns=True),
        "ar1_return_expanding": AutoRegression(returns=True),
    }
    for name, model in factories.items():
        first = max(0, cutoff - 730) if name.endswith("_730") else 0
        model.fit(levels[first:cutoff])
        point = np.array([model.predict(size, recent=levels[i - 1 : i + 1]).mean for i in origins])
        result[name] = apply_calendar(point, last, mask, pool=True)

    log_change = np.diff(np.log(levels), prepend=np.nan)
    columns = pd.concat([pd.Series(log_change).shift(lag) for lag in range(3)], axis=1).to_numpy()
    forecasts = []
    for h in HORIZONS:
        train = eligible_rows(cutoff, h, 730)
        train = train[train >= 3]
        target = levels[train + h] / levels[train] - 1
        weights = (levels[train] / np.median(levels[train])) ** 2
        regression = Ridge(alpha=0.01 * len(train)).fit(
            columns[train] * 100, target, sample_weight=weights
        )
        forecasts.append(last * (1 + regression.predict(columns[origins] * 100)))
    result["ar3_direct_730"] = apply_calendar(np.column_stack(forecasts), last, mask, pool=True)
    return result


def fold_ranking(panel, origins, actual, baseline, last):
    """Рассчитать TreeSHAP и gain на внутреннем окне с отсечкой целей."""
    probe = origins[-90:]
    fit = matured(origins, probe[0])
    x = panel.iloc[origins]
    available = x.loc[fit].notna().mean().ge(0.5) & x.loc[fit].nunique().gt(1)
    x = x.loc[:, available]
    if x.shape[1] < 8:
        raise ValueError("Менее восьми признаков доступны для отбора поправки")
    target = 100 * np.log(actual / baseline)
    weights = (last[fit] / np.median(last[fit])) ** 2
    scores = [
        tree_importance(x.loc[fit], target[fit, h], x.iloc[-90:], weights)
        for h in (0, len(HORIZONS) - 1)
    ]
    return fuse_importance(scores).head(8).index.tolist()


def lag_errors(levels, archive):
    """Перенести остатки на первую дату, когда известна соответствующая цель."""
    values = pd.Series(levels)
    errors = {}
    for column, h in enumerate(HORIZONS):
        previous = (values.shift(-h) - archive[:, column]) / values
        errors[h] = previous.shift(h + 1).ffill(limit=14)
    return pd.DataFrame(errors).to_numpy()


def level_models(panel, levels, archive, train, origins, actual, lagged):
    """Сравнить прямую цель в рублях и поправки на одинаковых входах."""
    features = panel[list(COMPACT_MACRO_FEATURES)]
    signs = {key: FUNDAMENTAL_SIGNS[key] for key in features if key in FUNDAMENTAL_SIGNS}
    last, prior = levels[origins], levels[train]
    target = actual - prior[:, None]
    baseline = archive[train] - prior[:, None]
    out = {name: [] for name in ("xgb_level", "xgb_direct", "ridge_level", "ridge_lag")}
    for j, _ in enumerate(HORIZONS):
        x_train = features.iloc[train].assign(anchor_return=baseline[:, j] / prior)
        x_test = features.iloc[origins].assign(anchor_return=archive[origins, j] / last - 1)
        residual = target[:, j] - baseline[:, j]
        for name, y in (("xgb_level", residual), ("xgb_direct", target[:, j])):
            model = XGBRegressor(**XGB_PARAMS, monotone_constraints=signs).fit(x_train, y)
            anchor = archive[origins, j] if name == "xgb_level" else last
            out[name].append(anchor + model.predict(x_test))
        for name in ("ridge_level", "ridge_lag"):
            fit, query = x_train, x_test
            if name == "ridge_lag":
                fit = fit.assign(previous_error=lagged[train, j])
                query = query.assign(previous_error=lagged[origins, j])
            columns = fit.columns[fit.notna().any()]
            fit, query = fit[columns], query[columns]
            scaler = StandardScaler().fit(fit)
            x = np.nan_to_num(scaler.transform(fit)).clip(-8, 8)
            x_query = np.nan_to_num(scaler.transform(query)).clip(-8, 8)
            model = Ridge(alpha=0.15 * len(train)).fit(x, residual)
            out[name].append(archive[origins, j] + model.predict(x_query))
    return {name: np.column_stack(columns) for name, columns in out.items()}


def corrections(panel, levels, archive, cutoff, origins, volatility, mask):
    """Обучить поправки только на известных вневыборочных ошибках."""
    train = np.flatnonzero(np.isfinite(archive[:, -1]) & matured(np.arange(len(levels)), cutoff))
    actual = actual_at(levels, train, HORIZONS)
    baseline, query = archive[train], archive[origins]
    last = levels[origins]
    selected = fold_ranking(panel, train, actual, baseline, levels[train])
    layouts = {
        "xgb_fixed8": list(COMPACT_MACRO_FEATURES),
        "xgb_full21": list(panel),
        "xgb_fold8": selected,
    }
    corrected = {}
    for name, keys in layouts.items():
        signs = {key: FUNDAMENTAL_SIGNS[key] for key in keys if key in FUNDAMENTAL_SIGNS}
        x = panel[keys]
        models = fit_corrections(x.iloc[train], actual, baseline, levels[train], signs)
        corrected[name] = predict_levels(models, x.iloc[origins], query, last)
    lagged = lag_errors(levels, archive)
    corrected.update(level_models(panel, levels, archive, train, origins, actual, lagged))

    threshold = np.quantile(volatility[train], 0.75)
    gate = np.where(volatility[origins] > threshold, 0.5, 1.0)[:, None]
    output = {}
    for name, point in corrected.items():
        if name == "xgb_direct":
            output[name] = apply_calendar(point, last, mask, pool=True)
            continue
        delta = point - query
        for fraction in FRACTIONS:
            key = f"{name}/{fraction:g}"
            output[key] = apply_calendar(query + fraction * delta, last, mask, pool=True)
            if name in {"xgb_fixed8", "ridge_lag"}:
                output[f"{name}_gated/{fraction:g}"] = apply_calendar(
                    query + fraction * gate * delta, last, mask, pool=True
                )
    return output, selected, len(train), threshold
