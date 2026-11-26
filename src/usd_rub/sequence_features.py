"""Стандартизация и отбор признаков для моделей USD/RUB."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from usd_rub.ml_experiment import actual_at, eligible_rows

LOOKBACK = 16
TRAIN_WINDOW = 1460
TOP_K = 64


def normalized_targets(levels, train, horizons):
    """Подготовить доходности и веса для обучения поправки к текущему курсу."""
    targets = actual_at(levels, train, horizons) / levels[train, None] - 1
    weights = (levels[train] / np.median(levels[train])) ** 2
    scale = np.sqrt(np.average(targets**2, axis=0, weights=weights)).clip(min=1e-8)
    return (targets / scale).clip(-5, 5), weights, scale


def prepare_fold(panel, levels, cutoff, horizons, min_hit_rate):
    """Стандартизация и ранжирование признаков по известным обучающим целям."""
    train = eligible_rows(cutoff, int(horizons[-1]), TRAIN_WINDOW)
    train = train[train >= LOOKBACK - 1]
    values = panel.to_numpy(dtype=float)
    history = values[train]
    finite = np.isfinite(history)
    coverage = finite.mean(axis=0)
    history_end = cutoff - int(horizons[-1])
    expanding_coverage = np.isfinite(values[:history_end]).mean(axis=0)
    eligible = np.flatnonzero(
        (coverage >= min_hit_rate)
        & (expanding_coverage >= min_hit_rate)
        & (finite.sum(axis=0) >= 60)
    )
    if not len(eligible):
        raise ValueError("Недостаточно истории для отбора признаков")
    history = np.where(finite[:, eligible], history[:, eligible], np.nan)
    center, spread = np.nanmean(history, axis=0), np.nanstd(history, axis=0)
    variable = spread > 1e-10
    keys = eligible[variable]
    standardized = (values[:, keys] - center[variable]) / spread[variable]
    standardized = np.where(np.isfinite(standardized), standardized, 0).clip(-8, 8)
    if len(keys) < TOP_K:
        raise ValueError(f"Для отбора {TOP_K} признаков доступны только {len(keys)}")

    targets, weights, scale = normalized_targets(levels, train, horizons)
    ranking = Ridge(alpha=0.15 * len(train))
    ranking.fit(standardized[train], targets, sample_weight=weights)
    importance = np.sqrt(np.mean(ranking.coef_**2, axis=0))
    order = np.argsort(-importance, kind="stable")
    columns = pd.DataFrame(
        {
            "rank": np.arange(1, len(keys) + 1),
            "feature": panel.columns[keys[order]],
            "importance": importance[order],
            "train_hit_rate": coverage[keys[order]],
            "history_hit_rate": expanding_coverage[keys[order]],
        }
    )
    return train, standardized, targets, weights, scale, order, columns
