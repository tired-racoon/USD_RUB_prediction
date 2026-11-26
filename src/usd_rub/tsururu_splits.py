"""Временные фолды Tsururu для календарных целей USD/RUB."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from usd_rub.fixing_calendar import HORIZONS


@dataclass(frozen=True)
class Fold:
    name: str
    cutoff: int
    stop: int
    warmup: bool = False

    @property
    def origins(self):
        return np.arange(self.cutoff, self.stop - HORIZONS[-1])


def temporal_folds(index, holdout, *, n_splits=6, block_days=365, warmup=2):
    """Разбить предтестовые даты на равные блоки и оставить зрелые цели."""
    from tsururu.model_training.validator import TimeSeriesValidator

    if not index.equals(pd.date_range(index[0], index[-1], freq="D")):
        raise ValueError("Для сплитов нужен непрерывный календарный индекс")
    stop = index.get_loc(pd.Timestamp(holdout))
    start = stop - (n_splits + 1) * block_days
    if not 0 < warmup < n_splits or block_days <= HORIZONS[-1] or start < 0:
        raise ValueError("Недостаточно истории или некорректные параметры фолдов")
    positions = np.arange(start, stop)
    validator = TimeSeriesValidator(n_splits=n_splits)
    folds = []
    for i, (train, _, valid, _) in enumerate(validator.get_split(positions)):
        cutoff = int(train[-1]) + 1
        if cutoff != valid[0] or len(valid) != block_days:
            raise ValueError("Границы Tsururu не соответствуют заданным блокам")
        folds.append(Fold(f"tsururu_{i + 1}", cutoff, int(valid[-1]) + 1, i < warmup))
    return folds, start


def matured(origins, cutoff, horizon=HORIZONS[-1]):
    """Оставить даты, чья цель известна строго до границы."""
    return origins + horizon < cutoff
