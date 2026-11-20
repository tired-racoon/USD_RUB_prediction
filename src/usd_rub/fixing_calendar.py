"""Календарь вступления официального курса ЦБ в силу."""

import numpy as np
import pandas as pd

HORIZONS = np.arange(1, 8)


def fixing_mask(origins, calendar, official):
    """Проверить рабочий день расчёта до каждой прогнозируемой календарной даты."""
    fixed = set(calendar["fixed_days_off"])
    years = {
        int(year): (
            pd.Timestamp(info["available_from"]),
            set(info["days_off"]),
            set(info["workdays"]),
        )
        for year, info in calendar["years"].items()
    }
    mask = np.empty((len(origins), len(HORIZONS)), dtype=bool)
    for i, origin in enumerate(origins):
        for j, h in enumerate(HORIZONS):
            day = origin + pd.Timedelta(days=int(h) - 1)
            working = day.dayofweek < 5
            if official:
                stamp = day.strftime("%m-%d")
                working = working and stamp not in fixed
                if day.year in years:
                    available, days_off, workdays = years[day.year]
                    if origin >= available:
                        working = (working and stamp not in days_off) or stamp in workdays
            mask[i, j] = working
    return mask


def apply_calendar(prediction, last, changes, pool):
    """Связать даты с одним курсом и сохранить известный курс до нового расчёта."""
    adjusted = np.empty_like(prediction)
    ids = np.cumsum(changes, axis=1)
    for i, (row, groups) in enumerate(zip(prediction, ids, strict=True)):
        for group in np.unique(groups):
            same = groups == group
            value = row[same].mean() if pool else row[same][0]
            adjusted[i, same] = last[i] if group == 0 else value
    return adjusted
