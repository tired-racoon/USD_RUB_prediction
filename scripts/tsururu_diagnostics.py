"""Интервалы и проверки из лекции о гибридном прогнозировании."""

import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from usd_rub.fixing_calendar import HORIZONS
from usd_rub.tsururu_splits import matured

ALPHA = 0.10
CALIBRATION = 365
HAC_LAG = 28


def conformal_radius(errors, coverage):
    """Взять порядковую статистику с конечновыборочной поправкой."""
    n = len(errors)
    rank = math.ceil((n + 1) * coverage) - 1
    if not 0 <= rank < n:
        raise ValueError("Мало известных ошибок для калибровки интервала")
    return np.partition(errors, rank)[rank]


def intervals(levels, origins, point, actual, test_rows, volatility):
    """Оценить скользящие 90%-интервалы по дозревшим ошибкам прогноза."""
    error = np.abs(actual - point)
    vol_scale = (levels[origins] * volatility[origins])[:, None] * np.sqrt(HORIZONS)
    table, bounds = [], []
    for name, scale in (("absolute", np.ones_like(error)), ("volatility", vol_scale)):
        radius = np.empty_like(error[test_rows])
        scores = error / scale
        for j, h in enumerate(HORIZONS):
            for i, pos in enumerate(test_rows):
                eligible = np.flatnonzero(matured(origins, origins[pos], h))[-CALIBRATION:]
                q = conformal_radius(scores[eligible, j], 1 - ALPHA)
                radius[i, j] = q * scale[pos, j]
        included = error[test_rows] <= radius
        score = 2 * radius + 2 / ALPHA * np.maximum(error[test_rows] - radius, 0)
        for j, h in enumerate(HORIZONS):
            table.append(
                (name, h, included[:, j].mean(), 2 * radius[:, j].mean(), score[:, j].mean())
            )
        p = point[test_rows]
        bounds.append(
            pd.DataFrame(
                {
                    "method": name,
                    "position": np.repeat(origins[test_rows], len(HORIZONS)),
                    "horizon": np.tile(HORIZONS, len(test_rows)),
                    "actual": actual[test_rows].ravel(),
                    "point": p.ravel(),
                    "lower": (p - radius).ravel(),
                    "upper": (p + radius).ravel(),
                }
            )
        )
    return pd.DataFrame(
        table, columns=["method", "horizon", "coverage", "width_rub", "interval_score_rub"]
    ), pd.concat(bounds, ignore_index=True)


def dm_hln(loss_gain, horizon):
    """Парный тест разности квадратов ошибок с Newey–West и поправкой HLN."""
    n = len(loss_gain)
    centered = loss_gain - loss_gain.mean()
    variance = centered @ centered / n
    bandwidth = min(n - 1, max(HAC_LAG, horizon - 1))
    for lag in range(1, bandwidth + 1):
        variance += 2 * (1 - lag / (bandwidth + 1)) * (centered[lag:] @ centered[:-lag]) / n
    if variance <= 0:
        return np.nan, np.nan
    factor = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic = loss_gain.mean() / np.sqrt(variance / n) * factor
    return statistic, 2 * t.sf(abs(statistic), n - 1)


def compare_points(points, actual, pairs):
    """Сравнить RMSE на одинаковых датах, используя DM и парный бутстрэп."""
    n = len(actual)
    errors = {name: (value - actual) ** 2 for name, value in points.items()}
    bootstrap = {}
    rng = np.random.default_rng(191)
    for length in (14, 28, 56):
        starts = rng.integers(0, n - length + 1, (2500, math.ceil(n / length)))
        bootstrap[length] = (starts[:, :, None] + np.arange(length)).reshape(2500, -1)[:, :n]
    rows = []
    for reference, candidate in pairs:
        for h in (1, 7):
            a, b = errors[reference][:, h - 1], errors[candidate][:, h - 1]
            dm, p = dm_hln(a - b, h)
            row = dict(reference=reference, candidate=candidate, horizon=h, dm_hln=dm, p_value=p)
            row["rmse_gain_rub"] = np.sqrt(a.mean()) - np.sqrt(b.mean())
            for length, sample in bootstrap.items():
                gain = np.sqrt(a[sample].mean(axis=1)) - np.sqrt(b[sample].mean(axis=1))
                row[f"gain_l95_block{length}"], row[f"gain_u95_block{length}"] = np.quantile(
                    gain, (0.025, 0.975)
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    valid = frame.p_value.notna()
    frame.loc[valid, "p_holm"] = multipletests(frame.loc[valid, "p_value"], method="holm")[1]
    return frame


def residual_tests(pretest, actual, point):
    """Проверить стационарность курса и зависимость ошибок вне выборки."""
    rows = []
    levels = np.log(pretest)
    for name, series in (("log_levels", levels), ("daily_log_returns", np.diff(levels))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            warnings.filterwarnings(
                "ignore", message=r"(adfuller|kpss) currently returns", category=FutureWarning
            )
            adf = adfuller(series, autolag="AIC")
            kp = kpss(series, regression="c", nlags="auto")
        rows += [(name, "ADF", adf[2], adf[0], adf[1]), (name, "KPSS", kp[2], kp[0], kp[1])]
    for h in (1, 7):
        error = actual[:, h - 1] - point[:, h - 1]
        for name, series in (("error", error), ("error_squared", error**2)):
            tests = acorr_ljungbox(series, lags=[7, 28])
            rows.extend(
                (f"{name}_h{h}", "Ljung–Box", lag, row.lb_stat, row.lb_pvalue)
                for lag, row in tests.iterrows()
            )
    return pd.DataFrame(rows, columns=["series", "test", "lag", "statistic", "p_value"])
