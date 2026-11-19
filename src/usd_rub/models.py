"""Эталонные одномерные модели прогноза."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class Forecast:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def validate_history(y, minimum=3):
    values = np.asarray(y, dtype=float)
    if values.ndim != 1 or len(values) < minimum:
        raise ValueError(f"Требуется одномерный ряд длиной не менее {minimum}")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Курс должен быть конечным и положительным, без пропусков")
    return values


def interval(mean, variance, alpha, log=False):
    if not 0 < alpha < 1:
        raise ValueError("alpha должна быть между 0 и 1")
    mean, variance = np.asarray(mean), np.asarray(variance)
    radius = norm.ppf(1 - alpha / 2) * np.sqrt(np.maximum(variance, 0))
    if log:
        return Forecast(np.exp(mean + variance / 2), np.exp(mean - radius), np.exp(mean + radius))
    return Forecast(mean, mean - radius, mean + radius)


class Baseline:
    def fit(self, y):
        self.y = validate_history(y)
        return self

    def _steps(self, horizon):
        if not hasattr(self, "y"):
            raise ValueError("Сначала вызовите fit")
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon должен быть положительным целым числом")
        return np.arange(1, horizon + 1)


class RandomWalk(Baseline):
    def __init__(self, drift=False):
        self.drift = drift

    def predict(self, horizon=7, alpha=0.05, *, recent=None):
        h = self._steps(horizon)
        changes = np.diff(self.y)
        drift = changes.mean() if self.drift else 0
        residuals = changes - drift
        variance = np.dot(residuals, residuals) / (len(changes) - int(self.drift))
        last = self.y[-1] if recent is None else recent[-1]
        mean = last + h * drift
        error_variance = variance * h
        if self.drift:
            error_variance *= 1 + h / len(changes)
        return interval(mean, error_variance, alpha)


class AutoRegression(Baseline):
    def __init__(self, log=False, returns=False):
        self.log = log or returns
        self.returns = returns

    def fit(self, y):
        super().fit(y)
        z = np.log(self.y) if self.log else self.y
        self.last_log = z[-1]
        if self.returns:
            z = np.diff(z)
        if len(z) < 4:
            raise ValueError("Для AR(1) требуется минимум 4 наблюдения после преобразования")
        x = z[:-1]
        centered = x - x.mean()
        denominator = np.dot(centered, centered)
        if denominator <= np.finfo(float).eps * max(1, np.dot(x, x)):
            self.phi = 0.0
        else:
            self.phi = float(np.dot(centered, z[1:] - z[1:].mean()) / denominator)
        self.intercept = float(z[1:].mean() - self.phi * x.mean())
        residuals = z[1:] - self.intercept - self.phi * x
        self.sigma2 = float(np.dot(residuals, residuals) / max(1, len(residuals) - 2))
        self.last = z[-1]
        return self

    def predict(self, horizon=7, alpha=0.05, *, recent=None):
        h = self._steps(horizon)
        mean, variance = [], []
        current = self.last
        cumulative = self.last_log
        if recent is not None:
            current = recent[-1]
            if self.log:
                cumulative = np.log(current)
                current = cumulative - np.log(recent[-2]) if self.returns else cumulative
        for step in h:
            current = self.intercept + self.phi * current
            if self.returns:
                cumulative += current
                mean.append(cumulative)
                weights = np.cumsum(self.phi ** np.arange(step))
            else:
                mean.append(current)
                weights = self.phi ** np.arange(step)
            variance.append(self.sigma2 * np.dot(weights, weights))
        return interval(mean, variance, alpha, log=self.log)


class HistoricalMean(Baseline):
    def predict(self, horizon=7, alpha=0.05, *, recent=None):
        h = self._steps(horizon)
        variance = np.var(self.y, ddof=1) * (1 + 1 / len(self.y))
        return interval(np.full(len(h), self.y.mean()), np.full(len(h), variance), alpha)


class SeasonalNaive(Baseline):
    def __init__(self, period=7):
        if not isinstance(period, int) or period < 1:
            raise ValueError("period должен быть положительным целым")
        self.period = period

    def fit(self, y):
        self.y = validate_history(y, self.period + 2)
        return self

    def predict(self, horizon=7, alpha=0.05, *, recent=None):
        h = self._steps(horizon)
        errors = self.y[self.period :] - self.y[: -self.period]
        variance = np.mean(errors**2)
        levels = self.y if recent is None else recent
        mean = np.array([levels[-self.period + ((step - 1) % self.period)] for step in h])
        return interval(mean, variance * np.ceil(h / self.period), alpha)


MODEL_FACTORIES = {
    "rw": RandomWalk,
    "rw_drift": lambda: RandomWalk(drift=True),
    "ar1": AutoRegression,
    "log_ar1": lambda: AutoRegression(log=True),
    "returns_ar1": lambda: AutoRegression(returns=True),
    "seasonal_naive_7": SeasonalNaive,
    "historical_mean": HistoricalMean,
}
