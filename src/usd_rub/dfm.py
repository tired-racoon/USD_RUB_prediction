"""Двухшаговая динамическая факторная модель Giannone et al. (2008)."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from usd_rub.models import interval, validate_history


def symmetric(matrix):
    return (matrix + matrix.T) / 2


@dataclass
class FactorStates:
    filtered_mean: np.ndarray
    filtered_covariance: np.ndarray
    smoothed_mean: np.ndarray
    smoothed_covariance: np.ndarray


def kalman_smoother(data, loadings, transition, process_covariance, observation_variance):
    """Пропущенные измерения исключаются из обновления Калмана."""
    data = np.asarray(data, dtype=float)
    count, factors = len(data), transition.shape[0]
    predicted_mean = np.zeros((count, factors))
    predicted_covariance = np.zeros((count, factors, factors))
    filtered_mean = np.zeros_like(predicted_mean)
    filtered_covariance = np.zeros_like(predicted_covariance)
    mean = np.zeros(factors)
    covariance = symmetric(solve_discrete_lyapunov(transition, process_covariance))
    for t, row in enumerate(data):
        if t:
            mean = transition @ mean
            covariance = symmetric(transition @ covariance @ transition.T + process_covariance)
        predicted_mean[t], predicted_covariance[t] = mean, covariance
        observed = np.isfinite(row)
        if observed.any():
            loading = loadings[observed]
            weighted = loading.T / observation_variance[observed]
            information = weighted @ loading
            innovation = row[observed] - loading @ mean
            posterior = np.linalg.solve(np.eye(factors) + covariance @ information, covariance)
            mean = mean + posterior @ weighted @ innovation
            covariance = symmetric(posterior)
        filtered_mean[t], filtered_covariance[t] = mean, covariance
    smoothed_mean = filtered_mean.copy()
    smoothed_covariance = filtered_covariance.copy()
    for t in range(count - 2, -1, -1):
        cross = transition @ filtered_covariance[t]
        try:
            gain = np.linalg.solve(predicted_covariance[t + 1], cross).T
        except np.linalg.LinAlgError:
            gain = (np.linalg.pinv(predicted_covariance[t + 1]) @ cross).T
        smoothed_mean[t] += gain @ (smoothed_mean[t + 1] - predicted_mean[t + 1])
        smoothed_covariance[t] = symmetric(
            filtered_covariance[t]
            + gain @ (smoothed_covariance[t + 1] - predicted_covariance[t + 1]) @ gain.T
        )
    return FactorStates(filtered_mean, filtered_covariance, smoothed_mean, smoothed_covariance)


class TwoStepFactorModel:
    def __init__(
        self, factors=2, shocks=2, min_balanced=120, stability_radius=0.999, variance_floor=1e-8
    ):
        if not 1 <= shocks <= factors or min_balanced < factors + 3:
            raise ValueError("DFM: требуются 1 <= shocks <= factors и достаточная история")
        if not 0 < stability_radius < 1 or variance_floor <= 0:
            raise ValueError("DFM: некорректные численные ограничения")
        self.factors = factors
        self.shocks = shocks
        self.min_balanced = min_balanced
        self.stability_radius = stability_radius
        self.variance_floor = variance_floor

    def fit(self, panel, target):
        """PCA и VAR оцениваются до обработки неполного края панели."""
        data = np.asarray(panel, dtype=float)
        target = validate_history(target)
        if data.ndim != 2 or len(data) != len(target) or np.isinf(data).any():
            raise ValueError("DFM: панель должна совпадать с историей цели и не содержать inf")
        available = np.isfinite(data)
        eligible = np.flatnonzero(available.sum(axis=0) >= self.min_balanced)
        if len(eligible) < self.factors:
            raise ValueError("DFM: недостаточно доступных рядов для факторов")
        data = data[:, eligible]
        complete = np.isfinite(data).all(axis=1)
        boundaries = np.diff(np.r_[False, complete, False].astype(int))
        blocks = [
            (first, stop - 1)
            for first, stop in zip(
                np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1), strict=True
            )
            if stop - first >= self.min_balanced
        ]
        if not blocks:
            raise ValueError("DFM: недостаточно наблюдений сбалансированной панели")
        first, last = blocks[-1]
        balanced = data[first : last + 1]
        mean, scale = balanced.mean(axis=0), balanced.std(axis=0, ddof=0)
        varying = scale > 1e-12
        self.columns_ = eligible[varying]
        if varying.sum() < self.factors:
            raise ValueError("DFM: недостаточно непостоянных рядов")
        data = data[:, varying]
        self.mean_, self.scale_ = mean[varying], scale[varying]
        standardized = (data - self.mean_) / self.scale_
        balanced = standardized[first : last + 1]
        eigenvalues, eigenvectors = np.linalg.eigh(balanced.T @ balanced / len(balanced))
        if eigenvalues[-self.factors] <= 1e-10:
            raise ValueError("DFM: недостаточный ранг панели")
        self.loadings_ = eigenvectors[:, -self.factors :][:, ::-1]
        scores = balanced @ self.loadings_
        self.transition_ = np.linalg.lstsq(scores[:-1], scores[1:], rcond=None)[0].T
        self.raw_spectral_radius_ = float(np.max(np.abs(np.linalg.eigvals(self.transition_))))
        if self.raw_spectral_radius_ > self.stability_radius:
            self.transition_ *= self.stability_radius / self.raw_spectral_radius_
        innovations = scores[1:] - scores[:-1] @ self.transition_.T
        covariance = symmetric(innovations.T @ innovations / len(innovations))
        values, vectors = np.linalg.eigh(covariance)
        self.process_covariance_ = (
            vectors[:, -self.shocks :] * np.maximum(values[-self.shocks :], self.variance_floor)
        ) @ vectors[:, -self.shocks :].T
        residual = balanced - scores @ self.loadings_.T
        self.observation_variance_ = np.maximum(np.mean(residual**2, axis=0), self.variance_floor)
        filter_first = int(np.flatnonzero(np.isfinite(standardized).any(axis=1))[0])
        self.states_ = kalman_smoother(
            standardized[filter_first:],
            self.loadings_,
            self.transition_,
            self.process_covariance_,
            self.observation_variance_,
        )
        returns = np.diff(np.log(target), prepend=np.nan)[filter_first:]
        valid = np.isfinite(returns)
        design = np.column_stack([np.ones(valid.sum()), self.states_.smoothed_mean[valid]])
        coefficients = np.linalg.lstsq(design, returns[valid], rcond=None)[0]
        self.intercept_, self.bridge_ = coefficients[0], coefficients[1:]
        errors = returns[valid] - design @ coefficients
        self.bridge_variance_ = float(errors @ errors / max(1, len(errors) - design.shape[1]))
        self.last_ = target[-1]
        self.balanced_first_ = int(first)
        self.balanced_last_ = int(last)
        self.balanced_size_ = len(balanced)
        self.explained_variance_ratio_ = float(
            eigenvalues[-self.factors :].sum() / eigenvalues.sum()
        )
        return self

    def predict(self, horizon=7, alpha=0.05):
        """Ковариация суммарной доходности учитывает общие будущие шоки."""
        if not hasattr(self, "states_"):
            raise ValueError("Сначала вызовите fit")
        if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("horizon должен быть положительным целым числом")
        dimension = self.factors + 1
        transition = np.zeros((dimension, dimension))
        transition[0, 0] = 1
        transition[0, 1:] = self.bridge_ @ self.transition_
        transition[1:, 1:] = self.transition_
        noise_map = np.vstack([self.bridge_, np.eye(self.factors)])
        process = noise_map @ self.process_covariance_ @ noise_map.T
        process[0, 0] += self.bridge_variance_
        mean = np.r_[0.0, self.states_.filtered_mean[-1]]
        covariance = np.zeros((dimension, dimension))
        covariance[1:, 1:] = self.states_.filtered_covariance[-1]
        means, variances = [], []
        for _ in range(horizon):
            mean = transition @ mean
            mean[0] += self.intercept_
            covariance = symmetric(transition @ covariance @ transition.T + process)
            means.append(np.log(self.last_) + mean[0])
            variances.append(covariance[0, 0])
        return interval(means, variances, alpha, log=True)
