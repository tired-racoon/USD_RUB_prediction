"""Последовательная вневыборочная оценка эталонных моделей."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from usd_rub.models import MODEL_FACTORIES, validate_history


def daily_target(frame: pd.DataFrame, start=None, end=None):
    if frame.empty:
        raise ValueError("Нет целевого ряда")
    frame = frame.sort_values(["date", "ingested_at"]).drop_duplicates("date", keep="last")
    series = frame.set_index("date")["value"].sort_index()
    lower = pd.Timestamp(start) if start else series.index.min()
    upper = pd.Timestamp(end) if end else series.index.max()
    if upper > series.index.max():
        raise ValueError("Нельзя продолжать целевой ряд после последней известной даты")
    index = pd.date_range(min(lower, series.index.min()), upper, freq="D")
    result = series.reindex(series.index.union(index)).sort_index().ffill().reindex(index)
    result = result.loc[lower:upper]
    validate_history(result)
    return result


def walk_forward(
    y: pd.Series,
    models=None,
    min_train=365,
    window=None,
    horizon=7,
    step=1,
    alpha=0.05,
    test_start=None,
):
    validate_history(y)
    expected = pd.date_range(y.index.min(), y.index.max(), freq="D")
    if not isinstance(y.index, pd.DatetimeIndex) or not y.index.equals(expected):
        raise ValueError("Backtest требует непрерывный календарный дневной индекс")
    models = list(models or MODEL_FACTORIES)
    if len(set(models)) != len(models) or any(name not in MODEL_FACTORIES for name in models):
        raise ValueError("Неизвестные или повторяющиеся модели")
    if "rw" not in models:
        models.insert(0, "rw")
    if min_train < 10 or step < 1 or not 1 <= horizon <= 7:
        raise ValueError("min_train>=10, step>=1, horizon=1..7")
    if window is not None and window < min_train:
        raise ValueError("Окно короче минимальной обучающей выборки")
    start = min_train - 1
    if test_start:
        start = max(start, int(y.index.searchsorted(pd.Timestamp(test_start))))
    rows = []
    for origin in range(start, len(y) - horizon, step):
        first = 0 if window is None else max(0, origin + 1 - window)
        history = y.iloc[first : origin + 1].to_numpy()
        scale = float(np.mean(np.abs(np.diff(history))))
        for name in models:
            prediction = MODEL_FACTORIES[name]().fit(history).predict(horizon, alpha)
            if not np.isfinite(prediction.mean).all():
                raise ValueError(f"Неконечный прогноз {name}, {y.index[origin]}")
            for h in range(1, horizon + 1):
                rows.append(
                    {
                        "model": name,
                        "origin": y.index[origin],
                        "horizon": h,
                        "target_date": y.index[origin + h],
                        "last": history[-1],
                        "actual": y.iloc[origin + h],
                        "prediction": prediction.mean[h - 1],
                        "lower": prediction.lower[h - 1],
                        "upper": prediction.upper[h - 1],
                        "mase_scale": scale,
                        "train_size": len(history),
                    }
                )
    if not rows:
        raise ValueError("Недостаточно данных для backtest с заданными параметрами")
    return pd.DataFrame(rows)


def dm_test(model_loss, rw_loss, lag):
    differential = np.asarray(model_loss) - np.asarray(rw_loss)
    n = len(differential)
    if n < max(30, lag + 2):
        return np.nan, np.nan
    centered = differential - differential.mean()
    lrv = np.dot(centered, centered) / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * np.dot(centered[k:], centered[:-k]) / n
    if lrv <= np.finfo(float).eps:
        return np.nan, np.nan
    statistic = differential.mean() / np.sqrt(lrv / n)
    return float(statistic), float(2 * norm.sf(abs(statistic)))


def score(predictions: pd.DataFrame, origin_step=1):
    rows = []
    rw = predictions.loc[predictions["model"] == "rw"].set_index(["origin", "horizon"])
    for (name, h), group in predictions.groupby(["model", "horizon"], sort=True):
        group = group.sort_values("origin")
        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["prediction"].to_numpy(dtype=float)
        if not np.isfinite(actual).all() or (actual <= 0).any():
            raise ValueError("Процентные метрики курса требуют конечных положительных actual")
        if not np.isfinite(predicted).all():
            raise ValueError("Метрики требуют конечных прогнозов")
        error = predicted - actual
        percentage_error = 100 * error / actual
        symmetric_percentage_error = 200 * np.abs(error) / (np.abs(actual) + np.abs(predicted))
        baseline = rw.loc[pd.MultiIndex.from_frame(group[["origin", "horizon"]])]
        rw_error = (baseline["prediction"] - baseline["actual"]).to_numpy()
        rmse = float(np.sqrt(np.mean(error**2)))
        rw_rmse = float(np.sqrt(np.mean(rw_error**2)))
        direction = np.sign(group["actual"] - group["last"]).to_numpy()
        predicted_direction = np.sign(group["prediction"] - group["last"]).to_numpy()
        moving = direction != 0
        positive = group["prediction"].to_numpy() > 0
        log_error = np.log(group["prediction"].to_numpy()[positive]) - np.log(
            group["actual"].to_numpy()[positive]
        )
        scale = group["mase_scale"].to_numpy()
        valid_scale = scale > 0
        dm_stat, dm_p = dm_test(
            error**2, rw_error**2, lag=max(0, int(np.ceil(h / origin_step)) - 1)
        )
        rows.append(
            {
                "model": name,
                "horizon": h,
                "n": len(group),
                "mae": float(np.mean(np.abs(error))),
                "rmse": rmse,
                "mape_pct": float(np.mean(np.abs(percentage_error))),
                "smape_pct": float(np.mean(symmetric_percentage_error)),
                "rmspe_pct": float(np.sqrt(np.mean(percentage_error**2))),
                "mase": float(np.mean(np.abs(error[valid_scale]) / scale[valid_scale]))
                if valid_scale.any()
                else np.nan,
                "log_return_mae": float(np.mean(np.abs(log_error))) if len(log_error) else np.nan,
                "log_return_rmse": float(np.sqrt(np.mean(log_error**2)))
                if len(log_error)
                else np.nan,
                "nonpositive_predictions": int((~positive).sum()),
                "direction_accuracy": float(np.mean(direction == predicted_direction)),
                "direction_accuracy_changed": float(
                    np.mean(direction[moving] == predicted_direction[moving])
                )
                if moving.any()
                else np.nan,
                "n_changed": int(moving.sum()),
                "mae_changed": float(np.mean(np.abs(error[moving]))) if moving.any() else np.nan,
                "rmse_changed": float(np.sqrt(np.mean(error[moving] ** 2)))
                if moving.any()
                else np.nan,
                "mape_changed_pct": float(np.mean(np.abs(percentage_error[moving])))
                if moving.any()
                else np.nan,
                "interval_coverage": float(
                    np.mean(
                        (group["actual"] >= group["lower"]) & (group["actual"] <= group["upper"])
                    )
                ),
                "interval_width": float(np.mean(group["upper"] - group["lower"])),
                "rmse_skill_vs_rw": 1 - rmse / rw_rmse if rw_rmse > 0 else np.nan,
                "dm_stat_squared_loss": dm_stat,
                "dm_pvalue_squared_loss": dm_p,
            }
        )
    return pd.DataFrame(rows)


def regime_scores(predictions, origin_step=1):
    dates = pd.to_datetime(predictions["origin"])
    groups = {
        "before_2022": dates < "2022-02-24",
        "2022_to_moex_stop": dates.between("2022-02-24", "2024-06-12"),
        "otc_since_2024": dates >= "2024-06-13",
    }
    parts = [
        score(predictions.loc[mask], origin_step).assign(regime=name)
        for name, mask in groups.items()
        if mask.any()
    ]
    return pd.concat(parts, ignore_index=True)
