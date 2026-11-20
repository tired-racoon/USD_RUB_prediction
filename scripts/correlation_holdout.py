"""Корреляции будущих изменений USD/RUB и парная проверка прогнозных правил."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from usd_rub.ml_dataset import feature_panels, read_dataset
from usd_rub.ml_experiment import relative_rmse, selection_loss
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints

HORIZONS = np.arange(1, 8)
FOCUS = {
    "fx__cny_basis": "Спред CNY/RUB",
    "global__synthetic_basis": "Кросс USD/RUB через юань",
    "fx__ecb_cross_close_basis": "Спред кросса ЕЦБ",
    "fx__si_front_settle_basis": "Базис фьючерса Si",
    "fx__cny_tom_close_change_1": "Дневное изменение CNY/RUB",
    "fx__si_front_settle_change_1": "Дневное изменение Si",
    "global__brent_change_1": "Дневное изменение Brent",
    "markets__rgbi_change_5": "Доходность RGBI за 5 дней",
    "own__return_1": "Вчерашнее изменение USD/RUB",
    "own__vol_60": "Волатильность курса за 60 дней",
}
CALENDAR_RULES = (
    ("weekend_carry", False, False),
    ("weekend_pool", False, True),
    ("calendar_carry", True, False),
    ("calendar_pool", True, True),
)


def forecasts(path, name, target):
    """Восстановить матрицы из сопоставимых прогнозов на горизонтах 1–7."""
    frame = pd.read_csv(path, parse_dates=["origin", "target_date"], float_precision="round_trip")
    frame = frame.loc[frame.model.eq(name)].sort_values(["origin", "horizon"])
    dates = pd.DatetimeIndex(frame.origin.unique())
    origins = target.index.get_indexer(dates)
    if not len(dates) or origins.min() < 0 or origins.max() + HORIZONS[-1] >= len(target):
        raise ValueError(f"Период прогноза вне снимка: {path}")
    expected_pos = origins[:, None] + HORIZONS
    np.testing.assert_array_equal(frame.horizon, np.tile(HORIZONS, len(origins)))
    np.testing.assert_array_equal(frame.origin, np.repeat(dates, len(HORIZONS)))
    np.testing.assert_array_equal(frame.target_date, target.index.to_numpy()[expected_pos].ravel())
    levels = target.to_numpy()
    last, actual = levels[origins], levels[expected_pos]
    np.testing.assert_allclose(frame["last"], np.repeat(last, len(HORIZONS)), rtol=1e-12)
    np.testing.assert_allclose(frame.actual, actual.ravel(), rtol=1e-12)
    base = frame.prediction.to_numpy().reshape(actual.shape)
    if not np.isfinite(base).all():
        raise ValueError(f"Неконечный прогноз: {path}")
    return dates, actual, last, base


def paired_correlations(x, y, minimum):
    """Вычислить связи по конечным парам при достаточном числе наблюдений."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < minimum or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return n, np.nan, np.nan
    _, counts = np.unique(x, return_counts=True)
    if n - counts.max() < max(8, n * 0.02):
        return n, np.nan, np.nan
    r = np.corrcoef(rankdata(x), rankdata(y))[0, 1]
    p = np.corrcoef(x, y)[0, 1]
    return n, r, p


def screen(panel, calibration, test):
    """Сравнить скрининг до 2025 года с отдельной оценкой 2025–2026."""
    train_days, train_actual, train_last, train_pred = calibration
    test_days, test_actual, test_last, test_pred = test
    targets = {
        "return": (
            (train_actual / train_last[:, None]) - 1,
            (test_actual / test_last[:, None]) - 1,
        ),
        "residual": (
            (train_actual - train_pred) / train_last[:, None],
            (test_actual - test_pred) / test_last[:, None],
        ),
    }
    train_x, test_x = panel.reindex(train_days).to_numpy(), panel.reindex(test_days).to_numpy()
    eligible = np.flatnonzero(np.isfinite(train_x).sum(axis=0) >= 300)
    years = [train_days.year == year for year in sorted(train_days.year.unique())]
    rows = []
    for key, (train_y, test_y) in targets.items():
        for h, xx, yy in zip(HORIZONS, train_y.T, test_y.T, strict=True):
            for i in eligible:
                x, z = train_x[:, i], test_x[:, i]
                n1, rho1, p1 = paired_correlations(x, xx, 300)
                if not np.isfinite(rho1):
                    continue
                n2, rho2, p2 = paired_correlations(z, yy, 100)
                annual = np.array([paired_correlations(x[v], xx[v], 120)[1] for v in years])
                annual = annual[np.isfinite(annual)]
                rows.append(
                    {
                        "feature": panel.columns[i],
                        "target": key,
                        "horizon": h,
                        "train_n": n1,
                        "test_n": n2,
                        "train_spearman": rho1,
                        "test_spearman": rho2,
                        "train_pearson": p1,
                        "test_pearson": p2,
                        "train_years": len(annual),
                        "annual_median_spearman": np.median(annual) if len(annual) else np.nan,
                        "annual_sign_agreement": np.mean(np.sign(annual) == np.sign(rho1))
                        if len(annual)
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def diagnostics(panel, calibration, test, calendar):
    """Посмотреть задержки, рыночные режимы и свежесть валютной котировки."""
    rows, lags, bins = [], [], []
    calibration_vol = panel.loc[calibration[0], "own__vol_21"].median()
    for phase, pack in (("train", calibration), ("test", test)):
        dates, actual, last, baseline = pack
        cv = panel.loc[dates, "fx__cny_tom_close_age"]
        vol = panel.loc[dates, "own__vol_21"]
        fixing = fixing_mask(dates, calendar, True)[:, 0]
        regimes = {
            "all": np.ones(len(dates), dtype=bool),
            "before_2022_02_24": dates < "2022-02-24",
            "2022_02_24_to_2024_06_12": (dates >= "2022-02-24") & (dates < "2024-06-13"),
            "post_moex": dates >= "2024-06-13",
            "fresh_cny": cv.le(1).to_numpy(),
            "stale_cny": cv.gt(1).to_numpy(),
            "fresh_cny_fixing_day": cv.le(1).to_numpy() & fixing,
            "stale_cny_fixing_day": cv.gt(1).to_numpy() & fixing,
            "high_vol": vol.gt(calibration_vol).to_numpy(),
            "low_vol": vol.le(calibration_vol).to_numpy(),
        }
        targets = {
            "return": actual / last[:, None] - 1,
            "residual": (actual - baseline) / last[:, None],
        }
        for name in FOCUS:
            x = panel.loc[dates, name].to_numpy()
            train_quantiles = panel.loc[calibration[0], name].quantile([0.2, 0.8]).to_numpy()
            for target_name, outcomes in targets.items():
                for h in (1, 3, 7):
                    y = outcomes[:, h - 1]
                    for regime, mask in regimes.items():
                        n, rho, _ = paired_correlations(x[mask], y[mask], 30)
                        rows.append(
                            {
                                "feature": name,
                                "phase": phase,
                                "target": target_name,
                                "horizon": h,
                                "regime": regime,
                                "n": n,
                                "spearman": rho,
                            }
                        )
                    for lag in (0, 1, 3, 7):
                        shifted = panel[name].shift(lag).reindex(dates).to_numpy()
                        n, rho, _ = paired_correlations(shifted, y, 100)
                        lags.append(
                            {
                                "feature": name,
                                "phase": phase,
                                "target": target_name,
                                "horizon": h,
                                "lag_days": lag,
                                "n": n,
                                "spearman": rho,
                            }
                        )
                    for tail, mask in (
                        ("low", x <= train_quantiles[0]),
                        ("high", x >= train_quantiles[1]),
                    ):
                        z = y[mask & np.isfinite(y)]
                        if len(z):
                            bins.append(
                                {
                                    "feature": name,
                                    "phase": phase,
                                    "target": target_name,
                                    "horizon": h,
                                    "tail": tail,
                                    "n": len(z),
                                    "median_pct": 100 * np.median(z),
                                    "positive_share": np.mean(z > 0),
                                }
                            )
    return pd.DataFrame(rows), pd.DataFrame(lags), pd.DataFrame(bins)


def supplementary_signals(panel, calibration, test):
    """Проверить разрыв ожиданий и курса, а также связи с величиной ошибки."""
    rows = []
    expected = panel["macro__cbr_survey_usd_rub_current_year_median"]
    for phase, (dates, actual, last, pred) in (("train", calibration), ("test", test)):
        level = expected.reindex(dates).to_numpy()
        features = {
            "survey_level": level,
            "survey_gap_to_spot": np.log(level / last),
            "survey_revision": panel.loc[dates, "macro__cbr_survey_usd_rub_current_year_revision"],
            "own_vol_21": panel.loc[dates, "own__vol_21"],
            "own_vol_60": panel.loc[dates, "own__vol_60"],
        }
        for h in (1, 3, 7):
            residual = (actual[:, h - 1] - pred[:, h - 1]) / last
            outcome = actual[:, h - 1] / last - 1
            targets = {
                "return": outcome,
                "residual": residual,
                "abs_return": abs(outcome),
                "abs_residual": abs(residual),
            }
            for feature, x in features.items():
                for target, y in targets.items():
                    n, rho, _ = paired_correlations(np.asarray(x), y, 100)
                    rows.append(
                        {
                            "feature": feature,
                            "phase": phase,
                            "target": target,
                            "horizon": h,
                            "n": n,
                            "spearman": rho,
                        }
                    )
    return pd.DataFrame(rows)


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


def residual_adjustment(panel, train, query, use_embedding=False):
    """Найти устойчивые связи с ошибкой ансамбля и обучить простую поправку."""
    dates, actual, last, baseline = train
    names = [c for c in panel if use_embedding or "_embedding_" not in c]
    x, q = panel.loc[dates, names].to_numpy(), panel.loc[query, names].to_numpy()
    annual = [dates.year == y for y in sorted(dates.year.unique())]
    eligible = np.flatnonzero(np.isfinite(x).mean(axis=0) >= 0.6)
    values = (actual - baseline) / last[:, None]
    weight = (last / np.median(last)) ** 2
    output = np.zeros((len(query), len(HORIZONS)))
    rows = []
    for j, target in enumerate(values.T):
        ranking = []
        for k in eligible:
            _, rho, _ = paired_correlations(x[:, k], target, 300)
            if not np.isfinite(rho):
                continue
            signs = np.array([paired_correlations(x[m, k], target[m], 120)[1] for m in annual])
            signs = signs[np.isfinite(signs)]
            if len(signs) >= 3 and np.mean(np.sign(signs) == np.sign(rho)) >= 0.75:
                ranking.append((min(abs(rho), abs(np.median(signs))), k, rho))
        keys = []
        for _, k, rho in sorted(ranking, reverse=True):
            if any(abs(paired_correlations(x[:, k], x[:, other], 200)[1]) > 0.90 for other in keys):
                continue
            keys.append(k)
            rows.append({"horizon": j + 1, "feature": names[k], "train_spearman": rho})
            if len(keys) == 8:
                break
        if not keys:
            continue
        train_x, query_x = x[:, keys], q[:, keys]
        center, spread = np.nanmean(train_x, axis=0), np.nanstd(train_x, axis=0)
        spread[spread < 1e-10] = 1
        train_x = np.nan_to_num((train_x - center) / spread).clip(-6, 6)
        query_x = np.nan_to_num((query_x - center) / spread).clip(-6, 6)
        reg = Ridge(alpha=0.3 * len(train_x), fit_intercept=False)
        reg.fit(train_x, target, sample_weight=weight)
        output[:, j] = reg.predict(query_x)
    return output, rows


def subset(pack, mask):
    """Сохранить одинаковые даты для всех горизонтов."""
    return tuple(part[mask] for part in pack)


def calibrate(panel, train, test, calendar):
    """Выбрать календарное правило и вес поправки на годовых фолдах."""
    folds, splits = [], []
    for year in range(2021, 2025):
        valid = subset(train, train[0].year == year)
        folds.append(valid)
        dates, actual, last, _ = valid
        splits.append((dates, actual, last))
    val_scores = {"ensemble": selection_loss([f[3] for f in folds], splits)}
    official_masks = {
        official: [fixing_mask(f[0], calendar, official) for f in folds]
        for official in (False, True)
    }
    for name, official, pool in CALENDAR_RULES:
        val = [
            apply_calendar(f[3], f[2], mask, pool)
            for f, mask in zip(folds, official_masks[official], strict=True)
        ]
        val_scores[name] = selection_loss(val, splits)
    selected = min(val_scores, key=val_scores.get)
    selection = {name: (official, pool) for name, official, pool in CALENDAR_RULES}

    def project(pred, last, dates):
        if selected == "ensemble":
            return pred
        official, pool = selection[selected]
        return apply_calendar(pred, last, fixing_mask(dates, calendar, official), pool)

    raw = [
        residual_adjustment(panel, subset(train, train[0].year < f[0][0].year), f[0])[0]
        for f in folds
    ]
    grid = (0, 0.05, 0.1, 0.25, 0.5, 1)
    factor_scores = {
        factor: selection_loss(
            [
                project(f[3] + factor * raw_fold * f[2][:, None], f[2], f[0])
                for f, raw_fold in zip(folds, raw, strict=True)
            ],
            splits,
        )
        for factor in grid
    }
    factor = min(grid, key=factor_scores.get)
    dates, actual, last, base = test
    forecasts = {"ensemble": base, "rw": np.broadcast_to(last[:, None], base.shape)}
    calendar_mask = {v: fixing_mask(dates, calendar, v) for v in (True, False)}
    for name, official, pool in CALENDAR_RULES:
        forecasts[name] = apply_calendar(base, last, calendar_mask[official], pool)
    correction, features = residual_adjustment(panel, train, dates)
    forecasts["calendar_and_residual"] = project(
        base + factor * correction * last[:, None], last, dates
    )
    forecasts["calendar_residual_weight_one"] = project(
        base + correction * last[:, None], last, dates
    )
    exceptions = np.diff(np.column_stack([last, actual]), axis=1)[~calendar_mask[True]]
    metadata = {
        "calendar_cv": val_scores,
        "selected_calendar": selected,
        "residual_cv": factor_scores,
        "residual_weight": factor,
        "calendar_test_carried_pairs": int(len(exceptions)),
        "calendar_test_exceptions": int(np.count_nonzero(np.abs(exceptions) > 1e-9)),
    }
    return forecasts, pd.DataFrame(features), metadata


def evaluate(test, estimates, selected):
    """Оценить те же даты, горизонты и парный интервал для поправки."""
    dates, actual, last, base = test
    rows, predictions = [], []
    for name, forecast in estimates.items():
        ratios = relative_rmse(forecast, actual, last)
        rmse = np.sqrt(((forecast - actual) ** 2).mean(axis=0))
        rows.extend(
            {"model": name, "horizon": h, "rmse": error, "rmse_to_rw": ratio}
            for h, error, ratio in zip(HORIZONS, rmse, ratios, strict=True)
        )
        predictions.append(
            pd.DataFrame(
                {
                    "origin": np.repeat(dates, len(HORIZONS)),
                    "horizon": np.tile(HORIZONS, len(dates)),
                    "model": name,
                    "last": np.repeat(last, len(HORIZONS)),
                    "actual": actual.ravel(),
                    "prediction": forecast.ravel(),
                }
            )
        )
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("model").rmse_to_rw.mean().to_frame()
    for h in (1, 3, 7):
        summary[f"rmse_{h}d"] = metrics.loc[metrics.horizon.eq(h)].set_index("model").rmse
    rng = np.random.default_rng(191)
    n = len(last)
    starts = rng.integers(0, n, size=(2000, (n + 27) // 28))
    positions = ((starts[:, :, None] + np.arange(28)) % n).reshape(len(starts), -1)[:, :n]
    rmse_base = np.sqrt(((base - actual) ** 2)[positions].mean(axis=1))
    rmse_selected = np.sqrt(((estimates[selected] - actual) ** 2)[positions].mean(axis=1))
    rmse_rw = np.sqrt(((last[:, None] - actual) ** 2)[positions].mean(axis=1))
    gain = 100 * ((rmse_base - rmse_selected) / rmse_rw).mean(axis=1)
    return (
        metrics,
        summary.sort_values("rmse_to_rw"),
        pd.concat(predictions, ignore_index=True),
        np.quantile(gain, [0.025, 0.975]).tolist(),
    )


def plot_results(correlations, metrics, selected, directory):
    """Построить рисунок для связи кроссов и результатов календарной поправки."""
    from matplotlib import pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    keys = (
        "fx__cny_basis",
        "global__synthetic_basis",
        "fx__ecb_cross_close_basis",
        "fx__si_front_settle_basis",
    )
    for feature in keys:
        rows = correlations.loc[correlations.feature.eq(feature) & correlations.target.eq("return")]
        sign = -1 if feature == "fx__ecb_cross_close_basis" else 1
        label = FOCUS[feature] if sign == 1 else f"{FOCUS[feature]} (обратный знак)"
        a.plot(rows.horizon, sign * rows.test_spearman, marker="o", label=label)
    a.set(
        title="Связь рыночных спредов с движением курса",
        xlabel="Горизонт, дней",
        ylabel="Корреляция Спирмена на тесте",
        xticks=HORIZONS,
    )
    a.legend(fontsize=8, loc="best")
    for name, label in dict.fromkeys(
        [
            ("ensemble", "Прежний ансамбль"),
            (selected, "С календарём"),
            ("rw", "Случайное блуждание"),
        ]
    ):
        rows = metrics.loc[metrics.model.eq(name)]
        b.plot(rows.horizon, rows.rmse, marker="o", label=label)
    b.set(
        title="Ошибка прогноза официального курса",
        xlabel="Горизонт, дней",
        ylabel="RMSE, ₽",
        xticks=HORIZONS,
    )
    b.legend(fontsize=8, loc="best")
    for ax in (a, b):
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(directory / "correlations_and_calendar.png", dpi=200)
    plt.close(fig)


def run(args):
    """Выпустить таблицы связей, исходные прогнозы и проверяемые метрики."""
    base = read_dataset(args.dataset)
    training = forecasts(args.oof, "ensemble_oof", base.usd_rub)
    test = forecasts(args.holdout, "ensemble", base.usd_rub)
    if training[0].max() + pd.Timedelta(days=7) >= test[0].min():
        raise ValueError("Цели обучения пересекают начало теста")
    panel = feature_panels(base)["extended"].join(read_dataset(args.events), validate="one_to_one")
    schedule = json.loads(args.calendar.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    correlations = screen(panel, training, test)
    write_csv(correlations, args.output / "correlations.csv.gz")
    trends, lags, bins = diagnostics(panel, training, test, schedule)
    write_csv(trends, args.output / "conditional_correlations.csv")
    write_csv(lags, args.output / "lead_lag.csv")
    write_csv(bins, args.output / "quantile_signals.csv")
    write_csv(
        supplementary_signals(panel, training, test), args.output / "supplementary_signals.csv"
    )
    print(f"Готово {len(correlations):,} оценок корреляции", flush=True)
    estimates, features, metadata = calibrate(panel, training, test, schedule)
    selected = metadata["selected_calendar"]
    metrics, summary, predictions, ci = evaluate(test, estimates, selected)
    write_csv(features, args.output / "residual_features.csv")
    write_csv(metrics, args.output / "projection_metrics.csv")
    write_csv(summary, args.output / "projection_summary.csv", index=True)
    write_csv(predictions, args.output / "projection_predictions.csv.gz")
    dates, actual, last, _ = test
    weekdays = (
        pd.DataFrame(
            {
                "weekday": dates.dayofweek,
                "unchanged_next_day": np.isclose(actual[:, 0], last, rtol=0),
            }
        )
        .groupby("weekday")
        .unchanged_next_day.agg(["size", "sum", "mean"])
    )
    write_csv(weekdays, args.output / "weekdays.csv", index=True)
    metadata.update(
        first_origin=str(dates[0].date()),
        last_origin=str(dates[-1].date()),
        test_origins=len(dates),
        training_origins=len(training[0]),
        candidates=panel.shape[1],
        calendar_gain_ci_pp=ci,
        input_files=input_fingerprints(
            [args.dataset, args.events, args.calendar, args.oof, args.holdout]
        ),
    )
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    if args.plot:
        plot_results(correlations, metrics, selected, args.output)
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/usd_rub_2011_2026_full.csv.gz")
    )
    parser.add_argument(
        "--events", type=Path, default=Path("datasets/llm_news_event_features_2016_2026.csv.gz")
    )
    parser.add_argument("--calendar", type=Path, default=Path("config/cbr_fixing_calendar.json"))
    parser.add_argument(
        "--oof", type=Path, default=Path("data/reports/llm_news_holdout/baseline_oof.csv.gz")
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        default=Path("data/reports/ml_holdout_minfin_rosstat/with_file/predictions.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/reports/correlations"))
    parser.add_argument("--plot", action="store_true")
    with threadpool_limits(limits=2):
        run(parser.parse_args())
