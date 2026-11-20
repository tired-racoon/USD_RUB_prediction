"""Проверка режимов, динамики спредов и весов истории на горизонте семь дней."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from correlation_holdout import HORIZONS, apply_calendar, fixing_mask, forecasts
from forecast_stacking import ARCHIVE_FILES, read_panel
from foundation_ensemble import CLASSICAL, RW
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from usd_rub.ml_dataset import clean_dataset, feature_panels, read_dataset
from usd_rub.ml_experiment import candidates, fit_predict, specific_features, to_levels
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints

RIDGE = "ridge_pilot_full_0.15_direct"
GAMMA = "t0_gamma/market_calendar"
YEARS = (2021, 2022, 2023, 2024)
BREAK = pd.Timestamp("2024-06-13")
HORIZON = pd.Timedelta(days=7)
PENALTIES = (0.15, 0.5, 2.0)
STRENGTHS = (0.25, 0.5, 1.0)
HISTORY = {"window_1460": 0, **{f"decay_{n}": n for n in (365, 730, 1460, 2920)}}


def residual_features(panel):
    """Построить признаки по срезу и его прошлым значениям."""
    cny = panel["fx__cny_basis"]
    funding = panel["rates__ruonia_spread"]
    age = panel["fx__cny_tom_close_age"]
    volatility = panel["own__vol_21"]
    typical = volatility.rolling(252, min_periods=60).median()
    stress = np.log(volatility.where(volatility > 0) / typical.where(typical > 0))
    static = pd.DataFrame(
        {
            "cny_basis": cny,
            "ecb_basis": panel["fx__ecb_cross_close_basis"],
            "funding": funding,
            "volatility": volatility,
            "quote_age": age,
        }
    )
    regime = static.assign(
        stress=stress,
        basis_stress=cny * stress,
        basis_quote_age=cny * age,
    )
    rvi = np.log(panel["markets__rvi_level"])
    rvi_shock = rvi - rvi.rolling(252, min_periods=60).median()
    dynamics = pd.DataFrame(
        {
            f"{name}_{kind}": change
            for name, spread in {"cny": cny, "ruonia": funding}.items()
            for kind, change in {
                "change_1": spread.diff(),
                "change_5": spread.diff(5),
                "acceleration": spread.diff().diff(),
            }.items()
        }
    )
    return {
        "scale": static.iloc[:, :0],
        "static": static,
        "regime": regime,
        "regime_rvi": regime.assign(rvi=rvi, rvi_shock=rvi_shock, basis_rvi=cny * rvi_shock),
        "cny_dynamics": static.join(dynamics.filter(like="cny_")),
        "ruonia_dynamics": static.join(dynamics.filter(like="ruonia_")),
        "spread_dynamics": static.join(dynamics),
        "regime_and_dynamics": regime.join(dynamics),
    }


def base_frame(dates, actual, last, raw, fixes):
    """Сохранить цель, проекцию ансамбля и наличие нового фиксинга."""
    return pd.DataFrame(
        {
            "actual": actual[:, -1],
            "last": last,
            "ensemble": apply_calendar(raw, last, fixes, pool=True)[:, -1],
            "new_fixing": fixes.any(axis=1),
        },
        index=dates,
    )


def residual_fit(features, history, query, cutoff, penalties):
    """Оценить поправку только по созревшим архивным прогнозам."""
    train = history.loc[(history.index + HORIZON < cutoff) & history.new_fixing]
    data = features.loc[train.index]
    selected = data.columns[data.notna().mean().ge(0.5)]
    data = data[selected]
    center = data.mean()
    std = data.std(ddof=0).where(lambda s: s > 1e-12, 1)
    error = ((train.actual - train.ensemble) / train["last"]).to_numpy()
    scale = ((train.ensemble - train["last"]) / train["last"]).std(ddof=0)

    def matrix(rows):
        signal = ((features.loc[rows.index, selected] - center) / std).fillna(0).clip(-6, 6)
        base = (rows.ensemble - rows["last"]) / (rows["last"] * scale)
        return np.column_stack([base, signal.to_numpy()])

    x, q = matrix(train), matrix(query)
    weights = (train["last"] / train["last"].median()).pow(2).to_numpy()
    scale_to_rub = query["last"].where(query.new_fixing, 0).to_numpy()
    settings = {}
    for penalty in penalties:
        model = Ridge(alpha=penalty * len(train), fit_intercept=False)
        model.fit(x, error, sample_weight=weights)
        settings[penalty] = model.predict(q) * scale_to_rub
    return settings


def rmse(prediction, actual):
    return np.sqrt(np.mean((prediction - actual) ** 2))


def annual_score(prediction, reference):
    """Усреднить отношение RMSE к случайному блужданию по годам."""
    return float(
        np.mean(
            [
                rmse(prediction.loc[part.index], part.actual) / rmse(part["last"], part.actual)
                for _, part in reference.groupby(reference.index.year)
            ]
        )
    )


def compare_residuals(designs, history, calibration, test):
    grid, valid, final = [], {}, {}
    for name, features in designs.items():
        folds = [
            (
                query.index,
                residual_fit(features, history, query, pd.Timestamp(year, 1, 1), PENALTIES),
            )
            for year, query in calibration.groupby(calibration.index.year)
        ]
        for penalty in PENALTIES:
            change = pd.concat([pd.Series(values[penalty], index=rows) for rows, values in folds])
            for strength in STRENGTHS:
                prediction = calibration.ensemble + strength * change
                grid.append(
                    {
                        "model": name,
                        "penalty": penalty,
                        "strength": strength,
                        "validation_rmse_to_rw": annual_score(prediction, calibration),
                    }
                )
        best = min(
            (row for row in grid if row["model"] == name),
            key=lambda row: row["validation_rmse_to_rw"],
        )
        penalty, strength = best["penalty"], best["strength"]
        valid[name] = calibration.ensemble + strength * pd.concat(
            [pd.Series(values[penalty], index=rows) for rows, values in folds]
        )
        test_delta = residual_fit(features, history, test, test.index[0], (penalty,))[penalty]
        final[name] = test.ensemble + strength * test_delta
    return valid, final, pd.DataFrame(grid)


def history_predictions(dataset, archive, reference, shrink, half_lives, folds):
    """Заменить только прогноз Ridge при сохранении других компонентов."""
    coverage = reference["min_hit_rate"]
    panels = feature_panels(clean_dataset(dataset, coverage))
    x = specific_features(panels, "pilot").to_numpy()
    y = dataset.usd_rub.to_numpy()
    ridge = next(item for item in candidates() if item.name == RIDGE)
    weight = reference["ensemble_weights"][RIDGE]
    raw_ridge = archive.raw[RIDGE]
    result = {name: [] for name in half_lives}
    for rows in folds:
        dates = archive.dates[rows]
        origins = dataset.index.get_indexer(dates)
        cutoff = dataset.index.searchsorted(dates[0])
        for name, half_life in half_lives.items():
            candidate = replace(ridge, window=1460) if half_life == 0 else ridge
            raw, _ = fit_predict(
                x,
                y,
                candidate,
                cutoff,
                origins,
                HORIZONS,
                min_hit_rate=coverage,
                half_life_days=half_life or None,
            )
            forecast = to_levels(raw, archive.last[rows], shrink)
            if half_life is None:
                np.testing.assert_allclose(forecast, raw_ridge[rows], atol=1e-10, rtol=1e-10)
            combined = archive.raw[CLASSICAL][rows] + weight * (forecast - raw_ridge[rows])
            result[name].append(pd.Series(archive.project(combined, rows)[:, -1], index=dates))
    return {name: pd.concat(folds) for name, folds in result.items()}


def gamma_frame(archive):
    """Воспроизвести прежнюю смесь с весом t0-gamma на горизонтах 4–7."""
    base = archive.raw[CLASSICAL]
    mixed = base + np.where(HORIZONS >= 4, 0.15, 0) * (archive.raw[GAMMA] - base)
    return base_frame(archive.dates, archive.actual, archive.last, mixed, archive.fixes)


def summarize(reference, predictions):
    """Сравнить RMSE и MAE на одинаковых датах с парным бутстрэпом."""
    actual = reference.actual.to_numpy()
    base = reference.ensemble.to_numpy()
    n, block = len(reference), 28
    start = np.random.default_rng(191).integers(n, size=(4000, (n + block - 1) // block))
    blocks = ((start[:, :, None] + np.arange(block)) % n).reshape(len(start), -1)[:, :n]
    base_rmse = np.sqrt(np.mean((base - actual)[blocks] ** 2, axis=1))
    rows = []
    for name, forecast in predictions.items():
        error = forecast.loc[reference.index].to_numpy() - actual
        values = np.sqrt(np.mean(error[blocks] ** 2, axis=1))
        low, high = np.quantile(base_rmse - values, (0.025, 0.975))
        rows.append(
            {
                "model": name,
                "dates": n,
                "rmse_rub": rmse(error, 0),
                "mae_rub": np.abs(error).mean(),
                "rmse_gain_rub": rmse(base, actual) - rmse(error, 0),
                "gain_ci_lo_rub": low,
                "gain_ci_hi_rub": high,
            }
        )
    return pd.DataFrame(rows)


def run(args):
    dataset = read_dataset(args.dataset)
    dates = dataset.index
    if not dates.equals(pd.date_range(dates[0], dates[-1])):
        raise ValueError("Нужен непрерывный дневной датасет")
    archive_metadata = args.archive / "archives.json"
    groups = json.loads(archive_metadata.read_text())["groups"]
    paths = [args.archive / filename for filename in ARCHIVE_FILES]
    names = [CLASSICAL, RW, *sum(groups.values(), [])]
    calibration, holdout = (read_panel(path, names) for path in paths)
    for archive in (calibration, holdout):
        np.testing.assert_allclose(archive.last, dataset.usd_rub.loc[archive.dates], rtol=1e-12)
        np.testing.assert_allclose(
            archive.actual[:, -1], dataset.usd_rub.loc[archive.dates + HORIZON], rtol=1e-12
        )
    if (
        calibration.dates.max() + HORIZON >= holdout.dates.min()
        or tuple(calibration.dates.year.unique()) != YEARS
    ):
        raise ValueError("Некорректные периоды архивов")
    valid, test = [
        base_frame(v.dates, v.actual, v.last, v.raw[CLASSICAL], v.fixes)
        for v in (calibration, holdout)
    ]
    oof_dates, oof_actual, oof_last, oof_raw = forecasts(args.oof, "ensemble_oof", dataset.usd_rub)
    schedule = json.loads(args.calendar.read_text())
    oof = base_frame(
        oof_dates, oof_actual, oof_last, oof_raw, fixing_mask(oof_dates, schedule, True)
    )
    if oof.index.max() + HORIZON >= holdout.dates.min():
        raise ValueError("Архив остатков пересекает тест")
    np.testing.assert_allclose(oof.loc[valid.index, "ensemble"], valid.ensemble, rtol=1e-12)
    designs = residual_features(feature_panels(dataset)["extended"])
    residual_val, residual_test, grid = compare_residuals(designs, oof, valid, test)
    gamma_calibration, gamma_test = map(gamma_frame, (calibration, holdout))
    gamma_corrections = {}
    for label in ("cny_dynamics", "spread_dynamics"):
        best = grid.loc[grid.model.eq(label)].nsmallest(1, "validation_rmse_to_rw").iloc[0]
        delta = residual_fit(
            designs[label], gamma_calibration, gamma_test, test.index[0], (best.penalty,)
        )[best.penalty]
        gamma_corrections[f"gamma_{label}"] = gamma_test.ensemble + best.strength * delta
    print("Поправки по режимам и спредам рассчитаны", flush=True)

    ref_path = args.reference / "run.json"
    settings_path = args.reference / "validation_candidates.csv"
    reference = json.loads(ref_path.read_text())
    shrink = pd.read_csv(settings_path).set_index("name").at[RIDGE, "shrink"]
    history_val = history_predictions(
        dataset,
        calibration,
        reference,
        shrink,
        {"uniform": None, **HISTORY},
        [calibration.dates.year == year for year in YEARS],
    )
    history_scores = {name: annual_score(values, valid) for name, values in history_val.items()}
    selected_decay = min(
        (name for name in HISTORY if name.startswith("decay_")), key=history_scores.get
    )
    history_test = history_predictions(
        dataset,
        holdout,
        reference,
        shrink,
        {
            "uniform": None,
            "window_1460": 0,
            selected_decay: HISTORY[selected_decay],
        },
        [np.ones(len(holdout.dates), dtype=bool)],
    )
    predictions = {
        "ensemble": test.ensemble,
        "hybrid_15_gamma": gamma_test.ensemble,
        **gamma_corrections,
        **residual_test,
        **{k: v for k, v in history_test.items() if k != "uniform"},
    }
    validation = {
        "ensemble": valid.ensemble,
        **residual_val,
        **{k: v for k, v in history_val.items() if k != "uniform"},
    }
    scores = pd.Series(
        {name: annual_score(prediction, valid) for name, prediction in validation.items()},
        name="validation_rmse_to_rw",
    ).sort_values()
    periods = {
        "validation_before_break": (valid.loc[valid.index + HORIZON < BREAK], validation),
        "validation_after_break": (valid.loc[valid.index >= BREAK], validation),
        "test": (test, predictions),
        "test_2025": (test.loc["2025"], predictions),
        "test_2026": (test.loc["2026"], predictions),
    }
    metrics = pd.concat(
        [summarize(part, models).assign(period=name) for name, (part, models) in periods.items()],
        ignore_index=True,
    )
    comparisons = {
        "static": ("regime", "cny_dynamics", "ruonia_dynamics", "spread_dynamics"),
        "regime": ("regime_rvi",),
        "hybrid_15_gamma": ("cny_dynamics", "spread_dynamics", *gamma_corrections),
        "window_1460": (selected_decay,),
    }
    paired = pd.concat(
        [
            summarize(
                test.assign(ensemble=predictions[base]), {name: predictions[name] for name in names}
            ).assign(reference=base)
            for base, names in comparisons.items()
        ],
        ignore_index=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(grid, args.output / "residual_grid.csv")
    write_csv(scores.rename_axis("model").reset_index(), args.output / "validation.csv")
    write_csv(metrics, args.output / "metrics.csv")
    write_csv(paired, args.output / "paired_comparisons.csv")
    predictions = pd.DataFrame(predictions).join(test[["actual", "last"]])
    write_csv(predictions.rename_axis("origin").reset_index(), args.output / "predictions.csv.gz")
    metadata = {
        "selected_decay": selected_decay,
        "selected_on_validation": scores.index[0],
        "ridge_weight": reference["ensemble_weights"][RIDGE],
        "validation_dates": len(valid),
        "test_dates": len(test),
        "inputs": input_fingerprints(
            [
                args.dataset,
                archive_metadata,
                *paths,
                args.oof,
                args.calendar,
                ref_path,
                settings_path,
            ]
        ),
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(scores.to_string())
    print(metrics.loc[metrics.period.eq("test")].round(5).to_string(index=False))


if __name__ == "__main__":
    root = Path("data/reports")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/datasets/usd_rub_2011_2026_full_rvi.csv.gz")
    )
    parser.add_argument("--archive", type=Path, default=root / "forecast_stacking")
    parser.add_argument("--oof", type=Path, default=root / "llm_news_holdout/baseline_oof.csv.gz")
    parser.add_argument(
        "--reference", type=Path, default=root / "ml_holdout_minfin_rosstat/with_file"
    )
    parser.add_argument("--calendar", type=Path, default=Path("config/cbr_fixing_calendar.json"))
    parser.add_argument("--output", type=Path, default=root / "weekly_literature_holdout")
    with threadpool_limits(limits=2):
        run(parser.parse_args())
