"""Сравнить почасовые и дневные валютные признаки на прогнозе USD/RUB."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from forecast_extensions import annual_folds, load_inputs
from forecast_stacking import panel_index
from foundation_ensemble import RW
from llm_news_holdout import bootstrap
from scipy.linalg import solve
from theory_holdout import PRIMARY, references
from threadpoolctl import threadpool_limits

from usd_rub.features import asof_series
from usd_rub.fixing_calendar import HORIZONS, apply_calendar
from usd_rub.fx_candles import candle_features
from usd_rub.market_correction import (
    MACRO_BUNDLE,
    corrected_forecast,
    fit_corrections,
    load_correction_bundle,
)
from usd_rub.ml_dataset import feature_panels
from usd_rub.ml_experiment import relative_rmse, selection_loss
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints, read_series
from usd_rub.theory_features import FUNDAMENTAL_SIGNS, MACRO_CORRECTION_FEATURES

INCUMBENT = "incumbent_macro"
CHANGE = pd.Timestamp("2024-06-13")
LAGS = {
    "expanding": None,
    "annual_730": 730,
    "rolling_365": 365,
    "rolling_730": 730,
    "decay_180": 1095,
    "decay_365": 1460,
    "regime": None,
}


@dataclass(frozen=True)
class Experiment:
    """Одна комбинация признаков, модели и истории обучения."""

    features: str
    model: str
    anchor: str
    history: str
    penalty: float

    @property
    def name(self):
        return f"{self.features}/{self.model}/{self.anchor}/{self.history}/{self.penalty:g}"


def experiments():
    """Задать сравнения источников, моделей и скорости адаптации."""
    result = [
        Experiment(features, "ridge", anchor, history, 0.12)
        for features in ("daily", "spot", "futures", "candles")
        for anchor in ("direct", "residual")
        for history in ("expanding", "rolling_365", "decay_180")
    ]
    result += [
        Experiment(features, model, anchor, history, penalty)
        for features in ("daily", "candles")
        for anchor in ("direct", "residual")
        for model, history, penalty in (
            ("pool", "expanding", 0.10),
            ("pool", "rolling_730", 0.10),
            ("pool", "decay_180", 0.06),
            ("huber", "rolling_730", 0.10),
            ("xgb", "expanding", 2),
            ("xgb", "annual_730", 3),
        )
    ]
    result += [
        Experiment(features, model, anchor, history, penalty)
        for features in ("daily_compact", "candles_compact")
        for anchor in ("direct", "residual")
        for model, history, penalty in (
            ("ridge", "decay_365", 0.035),
            ("ridge", "rolling_365", 0.04),
            ("pool", "regime", 0.03),
            ("pool", "decay_365", 0.035),
            ("huber", "regime", 0.03),
        )
    ]
    result += [
        Experiment(features, model, anchor, history, penalty)
        for features in ("nbrb", "candles_nbrb", "basis", "basis_nbrb")
        for anchor in ("direct", "residual")
        for model, history, penalty in (
            ("ridge", "decay_365", 0.035),
            ("ridge", "rolling_365", 0.045),
            ("pool", "regime", 0.05),
        )
    ]
    return result


def daily_sources(root, dates):
    """Выровнять дневные котировки и курсы ЦБ по срезу 00:00 МСК."""
    names = (
        "cny_rub",
        "cny_tom_close",
        "cny_tom_high",
        "cny_tom_low",
        "si_front_close",
        "usdrubf_close",
        "cnyrubf_close",
        "usdrubf_oi",
        "cnyrubf_oi",
        "usdrubf_swaprate",
        "cnyrubf_swaprate",
    )
    cutoffs = dates.tz_localize("Europe/Moscow")
    daily = {
        name: asof_series(
            read_series(root, name), cutoffs, max_age_days=15 if name == "cny_rub" else 5
        ).value.set_axis(dates)
        for name in names
    }
    return pd.DataFrame(daily), [root / "normalized" / f"{name}.csv" for name in names]


def nbrb_features(path, official):
    """Взять кросс через BYN с даты вступления курса НБРБ в силу."""
    rates = pd.read_csv(path, parse_dates=["date"]).pivot(
        index="date", columns="currency", values="byn_per_unit"
    )
    cross = np.log(rates.USD / rates.RUB).reindex(official.index)
    spread = cross - np.log(official)
    return pd.DataFrame(
        {
            "nbrb__basis": spread,
            "nbrb__basis_centered": spread - spread.ewm(span=21, min_periods=7).mean(),
            "nbrb__basis_change": spread.diff(),
            "nbrb__cross_return_1": cross.diff(),
            "nbrb__cross_return_5": cross.diff(5),
        }
    )


def panels(dataset, candles, source, nbrb):
    """Подготовить равные наборы дневных признаков и расширения свечами."""
    strict = feature_panels(dataset)["strict"]
    daily = strict.loc[:, strict.columns.str.startswith(("own__", "fx__", "markets__"))]
    daily = daily.loc[:, ~daily.columns.str.endswith("_age")]
    logs = np.log(source.where(source > 0))
    daily = daily.assign(
        fx__usdrubf_close_basis=logs.usdrubf_close - np.log(dataset.usd_rub),
        fx__cny_perpetual_basis=logs.cnyrubf_close - logs.cny_tom_close,
        fx__cny_tom_range=logs.cny_tom_high - logs.cny_tom_low,
        fx__si_close_basis=logs.si_front_close - np.log(dataset.usd_rub),
    )
    for name in ("usdrubf_close", "cnyrubf_close", "usdrubf_oi", "cnyrubf_oi"):
        for days in (1, 5):
            daily[f"fx__{name}_change_{days}"] = logs[name].diff(days)
    for name in ("usdrubf_swaprate", "cnyrubf_swaprate"):
        daily[f"fx__{name}_level"] = source[name]
    compact = daily[
        [
            "own__return_1",
            "own__return_2",
            "own__return_5",
            "own__return_21",
            "own__vol_21",
            "fx__cny_basis",
            "fx__cny_tom_close_change_1",
            "fx__cny_tom_close_change_5",
            "fx__si_front_settle_basis",
            "fx__si_front_settle_change_1",
            "fx__si_front_settle_change_5",
            "fx__usdrubf_close_basis",
            "fx__cny_perpetual_basis",
        ]
    ]
    context = {}
    for column in ("fx__cny_basis", "fx__si_front_settle_basis", "fx__usdrubf_close_basis"):
        signal = daily[column]
        context[f"context__{column}_centered"] = signal - signal.ewm(span=21, min_periods=7).mean()
        context[f"context__{column}_change"] = signal.diff()
    context["context__post_otc"] = (dataset.index >= CHANGE).astype(float)
    daily = daily.assign(**context)
    compact = compact.assign(**context)
    micro = candle_features(candles, dataset.usd_rub, source.cny_rub)
    spot = micro.loc[:, micro.columns.str.contains("_spot__")]
    futures = micro.loc[:, ~micro.columns.isin(spot.columns)]
    select = micro.columns.str.endswith(
        (
            "__basis_centered",
            "__since_afternoon",
            "__since_early_evening",
            "__signed_volume",
            "__perp_basis",
        )
    )
    basis = compact[["fx__cny_basis", "own__return_1", "own__vol_21"]].join(
        micro[
            [
                "usd_spot__basis",
                "usd_perp__basis_centered",
                "usd_perp__since_afternoon",
                "cny_spot__since_afternoon",
                "cny_perp__since_afternoon",
                "cross__perp_basis",
            ]
        ]
    )
    cross = nbrb_features(nbrb, dataset.usd_rub)
    return {
        "daily": daily,
        "spot": daily.join(spot),
        "futures": daily.join(futures),
        "candles": daily.join(micro),
        "daily_compact": compact,
        "candles_compact": compact.join(micro.loc[:, select]),
        "nbrb": compact.join(cross),
        "candles_nbrb": compact.join(micro.loc[:, select]).join(cross),
        "basis": basis,
        "basis_nbrb": basis.join(cross),
    }, micro.join(cross)


def time_weight(dates, cutoff, cfg):
    """Взять только уже известные цели и назначить веса по давности."""
    age = (cutoff - dates).days.to_numpy()
    allowed = age > HORIZONS[-1]
    window = LAGS[cfg.history]
    if window:
        allowed &= age <= window
    weight = np.ones(len(dates))
    if cfg.history.startswith("decay"):
        weight = np.exp2(-age / int(cfg.history.split("_")[-1]))
    if cfg.history == "regime" and cutoff >= CHANGE:
        weight[dates < CHANGE] *= 0.25
        weight *= np.exp2(-age / 365)
    return allowed, weight[allowed]


def standardize(train, query):
    """Стандартизовать по обучению и обнулить неизвестные значения."""
    supported = np.isfinite(train).sum(axis=0) >= 45
    history = train[:, supported]
    center = np.nanmedian(history, axis=0)
    spread = np.nanstd(history, axis=0)
    spread[spread < 1e-12] = 1
    return tuple(
        np.nan_to_num((x[:, supported] - center) / spread).clip(-6, 6) for x in (train, query)
    )


def ridge(train, target, weight, query, penalty, robust=False):
    """Решить взвешенную Ridge с необязательной функцией потерь Хьюбера."""
    base_weight = weight * (len(weight) / weight.sum())
    shrink = np.eye(train.shape[1]) * penalty * len(weight)
    shrink[0, 0] *= 0.25
    for _ in range(4 if robust else 1):
        weighted = train.T * weight
        beta = solve(weighted @ train + shrink, weighted @ target, assume_a="pos")
        if robust:
            error = np.abs(target - train @ beta)
            k = max(0.15, 2.5 * np.median(error))
            weight = base_weight * np.minimum(1, k / np.maximum(error, 1e-9))
    return query @ beta


def linear_design(values, baseline, last, fix_count):
    """Разрешить коэффициентам зависеть от числа предстоящих фиксингов."""
    height = len(HORIZONS)
    rows = np.repeat(values, height, axis=0)
    progress = (fix_count.ravel() - 1) / 4
    anchor = 100 * np.log(baseline.ravel() / np.repeat(last, height))
    common = np.column_stack((np.ones(len(rows)), rows, anchor))
    return np.column_stack((common, common * progress[:, None], common * progress[:, None] ** 2))


def predict(
    cfg,
    train_x,
    query_x,
    actual,
    train_base,
    query_base,
    train_last,
    query_last,
    train_fixes,
    query_fixes,
    time_weights,
):
    """Обучить отдельно по горизонту или совместно по календарю фиксинга."""
    train_x, query_x = standardize(train_x, query_x)
    target = 100 * np.log(actual / train_base)
    weight = time_weights * (train_last / np.median(train_last)) ** 2
    weight *= len(weight) / weight.sum()
    train_fixes, query_fixes = train_fixes.cumsum(axis=1), query_fixes.cumsum(axis=1)
    active = train_fixes > 0
    if cfg.model in ("pool", "xgb"):
        if cfg.model == "pool":
            train = linear_design(train_x, train_base, train_last, train_fixes)
            query = linear_design(query_x, query_base, query_last, query_fixes)
        else:

            def tree_design(x, base, last, fixes):
                return np.column_stack(
                    (
                        np.repeat(x, len(HORIZONS), axis=0),
                        fixes.ravel(),
                        np.tile(HORIZONS, len(x)),
                        100 * np.log(base.ravel() / np.repeat(last, len(HORIZONS))),
                    )
                )

            train = tree_design(train_x, train_base, train_last, train_fixes)
            query = tree_design(query_x, query_base, query_last, query_fixes)
        pooled_weight = weight[:, None] / np.sqrt(HORIZONS)
        used = active.ravel()
        target = target.ravel()[used]
        weight = pooled_weight.ravel()[used]
        train = train[used]
        if cfg.model == "pool":
            estimate = ridge(train, target, weight, query, cfg.penalty)
        else:
            from xgboost import XGBRegressor

            model = XGBRegressor(
                n_estimators=72,
                max_depth=int(cfg.penalty),
                max_bin=64,
                learning_rate=0.04,
                min_child_weight=25,
                reg_lambda=20,
                subsample=0.85,
                colsample_bynode=0.8,
                tree_method="hist",
                n_jobs=1,
                random_state=191,
            )
            estimate = model.fit(train, target, sample_weight=weight).predict(query)
        estimate = estimate.reshape(query_base.shape)
    else:
        estimate = []
        for idx in range(len(HORIZONS)):
            train = np.column_stack(
                (
                    np.ones(len(train_x)),
                    train_x,
                    100 * np.log(train_base[:, idx] / train_last),
                    (train_fixes[:, idx] - 1) / 4,
                )
            )
            query = np.column_stack(
                (
                    np.ones(len(query_x)),
                    query_x,
                    100 * np.log(query_base[:, idx] / query_last),
                    (query_fixes[:, idx] - 1) / 4,
                )
            )
            used = active[:, idx]
            estimate.append(
                ridge(
                    train[used],
                    target[used, idx],
                    weight[used],
                    query,
                    cfg.penalty,
                    robust=cfg.model == "huber",
                )
            )
        estimate = np.column_stack(estimate)
    return query_base * np.exp(estimate.clip(-15, 15) / 100)


def prediction_for(cfg, x, dates, actual, last, fixes, baseline, rows):
    """Вернуть прогнозы с переобучением в начале заданного периода."""
    base = np.broadcast_to(last[:, None], actual.shape) if cfg.anchor == "direct" else baseline
    split_dates = dates[rows]
    periods = split_dates.to_period("Y" if cfg.history in ("expanding", "annual_730") else "M")
    result = np.empty_like(actual[rows])
    for period in periods.unique():
        selection = periods == period
        query = rows[selection]
        cutoff = split_dates[selection][0]
        train, weight = time_weight(dates, cutoff, cfg)
        output = predict(
            cfg,
            x[train],
            x[query],
            actual[train],
            base[train],
            base[query],
            last[train],
            last[query],
            fixes[train],
            fixes[query],
            weight,
        )
        result[selection] = apply_calendar(output, last[query], fixes[query], pool=True)
    return result


def metrics(prediction, actual, last):
    """Рассчитать основные ошибки на всех семи горизонтах."""
    rmse = np.sqrt(((prediction - actual) ** 2).mean(axis=0))
    mae = np.abs(prediction - actual).mean(axis=0)
    return {
        "mean_rmse_to_rw": float(relative_rmse(prediction, actual, last).mean()),
        **{f"rmse_{h}d": rmse[h - 1] for h in HORIZONS},
        **{f"mae_{h}d": mae[h - 1] for h in HORIZONS},
    }


def macro_reference(dataset, calibration, test, baseline, annual):
    """Воспроизвести текущую модель и её прежние валидационные фолды."""
    macro = feature_panels(dataset)["extended"][list(MACRO_CORRECTION_FEATURES)]
    covered = calibration.dates.append(test.dates)
    frame = macro.loc[covered]
    fit = []
    for train, query in annual:
        models = fit_corrections(
            frame.loc[calibration.dates[train]],
            calibration.actual[train],
            baseline[: len(calibration.dates)][train],
            calibration.last[train],
            FUNDAMENTAL_SIGNS,
        )
        fit.append(
            corrected_forecast(
                models,
                frame.loc[calibration.dates[query]],
                baseline[: len(calibration.dates)][query],
                calibration.last[query],
                calibration.fixes[query],
            )
        )
    models, _ = load_correction_bundle(MACRO_BUNDLE, MACRO_CORRECTION_FEATURES, FUNDAMENTAL_SIGNS)
    fit.append(
        corrected_forecast(
            models,
            frame.loc[test.dates],
            baseline[len(calibration.dates) :],
            test.last,
            test.fixes,
        )
    )
    return fit


def run(args):
    """Запустить сетку и сохранить результаты и происхождение входов."""
    dataset, calibration, test, _, paths = load_inputs(args)
    dates = calibration.dates.append(test.dates)
    actual = np.vstack((calibration.actual, test.actual))
    last = np.r_[calibration.last, test.last]
    fixes = np.vstack((calibration.fixes, test.fixes))
    baseline = np.vstack([references(archive)[PRIMARY] for archive in (calibration, test)])
    source, source_paths = daily_sources(args.data_dir, dataset.index)
    features, micro = panels(dataset, args.candles, source, args.nbrb)
    _, annual = annual_folds(calibration)
    validation = [np.flatnonzero(query) for _, query in annual]
    holdout = np.arange(len(calibration.dates), len(dates))
    splits = [calibration.split(query) for _, query in annual]
    arrays = {key: panel.loc[dates].to_numpy() for key, panel in features.items()}
    candidates = experiments()
    controls = {
        RW: [np.broadcast_to(last[idx, None], actual[idx].shape) for idx in (*validation, holdout)],
        PRIMARY: [baseline[idx] for idx in (*validation, holdout)],
        INCUMBENT: macro_reference(dataset, calibration, test, baseline, annual),
    }

    def execute(cfg):
        x = arrays[cfg.features]
        result = [
            prediction_for(cfg, x, dates, actual, last, fixes, baseline, rows)
            for rows in (*validation, holdout)
        ]
        print(cfg.name, flush=True)
        return cfg.name, result

    with threadpool_limits(limits=1), ThreadPoolExecutor(args.jobs) as pool:
        fitted = dict(pool.map(execute, candidates))
    outputs = controls | fitted
    summaries, yearly, choices = [], [], {}
    latest = validation[-1]
    post_otc = dates[latest] >= CHANGE
    for name, prediction in outputs.items():
        val_score = selection_loss(prediction[:-1], splits)
        summaries.append(
            {
                "experiment": name,
                "validation": val_score,
                "validation_post_otc": relative_rmse(
                    prediction[-2][post_otc], actual[latest][post_otc], last[latest][post_otc]
                ).mean(),
                **metrics(
                    prediction[-1],
                    actual[holdout],
                    last[holdout],
                ),
            }
        )
        for period, values in zip((*validation, holdout), prediction, strict=True):
            for year in sorted(set(dates[period].year)):
                mask = dates[period].year == year
                yearly.append(
                    {
                        "experiment": name,
                        "year": year,
                        **metrics(
                            values[mask],
                            actual[period][mask],
                            last[period][mask],
                        ),
                    }
                )
        choices[name] = np.vstack(prediction)
    table = pd.DataFrame(summaries).sort_values("validation")
    chosen = table.loc[table.experiment.isin(fitted)].iloc[0].experiment
    best_test = (
        table.loc[table.experiment.isin(fitted)].nsmallest(1, "mean_rmse_to_rw").iloc[0].experiment
    )
    comparisons = [
        {
            "experiment": name,
            "gain_ci_pp": bootstrap(
                controls[INCUMBENT][-1],
                outputs[name][-1],
                test.actual,
                test.last,
            ),
        }
        for name in dict.fromkeys((chosen, best_test))
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(table, args.output / "summary.csv")
    write_csv(pd.DataFrame(yearly), args.output / "by_year.csv")
    write_csv(micro.rename_axis("date").reset_index(), args.output / "candle_features.csv.gz")
    evaluated = np.concatenate((*validation, holdout))
    index = panel_index(dates[evaluated])
    predictions = pd.DataFrame(
        {
            "actual": actual[evaluated].ravel(),
            "last": np.repeat(last[evaluated], len(HORIZONS)),
            "fixing": fixes[evaluated].ravel(),
            **{name: values.ravel() for name, values in choices.items()},
        },
        index=index,
    )
    write_csv(predictions.reset_index(), args.output / "predictions.csv.gz")
    metadata = {
        "inputs": input_fingerprints(
            [
                *paths,
                args.candles,
                args.nbrb,
                *source_paths,
                MACRO_BUNDLE,
                Path(__file__),
                Path(candle_features.__code__.co_filename),
            ]
        ),
        "experiments": [asdict(candidate) | {"name": candidate.name} for candidate in candidates],
        "chosen_on_validation": chosen,
        "lowest_test_rmse": best_test,
        "paired_vs_incumbent": comparisons,
        "validation_origins": [len(rows) for rows in validation],
        "validation_post_otc_origins": int(post_otc.sum()),
        "test_origins": len(holdout),
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(table.head(12).to_string(index=False), flush=True)
    print(table.nsmallest(12, "mean_rmse_to_rw").to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/reports/theory_holdout/dataset.csv.gz")
    )
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument("--archive", type=Path, default=Path("data/reports/forecast_stacking"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--candles", type=Path, default=Path("data/research/fx_candles.csv.gz"))
    parser.add_argument("--nbrb", type=Path, default=Path("data/research/nbrb_fx.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/fx_candles_holdout"))
    parser.add_argument("--jobs", type=int, default=3)
    run(parser.parse_args())
