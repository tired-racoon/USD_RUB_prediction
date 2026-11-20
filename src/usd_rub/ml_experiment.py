"""Замороженный тест моделей поправки к случайному блужданию."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from usd_rub.backtest import daily_target, score
from usd_rub.meeting_features import CALENDAR_PATH
from usd_rub.ml_dataset import clean_dataset, feature_panels, read_dataset
from usd_rub.ml_features import load_panel
from usd_rub.models import MODEL_FACTORIES
from usd_rub.storage import input_fingerprints, read_series


@dataclass(frozen=True)
class Candidate:
    model: str
    setup: str
    window: int | None
    complexity: float
    pooled: bool = False

    @property
    def name(self):
        kind = "pooled" if self.pooled else "direct"
        return f"{self.model}_{self.setup}_{self.window or 'full'}_{self.complexity:g}_{kind}"


SETUPS = {
    "pilot": ("strict", None),
    "own": ("strict", {"own", "calendar"}),
    "market": ("strict", {"own", "calendar", "fx", "markets"}),
    "strict_all": ("strict", None),
    "extended": ("extended", None),
    "extended_no_news": (
        "extended",
        {"own", "calendar", "fx", "markets", "global", "rates", "macro"},
    ),
}


def candidates():
    result = [
        Candidate("ridge", setup, window, penalty)
        for setup in SETUPS
        for window, penalty in [(730, 0.15), (1460, 0.15), (None, 0.15)]
    ]
    result.extend(
        Candidate("ridge", setup, 1460, penalty)
        for setup in ["market", "extended"]
        for penalty in [0.04, 0.5]
    )
    result.extend(
        Candidate(model, setup, window, depth, pooled)
        for model in ["catboost", "extra_trees", "histboost"]
        for setup, window, depth, pooled in [
            ("market", 1460, 3, True),
            ("strict_all", None, 4, True),
            ("extended", 1460, 3, True),
            ("extended", None, 4, True),
            ("extended_no_news", 1460, 3, True),
        ]
    )
    result.extend(Candidate(model, "market", 1460, 3) for model in ["catboost", "histboost"])
    return result


def specific_features(panels, setup):
    mode, groups = SETUPS[setup]
    panel = panels[mode]
    if setup == "pilot":
        selected = [
            "own__return_1",
            "own__return_5",
            "own__vol_21",
            "fx__si_front_settle_basis",
            "fx__cny_basis",
        ]
        for block, names in [
            ("fx", ["cny_rub", "eur_rub", "si_front_settle", "cny_tom_close"]),
            ("markets", ["imoex", "vix"]),
        ]:
            selected.extend(f"{block}__{name}_change_{lag}" for name in names for lag in (1, 5))
        return panel[[name for name in selected if name in panel]]
    if groups is None:
        return panel
    return panel.loc[:, [col for col in panel if col.split("__")[0] in groups]]


def eligible_rows(split, horizon, window=None):
    first = max(0, split - window) if window else 0
    return np.arange(first, max(first, split - horizon))


def fit_predict(x, y, cfg, split, origins, horizons, *, min_hit_rate=0, half_life_days=None):
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    if half_life_days is not None and (not np.isfinite(half_life_days) or half_life_days <= 0):
        raise ValueError("Период полураспада весов должен быть положительным и конечным")
    window_start = max(0, split - cfg.window) if cfg.window else 0
    observed = x[window_start : split - 1]
    finite = np.isfinite(observed)
    counts = finite.sum(axis=0)
    keep = counts >= max(60, min_hit_rate * len(observed))
    if not keep.any():
        raise ValueError(f"Нет признаков с данными в обучении {cfg.name}")
    keys = np.flatnonzero(keep)
    x = x[:, keys]
    history = observed[:, keys]
    if cfg.model == "ridge":
        center = np.nanmean(history, axis=0)
        spread = np.nanstd(history, axis=0)
        spread = np.where(spread < 1e-10, 1, spread)
        x = np.nan_to_num((x - center) / spread).clip(-8, 8)
    elif cfg.model == "extra_trees":
        x = np.where(np.isfinite(x), x, np.nanmedian(history, axis=0))

    def train_model(train, target, weight, query):
        if cfg.model == "ridge":
            model = Ridge(alpha=cfg.complexity * len(target), fit_intercept=False)
        elif cfg.model == "extra_trees":
            model = ExtraTreesRegressor(
                n_estimators=96,
                max_depth=int(cfg.complexity) + 3,
                min_samples_leaf=20,
                max_features=0.65,
                random_state=191,
                n_jobs=2,
            )
        elif cfg.model == "histboost":
            model = HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.035,
                min_samples_leaf=40,
                max_leaf_nodes=2 ** int(cfg.complexity),
                max_bins=32,
                l2_regularization=4,
                early_stopping=False,
                random_state=191,
            )
        else:
            from catboost import CatBoostRegressor

            model = CatBoostRegressor(
                iterations=145,
                depth=int(cfg.complexity),
                learning_rate=0.035,
                l2_leaf_reg=12,
                border_count=32,
                random_seed=191,
                has_time=True,
                thread_count=2,
                verbose=False,
                allow_writing_files=False,
            )
        model.fit(train, target, sample_weight=weight)
        return model.predict(query)

    def data_for(h):
        train = eligible_rows(split, h, cfg.window)
        delta = y[train + h] / y[train] - 1
        weight = (y[train] / np.median(y[train])) ** 2
        if half_life_days is not None:
            age = train - train[-1]
            decay = np.exp2(age / half_life_days)
            weight = weight * decay * (weight.sum() / (weight @ decay))
        return train, delta, weight

    if not cfg.pooled:
        predictions = []
        for h in horizons:
            train, delta, weight = data_for(h)
            predictions.append(train_model(x[train], delta, weight, x[origins]))
        return np.column_stack(predictions), keys

    train_rows, labels, weights, feature_rows = [], [], [], []
    for h in horizons:
        train, delta, weight = data_for(h)
        train_rows.append(train)
        labels.append(delta * 100)
        weights.append(weight / h)
        feature_rows.append(np.column_stack([x[train], np.full(len(train), h)]))
    ordering = np.argsort(np.concatenate(train_rows), kind="stable")
    query = np.vstack([np.column_stack([x[origins], np.full(len(origins), h)]) for h in horizons])
    prediction = train_model(
        np.vstack(feature_rows)[ordering],
        np.concatenate(labels)[ordering],
        np.concatenate(weights)[ordering],
        query,
    )
    return prediction.reshape(len(horizons), len(origins)).T / 100, keys


def actual_at(y, origins, horizons):
    return y[origins[:, None] + horizons[None, :]]


def relative_rmse(predicted, actual, last):
    rmse = np.sqrt(np.mean((predicted - actual) ** 2, axis=0))
    rw_rmse = np.sqrt(np.mean((last[:, None] - actual) ** 2, axis=0))
    return rmse / rw_rmse


def selection_loss(predicted, splits):
    return float(
        np.mean(
            [
                relative_rmse(prediction, actual, last).mean()
                for prediction, (_, actual, last) in zip(predicted, splits, strict=True)
            ]
        )
    )


def to_levels(relative, last, shrink=1):
    return last[:, None] * (1 + shrink * relative)


def choose_shrink(relative, splits):
    options = [
        (
            selection_loss(
                [to_levels(a, s[2], k) for a, s in zip(relative, splits, strict=True)], splits
            ),
            k,
        )
        for k in (0, 0.25, 0.5, 0.75, 1)
    ]
    return min(options)


def blend_weights(predictions, splits):
    count = len(predictions)

    def objective(weights):
        blended = [
            sum(w * a for w, a in zip(weights, fold, strict=True))
            for fold in zip(*predictions, strict=True)
        ]
        return selection_loss(blended, splits) + 0.002 * np.dot(weights[1:], weights[1:])

    result = minimize(
        objective,
        np.r_[1.0, np.zeros(count - 1)],
        method="SLSQP",
        bounds=[(0, 1)] * count,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Не подобраны веса ансамбля: {result.message}")
    weights = result.x.clip(0, 1)
    return weights / weights.sum()


def baseline_predictions(y, split, origins, horizons, window):
    values = y.to_numpy()
    training = values[max(0, split - window) : split]
    forecasts = {}
    for name, create in MODEL_FACTORIES.items():
        fitted = create().fit(training)
        forecasts[name] = [
            fitted.predict(len(horizons), recent=values[origin - 6 : origin + 1])
            for origin in origins
        ]
    return forecasts


def prediction_rows(name, y, origins, horizons, prediction, lower, upper, scale, train_size):
    dates = y.index.to_numpy()
    actual_pos = origins[:, None] + horizons[None, :]
    values = y.to_numpy()
    return pd.DataFrame(
        {
            "model": name,
            "origin": np.repeat(dates[origins], len(horizons)),
            "horizon": np.tile(horizons, len(origins)),
            "target_date": dates[actual_pos].ravel(),
            "last": np.repeat(values[origins], len(horizons)),
            "actual": values[actual_pos].ravel(),
            "prediction": prediction.ravel(),
            "lower": lower.ravel(),
            "upper": upper.ravel(),
            "mase_scale": scale,
            "train_size": train_size,
        }
    )


def interval_from_validation(prediction, last, validation, splits):
    errors = np.concatenate(
        [(s[1] - estimate) / s[2][:, None] for estimate, s in zip(validation, splits, strict=True)]
    )
    lower, upper = np.quantile(errors, [0.025, 0.975], axis=0)
    return prediction + last[:, None] * lower, prediction + last[:, None] * upper


def bootstrap_gain(prediction, actual, last, seed=109, draws=1000, block=28):
    rng = np.random.default_rng(seed)
    n = len(last)
    starts = rng.integers(n, size=(draws, (n + block - 1) // block))
    positions = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    rw_loss = (actual - last[:, None]) ** 2
    model_loss = (actual - prediction) ** 2
    ratios = np.sqrt(model_loss[positions].sum(axis=1) / rw_loss[positions].sum(axis=1))
    return np.quantile(100 * (1 - ratios.mean(axis=1)), [0.025, 0.975]).tolist()


def compare(config_path, data_dir, output, *, dataset_path=None):
    started = time.monotonic()
    config_path, root, output = map(Path, [config_path, data_dir, output])
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    min_hit_rate = 0
    if dataset_path is None:
        target_path = root / "normalized" / "usd_rub.csv"
        target = daily_target(read_series(root, "usd_rub"), config["start"], config["end"])
        print("Сборка двух панелей признаков в 00:00 МСК", flush=True)
        panels, paths = load_panel(root, config, target)
        data_inputs = [target_path, CALENDAR_PATH, *paths]
    else:
        dataset_path = Path(dataset_path)
        min_hit_rate = config.get("min_hit_rate", 0.5)
        dataset = clean_dataset(read_dataset(dataset_path), min_hit_rate)
        dates = pd.date_range(config["start"], config["end"])
        test_start = pd.Timestamp(config["test_start"])
        if not dataset.index.equals(dates) or not np.array_equal(
            dataset["split"], np.where(dates < test_start, "train", "test")
        ):
            raise ValueError("Даты или разбиение CSV расходятся с конфигурацией")
        target = dataset["usd_rub"]
        if not (np.isfinite(target) & target.gt(0)).all():
            raise ValueError("Некорректный курс в CSV")
        panels = feature_panels(dataset)
        data_inputs = [dataset_path]
    values = target.to_numpy()
    horizons = np.arange(1, config["horizon"] + 1)
    split = target.index.searchsorted(config["test_start"])
    origins = np.arange(split, len(target) - horizons[-1])
    if not len(origins):
        raise ValueError("Тестовый период не содержит завершённых прогнозов")
    source_files = [
        Path(__file__),
        *[
            Path(__file__).with_name(name)
            for name in [
                "ml_features.py",
                "ml_dataset.py",
                "features.py",
                "meeting_features.py",
                "freshness.py",
                "backtest.py",
                "models.py",
            ]
        ],
    ]
    input_paths = [*data_inputs, config_path, *source_files]
    inputs = input_fingerprints(input_paths)

    datasets = {setup: specific_features(panels, setup) for setup in SETUPS}
    matrices = {setup: frame.to_numpy(dtype=float) for setup, frame in datasets.items()}
    splits = []
    for year in config["validation_years"]:
        boundary = target.index.searchsorted(f"{year}-01-01")
        next_boundary = target.index.searchsorted(f"{year + 1}-01-01")
        if next_boundary > split:
            raise ValueError("Валидация пересекается с тестовым периодом")
        valid = np.arange(boundary, next_boundary - horizons[-1])
        splits.append((valid, actual_at(values, valid, horizons), values[valid]))
    rw_valid = [np.repeat(last[:, None], len(horizons), axis=1) for _, _, last in splits]

    def validate(candidate, matrix):
        return [
            fit_predict(
                matrix, values, candidate, valid[0], valid, horizons, min_hit_rate=min_hit_rate
            )[0]
            for valid, _, _ in splits
        ]

    validation, validation_rows, lookup, shrinks = {}, [], {}, {}
    for candidate in candidates():
        name = candidate.name
        lookup[name] = candidate
        raw = validate(candidate, matrices[candidate.setup])
        loss, k = choose_shrink(raw, splits)
        shrinks[name] = k
        validation[name] = [to_levels(a, s[2], k) for a, s in zip(raw, splits, strict=True)]
        raw_loss = selection_loss(
            [to_levels(a, s[2]) for a, s in zip(raw, splits, strict=True)], splits
        )
        validation_rows.append(
            {
                "name": name,
                **asdict(candidate),
                "shrink": k,
                "cv_score": loss,
                "raw_cv_score": raw_loss,
            }
        )
        print(f"{name} CV {loss:.6f} ×{k:g} ({time.monotonic() - started:.0f} с)", flush=True)
    ranking = pd.DataFrame(validation_rows).sort_values(["cv_score", "name"])
    ranking.to_csv(output / "validation_candidates.csv", index=False)
    family_winners = ranking.drop_duplicates("model").name.tolist()
    setup_winners = ranking.drop_duplicates("setup").name.tolist()
    selected = list(dict.fromkeys(family_winners + setup_winners))
    components = ["rw", *family_winners]
    ensemble_valid = [rw_valid, *[validation[name] for name in family_winners]]
    weights = blend_weights(ensemble_valid, splits)
    validation["ensemble"] = [
        sum(w * pred for w, pred in zip(weights, fold, strict=True))
        for fold in zip(*ensemble_valid, strict=True)
    ]
    ensemble_score = selection_loss(validation["ensemble"], splits)
    scores = {"rw": 1.0, **ranking.set_index("name").cv_score.to_dict(), "ensemble": ensemble_score}
    champion = min(["rw", *selected, "ensemble"], key=scores.__getitem__)
    best_individual = ranking.iloc[0]["name"]
    rich_name = ranking.loc[ranking.setup.eq("extended"), "name"].iloc[0]
    decision = {
        "champion": champion,
        "best_individual": best_individual,
        "family_winners": family_winners,
        "setup_winners": setup_winners,
        "ensemble_weights": dict(zip(components, weights.tolist(), strict=True)),
        "ensemble_cv_score": ensemble_score,
        "champion_cv_score": scores[champion],
        "test_start": config["test_start"],
        "training_last_date": str(target.index[split - 1].date()),
        "min_hit_rate": min_hit_rate,
        "validation_folds": [
            {"first": str(target.index[s[0][0]].date()), "last": str(target.index[s[0][-1]].date())}
            for s in splits
        ],
        "inputs": inputs,
    }
    (output / "selection.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"Зафиксирован выбор до подсчёта тестовых метрик: {champion}", flush=True)

    observed = values[origins]
    actual = actual_at(values, origins, horizons)
    baseline_window = config["baseline_window"]
    scale = float(np.abs(np.diff(values[split - baseline_window : split])).mean())
    predictions, records, feature_counts = {}, [], {}
    baselines = baseline_predictions(target, split, origins, horizons, baseline_window)
    for name, forecasts in baselines.items():
        predictions[name] = np.array([f.mean for f in forecasts])
        records.append(
            prediction_rows(
                name,
                target,
                origins,
                horizons,
                predictions[name],
                np.array([f.lower for f in forecasts]),
                np.array([f.upper for f in forecasts]),
                scale,
                baseline_window,
            )
        )
    for name in selected:
        candidate = lookup[name]
        raw, keys = fit_predict(
            matrices[candidate.setup],
            values,
            candidate,
            split,
            origins,
            horizons,
            min_hit_rate=min_hit_rate,
        )
        predictions[name] = to_levels(raw, observed, shrinks[name])
        feature_counts[name] = datasets[candidate.setup].columns[keys].tolist()
    predictions["ensemble"] = sum(
        w * predictions[name] for w, name in zip(weights, components, strict=True)
    )
    for name in [*selected, "ensemble"]:
        lower, upper = interval_from_validation(
            predictions[name], observed, validation[name], splits
        )
        cfg = lookup.get(name)
        train_size = min(split, cfg.window or split) if cfg else None
        records.append(
            prediction_rows(
                name, target, origins, horizons, predictions[name], lower, upper, scale, train_size
            )
        )
    combined = pd.concat(records, ignore_index=True)
    if not np.isfinite(combined.prediction).all():
        raise ValueError("Неконечный прогноз")
    combined.to_csv(output / "predictions.csv", index=False)
    metrics = score(combined)
    metrics.to_csv(output / "metrics.csv", index=False)
    by_year = combined.groupby(combined.origin.dt.year)
    pd.concat([score(part).assign(year=year) for year, part in by_year]).to_csv(
        output / "metrics_by_year.csv", index=False
    )
    summary = (
        metrics.assign(rmse_ratio=1 - metrics.rmse_skill_vs_rw)
        .groupby("model")[["rmse_ratio", "mape_pct", "smape_pct", "rmspe_pct"]]
        .mean()
        .sort_values("rmse_ratio")
    )
    summary["cv_score"] = pd.Series(scores)
    summary.to_csv(output / "summary.csv")
    print(summary.to_string(), flush=True)

    tests = [
        ("fx", ("fx__",)),
        ("si", ("fx__si_",)),
        ("cny", ("fx__cny_", "fx__cnyrubf_", "fx__cny_tom_", "fx__cny_rub_")),
        ("markets", ("markets__",)),
        ("global", ("global__",)),
        ("rates", ("rates__",)),
        ("macro", ("macro__",)),
        ("news", ("news__",)),
        ("gpr", ("news__gpr",)),
        ("calendar", ("calendar__",)),
        ("meetings", ("calendar__days_to_cbr_", "calendar__days_since_cbr_", "calendar__cbr_")),
    ]
    ablations = []
    for base in dict.fromkeys([best_individual, rich_name]):
        candidate = lookup[base]
        dataset = datasets[candidate.setup]
        base_score = relative_rmse(predictions[base], actual, observed).mean()
        for group, prefixes in tests:
            removed = [col for col in feature_counts[base] if col.startswith(prefixes)]
            if not removed:
                continue
            subset = dataset.loc[:, ~dataset.columns.str.startswith(prefixes)].to_numpy(dtype=float)
            raw, _ = fit_predict(
                subset, values, candidate, split, origins, horizons, min_hit_rate=min_hit_rate
            )
            omitted = to_levels(raw, observed, shrinks[base])
            ratios = relative_rmse(omitted, actual, observed)
            base_ratios = relative_rmse(predictions[base], actual, observed)
            ablations.append(
                {
                    "model": base,
                    "group": group,
                    "removed_columns": len(removed),
                    "ratio_without": ratios.mean(),
                    "gain_from_group_pp": 100 * (ratios.mean() - base_score),
                    "gain_h1_pp": 100 * (ratios[0] - base_ratios[0]),
                    "gain_h7_pp": 100 * (ratios[-1] - base_ratios[-1]),
                }
            )
            print(
                f"Проверка вклада {base} − {group}: {ablations[-1]['gain_from_group_pp']:+.3f} п.п.",
                flush=True,
            )
    pd.DataFrame(ablations).to_csv(output / "ablation.csv", index=False)
    pd.DataFrame(
        [{"model": name, "feature": col} for name, names in feature_counts.items() for col in names]
    ).to_csv(output / "features_used.csv", index=False)
    decision["test_first"] = str(target.index[origins[0]].date())
    decision["test_last_origin"] = str(target.index[origins[-1]].date())
    decision["test_last_target"] = str(target.index[origins[-1] + horizons[-1]].date())
    decision["test_origins"] = len(origins)
    decision["candidate_count"] = len(ranking)
    decision["bootstrap_gain_95_pct"] = bootstrap_gain(predictions[champion], actual, observed)
    if inputs != input_fingerprints(input_paths):
        raise ValueError("Входные файлы изменились во время эксперимента")
    (output / "run.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2))
    return decision


def main():
    from threadpoolctl import threadpool_limits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/ml_holdout.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", help="Единый CSV или CSV.GZ с признаками")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = args.output or (
        "data/reports/ml_holdout_50pct" if args.dataset else "data/reports/ml_holdout_2025_2026"
    )
    with threadpool_limits(limits=2):
        compare(args.config, args.data_dir, output, dataset_path=args.dataset)


if __name__ == "__main__":
    main()
