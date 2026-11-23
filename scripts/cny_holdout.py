"""CNY/RUB transfer of the frozen USD/RUB model families, with yuan market features.

The 2021–2024 folds choose shrinkage, ensemble weights and feature variant. The
2025–2026 period is evaluated once after those choices are frozen.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from usd_rub.backtest import daily_target
from usd_rub.features import asof_series
from usd_rub.ml_dataset import feature_panels, read_dataset
from usd_rub.ml_experiment import (
    Candidate,
    actual_at,
    blend_weights,
    choose_shrink,
    fit_predict,
    relative_rmse,
    selection_loss,
    specific_features,
    to_levels,
)
from usd_rub.storage import input_fingerprints, read_series


USD_TARGET_BASES = {
    "fx__si_front_settle_basis",
    "fx__usdrubf_close_basis",
    "fx__moex_usd_fixing_basis",
    "global__synthetic_basis",
}
COMPONENTS = (
    Candidate("histboost", "market", 1460, 3),
    Candidate("ridge", "pilot", None, 0.15),
    Candidate("extra_trees", "market", 1460, 3, pooled=True),
)


def own_features(target):
    log = np.log(target)
    result = {}
    for lag in (1, 2, 5, 10, 21):
        result[f"own__return_{lag}"] = log.diff(lag)
    for window in (7, 21, 60):
        result[f"own__vol_{window}"] = log.diff().rolling(window).std()
    return pd.DataFrame(result, index=target.index)


def aligned_series(root, name, index):
    path = root / "normalized" / f"{name}.csv"
    if not path.exists():
        return pd.Series(np.nan, index=index, name=name)
    frame = read_series(root, name)
    cutoffs = index.tz_localize("Europe/Moscow")
    aligned = asof_series(frame, cutoffs)
    return pd.Series(aligned.value.to_numpy(dtype=float), index=index, name=name)


def yuan_features(root, target):
    """Only observations available at the 00:00 Moscow forecast cutoff."""
    index = target.index
    names = (
        "cny_tom_open",
        "cny_tom_high",
        "cny_tom_low",
        "cnyrubf_close",
        "cnyrubf_oi",
        "cnyrubf_volume",
        "cnyrubf_swaprate",
        "cny_implied_overnight_rate",
        "rusfar_cny",
    )
    series = {name: aligned_series(root, name, index) for name in names}
    result = {}
    for name in ("cny_tom_open", "cny_tom_high", "cny_tom_low", "cnyrubf_close"):
        log = np.log(series[name].where(series[name] > 0))
        for lag in (1, 5):
            result[f"fx__{name}_change_{lag}"] = log.diff(lag)
    result["fx__cny_tom_intraday_range"] = np.log(
        series["cny_tom_high"].where(series["cny_tom_high"] > 0)
        / series["cny_tom_low"].where(series["cny_tom_low"] > 0)
    )
    result["fx__cnyrubf_target_basis"] = np.log(
        series["cnyrubf_close"].where(series["cnyrubf_close"] > 0) / target
    )
    for name in ("cnyrubf_oi", "cnyrubf_volume"):
        value = np.log1p(series[name].where(series[name] >= 0))
        result[f"fx__{name}_change_1"] = value.diff()
        result[f"fx__{name}_level"] = value
    for name in ("cnyrubf_swaprate", "cny_implied_overnight_rate", "rusfar_cny"):
        value = series[name]
        result[f"fx__{name}_level"] = value
        result[f"fx__{name}_change_1"] = value.diff()
    return pd.DataFrame(result, index=index).replace([np.inf, -np.inf], np.nan)


def panels_for_cny(snapshot, target, root):
    strict = feature_panels(snapshot.loc[target.index])["strict"].copy()
    strict = strict.drop(columns=list(USD_TARGET_BASES & set(strict)))
    # The saved USD panel has the same field names, but those fields must contain
    # CNY history once CNY becomes the target.
    own = own_features(target)
    for name in own:
        strict[name] = own[name]
    usd_quotes = [
        name
        for name in strict
        if name.startswith(("fx__si_front_", "fx__usdrubf_", "fx__moex_usd_fixing_"))
    ]
    enriched = strict.drop(columns=usd_quotes).join(yuan_features(root, target))
    return {"base": strict, "yuan": enriched}


def model_features(panel, candidate):
    selected = specific_features({"strict": panel}, candidate.setup)
    if candidate.setup == "pilot" and "fx__cnyrubf_target_basis" in panel:
        selected = selected.join(
            panel[
                [
                    "fx__cnyrubf_target_basis",
                    "fx__cnyrubf_close_change_1",
                    "fx__cnyrubf_close_change_5",
                ]
            ]
        )
    return selected


def metrics(prediction, actual, last):
    error = prediction - actual
    rw_error = last[:, None] - actual
    return pd.DataFrame(
        {
            "horizon": np.arange(1, actual.shape[1] + 1),
            "n": len(last),
            "rmse": np.sqrt(np.mean(error**2, axis=0)),
            "rw_rmse": np.sqrt(np.mean(rw_error**2, axis=0)),
            "mape_pct": 100 * np.mean(np.abs(error / actual), axis=0),
            "rmse_ratio": relative_rmse(prediction, actual, last),
        }
    )


def run(snapshot_path, root, output):
    snapshot = read_dataset(snapshot_path)
    target = daily_target(read_series(root, "cny_rub"), "2015-01-01", "2026-11-18")
    if not target.index.isin(snapshot.index).all():
        raise ValueError("Снимок USD не покрывает весь целевой CNY/RUB")
    panels = panels_for_cny(snapshot, target, root)
    values = target.to_numpy(dtype=float)
    horizons = np.arange(1, 8)
    test_split = target.index.searchsorted("2025-01-01")
    validation = []
    for year in (2021, 2022, 2023, 2024):
        first = target.index.searchsorted(f"{year}-01-01")
        stop = target.index.searchsorted(f"{year + 1}-01-01")
        origins = np.arange(first, stop - 7)
        validation.append((origins, actual_at(values, origins, horizons), values[origins]))
    decisions = {}
    for variant, panel in panels.items():
        print(f"Валидация {variant}: {len(panel.columns)} признаков", flush=True)
        predictions = {}
        shrinks = {}
        feature_sets = {}
        for candidate in COMPONENTS:
            frame = model_features(panel, candidate)
            x = frame.to_numpy(dtype=float)
            raw = []
            for origins, _, _ in validation:
                prediction, _ = fit_predict(
                    x, values, candidate, origins[0], origins, horizons, min_hit_rate=0.1
                )
                raw.append(prediction)
            if not all(np.isfinite(p).all() for p in raw):
                raise ValueError(f"Неконечный прогноз на валидации: {variant} / {candidate.name}")
            _, shrink = choose_shrink(raw, validation)
            predictions[candidate.name] = [
                to_levels(p, fold[2], shrink)
                for p, fold in zip(raw, validation, strict=True)
            ]
            shrinks[candidate.name] = shrink
            cv_score = selection_loss(predictions[candidate.name], validation)
            print(f"  {candidate.name}: {cv_score:.4f}", flush=True)
            _, keys = fit_predict(
                x,
                values,
                candidate,
                test_split,
                np.array([test_split]),
                horizons,
                min_hit_rate=0.1,
            )
            feature_sets[candidate.name] = frame.columns[keys].tolist()
        rw = [np.broadcast_to(fold[2][:, None], fold[1].shape) for fold in validation]
        names = ["rw", *(candidate.name for candidate in COMPONENTS)]
        weights = blend_weights([rw, *(predictions[name] for name in names[1:])], validation)
        ensemble = [
            sum(weight * pred for weight, pred in zip(weights, fold, strict=True))
            for fold in zip(rw, *(predictions[name] for name in names[1:]), strict=True)
        ]
        losses = {"rw": 1.0, "ensemble": selection_loss(ensemble, validation)}
        losses.update({name: selection_loss(predictions[name], validation) for name in names[1:]})
        decisions[variant] = {
            "cv_scores": losses,
            "champion": min(losses, key=losses.get),
            "weights": dict(zip(names, weights.tolist(), strict=True)),
            "shrinks": shrinks,
            "features_used": feature_sets,
        }
    selected_variant = min(
        decisions,
        key=lambda variant: decisions[variant]["cv_scores"][decisions[variant]["champion"]],
    )
    origins = np.arange(test_split, len(target) - 7)
    actual = actual_at(values, origins, horizons)
    last = values[origins]
    rows = []
    scores = []
    for variant, panel in panels.items():
        model_preds = {"rw": np.broadcast_to(last[:, None], actual.shape)}
        for candidate in COMPONENTS:
            x = model_features(panel, candidate).to_numpy(dtype=float)
            raw, _ = fit_predict(
                x, values, candidate, test_split, origins, horizons, min_hit_rate=0.1
            )
            if not np.isfinite(raw).all():
                raise ValueError(f"Неконечный прогноз на тесте: {variant} / {candidate.name}")
            model_preds[candidate.name] = to_levels(
                raw, last, decisions[variant]["shrinks"][candidate.name]
            )
        model_preds["ensemble"] = sum(
            weight * model_preds[name]
            for name, weight in decisions[variant]["weights"].items()
        )
        for name, pred in model_preds.items():
            part = metrics(pred, actual, last)
            part.insert(0, "model", name)
            part.insert(0, "variant", variant)
            scores.append(part)
            rows.append(
                pd.DataFrame(
                    {
                        "variant": variant,
                        "model": name,
                        "origin": np.repeat(target.index[origins], 7),
                        "target_date": target.index[(origins[:, None] + horizons).ravel()],
                        "horizon": np.tile(horizons, len(origins)),
                        "last": np.repeat(last, 7),
                        "actual": actual.ravel(),
                        "prediction": pred.ravel(),
                    }
                )
            )
    output.mkdir(parents=True, exist_ok=True)
    decision = {
        "target": "cny_rub",
        "origin_cutoff": "00:00 Europe/Moscow",
        "validation_years": [2021, 2022, 2023, 2024],
        "test_start": "2025-01-01",
        "test_last_origin": str(target.index[origins[-1]].date()),
        "selected_variant": selected_variant,
        "variants": decisions,
        "inputs": input_fingerprints(
            [
                snapshot_path,
                root / "normalized" / "cny_rub.csv",
                *[
                    root / "normalized" / f"{name}.csv"
                    for name in (
                        "cny_tom_open",
                        "cny_tom_high",
                        "cny_tom_low",
                        "cnyrubf_close",
                        "cnyrubf_oi",
                        "cnyrubf_volume",
                        "cnyrubf_swaprate",
                        "cny_implied_overnight_rate",
                        "rusfar_cny",
                    )
                    if (root / "normalized" / f"{name}.csv").exists()
                ],
                Path(__file__),
                Path(__file__).parents[1] / "src" / "usd_rub" / "ml_experiment.py",
            ]
        ),
    }
    (output / "selection.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2))
    pd.concat(scores, ignore_index=True).to_csv(output / "metrics.csv", index=False)
    pd.concat(rows, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    panel = panels[selected_variant]
    latest = np.array([len(target) - 1])
    latest_level = values[latest]
    forecast = {"rw": np.broadcast_to(latest_level[:, None], (1, len(horizons)))}
    for candidate in COMPONENTS:
        x = model_features(panel, candidate).to_numpy(dtype=float)
        raw, _ = fit_predict(x, values, candidate, len(target), latest, horizons, min_hit_rate=0.1)
        if not np.isfinite(raw).all():
            raise ValueError(f"Неконечный итоговый прогноз: {candidate.name}")
        forecast[candidate.name] = to_levels(
            raw, latest_level, decisions[selected_variant]["shrinks"][candidate.name]
        )
    forecast["ensemble"] = sum(
        weight * forecast[name]
        for name, weight in decisions[selected_variant]["weights"].items()
    )
    pd.DataFrame(
        [
            {
                "origin": target.index[-1],
                "target_date": target.index[-1] + pd.Timedelta(days=int(horizon)),
                "horizon": horizon,
                "model": name,
                "prediction": values[-1] if name == "rw" else float(pred[0, horizon - 1]),
            }
            for name, pred in forecast.items()
            for horizon in horizons
        ]
    ).to_csv(output / "forecast.csv", index=False)
    print(f"Выбрано на валидации: {selected_variant} / {decisions[selected_variant]['champion']}")
    print(pd.concat(scores).loc[lambda x: x.horizon.isin([1, 7])].to_string(index=False))
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("datasets/usd_rub_2011_2026.csv.gz"))
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("experiments/cny_holdout_20261119"))
    args = parser.parse_args()
    run(args.snapshot, args.data_dir, args.output)
