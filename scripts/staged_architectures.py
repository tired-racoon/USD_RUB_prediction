"""H6: linear, boosting, and time-OOF residual models on identical folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

from staged_features_pca import variants
from staged_ridge_h1_h3 import (
    HORIZONS,
    evaluate,
    fitted_features,
    fold_score,
    make_folds,
    validation_safe_dataset,
)


def hgb():
    return HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.035, min_samples_leaf=40,
        max_leaf_nodes=8, l2_regularization=4, early_stopping=False,
        random_state=191,
    )


def train_fold(x, y, dates, boundary, origins):
    transform, _ = fitted_features(x, np.arange(0, boundary - 1))
    zquery = transform(x[origins])
    model_names = (
        "ridge", "elasticnet", "histboost", "catboost",
        "ridge_plus_insample_residual_boost", "ridge_plus_oof_residual_boost",
    ) + (("lightgbm",) if LGBMRegressor is not None else ())
    output = {name: np.empty((len(origins), len(HORIZONS))) for name in model_names}
    for j, horizon in enumerate(HORIZONS):
        rows = np.arange(0, boundary - horizon)
        ztrain = transform(x[rows])
        label = np.log(y[rows + horizon] / y[rows])
        ridge = Ridge(alpha=0.15 * len(rows), fit_intercept=True).fit(ztrain, label)
        elastic = ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=5000,
                             random_state=191).fit(ztrain, label)
        boost = hgb().fit(ztrain, label)
        cat = CatBoostRegressor(
            iterations=145, depth=3, learning_rate=0.035, l2_leaf_reg=12,
            border_count=32, random_seed=191, has_time=True, thread_count=2,
            verbose=False, allow_writing_files=False,
        ).fit(ztrain, label)
        ridge_query = ridge.predict(zquery)
        raw = {
            "ridge": ridge_query,
            "elasticnet": elastic.predict(zquery),
            "histboost": boost.predict(zquery),
            "catboost": cat.predict(zquery),
        }
        if LGBMRegressor is not None:
            lightgbm = LGBMRegressor(
                n_estimators=145, max_depth=3, num_leaves=8,
                learning_rate=0.035, min_child_samples=40,
                reg_lambda=4, verbosity=-1, n_jobs=2, random_state=191,
            ).fit(ztrain, label)
            raw["lightgbm"] = lightgbm.predict(zquery)

        # OOF predictions from successively earlier calendar-year training sets.
        oof_rows, oof_predictions = [], []
        for year in range(2016, dates[boundary].year):
            inner_boundary = dates.searchsorted(pd.Timestamp(f"{year}-01-01"))
            inner_stop = min(dates.searchsorted(pd.Timestamp(f"{year + 1}-01-01")),
                             boundary - horizon)
            if inner_boundary >= inner_stop:
                continue
            inner_train = np.arange(0, inner_boundary - horizon)
            if len(inner_train) < 365:
                continue
            inner_transform, _ = fitted_features(x, np.arange(0, inner_boundary - 1))
            inner_label = np.log(y[inner_train + horizon] / y[inner_train])
            inner_ridge = Ridge(alpha=0.15 * len(inner_train), fit_intercept=True).fit(
                inner_transform(x[inner_train]), inner_label
            )
            indices = np.arange(inner_boundary, inner_stop)
            oof_rows.append(indices)
            oof_predictions.append(inner_ridge.predict(inner_transform(x[indices])))
        eligible = np.concatenate(oof_rows)
        oof = np.concatenate(oof_predictions)
        assert np.all(eligible + horizon < boundary)
        truth = np.log(y[eligible + horizon] / y[eligible])
        zeligible = transform(x[eligible])
        insample_residual = truth - ridge.predict(zeligible)
        oof_residual = truth - oof
        raw["ridge_plus_insample_residual_boost"] = (
            ridge_query + hgb().fit(zeligible, insample_residual).predict(zquery)
        )
        raw["ridge_plus_oof_residual_boost"] = (
            ridge_query + hgb().fit(zeligible, oof_residual).predict(zquery)
        )
        for name, predicted_log in raw.items():
            output[name][:, j] = y[origins] * np.exp(predicted_log)
    assert all(np.isfinite(value).all() for value in output.values())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/usd_rub_2011_2026.csv.gz")
    parser.add_argument("--previous", default="experiments/staged_features_pca_20261119/selection.json")
    parser.add_argument("--output", default="experiments/staged_architectures_20261119")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    previous = json.loads(Path(args.previous).read_text())
    alpha = previous["shrink_alpha"]
    data = validation_safe_dataset(args.dataset)
    dates = data.index
    y = data.usd_rub.to_numpy(dtype=float)
    panel = variants(data)[previous["h4_selected"]["variant"]]
    x = panel.to_numpy(dtype=float)
    folds, (test_boundary, test_origins) = make_folds(dates, "2025-01-01")
    fold_predictions = [train_fold(x, y, dates, boundary, origins)
                        for boundary, origins in folds]
    names = list(fold_predictions[0])
    cv = []
    for name in names:
        predictions = [fold[name] for fold in fold_predictions]
        cv.append({"model": name, "cv_relative_rmse": fold_score(predictions, folds, y, alpha)})
    # Simple static blend is an H6 architecture; its weight is validation-only.
    for boost_weight in np.linspace(0, 1, 5):
        forecasts = [
            (1 - boost_weight) * fold["ridge"] + boost_weight * fold["histboost"]
            for fold in fold_predictions
        ]
        cv.append({"model": f"ridge_histboost_blend_{boost_weight:g}",
                   "cv_relative_rmse": fold_score(forecasts, folds, y, alpha)})
    ranking = pd.DataFrame(cv).sort_values("cv_relative_rmse")
    ranking.to_csv(output / "validation.csv", index=False)
    winner = ranking.iloc[0].to_dict()
    decision = {"winner": winner, "alpha": alpha, "feature_panel": previous["h4_selected"]["variant"],
                "test_start": "2025-01-01", "test_count": len(test_origins),
                "lightgbm_status": "evaluated" if LGBMRegressor is not None else "not_installed"}
    (output / "selection.json").write_text(json.dumps(decision, indent=2))
    records = []
    validation_metrics = []
    for year, fold, (_, origins) in zip((2021, 2022, 2023, 2024), fold_predictions, folds,
                                        strict=True):
        for name, forecast in fold.items():
            last = y[origins, None]
            shrunk = last + alpha * (forecast - last)
            validation_metrics.extend({**row, "year": year} for row in
                                      evaluate(name, shrunk, origins, y))
            for j, horizon in enumerate(HORIZONS):
                records.append(pd.DataFrame({
                    "year": year, "model": name, "origin": dates[origins],
                    "horizon": horizon, "last": y[origins],
                    "actual": y[origins + horizon], "prediction": forecast[:, j],
                }))
    pd.concat(records, ignore_index=True).to_csv(output / "validation_predictions.csv", index=False)
    pd.DataFrame(validation_metrics).to_csv(output / "validation_metrics.csv", index=False)
    # Only the preselected architecture is scored on the historical holdout.
    fitted = train_fold(x, y, dates, test_boundary, test_origins)
    if winner["model"].startswith("ridge_histboost_blend_"):
        weight = float(winner["model"].rsplit("_", 1)[1])
        selected = (1 - weight) * fitted["ridge"] + weight * fitted["histboost"]
    else:
        selected = fitted[winner["model"]]
    rw = np.repeat(y[test_origins, None], len(HORIZONS), axis=1)
    selected = rw + alpha * (selected - rw)
    pd.DataFrame(evaluate("h6_selected", selected, test_origins, y)).to_csv(
        output / "test_metrics.csv", index=False
    )
    pd.concat([
        pd.DataFrame({"model": "h6_selected", "origin": dates[test_origins],
                      "horizon": horizon, "last": y[test_origins],
                      "actual": y[test_origins + horizon], "prediction": selected[:, j]})
        for j, horizon in enumerate(HORIZONS)
    ], ignore_index=True).to_csv(output / "test_predictions.csv", index=False)
    # Unscored component forecasts make later H7 dynamic weighting reproducible.
    pd.concat([
        pd.DataFrame({"model": name, "origin": dates[test_origins], "horizon": horizon,
                      "last": y[test_origins], "actual": y[test_origins + horizon],
                      "prediction": forecast[:, j]})
        for name, forecast in fitted.items() for j, horizon in enumerate(HORIZONS)
    ], ignore_index=True).to_csv(output / "test_components.csv", index=False)
    print(ranking.to_string(index=False))
    print(decision)


if __name__ == "__main__":
    main()
