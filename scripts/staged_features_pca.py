"""Validation-only H4 group additions and H5 PCA after the H1-H3 selection."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from staged_ridge_h1_h3 import (
    HORIZONS,
    evaluate,
    fitted_features,
    fold_score,
    make_folds,
    validation_safe_dataset,
)
from usd_rub.ml_dataset import feature_panels
from usd_rub.ml_experiment import specific_features


def predict(x, names, y, boundary, origins, mode):
    history = np.arange(0, boundary - 1)
    normalize, keep = fitted_features(x, history)
    column_names = np.array(names)[keep]
    train_x = normalize(x[history])
    query_x = normalize(x[origins])

    def components(train, query, group_mode=False):
        if not group_mode:
            pca = PCA(n_components=min(5, train.shape[1]), svd_solver="full").fit(train)
            return pca.transform(train), pca.transform(query)
        train_parts, query_parts = [], []
        groups = np.array([name.split("__", 1)[0] for name in column_names])
        for group in np.unique(groups):
            selected = groups == group
            sub_train, sub_query = train[:, selected], query[:, selected]
            pca = PCA(n_components=min(2, sub_train.shape[1]), svd_solver="full").fit(
                sub_train
            )
            train_parts.append(pca.transform(sub_train))
            query_parts.append(pca.transform(sub_query))
        return np.column_stack(train_parts), np.column_stack(query_parts)

    if mode != "raw":
        factors_train, factors_query = components(train_x, query_x, mode == "group_pca")
        if mode == "pca":
            train_x, query_x = factors_train, factors_query
        else:
            train_x = np.column_stack([train_x, factors_train])
            query_x = np.column_stack([query_x, factors_query])

    result = np.empty((len(origins), len(HORIZONS)))
    for j, h in enumerate(HORIZONS):
        rows = np.arange(0, boundary - h)
        label = np.log(y[rows + h] / y[rows])
        model = Ridge(alpha=0.15 * len(rows), fit_intercept=True)
        model.fit(train_x[: len(rows)], label)
        result[:, j] = y[origins] * np.exp(model.predict(query_x))
    assert np.isfinite(result).all()
    return result


def variants(dataset):
    strict = feature_panels(dataset)["strict"]
    pilot = specific_features(feature_panels(dataset), "pilot")
    base = pilot.assign(current_usd_rub=dataset.usd_rub)
    output = {"pilot": base}
    for group in ("own", "fx", "markets", "calendar", "rates", "news"):
        extra = strict.loc[:, strict.columns.str.startswith(group + "__")]
        extra = extra.loc[:, ~extra.columns.isin(base.columns)]
        if not extra.empty:
            output["pilot_plus_" + group] = pd.concat([base, extra], axis=1)
    output["all_strict"] = pd.concat(
        [base, strict.loc[:, ~strict.columns.isin(base.columns)]], axis=1
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/usd_rub_2011_2026.csv.gz")
    parser.add_argument("--previous", default="experiments/staged_ridge_h1_h3_20261119/selection.json")
    parser.add_argument("--output", default="experiments/staged_features_pca_20261119")
    args = parser.parse_args()
    previous = json.loads(Path(args.previous).read_text())
    assert previous["h1_target"] == "log_return"
    alpha = previous["h2_alpha"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = validation_safe_dataset(args.dataset)
    dates = data.index
    y = data.usd_rub.to_numpy(dtype=float)
    folds, (test_boundary, test_origins) = make_folds(dates, "2025-01-01")
    panels = variants(data)
    validation = []
    for name, panel in panels.items():
        x = panel.to_numpy(dtype=float)
        forecasts = [predict(x, panel.columns, y, boundary, origins, "raw")
                     for boundary, origins in folds]
        validation.append({"stage": "H4", "variant": name, "columns": len(panel.columns),
                           "cv_relative_rmse": fold_score(forecasts, folds, y, alpha)})
    winner4 = min(validation, key=lambda item: item["cv_relative_rmse"])
    panel = panels[winner4["variant"]]
    x = panel.to_numpy(dtype=float)
    for mode in ("pca", "raw_plus_pca", "group_pca"):
        forecasts = [predict(x, panel.columns, y, boundary, origins, mode)
                     for boundary, origins in folds]
        validation.append({"stage": "H5", "variant": mode, "columns": len(panel.columns),
                           "cv_relative_rmse": fold_score(forecasts, folds, y, alpha)})
    h5 = [{"stage": "H5", "variant": "raw", "columns": len(panel.columns),
           "cv_relative_rmse": winner4["cv_relative_rmse"]},
          *[row for row in validation if row["stage"] == "H5"]]
    winner5 = min(h5, key=lambda item: item["cv_relative_rmse"])
    pd.DataFrame(validation).to_csv(output / "validation.csv", index=False)
    yearly = []
    for label, selected_panel, mode in (
        ("pilot", panels["pilot"], "raw"),
        ("h4_features", panel, "raw"),
        ("h5_pca", panel, winner5["variant"]),
    ):
        selected_x = selected_panel.to_numpy(dtype=float)
        for year, (boundary, origins) in zip((2021, 2022, 2023, 2024), folds, strict=True):
            raw = predict(selected_x, selected_panel.columns, y, boundary, origins, mode)
            last = y[origins, None]
            forecast = last + alpha * (raw - last)
            actual = y[origins[:, None] + HORIZONS]
            ratio = np.sqrt(np.mean((forecast - actual) ** 2, axis=0)) / np.sqrt(
                np.mean((last - actual) ** 2, axis=0)
            )
            yearly.append({"year": year, "model": label,
                           "relative_rmse": float(ratio.mean())})
    pd.DataFrame(yearly).to_csv(output / "validation_by_year.csv", index=False)
    selection = {"h4_selected": winner4, "h5_selected": winner5,
                 "target": "log_return", "shrink_alpha": alpha,
                 "feature_screening_end_exclusive": "2021-01-01",
                 "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                 "test_first": str(dates[test_origins[0]].date()),
                 "test_last": str(dates[test_origins[-1]].date())}
    (output / "selection.json").write_text(json.dumps(selection, indent=2))

    # Evaluate the two frozen validation decisions once on the old holdout.
    predictions = {}
    for label, selected_panel, mode in (
        ("h4_features", panels[winner4["variant"]], "raw"),
        ("h5_pca", panel, winner5["variant"]),
    ):
        raw = predict(selected_panel.to_numpy(dtype=float), selected_panel.columns, y,
                      test_boundary, test_origins, mode)
        last = y[test_origins, None]
        predictions[label] = last + alpha * (raw - last)
    metrics = [row for name, forecast in predictions.items()
               for row in evaluate(name, forecast, test_origins, y)]
    pd.DataFrame(metrics).to_csv(output / "test_metrics.csv", index=False)
    pd.concat(
        [pd.DataFrame({"model": name, "origin": dates[test_origins], "horizon": h,
                       "last": y[test_origins], "actual": y[test_origins + h],
                       "prediction": forecast[:, j]})
         for name, forecast in predictions.items() for j, h in enumerate(HORIZONS)],
        ignore_index=True,
    ).to_csv(output / "test_predictions.csv", index=False)
    normalize, keep = fitted_features(x, np.arange(0, test_boundary - 1))
    importance = []
    for h in HORIZONS:
        rows = np.arange(0, test_boundary - h)
        label = np.log(y[rows + h] / y[rows])
        model = Ridge(alpha=0.15 * len(rows), fit_intercept=True).fit(normalize(x[rows]), label)
        importance.extend({"horizon": h, "feature": name, "standardized_coefficient": value}
                          for name, value in zip(np.array(panel.columns)[keep], model.coef_,
                                                 strict=True))
    pd.DataFrame(importance).to_csv(output / "ridge_coefficients.csv", index=False)
    print(json.dumps(selection, indent=2))
    print(pd.DataFrame(metrics).groupby("model").relative_rmse.mean())


if __name__ == "__main__":
    main()
