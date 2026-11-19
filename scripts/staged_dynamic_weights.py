"""H7: static and causal error/volatility/regime weights for RW, Ridge, HGB."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from staged_ridge_h1_h3 import HORIZONS, evaluate, fold_score, make_folds, validation_safe_dataset


NAMES = ("rw", "ridge", "histboost")


def frames(path, kind, alpha):
    raw = pd.read_csv(Path(path) / kind, parse_dates=["origin"])
    raw = raw.loc[raw.model.isin(NAMES[1:])]
    rw = raw.loc[raw.model.eq("ridge"), ["origin", "horizon", "last", "actual"]].copy()
    rw["model"] = "rw"
    rw["prediction"] = rw["last"]
    raw["prediction"] = raw["last"] + alpha * (raw["prediction"] - raw["last"])
    return pd.concat([rw, raw], ignore_index=True)


def cube(frame):
    piv = frame.pivot(index=["origin", "horizon"], columns="model", values="prediction")
    piv = piv.loc[:, NAMES].sort_index()
    anchor = frame.loc[frame.model.eq("rw")].set_index(["origin", "horizon"]).loc[piv.index]
    dates = pd.DatetimeIndex(piv.index.get_level_values("origin"))
    return piv.to_numpy(), anchor.actual.to_numpy(), dates, np.array(piv.index.get_level_values("horizon"))


def regime(dates):
    return np.select(
        [dates < pd.Timestamp("2022-02-24"), dates < pd.Timestamp("2024-06-13")],
        [0, 1], default=2,
    )


def dynamic_forecast(frame, base_weights, lookback, conditioning, vol, threshold):
    forecasts, actual, dates, horizons = cube(frame)
    result = np.empty(len(actual))
    vol_values = pd.Series(dates).map(vol).to_numpy(dtype=float)
    high = vol_values >= threshold
    regimes = regime(dates)
    for h in HORIZONS:
        positions = np.flatnonzero(horizons == h)
        days = dates[positions].to_numpy().astype("datetime64[D]").astype(int)
        loss = (forecasts[positions] - actual[positions, None]) ** 2
        for i, position in enumerate(positions):
            # A prediction at t becomes usable only when t+h < this origin.
            age = days[i] - days[:i]
            eligible = (age > h) & (age <= lookback)
            if conditioning in ("volatility", "regime"):
                same_vol = high[positions[:i]] == high[position]
                if np.count_nonzero(eligible & same_vol) >= 30:
                    eligible &= same_vol
            if conditioning == "regime":
                same_regime = regimes[positions[:i]] == regimes[position]
                if np.count_nonzero(eligible & same_regime) >= 30:
                    eligible &= same_regime
            if np.count_nonzero(eligible) < 30:
                weights = base_weights
            else:
                mse = loss[:i][eligible].mean(axis=0)
                score = 1 / np.maximum(mse, 1e-8)
                weights = score / score.sum()
            result[position] = forecasts[position] @ weights
    return result


def validation_loss(frame, prediction, folds, y):
    index = frame.loc[frame.model.eq("rw"), ["origin", "horizon"]].copy()
    index["prediction"] = prediction
    wide = index.pivot(index="origin", columns="horizon", values="prediction")
    estimates = [wide.loc[frame_dates].to_numpy() for _, origins in folds
                 for frame_dates in [pd.DatetimeIndex(y.index[origins])]]
    return fold_score(estimates, folds, y.to_numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h6", default="experiments/staged_architectures_20261119")
    parser.add_argument("--dataset", default="datasets/usd_rub_2011_2026.csv.gz")
    parser.add_argument("--output", default="experiments/staged_dynamic_weights_20261119")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    alpha = json.loads((Path(args.h6) / "selection.json").read_text())["alpha"]
    data = validation_safe_dataset(args.dataset)
    y = data.usd_rub
    folds, (_, test_origins) = make_folds(y.index, "2025-01-01")
    vol = data["own__vol_21"]
    threshold = float(vol.loc[vol.index < "2021-01-01"].quantile(0.75))
    validation = frames(args.h6, "validation_predictions.csv", alpha)
    test = frames(args.h6, "test_components.csv", alpha)
    validation = validation.sort_values(["origin", "horizon", "model"])
    test = test.sort_values(["origin", "horizon", "model"])
    val_forecasts, _, _, _ = cube(validation)
    rows = []
    for ridge_weight in np.linspace(0, 1, 5):
        for boost_weight in np.linspace(0, 1 - ridge_weight, int(round((1 - ridge_weight) * 4)) + 1):
            weights = np.array([1 - ridge_weight - boost_weight, ridge_weight, boost_weight])
            score = validation_loss(validation, val_forecasts @ weights, folds, y)
            rows.append({"kind": "static", "lookback": 0, "conditioning": "none",
                         "dynamic_fraction": 0, "weights": weights.tolist(),
                         "cv_relative_rmse": score})
    static = min(rows, key=lambda row: row["cv_relative_rmse"])
    static_weights = np.array(static["weights"])
    for lookback in (45, 90, 180):
        for conditioning in ("error", "volatility", "regime"):
            dynamic = dynamic_forecast(validation, static_weights, lookback, conditioning,
                                       vol, threshold)
            for fraction in (0.25, 0.5, 0.75, 1.0):
                combined = (1 - fraction) * (val_forecasts @ static_weights) + fraction * dynamic
                score = validation_loss(validation, combined, folds, y)
                rows.append({"kind": "dynamic", "lookback": lookback,
                             "conditioning": conditioning, "dynamic_fraction": fraction,
                             "weights": static_weights.tolist(), "cv_relative_rmse": score})
    ranking = pd.DataFrame(rows)
    ranking["complexity_rank"] = ranking.kind.eq("dynamic").astype(int)
    ranking = ranking.sort_values(["cv_relative_rmse", "complexity_rank"])
    ranking.to_csv(output / "validation.csv", index=False)
    winner = ranking.iloc[0].to_dict()
    selection = {"winner": winner, "static_weights": static_weights.tolist(),
                 "static_cv_relative_rmse": static["cv_relative_rmse"],
                 "volatility_threshold_pre_2021": threshold,
                 "availability_rule": "past forecast used only if origin + horizon < current origin"}
    (output / "selection.json").write_text(json.dumps(selection, indent=2))

    # The historical test's observed outcomes enter only once they mature.
    complete = pd.concat([validation, test], ignore_index=True).sort_values(
        ["origin", "horizon", "model"]
    )
    all_forecasts, _, all_dates, all_horizons = cube(complete)
    if winner["kind"] == "static":
        all_pred = all_forecasts @ np.array(winner["weights"])
    else:
        dynamic = dynamic_forecast(complete, static_weights, int(winner["lookback"]),
                                   winner["conditioning"], vol, threshold)
        fraction = winner["dynamic_fraction"]
        all_pred = (1 - fraction) * (all_forecasts @ static_weights) + fraction * dynamic
    origin_dates = y.index[test_origins]
    mask = all_dates.isin(origin_dates)
    result = pd.DataFrame({"origin": all_dates[mask], "horizon": all_horizons[mask],
                           "prediction": all_pred[mask]}).pivot(
        index="origin", columns="horizon", values="prediction"
    ).loc[origin_dates].to_numpy()
    pd.DataFrame(evaluate("h7_selected", result, test_origins, y.to_numpy())).to_csv(
        output / "test_metrics.csv", index=False
    )
    print(ranking.head(12).to_string(index=False))
    print(selection)


if __name__ == "__main__":
    main()
