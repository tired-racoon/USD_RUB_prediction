"""Combine the frozen baseline and staged experiments without refitting models."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from usd_rub.backtest import dm_test
from usd_rub.ml_dataset import read_dataset


def metrics(frame, rw):
    rows = []
    for (name, h), part in frame.groupby(["model", "horizon"]):
        reference = rw.loc[pd.MultiIndex.from_frame(part[["origin", "horizon"]])]
        error = part.prediction.to_numpy() - part.actual.to_numpy()
        rw_error = reference.prediction.to_numpy() - reference.actual.to_numpy()
        last = part["last"].to_numpy()
        actual = part.actual.to_numpy()
        prediction = part.prediction.to_numpy()
        moved = actual != last
        rmse = float(np.sqrt(np.mean(error**2)))
        rw_rmse = float(np.sqrt(np.mean(rw_error**2)))
        dm, p = dm_test(error**2, rw_error**2, lag=h - 1)
        rows.append({
            "model": name, "horizon": h, "n": len(part),
            "mae": float(np.mean(np.abs(error))), "rmse": rmse,
            "relative_rmse": rmse / rw_rmse,
            "rmse_reduction_pct": 100 * (1 - rmse / rw_rmse),
            "directional_accuracy": float(np.mean(np.sign(prediction - last)
                                                  == np.sign(actual - last))),
            "directional_accuracy_changed": float(np.mean(
                np.sign(prediction[moved] - last[moved])
                == np.sign(actual[moved] - last[moved]))),
            "dm_pvalue_vs_rw": p, "dm_stat_vs_rw": dm,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="experiments/baseline_reproduction_20261119")
    parser.add_argument("--h1h3", default="experiments/staged_ridge_h1_h3_20261119")
    parser.add_argument("--h4h5", default="experiments/staged_features_pca_20261119")
    parser.add_argument("--dataset", default="datasets/usd_rub_2011_2026.csv.gz")
    parser.add_argument("--output", default="experiments/research_summary_20261119")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(Path(args.baseline) / "predictions.csv", parse_dates=["origin"])
    baseline = baseline.loc[baseline.model.isin([
        "rw", "ensemble", "ridge_pilot_full_0.15_direct",
        "histboost_market_1460_3_direct",
    ]), ["model", "origin", "horizon", "last", "actual", "prediction"]]
    baseline["model"] = baseline.model.replace({
        "ridge_pilot_full_0.15_direct": "frozen_ridge",
        "histboost_market_1460_3_direct": "frozen_histboost",
        "ensemble": "frozen_ensemble",
    })
    stage1 = pd.read_csv(Path(args.h1h3) / "test_predictions.csv", parse_dates=["origin"])
    stage1 = stage1.loc[stage1.model.isin([
        "h1_level", "h1_difference", "h1_log_return", "h2_shrink",
    ])]
    stage2 = pd.read_csv(Path(args.h4h5) / "test_predictions.csv", parse_dates=["origin"])
    combined = pd.concat([baseline, stage1, stage2], ignore_index=True)
    assert not combined.duplicated(["model", "origin", "horizon"]).any()
    rw = combined.loc[combined.model.eq("rw")].set_index(["origin", "horizon"])
    summary = metrics(combined, rw)
    summary.to_csv(output / "test_metrics.csv", index=False)
    summary.groupby("model", as_index=False).agg(
        mean_relative_rmse=("relative_rmse", "mean"),
        mean_rmse_reduction_pct=("rmse_reduction_pct", "mean"),
        mean_directional_accuracy=("directional_accuracy", "mean"),
    ).sort_values("mean_relative_rmse").to_csv(output / "test_summary.csv", index=False)

    data = read_dataset(args.dataset)
    vol = data["own__vol_21"]
    cutoff = float(vol.loc[data.index < "2025-01-01"].quantile(0.75))
    origin_vol = combined.origin.map(vol)
    periods = {
        "2025": combined.origin.dt.year.eq(2025),
        "2026": combined.origin.dt.year.eq(2026),
        "lower_volatility": origin_vol.lt(cutoff),
        "higher_volatility": origin_vol.ge(cutoff),
    }
    period_rows = []
    for name, mask in periods.items():
        part = combined.loc[mask]
        reference = part.loc[part.model.eq("rw")].set_index(["origin", "horizon"])
        period_rows.append(metrics(part, reference).assign(period=name, vol_threshold=cutoff))
    pd.concat(period_rows, ignore_index=True).to_csv(output / "test_by_period.csv", index=False)

    for h in (1, 7):
        part = combined.loc[combined.horizon.eq(h) & combined.model.isin([
            "rw", "frozen_ensemble", "h2_shrink", "h4_features", "h5_pca",
        ])]
        pivot = part.pivot(index="origin", columns="model", values="prediction")
        actual = part.loc[part.model.eq("rw")].set_index("origin").actual
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(actual.index, actual, label="actual", color="black", lw=1.2)
        for name in ("rw", "frozen_ensemble", "h4_features"):
            ax.plot(pivot.index, pivot[name], label=name, lw=0.9, alpha=0.85)
        ax.set(title=f"USD/RUB, h={h} calendar day(s)", ylabel="RUB per USD")
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(output / f"forecast_h{h}.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(13, 4))
        for name in ("rw", "frozen_ensemble", "h2_shrink", "h4_features", "h5_pca"):
            ax.plot(pivot.index, np.cumsum((pivot[name] - actual) ** 2), label=name, lw=1)
        ax.set(title=f"Cumulative squared forecast error, h={h}", ylabel="RUB²")
        ax.legend(ncol=3)
        fig.tight_layout()
        fig.savefig(output / f"cumulative_squared_error_h{h}.png", dpi=140)
        plt.close(fig)
    print("Volatility cutoff (train 75th percentile):", cutoff)
    print(pd.read_csv(output / "test_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
