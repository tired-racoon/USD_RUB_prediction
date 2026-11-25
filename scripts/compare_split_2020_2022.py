"""Freeze models after 2022; validate in 2023-24 and test in 2025-26."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from usd_rub.fixing_calendar import HORIZONS, apply_calendar, fixing_mask
from usd_rub.market_correction import corrected_forecast, fit_corrections
from usd_rub.ml_dataset import feature_panels, read_dataset
from usd_rub.ml_experiment import Candidate, fit_predict, relative_rmse, specific_features, to_levels
from usd_rub.theory_features import COMPACT_FUNDAMENTAL_SIGNS, COMPACT_MACRO_FEATURES
from usd_rub.yuan_experiment import feature_sets


START = pd.Timestamp("2020-01-01")
VALIDATION = pd.Timestamp("2023-01-01")
TEST = pd.Timestamp("2025-01-01")
LAST_ORIGIN = pd.Timestamp("2026-11-11")
MIN_HIT_RATE = 0.5
CLASSICAL = (
    (Candidate("ridge", "pilot", None, 0.15), 0.43838276164287915, 1.0),
    (Candidate("histboost", "market", 1460, 3), 0.3844272108930251, 0.75),
    (Candidate("extra_trees", "market", 1460, 3, pooled=True), 0.17719002746409557, 1.0),
)


def classical_features(panels, candidate, cutoff):
    features = specific_features(panels, candidate.setup)
    features = features.loc[:, ~features.columns.str.contains("__theory_", regex=False)]
    supported = features.iloc[:cutoff].notna().mean().ge(MIN_HIT_RATE)
    return features.loc[:, supported]


def indices(index):
    train_end = index.searchsorted(VALIDATION)
    valid = np.flatnonzero((index >= VALIDATION) & (index + pd.Timedelta(days=7) < TEST))
    test = np.flatnonzero((index >= TEST) & (index <= LAST_ORIGIN))
    return train_end, valid, test


def measures(name, period, predicted, actual, last):
    rmse = np.sqrt(np.mean((predicted - actual) ** 2, axis=0))
    mae = np.mean(np.abs(predicted - actual), axis=0)
    mape = np.mean(np.abs(predicted - actual) / actual, axis=0) * 100
    ratio = relative_rmse(predicted, actual, last)
    return {
        "model": name,
        "period": period,
        "n_origins": len(last),
        "rmse_1d": rmse[0],
        "rmse_7d": rmse[-1],
        "mae_1d": mae[0],
        "mae_7d": mae[-1],
        "mape_1d_pct": mape[0],
        "mape_7d_pct": mape[-1],
        "mape_mean_pct": mape.mean(),
        "rmse_to_rw_mean": ratio.mean(),
        "rmse_7d_to_rw": ratio[-1],
    }


def classical_prediction(panel, values, cutoff, origins, dates, calendar):
    pred = np.zeros((len(origins), len(HORIZONS)))
    for candidate, weight, shrink in CLASSICAL:
        features = classical_features(panel, candidate, cutoff)
        raw, _ = fit_predict(
            features.to_numpy(dtype=float), values, candidate, cutoff, origins,
            HORIZONS, min_hit_rate=MIN_HIT_RATE,
        )
        pred += weight * to_levels(raw, values[origins], shrink)
    changes = fixing_mask(dates, calendar, True)
    return apply_calendar(pred, values[origins], changes, pool=True), changes


def main_experiment(root):
    dataset = read_dataset(root / "datasets/usd_rub_2011_2026_ensemble.csv.gz")
    dataset = dataset.loc[START:]
    panel = feature_panels(dataset)
    values = dataset.usd_rub.to_numpy(dtype=float)
    cutoff, valid, test = indices(dataset.index)
    origins = np.r_[valid, test]
    dates = dataset.index[origins]
    calendar = json.loads((root / "config/cbr_fixing_calendar.json").read_text())
    base, changes = classical_prediction(panel, values, cutoff, origins, dates, calendar)

    # OOF anchors: each year is forecast by a classical model trained only on prior
    # years starting in 2020. The XGBoost correction never sees in-sample errors.
    oof_origins, oof_baselines = [], []
    for year in (2021, 2022, 2023, 2024):
        boundary = dataset.index.searchsorted(f"{year}-01-01")
        block = np.flatnonzero(
            (dataset.index >= pd.Timestamp(f"{year}-01-01"))
            & (dataset.index + pd.Timedelta(days=7) < pd.Timestamp(f"{year + 1}-01-01"))
        )
        forecast, _ = classical_prediction(
            panel, values, boundary, block, dataset.index[block], calendar
        )
        oof_origins.append(block)
        oof_baselines.append(forecast)
    features = panel["extended"].loc[dataset.index, list(COMPACT_MACRO_FEATURES)]
    oof = np.concatenate(oof_origins[:2])
    oof_base = np.vstack(oof_baselines[:2])
    models = fit_corrections(
        features.iloc[oof],
        values[oof[:, None] + HORIZONS[None, :]],
        oof_base,
        values[oof],
        COMPACT_FUNDAMENTAL_SIGNS,
    )
    versions = {"main_classical": base}
    query_features = features.iloc[origins]
    for fraction in (0.25, 0.5, 1.0):
        versions[f"main_xgb_{fraction:g}"] = corrected_forecast(
            models, query_features, base, values[origins], changes, fraction=fraction
        )
    n_valid = len(valid)
    actual = values[origins[:, None] + HORIZONS[None, :]]
    last = values[origins]
    rows = []
    for name, pred in versions.items():
        rows += [
            measures(name, "validation_2023_2024", pred[:n_valid], actual[:n_valid], last[:n_valid]),
            measures(name, "test_2025_2026", pred[n_valid:], actual[n_valid:], last[n_valid:]),
        ]
    rw = np.broadcast_to(last[:, None], actual.shape)
    rows += [
        measures("random_walk", "validation_2023_2024", rw[:n_valid], actual[:n_valid], last[:n_valid]),
        measures("random_walk", "test_2025_2026", rw[n_valid:], actual[n_valid:], last[n_valid:]),
    ]
    best = min(
        versions,
        key=lambda name: next(
            row["rmse_7d_to_rw"] for row in rows
            if row["model"] == name and row["period"] == "validation_2023_2024"
        ),
    )
    refit_cutoff = dataset.index.searchsorted(TEST)
    refit_base, refit_changes = classical_prediction(
        panel, values, refit_cutoff, test, dataset.index[test], calendar
    )
    refit_oof = np.concatenate(oof_origins)
    refit_models = fit_corrections(
        features.iloc[refit_oof],
        values[refit_oof[:, None] + HORIZONS[None, :]],
        np.vstack(oof_baselines),
        values[refit_oof],
        COMPACT_FUNDAMENTAL_SIGNS,
    )
    refit_versions = {"main_classical": refit_base}
    for fraction in (0.25, 0.5):
        refit_versions[f"main_xgb_{fraction:g}"] = corrected_forecast(
            refit_models, features.iloc[test], refit_base, values[test], refit_changes,
            fraction=fraction,
        )
    for name, pred in refit_versions.items():
        rows.append(measures(name, "test_refit_2025_2026", pred, actual[n_valid:], last[n_valid:]))
    return rows, {
        "main_selected_by_validation_rmse_7d": best,
        "main_xgb_oof_origins_initial": len(oof),
        "main_xgb_oof_origins_refit": len(refit_oof),
    }


def yuan_panel(root):
    data = pd.read_csv(
        root / "datasets/usd_rub_2011_2026_papers.csv.gz",
        usecols=[
            "date", "usd_rub", "fx__ecb_cross_vwap_basis",
            "fx__ecb_cross_close_basis", "global__ecb_usd_cny_return_1d",
            "global__ecb_usd_cny_age", "fx__cny_tom_vwap_age",
        ],
        index_col="date", parse_dates=["date"], float_precision="round_trip",
    ).loc["2017-01-01":]
    usd = data.usd_rub
    ecb = np.exp(data.global__ecb_usd_cny_return_1d.fillna(0).cumsum())
    cross_vwap = usd * np.exp(-data.fx__ecb_cross_vwap_basis)
    cross_close = usd * np.exp(-data.fx__ecb_cross_close_basis)
    # The ECB leg has an arbitrary fixed scale. Both Ridge targets and features
    # use relative changes, while the cross and basis retain their exact levels.
    panel = pd.DataFrame(index=data.index)
    panel["usd_rub"] = usd
    panel["ecb_usd_cny"] = ecb
    panel["cny_tom_vwap"] = cross_vwap / ecb
    panel["cny_tom_close"] = cross_close / ecb
    panel["ecb_usd_cny__age_days"] = data.global__ecb_usd_cny_age
    panel["cny_tom_vwap__age_days"] = data.fx__cny_tom_vwap_age
    panel["cross_vwap"] = cross_vwap
    panel["cross_close"] = cross_close
    panel["basis_factor"] = np.exp(data.fx__ecb_cross_vwap_basis)
    panel["rusfar_cny"] = np.nan
    panel["rusfar_cny__age_days"] = np.nan
    return panel


def yuan_experiment(root):
    panel = yuan_panel(root)
    matrices = feature_sets(panel)
    panel = panel.loc[START:]
    base = {name: array[len(matrices["base"][name]) - len(panel):]
            for name, array in matrices["base"].items()}
    values = panel.usd_rub.to_numpy(dtype=float)
    cutoff, valid, test = indices(panel.index)
    origins = np.r_[valid, test]
    last = values[origins]
    ridge = Candidate("ridge", "yuan_experiment", window=1460, complexity=0.15)
    direct, _ = fit_predict(base["basis_factor"], values, ridge, cutoff, origins, HORIZONS)
    versions = {
        "yuan_direct_ridge": to_levels(direct, last),
        "yuan_cross_close": np.broadcast_to(panel.cross_close.to_numpy()[origins, None], (len(origins), 7)),
        "yuan_cross_vwap": np.broadcast_to(panel.cross_vwap.to_numpy()[origins, None], (len(origins), 7)),
    }
    factors = {}
    for leg in ("cny_tom_vwap", "ecb_usd_cny", "basis_factor"):
        change, _ = fit_predict(
            base[leg], panel[leg].to_numpy(dtype=float), ridge, cutoff, origins, HORIZONS
        )
        factors[leg] = 1 + change
    versions["yuan_three_legs"] = (
        last[:, None] * factors["cny_tom_vwap"] * factors["ecb_usd_cny"] * factors["basis_factor"]
    )
    actual = values[origins[:, None] + HORIZONS[None, :]]
    n_valid = len(valid)
    rows = []
    shrink = {}
    for name, pred in versions.items():
        if name in ("yuan_direct_ridge", "yuan_three_legs"):
            candidates = (0, 0.25, 0.5, 0.75, 1)
            fraction = min(
                candidates,
                key=lambda f: relative_rmse(
                    last[:n_valid, None] + f * (pred[:n_valid] - last[:n_valid, None]),
                    actual[:n_valid], last[:n_valid],
                ).mean(),
            )
            shrink[name] = fraction
            pred = last[:, None] + fraction * (pred - last[:, None])
        rows += [
            measures(name, "validation_2023_2024", pred[:n_valid], actual[:n_valid], last[:n_valid]),
            measures(name, "test_2025_2026", pred[n_valid:], actual[n_valid:], last[n_valid:]),
        ]
    refit_cutoff = panel.index.searchsorted(TEST)
    refit_direct, _ = fit_predict(
        base["basis_factor"], values, ridge, refit_cutoff, test, HORIZONS
    )
    refit_versions = {"yuan_direct_ridge": to_levels(refit_direct, values[test])}
    refit_factors = {}
    for leg in ("cny_tom_vwap", "ecb_usd_cny", "basis_factor"):
        change, _ = fit_predict(
            base[leg], panel[leg].to_numpy(dtype=float), ridge, refit_cutoff, test, HORIZONS
        )
        refit_factors[leg] = 1 + change
    refit_versions["yuan_three_legs"] = (
        values[test, None] * refit_factors["cny_tom_vwap"]
        * refit_factors["ecb_usd_cny"] * refit_factors["basis_factor"]
    )
    for name, pred in refit_versions.items():
        pred = values[test, None] + shrink[name] * (pred - values[test, None])
        rows.append(measures(name, "test_refit_2025_2026", pred, actual[n_valid:], last[n_valid:]))
    return rows, {"yuan_validation_shrink": shrink}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/reports/split_2020_2022"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with threadpool_limits(limits=2):
        main_rows, main_meta = main_experiment(root)
        yuan_rows, yuan_meta = yuan_experiment(root)
    rows = main_rows + yuan_rows
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "metrics.csv", index=False, float_format="%.9f")
    metadata = {
        "train_dates": ["2020-01-01", "2022-12-31"],
        "validation_dates": ["2023-01-01", "2024-12-24"],
        "test_dates": ["2025-01-01", "2026-11-11"],
        "test_2025_2026": "models frozen after 2022",
        "test_refit_2025_2026": "models refit on 2020-2024 using validation-selected settings",
        "correction_features": list(COMPACT_MACRO_FEATURES),
        "yuan_basis_reconstructed_from": "datasets/usd_rub_2011_2026_papers.csv.gz",
        **main_meta,
        **yuan_meta,
    }
    (args.output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
