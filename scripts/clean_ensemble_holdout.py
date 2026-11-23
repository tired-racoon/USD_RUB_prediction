"""Сравнить ансамбли без весов, опубликованных после начала теста."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from forecast_extensions import annual_folds, load_inputs
from forecast_stacking import panel_index
from fx_candles_blends import GROUPS
from fx_candles_holdout import experiments, prediction_for
from llm_news_holdout import bootstrap
from pre2025_foundation import PINS
from shap_holdout import fuse_importance, rank, score_row, tree_importance
from theory_holdout import PRIMARY, references
from threadpoolctl import threadpool_limits

from usd_rub.ensemble_forecast import GAMMA_WEIGHTS
from usd_rub.fixing_calendar import HORIZONS
from usd_rub.market_correction import fit_corrections, predict_levels
from usd_rub.ml_dataset import feature_panels, read_dataset
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints
from usd_rub.theory_features import FUNDAMENTAL_SIGNS, MACRO_CORRECTION_FEATURES

CLEAN_SPLIT = pd.Timestamp("2025-07-01")
WEIGHTS = (0.05, 0.15, 0.30)
CORRECTIONS = {
    "full": ("combined", None),
    "shap8": ("shap", 8),
    "gain8": ("gain", 8),
    "combined8": ("combined", 8),
    "combined14": ("combined", 14),
}
BASIS = (
    "usd_spot__basis",
    "usd_perp__basis_centered",
    "usd_perp__since_afternoon",
    "cny_spot__since_afternoon",
    "cny_perp__since_afternoon",
    "cross__perp_basis",
)


def archived_panel(path, dates):
    """Проверить даты и горизонты перед агрегацией."""
    frame = pd.read_csv(path, parse_dates=["origin"], float_precision="round_trip")
    frame = frame.set_index(["origin", "horizon"])
    pd.testing.assert_index_equal(frame.index, panel_index(dates))
    return frame


def rank_correction(x, actual, baseline, last):
    """Ранжировать макропризнаки на известных остатках вневыборочных прогнозов."""
    border = len(x) - 90
    y = 100 * np.log(actual / baseline)
    weights = (last[:border] / np.median(last[:border])) ** 2
    scores = [
        tree_importance(x.iloc[:border], y[:border, h], x.iloc[border:], weights)
        for h in (0, len(HORIZONS) - 1)
    ]
    return fuse_importance(scores)


def corrections(panel, calibration, test, parts, base, future):
    """Подобрать блок XGBoost и силу поправки на годах до 2025."""
    cal_x, test_x = panel.loc[calibration.dates], panel.loc[test.dates]
    designs = {}
    columns_for_test = {}
    for train, check, final in [(*fold, False) for fold in parts] + [
        (np.ones(len(base), dtype=bool), slice(None), True),
    ]:
        train_x = cal_x.loc[train]
        ranking = rank_correction(
            train_x, calibration.actual[train], base[train], calibration.last[train]
        )
        query, reference, features = (
            (test, future, test_x)
            if final
            else (
                calibration,
                base[check],
                cal_x.loc[check],
            )
        )
        for label, (method, limit) in CORRECTIONS.items():
            if limit is None:
                keys = list(panel)
            else:
                keys = (
                    ranking[method]
                    .sort_values(ascending=False, kind="stable")
                    .head(limit)
                    .index.tolist()
                )
            signs = {key: FUNDAMENTAL_SIGNS[key] for key in keys if key in FUNDAMENTAL_SIGNS}
            models = fit_corrections(
                train_x[keys],
                calibration.actual[train],
                base[train],
                calibration.last[train],
                signs,
            )
            point = predict_levels(models, features[keys], reference, query.last[check])
            for weight in (0.25, 0.50, 1.0):
                name = f"{label}/{weight:g}"
                adjusted = query.project(reference + weight * (point - reference), check)
                designs.setdefault(name, []).append(adjusted)
            if final:
                columns_for_test[label] = keys
    scores = {}
    for name, prediction in designs.items():
        history = [
            score_row(value, calibration.actual[check], calibration.last[check])
            for (_, check), value in zip(parts, prediction[:-1], strict=True)
        ]
        scores[name] = {
            "cv_rmse7_to_rw": float(np.mean([row["rmse7_to_rw"] for row in history])),
            "cv_rmse_to_rw": float(np.mean([row["rmse_to_rw"] for row in history])),
        }
    return {name: values[-1] for name, values in designs.items()}, scores, columns_for_test


def spreads(dataset, intraday, calibration, test, base, reference):
    """Пересчитать полный и сокращённый блок спредов без t0-gamma."""
    market = feature_panels(dataset)["strict"][["fx__cny_basis", "own__return_1", "own__vol_21"]]
    pd.testing.assert_index_equal(market.index, intraday.index)
    market = market.join(intraday[list(BASIS)])
    dates = calibration.dates.append(test.dates)
    levels = dataset.usd_rub.to_numpy()
    cutoff = dataset.index.get_loc(test.dates[0])
    shortlist = rank(market, levels, cutoff).head(5).index
    actual = np.vstack((calibration.actual, test.actual))
    last = np.r_[calibration.last, test.last]
    fixing = np.vstack((calibration.fixes, test.fixes))
    configs = {spec.name: spec for spec in experiments()}
    rows = np.arange(len(calibration.dates), len(dates))
    mask = test.fixes.cumsum(axis=1)
    output = {}
    for label, columns in (("full", market.columns), ("compact", shortlist)):
        x = market.loc[dates, columns].to_numpy(dtype=float)
        components = [
            prediction_for(configs[name], x, dates, actual, last, fixing, base, rows)
            for name in GROUPS["basis"]
        ]
        candidate = np.mean(components, axis=0)
        output[f"classical_spreads/{label}"] = reference + 0.5 * (candidate - reference) * (
            (mask > 0) & (mask <= 3)
        )
    return output, shortlist.tolist()


def run(args):
    """Сохранить чистое сравнение и правила выбора компонентов."""
    dataset, calibration, test, _, paths = load_inputs(args)
    inputs = [*paths, args.calendar, Path(__file__)]
    panel = feature_panels(dataset)["extended"][list(MACRO_CORRECTION_FEATURES)]
    previous = args.pruning / "run.json"
    pruned_path = args.pruning / "predictions.csv.gz"
    config = json.loads(previous.read_text())
    forecasts = archived_panel(pruned_path, calibration.dates.append(test.dates))
    inputs += [previous, pruned_path]
    _, folds = annual_folds(calibration)
    n = len(calibration.dates)

    def matrix(name):
        """Восстановить один прогноз на валидации и тесте."""
        raw = forecasts[name].to_numpy().reshape(-1, len(HORIZONS))
        return raw[:n], raw[n:]

    clean = "production_classical"
    compact = config["selected_market_mixture"]
    anchors = {"classical": matrix(clean), "classical_compact": matrix(compact)}
    model_names = {}
    for model, method in config["selected_by_model"].items():
        model_names[f"{model}_compact"] = f"{model}/{method}"
    holdout = {name: point for name, (_, point) in anchors.items()}
    holdout.update({key: matrix(name)[1] for key, name in model_names.items()})
    holdout["random_walk"] = matrix("random_walk")[1]
    holdout["legacy_t0_gamma"] = references(test)[PRIMARY]
    fx_file = args.legacy / "predictions.csv.gz"
    fx = pd.read_csv(
        fx_file, parse_dates=["origin"], usecols=["origin", "horizon", "incumbent_macro"]
    )
    fx = fx.loc[fx.origin.isin(test.dates)].set_index(["origin", "horizon"])
    pd.testing.assert_index_equal(fx.index, panel_index(test.dates))
    holdout["legacy_t0_gamma_macro"] = fx.incumbent_macro.to_numpy().reshape(test.actual.shape)
    inputs.append(fx_file)
    blends_file = args.legacy_blends / "predictions.csv.gz"
    blends = pd.read_csv(
        blends_file, parse_dates=["origin"], usecols=["origin", "horizon", "basis/3/half"]
    )
    blends = blends.loc[blends.origin.isin(test.dates)].set_index(["origin", "horizon"])
    pd.testing.assert_index_equal(blends.index, panel_index(test.dates))
    holdout["legacy_t0_gamma_spreads"] = (
        blends["basis/3/half"].to_numpy().reshape(test.actual.shape)
    )
    inputs.append(blends_file)

    diagnostics = {}
    finalists = {}
    for label, (cal, point) in anchors.items():
        raw, selected, columns = corrections(panel, calibration, test, folds, cal, point)
        keys = [*raw, "none"]
        z = [
            score_row(cal[check], calibration.actual[check], calibration.last[check])
            for _, check in folds
        ]
        selected["none"] = {
            "cv_rmse7_to_rw": float(np.mean([row["rmse7_to_rw"] for row in z])),
            "cv_rmse_to_rw": float(np.mean([row["rmse_to_rw"] for row in z])),
        }
        finalists[label] = min(
            keys,
            key=lambda key: (selected[key]["cv_rmse7_to_rw"], selected[key]["cv_rmse_to_rw"]),
        )
        diagnostics.update({f"{label}_macro/{key}": value for key, value in selected.items()})
        diagnostics[label] = selected["none"]
        holdout.update({f"{label}_macro/{key}": value for key, value in raw.items()})
        print(
            label,
            finalists[label],
            "retained",
            columns.get(finalists[label].split("/")[0], []),
            flush=True,
        )
        write_csv(
            pd.DataFrame(
                [(design, name) for design, names in columns.items() for name in names],
                columns=["variant", "feature"],
            ),
            args.output / f"{label}_macro_features.csv",
        )

    anchor_label = "classical"
    macro_key = finalists[anchor_label]
    macro_name = anchor_label if macro_key == "none" else f"{anchor_label}_macro/{macro_key}"
    spread_results, spread_columns = spreads(
        dataset,
        read_dataset(args.intraday),
        calibration,
        test,
        np.vstack(anchors[anchor_label]),
        holdout[macro_name],
    )
    holdout.update(spread_results)
    inputs.append(args.intraday)

    release_selection = {}
    prior = (test.dates + pd.Timedelta(days=int(HORIZONS[-1]))) < CLEAN_SPLIT
    final = test.dates >= CLEAN_SPLIT
    for label in anchors:
        release_selection[label] = [label]
    for model, (_, revision, release, expected) in PINS.items():
        metadata_path = args.foundation / f"{model}.json"
        path = args.foundation / f"{model}.csv.gz"
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata["revision"] != revision
            or metadata["revision_date"] != release
            or metadata["sha256"] != expected
            or pd.Timestamp(release) >= test.dates[0]
            or metadata["predictions"]["sha256"] != input_fingerprints([path])[0]["sha256"]
        ):
            raise ValueError(f"Не совпала зафиксированная ревизия или прогноз {model}")
        frame = archived_panel(path, test.dates)
        inputs += [metadata_path, path]
        for name, points in frame.items():
            raw = points.to_numpy().reshape(test.actual.shape)
            holdout[name] = test.project(raw)
            for label, (_, point) in anchors.items():
                for weight in WEIGHTS:
                    variant = f"{label}+{name}/{weight:g}"
                    prediction = test.project(point + weight * (GAMMA_WEIGHTS > 0) * (raw - point))
                    holdout[variant] = prediction
                    release_selection[label].append(variant)
    early = {
        name: score_row(prediction[prior], test.actual[prior], test.last[prior])
        for name, prediction in holdout.items()
        if not name.startswith("legacy_")
    }
    chosen = {
        label: min(names, key=lambda name: (early[name]["rmse7_to_rw"], early[name]["rmse_to_rw"]))
        for label, names in release_selection.items()
    }
    results = []
    for name, point in holdout.items():
        record = {
            "model": name,
            **score_row(point, test.actual, test.last),
            **diagnostics.get(name, {}),
        }
        for suffix, mask in (("h1_2025", prior), ("from_july_2025", final)):
            out = score_row(point[mask], test.actual[mask], test.last[mask])
            for key in ("rmse_1d", "rmse_7d", "rmse_to_rw"):
                record[f"{key}_{suffix}"] = out[key]
        results.append(record)
    reports = pd.DataFrame(results).sort_values("rmse_7d")
    relevant = {*anchors, *chosen.values(), *spread_results}
    for label, key in finalists.items():
        relevant.add(label if key == "none" else f"{label}_macro/{key}")
    write_csv(reports, args.output / "summary.csv")
    index = panel_index(test.dates)
    write_csv(
        pd.DataFrame(
            {name: np.ravel(point) for name, point in holdout.items()}, index=index
        ).reset_index(),
        args.output / "predictions.csv.gz",
    )
    delta = {
        name: bootstrap(holdout["classical"], holdout[name], test.actual, test.last)
        for name in relevant
        if name != "classical"
    }
    manifest = {
        "inputs": input_fingerprints(inputs),
        "selected_macro_before_2025": finalists,
        "selected_foundation_through_h1_2025": chosen,
        "compact_spread_features": spread_columns,
        "foundation_select_origins": int(prior.sum()),
        "foundation_evaluate_origins": int(final.sum()),
        "calibrated_foundation_holdout_start": str(CLEAN_SPLIT.date()),
        "gain_vs_classical_95_ci_pp": delta,
    }
    (args.output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subset = reports.loc[reports.model.isin(relevant | {"legacy_t0_gamma_macro"})]
    print(
        subset[["model", "rmse_1d", "rmse_7d", "rmse_to_rw", "rmse_7d_from_july_2025"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/reports/theory_holdout/dataset.csv.gz")
    )
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument("--archive", type=Path, default=Path("data/reports/forecast_stacking"))
    parser.add_argument("--calendar", type=Path, default=Path("config/cbr_fixing_calendar.json"))
    parser.add_argument("--pruning", type=Path, default=Path("data/reports/shap_holdout"))
    parser.add_argument("--foundation", type=Path, default=Path("data/reports/pre2025_foundation"))
    parser.add_argument("--legacy", type=Path, default=Path("data/reports/fx_candles_holdout"))
    parser.add_argument(
        "--legacy-blends", type=Path, default=Path("data/reports/fx_candles_blends")
    )
    parser.add_argument(
        "--intraday", type=Path, default=Path("datasets/usd_rub_2011_2026_fx_candles.csv.gz")
    )
    parser.add_argument("--output", type=Path, default=Path("data/reports/clean_ensemble_holdout"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        run(args)
