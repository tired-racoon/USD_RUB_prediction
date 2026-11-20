"""Собрать семь моделей поправки из вневыборочных прогнозов 2021–2024 годов."""

import argparse
import gzip
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import xgboost
from forecast_extensions import load_inputs
from forecast_stacking import ARCHIVE_FILES
from theory_holdout import PRIMARY, references

from usd_rub.fixing_calendar import HORIZONS
from usd_rub.market_correction import (
    MACRO_BUNDLE,
    MARKET_BUNDLE,
    MARKET_FEATURES,
    XGB_PARAMS,
    fit_corrections,
)
from usd_rub.ml_dataset import feature_panels
from usd_rub.storage import input_fingerprints
from usd_rub.theory_features import FUNDAMENTAL_SIGNS, MACRO_CORRECTION_FEATURES


def export(args):
    """Обучить на калибровке и записать переносимый архив деревьев."""
    dataset, calibration, _, _, _ = load_inputs(args)
    columns = MACRO_CORRECTION_FEATURES if args.macro else MARKET_FEATURES
    signs = FUNDAMENTAL_SIGNS if args.macro else None
    features = feature_panels(dataset)["extended"].loc[calibration.dates, list(columns)]
    baseline = references(calibration)[PRIMARY]
    models = fit_corrections(features, calibration.actual, baseline, calibration.last, signs)
    bundle = {
        "format_version": 1,
        "xgboost": xgboost.__version__,
        "trained_before": json.loads(args.config.read_text())["test_start"],
        "calibration_origins": len(calibration.dates),
        "features": [*columns, "anchor_return"],
        "horizons": HORIZONS.tolist(),
        "parameters": XGB_PARAMS,
        "inputs": input_fingerprints([args.dataset, args.config, args.archive / ARCHIVE_FILES[0]]),
        "boosters": [json.loads(model.save_raw(raw_format="json")) for model in models],
    }
    if signs:
        bundle["monotone_constraints"] = signs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = gzip.compress(json.dumps(bundle, separators=(",", ":")).encode("utf-8"), mtime=0)
    with NamedTemporaryFile(dir=args.output.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(payload)
            file.close()
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"{args.output}: {len(models)} моделей, {len(calibration.dates)} дат")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro", action="store_true", help="Собрать модель со знаками макроблока")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument("--archive", type=Path, default=Path("data/reports/forecast_stacking"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.dataset = args.dataset or Path(
        "data/reports/theory_holdout/dataset.csv.gz"
        if args.macro
        else "datasets/usd_rub_2011_2026_full.csv.gz"
    )
    args.output = args.output or (MACRO_BUNDLE if args.macro else MARKET_BUNDLE)
    export(args)
