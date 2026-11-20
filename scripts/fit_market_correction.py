"""Собрать семь рыночных моделей из вневыборочных прогнозов 2021–2024 годов."""

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
from usd_rub.market_correction import MARKET_BUNDLE, MARKET_FEATURES, XGB_PARAMS, fit_corrections
from usd_rub.ml_dataset import feature_panels
from usd_rub.storage import input_fingerprints


def export(args):
    """Обучить на калибровке и записать переносимый архив деревьев."""
    dataset, calibration, _, _, _ = load_inputs(args)
    features = feature_panels(dataset)["extended"].loc[calibration.dates, list(MARKET_FEATURES)]
    baseline = references(calibration)[PRIMARY]
    models = fit_corrections(features, calibration.actual, baseline, calibration.last)
    bundle = {
        "format_version": 1,
        "xgboost": xgboost.__version__,
        "trained_before": json.loads(args.config.read_text())["test_start"],
        "calibration_origins": len(calibration.dates),
        "features": [*MARKET_FEATURES, "anchor_return"],
        "horizons": HORIZONS.tolist(),
        "parameters": XGB_PARAMS,
        "inputs": input_fingerprints([args.dataset, args.config, args.archive / ARCHIVE_FILES[0]]),
        "boosters": [json.loads(model.save_raw(raw_format="json")) for model in models],
    }
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
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/usd_rub_2011_2026_full.csv.gz")
    )
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument("--archive", type=Path, default=Path("data/reports/forecast_stacking"))
    parser.add_argument("--output", type=Path, default=MARKET_BUNDLE)
    export(parser.parse_args())
