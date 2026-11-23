"""Выгрузить признаки основного ансамбля вместе с датами, курсом и целями."""

import argparse
import json
from pathlib import Path

import pandas as pd
from export_ml_dataset import write_dataset

from usd_rub.ensemble_forecast import CLASSICAL, classical_features
from usd_rub.ml_dataset import EXTENDED_PREFIX, FEATURE_GROUPS, feature_panels, read_dataset
from usd_rub.theory_features import COMPACT_MACRO_FEATURES

ENSEMBLE_DATASET = Path("datasets/usd_rub_2011_2026_ensemble.csv.gz")


def export(dataset, cutoff):
    """Сохранить порядок классических признаков и значения для поправки."""
    split = dataset.index.get_loc(cutoff)
    panels = feature_panels(dataset)
    required = {
        name
        for candidate, _, _ in CLASSICAL
        for name in classical_features(panels, candidate, split)
    }
    strict = panels["strict"].loc[:, panels["strict"].columns.isin(required)]
    extended = panels["extended"]
    correction = [
        name
        for name in COMPACT_MACRO_FEATURES
        if name not in strict or not extended[name].equals(strict[name])
    ]
    context = [
        name
        for name in dataset
        if name.removeprefix(EXTENDED_PREFIX).split("__", 1)[0] not in FEATURE_GROUPS
    ]
    return dataset[context].join(strict).join(extended[correction].add_prefix(EXTENDED_PREFIX))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/reports/theory_holdout/dataset.csv.gz")
    )
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument("--output", type=Path, default=ENSEMBLE_DATASET)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    snapshot = export(read_dataset(args.dataset), pd.Timestamp(config["test_start"]))
    write_dataset(snapshot, args.output)
    print(
        f"{args.output}: {len(snapshot)} дат, {feature_panels(snapshot)['extended'].shape[1]} признаков"
    )
