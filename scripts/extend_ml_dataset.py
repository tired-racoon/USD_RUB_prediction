"""Добавление признаков из обзора статей к сохранённому датасету USD/RUB."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from export_ml_dataset import write_dataset

from usd_rub.ml_dataset import read_dataset
from usd_rub.paper_features import MARKET_SERIES, build_paper_features
from usd_rub.storage import Client, input_fingerprints


def extend(input_path, output, root, report):
    if output.exists() or report.exists():
        raise FileExistsError("Укажите новые пути для датасета и отчёта")
    base = read_dataset(input_path)
    if not base.index.equals(pd.date_range(base.index.min(), base.index.max())):
        raise ValueError("Нужен непрерывный дневной индекс исходного датасета")
    with Client(root) as client:
        features, swaps, releases, sources = build_paper_features(client, root, base.usd_rub)
    if not base.columns.intersection(features.columns).empty:
        raise ValueError("В исходном датасете уже есть новые столбцы")
    if np.isinf(features.to_numpy()).any() or features.isna().all().any():
        raise ValueError("Признаки содержат бесконечности или пустые столбцы")
    train = base.split.eq("train")
    coverage = pd.DataFrame(
        {
            "first_date": features.apply(pd.Series.first_valid_index),
            "last_date": features.apply(pd.Series.last_valid_index),
            "train_hit_rate": features.loc[train].notna().mean(),
            "last_730_train_days_hit_rate": features.loc[train].tail(730).notna().mean(),
            "test_hit_rate": features.loc[~train].notna().mean(),
        }
    ).rename_axis("feature")
    report.mkdir(parents=True)
    write_dataset(pd.concat([base, features], axis=1), output)
    coverage.to_csv(report / "coverage.csv")
    swaps.to_csv(report / "swap_history.csv")
    releases.to_csv(report / "survey_releases.csv")
    metadata = {
        "rows": len(base),
        "original_columns_excluding_date": len(base.columns),
        "new_features": len(features.columns),
        "inputs": input_fingerprints(
            [input_path, *(root / "normalized" / f"{s}.csv" for s in MARKET_SERIES)]
        ),
        "sources": {name: {"url": p.url, "sha256": p.sha256} for name, p in sources.items()},
    }
    (report / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"{output} — {len(base)} дат, добавлено {len(features.columns)} признака")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/usd_rub_2011_2026.csv.gz"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    extend(args.input, args.output, args.data_dir, args.report)
