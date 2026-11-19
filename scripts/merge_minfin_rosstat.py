"""Объединить снимки USD/RUB и добавить месячные ряды Минфина и Росстата."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
from export_ml_dataset import write_dataset

from usd_rub.features import build_features
from usd_rub.ml_dataset import FEATURE_GROUPS, read_dataset
from usd_rub.sources.minfin import ytd_to_monthly
from usd_rub.storage import Payload, input_fingerprints, observations, utc_now

SERIES = {
    "oil_gas_revenue_ytd_bln_rub": ("oil_gas_revenue", "RUB bn"),
    "non_oil_gas_revenue_ytd_bln_rub": ("budget_non_oil_revenue", "RUB bn"),
    "cpi_ru_index_mom": ("cpi_ru_mom_index", "% previous month=100"),
}


def merge_snapshots(paths):
    """Соединить признаки с одинаковыми календарём, целями и общими столбцами."""
    base = read_dataset(paths[0])
    for path in paths[1:]:
        other = read_dataset(path)
        common = base.columns.intersection(other.columns)
        if not base.index.equals(other.index) or not base[common].equals(other[common]):
            raise ValueError(f"Общие даты или столбцы расходятся в {path}")
        new = other.loc[:, other.columns.difference(base.columns, sort=False)]
        if not all(
            col.removeprefix("extended__").split("__", 1)[0] in FEATURE_GROUPS for col in new
        ):
            raise ValueError(f"В {path} есть новые столбцы вне групп признаков")
        base = pd.concat([base, new], axis=1)
    return base


def fiscal_features(path, dates, config):
    """Преобразовать накопленные доходы и дать каждому месяцу лаг публикации."""
    content = path.read_bytes()
    table = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", dtype={"month": "string"})
    if (
        table.empty
        or set(table.columns) != {"month", *SERIES}
        or not table.month.str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").fillna(False).all()
    ):
        raise ValueError("Ожидаются месяц YYYY-MM и три ряда minfin_rosstat.csv")
    periods = pd.to_datetime(table.month, format="%Y-%m")
    if periods.duplicated().any():
        raise ValueError("В minfin_rosstat.csv повторяются месяцы")
    payload = Payload(content, str(path), hashlib.sha256(content).hexdigest(), utc_now())
    frames = {}
    for column, (name, unit) in SERIES.items():
        frame = observations(periods, table[column], name, unit, payload)
        frames[name] = ytd_to_monthly(frame) if "_ytd_" in column else frame
        if frames[name].empty:
            raise ValueError(f"В ряду {column} нет месячных значений")
    lags = {
        "oil_gas_revenue": config["assumed_lags"]["oil_gas_revenue"],
        "budget_non_oil_revenue": config["assumed_lags"]["oil_gas_revenue"],
        "cpi_ru_mom_index": config["assumed_lags"]["cpi_ru_mom_index"],
    }
    aligned = build_features(frames, dates[0], dates[-1], cutoff="00:00", assumed_lags=lags)
    features = {}
    for name in frames:
        key = f"extended__macro__{name}"
        values = aligned[name].set_axis(aligned.index.tz_localize(None)).reindex(dates)
        features[f"{key}_level"] = values
        for window in (30, 90):
            features[f"{key}_change_{window}"] = values.diff(window)
    return pd.DataFrame(features, index=dates), lags


def run(args):
    """Выгрузить контроль и расширенный датасет для парного сравнения."""
    inputs = [*args.snapshot, args.source, args.config]
    output_paths = {path.resolve() for path in (args.control, args.output)}
    if len(output_paths) != 2 or output_paths.intersection(path.resolve() for path in inputs):
        raise ValueError("Укажите разные пути для входов, контроля и расширенного датасета")
    fingerprints = input_fingerprints(inputs)
    config = json.loads(args.config.read_text())
    base = merge_snapshots(args.snapshot)
    dates = pd.date_range(config["start"], config["end"])
    if not base.index.equals(dates):
        raise ValueError("Снимок и конфигурация должны покрывать одинаковые календарные даты")
    features, lags = fiscal_features(args.source, base.index, config)
    overlap = base.columns.intersection(features.columns)
    extended = base.reindex(columns=base.columns.union(features.columns, sort=False))
    extended[features.columns] = features
    first = {key: values.first_valid_index() for key, values in features.items()}
    report = {
        "rows": len(extended),
        "input_files": fingerprints,
        "new_features": features.columns.difference(overlap, sort=False).tolist(),
        "updated_features": overlap.tolist(),
        "assumed_lags_from_month_start_days": lags,
        "coverage_train": features.loc[base.split.eq("train")].notna().mean().to_dict(),
        "first_observation": {
            key: str(day.date()) if day is not None else None for key, day in first.items()
        },
    }
    if fingerprints != input_fingerprints(inputs):
        raise ValueError("Входные файлы изменились при сборке")
    write_dataset(base, args.control)
    write_dataset(extended, args.output)
    args.control.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"{args.output} — {len(extended)} дат, {len(features.columns)} признаков из CSV")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/minfin_rosstat.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument("--snapshot", type=Path, action="append", required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
