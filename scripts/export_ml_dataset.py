"""Выгрузка признаков и целей эксперимента 2025–2026 в один CSV."""

import argparse
import gzip
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from usd_rub.backtest import daily_target
from usd_rub.ml_dataset import clean_dataset, read_dataset
from usd_rub.ml_features import load_panel
from usd_rub.storage import read_series


def build_dataset(config_path, data_dir):
    config = json.loads(config_path.read_text())
    target = daily_target(read_series(data_dir, "usd_rub"), config["start"], config["end"])
    panels, _ = load_panel(data_dir, config, target)
    strict, extended = panels["strict"], panels["extended"]
    revised = [
        name for name in extended if name not in strict or not extended[name].equals(strict[name])
    ]

    test_start = pd.Timestamp(config["test_start"])
    split = pd.Series("train", index=target.index).mask(target.index >= test_start, "test")
    labels = {"split": split, "usd_rub": target}
    labels["in_test_metrics"] = (target.index >= test_start) & target.shift(
        -config["horizon"]
    ).notna()
    dates = target.index.to_series()
    for horizon in range(1, config["horizon"] + 1):
        later = target.shift(-horizon)
        eligible = (split == "test") | (dates.shift(-horizon) < test_start)
        later = later.where(eligible)
        labels[f"target_usd_rub_{horizon}d"] = later
        labels[f"target_return_{horizon}d"] = later / target - 1

    return pd.concat(
        [pd.DataFrame(labels), strict, extended[revised].add_prefix("extended__")],
        axis=1,
    ).rename_axis("date")


def write_dataset(dataset, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=output.parent, delete=False) as file:
            temporary = Path(file.name)
            if output.suffix == ".gz":
                with gzip.GzipFile(fileobj=file, filename="", mode="wb", mtime=0) as compressed:
                    compressed.write(dataset.to_csv().encode("utf-8"))
            else:
                dataset.to_csv(file)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def add_news_features(dataset, path):
    """Добавить новостные признаки по уже рассчитанной дате прогноза."""
    news = pd.read_csv(path, index_col="date", parse_dates=["date"], float_precision="round_trip")
    if (
        not isinstance(news.index, pd.DatetimeIndex)
        or news.index.tz is not None
        or news.index.hasnans
        or not news.index.is_unique
        or not news.index.equals(news.index.normalize())
        or not len(news.columns)
        or not all(
            name.startswith(("news__press_", "news__smartlab_", "news__rbc_")) for name in news
        )
    ):
        raise ValueError("Некорректные даты прогноза или имена новостных признаков")
    return dataset.join(news, how="left", validate="one_to_one")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--input", type=Path, help="Ранее собранный CSV или CSV.GZ")
    parser.add_argument(
        "--news-features",
        type=Path,
        action="append",
        default=[],
        help="Дневные признаки news-features",
    )
    parser.add_argument("--min-hit-rate", type=float, default=0.5)
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/ml_holdout_2025_2026/dataset.csv")
    )
    args = parser.parse_args()
    source = read_dataset(args.input) if args.input else build_dataset(args.config, args.data_dir)
    for path in args.news_features:
        source = add_news_features(source, path)
    dataset = clean_dataset(source, args.min_hit_rate)
    write_dataset(dataset, args.output)
    rows, columns = dataset.shape
    print(f"{args.output}: {rows} строк, {columns + 1} столбцов")
