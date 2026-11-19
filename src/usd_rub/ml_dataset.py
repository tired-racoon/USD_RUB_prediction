"""Единый датасет для сравнения моделей USD/RUB."""

import pandas as pd

FEATURE_GROUPS = {"own", "calendar", "fx", "markets", "global", "rates", "macro", "news"}
EXTENDED_PREFIX = "extended__"


def read_dataset(path):
    return pd.read_csv(path, index_col="date", parse_dates=["date"], float_precision="round_trip")


def feature_hit_rates(dataset):
    """Доля заполненных дат среди строк train для каждого признака."""
    columns = [
        name
        for name in dataset
        if name.removeprefix(EXTENDED_PREFIX).split("__", 1)[0] in FEATURE_GROUPS
    ]
    train = dataset.loc[dataset["split"].eq("train"), columns]
    if train.empty:
        raise ValueError("В датасете нет обучающих признаков")
    return train.notna().mean()


def clean_dataset(dataset, min_hit_rate):
    if not 0 <= min_hit_rate <= 1:
        raise ValueError("Доля заполненных дат должна быть от 0 до 1")
    hit_rates = feature_hit_rates(dataset)
    return dataset.drop(columns=hit_rates.index[hit_rates.lt(min_hit_rate)])


def feature_panels(dataset):
    strict = dataset[[name for name in dataset if name.split("__", 1)[0] in FEATURE_GROUPS]]
    columns = dict(strict.items())
    columns.update(
        {
            name.removeprefix(EXTENDED_PREFIX): dataset[name]
            for name in dataset
            if name.startswith(EXTENDED_PREFIX)
        }
    )
    extended = pd.DataFrame(columns, index=dataset.index)
    return {"strict": strict, "extended": extended}
