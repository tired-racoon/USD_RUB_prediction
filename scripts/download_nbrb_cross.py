"""Скачать официальные котировки USD и RUB в белорусских рублях."""

import argparse
import json
from pathlib import Path

import pandas as pd

from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import Client

URL = "https://api.nbrb.by/exrates/rates/dynamics"
QUOTES = {"USD": (1, 145, 431), "RUB": (100, 298, 456)}
CHANGE = pd.Timestamp("2021-07-09")


def fetch(root, start, end):
    """Сохранить масштаб валют и хеш исходного ответа для каждой даты."""
    output = []
    dates = pd.date_range(start, end)
    chunks = dates.to_period("Q")
    with Client(root) as client:
        for currency, (scale, old, new) in QUOTES.items():
            for quarter in chunks.unique():
                days = dates[chunks == quarter]
                for subset, code in ((days[days < CHANGE], old), (days[days >= CHANGE], new)):
                    if not len(subset):
                        continue
                    payload = client.get(
                        f"{URL}/{code}",
                        {"startDate": str(subset[0].date()), "endDate": str(subset[-1].date())},
                    )
                    data = pd.DataFrame(json.loads(payload.content))
                    if data.empty or not data.Cur_ID.eq(code).all():
                        raise ValueError(f"Неполная выгрузка НБРБ для {currency} за {quarter}")
                    values = data.Cur_OfficialRate / scale
                    output.append(
                        pd.DataFrame(
                            {
                                "date": pd.to_datetime(data.Date),
                                "currency": currency,
                                "byn_per_unit": values,
                                "raw_sha256": payload.sha256,
                            }
                        )
                    )
    result = pd.concat(output, ignore_index=True).sort_values(["date", "currency"])
    if not result.pivot(index="date", columns="currency", values="byn_per_unit").index.equals(
        dates
    ):
        raise ValueError("Пропуски дат в выгрузке НБРБ")
    if result.byn_per_unit.le(0).any() or result.groupby("currency").size().ne(len(dates)).any():
        raise ValueError("Некорректный курс или пропуски валют в выгрузке НБРБ")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-11-18")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/research/nbrb_fx.csv.gz"))
    args = parser.parse_args()
    data = fetch(args.data_dir, args.start, args.end)
    write_csv(data, args.output)
    print(data.groupby("currency").date.agg(["size", "min", "max"]).to_string())
