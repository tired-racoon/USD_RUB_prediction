"""Скачать почасовые свечи валютного рынка для исторического бэктеста."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import Client

ISS = "https://iss.moex.com/iss/engines"
SPOT = "currency/markets/selt/boards/CETS/securities"
FUTURES = "futures/markets/forts/boards/RFUD/securities"
INSTRUMENTS = {
    "CNYRUB_TOM": (SPOT, "2019-01-01", None),
    "USD000UTSTOM": (SPOT, "2019-01-01", "2024-06-12"),
    "USDRUBF": (FUTURES, "2022-12-05", None),
    "CNYRUBF": (FUTURES, "2022-12-05", None),
}


def download(client, instrument, first, last):
    """Прочитать все страницы за указанный интервал."""
    venue = INSTRUMENTS[instrument][0]
    url = f"{ISS}/{venue}/{instrument}/candles.json"
    offset = 0
    frames = []
    while True:
        payload = client.get(
            url,
            {
                "from": first,
                "till": last,
                "interval": 60,
                "start": offset,
                "iss.meta": "off",
                "iss.only": "candles",
            },
        )
        block = json.loads(payload.content)["candles"]
        if not block["data"]:
            break
        frame = pd.DataFrame(block["data"], columns=block["columns"])
        if frames and frame.begin.iloc[0] <= frames[-1].begin.iloc[-1]:
            raise ValueError(f"ISS вернул повтор свечей для {instrument} с {first}")
        frames.append(frame.assign(instrument=instrument, raw_sha256=payload.sha256))
        offset += len(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch(args):
    """Сохранить свечи и хеши ответов ISS."""
    jobs = []
    for instrument, (_, first, last) in INSTRUMENTS.items():
        start, end = max(first, args.start), min(last or args.end, args.end)
        for month in pd.period_range(start, end, freq="M"):
            jobs.append(
                (
                    instrument,
                    max(start, str(month.start_time.date())),
                    min(end, str(month.end_time.date())),
                )
            )
    with Client(args.data_dir) as client, ThreadPoolExecutor(args.jobs) as pool:
        tasks = [pool.submit(download, client, *job) for job in jobs]
        frames = [task.result() for task in tasks]
    table = pd.concat(frames, ignore_index=True).sort_values(["instrument", "begin"])
    if table.duplicated(["instrument", "begin"]).any():
        raise ValueError("Повтор свечи в выгрузке ISS")
    write_csv(table, args.output)
    print(table.groupby("instrument").begin.agg(["size", "min", "max"]).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-11-18")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("data/research/fx_candles.csv.gz"))
    fetch(parser.parse_args())
