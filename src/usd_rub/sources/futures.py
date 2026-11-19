"""Фьючерсы MOEX и ряд ближайшего контракта с фиксированным роллированием."""

import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from usd_rub.sources.markets import fetch_moex, iss_table
from usd_rub.storage import Client


def contract_info(client, ticker, year):
    for name in (f"{ticker}_{year}", ticker):
        payload = client.get(
            f"https://iss.moex.com/iss/securities/{name}.json", {"iss.meta": "off"}
        )
        table = iss_table(json.loads(payload.content), "description")
        if table.empty:
            continue
        data = table.set_index("name")["value"].to_dict()
        expiry = pd.to_datetime(data.get("LSTTRADE"), errors="coerce")
        if pd.notna(expiry) and expiry.year == year:
            return {"secid": name, "expiry": expiry, "metadata_sha256": payload.sha256}
    return None


def front_windows(contracts, start, end, roll_days=5):
    cursor, end = pd.Timestamp(start), pd.Timestamp(end)
    result = []
    for contract in sorted(contracts, key=lambda c: c["expiry"]):
        last = min(end, contract["expiry"] - pd.Timedelta(days=roll_days))
        if cursor <= last:
            result.append((contract, cursor, last))
            cursor = last + pd.Timedelta(days=1)
        if cursor > end:
            break
    if cursor <= end:
        raise ValueError(f"Не найден ближайший контракт для конца периода: {cursor.date()}")
    return result


def fetch_si_front(client, start, end):
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 2)
    candidates = [(f"Si{code}{year % 10}", year) for year in years for code in "HMUZ"]

    def metadata(item):
        with Client(client.root, refresh=client.refresh) as local:
            return contract_info(local, *item)

    with ThreadPoolExecutor(max_workers=4) as pool:
        contracts = [item for item in pool.map(metadata, candidates) if item is not None]
    windows = front_windows(contracts, start, end)

    def download(entry):
        contract, first, last = entry
        config = {
            "path": f"history/engines/futures/markets/forts/boards/RFUD/securities/{contract['secid']}",
            "columns": {
                "CLOSE": ("si_front_close", "RUB per 1000 USD"),
                "SETTLEPRICE": ("si_front_settle", "RUB per 1000 USD"),
                "OPENPOSITION": ("si_front_oi", "contracts"),
                "VOLUME": ("si_front_volume", "contracts"),
            },
        }
        with Client(client.root, refresh=client.refresh) as local:
            frame = fetch_moex(local, config, first.date(), last.date())
        prices = frame["series"].isin(["si_front_close", "si_front_settle"])
        frame.loc[prices, "value"] /= 1000
        frame.loc[prices, "unit"] = "RUB/USD"
        frame["contract"] = contract["secid"]
        frame["contract_expiry"] = str(contract["expiry"].date())
        frame["contract_metadata_sha256"] = contract["metadata_sha256"]
        frame["roll_rule"] = (
            "nearest expiry, switch 5 calendar days before last trade, no adjustment"
        )
        return frame

    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = list(pool.map(download, windows))
    return pd.concat(frames, ignore_index=True)
