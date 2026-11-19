"""MOEX ISS, FRED и Cboe."""

from io import BytesIO

import pandas as pd

from usd_rub.storage import observations


def parse_fred(payload, series_id: str, name: str, unit: str):
    frame = pd.read_csv(BytesIO(payload.content), na_values=["."])
    if series_id not in frame or not ({"observation_date", "DATE"} & set(frame.columns)):
        raise ValueError(f"FRED: неожиданный формат {series_id}")
    date_col = "observation_date" if "observation_date" in frame else "DATE"
    return observations(frame[date_col], frame[series_id], name, unit, payload)


def fetch_fred(client, series_id: str, name: str, unit: str, start, end):
    payload = client.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        {
            "id": series_id,
            "cosd": str(start),
            "coed": str(end),
        },
    )
    return parse_fred(payload, series_id, name, unit)


def fetch_alfred(client, series_id: str, name: str, unit: str, start, end, vintage: str):
    payload = client.get(
        "https://api.stlouisfed.org/fred/series/observations",
        {
            "series_id": series_id,
            "file_type": "json",
            "observation_start": str(start),
            "observation_end": str(end),
            "realtime_start": vintage,
            "realtime_end": vintage,
            "api_key": __import__("os").environ["FRED_API_KEY"],
        },
    )
    import json

    rows = json.loads(payload.content).get("observations", [])
    result = observations(
        [r["date"] for r in rows], [r["value"] for r in rows], name, unit, payload
    )
    result["available_at"] = pd.Timestamp(vintage, tz="America/Chicago").tz_convert(
        "UTC"
    ) + pd.Timedelta(days=1)
    result["availability_basis"] = "alfred_vintage_end_of_day"
    result["vintage"] = vintage + ":" + payload.sha256
    return result


def parse_cboe(payload):
    frame = pd.read_csv(BytesIO(payload.content))
    if not {"DATE", "CLOSE"}.issubset(frame.columns):
        raise ValueError("Cboe: неожиданный формат VIX")
    return observations(
        pd.to_datetime(frame["DATE"], format="%m/%d/%Y"),
        frame["CLOSE"],
        "vix",
        "index points",
        payload,
        "conservative_daily_close",
        lag_days=2,
    )


def fetch_cboe(client, start, end):
    payload = client.get("https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv")
    return parse_cboe(payload)


def iss_table(document: dict, name: str):
    block = document.get(name)
    if block is None or "columns" not in block or "data" not in block:
        raise ValueError(f"MOEX: нет блока {name}")
    return pd.DataFrame(block["data"], columns=block["columns"])


def fetch_moex(client, config: dict, start, end):
    import json

    base = "https://iss.moex.com/iss/"
    path, block = config["path"], config.get("block", "history")
    cursor = 0
    frames = []
    while True:
        payload = client.get(
            base + path + ".json",
            {
                "from": str(start),
                "till": str(end),
                "start": cursor,
                "interval": 24,
                "iss.meta": "off",
            },
        )
        doc = json.loads(payload.content)
        table = iss_table(doc, block)
        if table.empty:
            break
        date_column = "begin" if block == "candles" else "TRADEDATE"
        if date_column not in table:
            raise ValueError(f"MOEX: нет {date_column}")
        for column, (name, unit) in config["columns"].items():
            if column not in table:
                raise ValueError(f"MOEX: поле {column} отсутствует в бесплатном ответе")
            values = table[column]
            if column.upper() in {
                "OPEN",
                "HIGH",
                "LOW",
                "CLOSE",
                "WAPRICE",
                "SETTLEPRICE",
            } and unit in {
                "RUB/CNY",
                "RUB/USD",
                "RUB per 1000 USD",
            }:
                values = pd.to_numeric(values, errors="raise").where(lambda x: x > 0)
            frames.append(
                observations(
                    table[date_column],
                    values,
                    name,
                    unit,
                    payload,
                    "conservative_daily_close",
                    lag_days=1,
                )
            )
        next_cursor = doc.get(f"{block}.cursor")
        cursor += len(table)
        if next_cursor:
            info = iss_table(doc, f"{block}.cursor").iloc[0]
            if cursor >= info["TOTAL"]:
                break
        if cursor > 1_000_000:
            raise ValueError("MOEX: превышен предел пагинации")
    if not frames:
        raise ValueError(f"MOEX: нет данных {path} за {start} — {end}")
    return pd.concat(frames, ignore_index=True)
