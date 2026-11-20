"""Постоянные сроки подразумеваемой волатильности из архива MOEX Si."""

import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.special import ndtri

from usd_rub.features import asof_series
from usd_rub.sources.markets import iss_table
from usd_rub.storage import Client, observations, read_series

HOST = "https://iss.moex.com"
CATALOG = HOST + "/iss/archives/files/options_coefficients_archive.json"
SERIES = {
    f"si_{name}_{days}d": unit
    for days in (7, 30)
    for name, unit in (
        ("iv", "% p.a."),
        ("rr25", "percentage points"),
    )
}
CONTRACT_CODE = r"^Si-(?P<future>\d{1,2}\.\d{2})M(?P<expiry>\d{6})(?P<side>[CP])A\s*\d"


def contracts(content):
    """Прочитать только маржируемые опционы Si и дату в полном коде."""
    names = ["tradedate", "isin", "option_type", "delta", "volat", "openposition"]
    filtered = []
    with ZipFile(BytesIO(content)) as archive, archive.open(archive.namelist()[0]) as source:
        for chunk in pd.read_csv(
            source,
            sep=";",
            skiprows=2,
            decimal=",",
            encoding="cp1251",
            usecols=names,
            dtype={name: float for name in ("delta", "volat", "openposition")},
            na_values=["-"],
            chunksize=150_000,
        ):
            subset = chunk.loc[chunk["isin"].str.startswith("Si-", na=False)].copy()
            code = subset["isin"].str.extract(CONTRACT_CODE)
            subset["expiry"] = pd.to_datetime(code.expiry, format="%d%m%y", errors="coerce")
            subset["future"] = code.future
            subset = subset.loc[code.side.eq(subset.option_type)]
            filtered.append(subset)
    result = pd.concat(filtered, ignore_index=True)
    result["date"] = pd.to_datetime(result.tradedate, format="%d.%m.%Y")
    result["days"] = (result.expiry - result.date).dt.days
    result["call_delta"] = result.delta + result.option_type.eq("P")
    return result.loc[
        result.days.between(1, 120)
        & result.volat.gt(0)
        & np.isfinite(result.volat)
        & result.call_delta.gt(0)
        & result.call_delta.lt(1)
    ]


def interp(x, y, point):
    """Интерполировать между наблюдениями внутри диапазона."""
    if len(x) < 2 or point < x.min() or point > x.max():
        return np.nan
    order = np.argsort(x)
    return float(np.interp(point, x[order], y[order]))


def expiry_volatility(contracts):
    """Восстановить ATM forward и две точки дельты на каждой экспирации."""
    rows = []
    for (date, days, future), series in contracts.groupby(["date", "days", "future"]):
        if not series.loc[series.call_delta.between(0.15, 0.85), "openposition"].gt(0).any():
            continue
        smile = series.groupby("call_delta", sort=True).volat.median()
        delta, volatility = smile.index.to_numpy(), smile.to_numpy()
        moneyness = ndtri(delta) - (volatility / 100) * np.sqrt(days / 365) / 2
        rows.append(
            (
                date,
                days,
                future,
                series.openposition.sum(),
                interp(moneyness, volatility, 0),
                interp(delta, volatility, 0.25),
                interp(delta, volatility, 0.75),
            )
        )
    return pd.DataFrame(rows, columns=["date", "days", "future", "oi", "atm", "call25", "put25"])


def constant_maturity(expirations):
    """Свести ближайшие серии к 7 и 30 дням по полной дисперсии."""
    rows = []
    usable = expirations.sort_values("oi", ascending=False).drop_duplicates(["date", "days"])
    for date, series in usable.groupby("date"):
        row = {"date": date}
        wings = series.dropna(subset=["call25", "put25"])
        for days in (7, 30):
            values = {}
            for column in ("atm", "call25", "put25"):
                selected = series.dropna(subset=[column]) if column == "atm" else wings
                points = selected.set_index("days")[column]
                maturity, vol = points.index.to_numpy(), points.to_numpy()
                if days in points:
                    values[column] = points[days]
                else:
                    values[column] = np.sqrt(interp(maturity, maturity * vol**2, days) / days)
            row[f"si_iv_{days}d"] = values["atm"]
            row[f"si_rr25_{days}d"] = values["call25"] - values["put25"]
        rows.append(row)
    return pd.DataFrame(rows, columns=["date", *SERIES])


def fetch_si_options(root, start, end, refresh=False, workers=4):
    """Собрать дневные признаки из месячных архивов и сохранить происхождение."""
    with Client(root, refresh=refresh) as client:
        listing = client.get(CATALOG, {"iss.meta": "off"})
    files = iss_table(json.loads(listing.content), "files")
    files = files.loc[
        files.extension.eq("csv") & files.date_till.ge(start) & files.date_from.le(end)
    ]
    files = files.sort_values("date_from", ascending=False)
    if files.empty:
        raise ValueError("MOEX: за указанный период нет архивов опционов")

    def month(item):
        with Client(root, refresh=refresh) as client:
            payload = client.get(HOST + item.url)
        rows = contracts(payload.content)
        levels = constant_maturity(expiry_volatility(rows.loc[rows.date.between(start, end)]))
        frames = [
            observations(
                levels.date,
                levels[name],
                name,
                unit,
                payload,
                "conservative_options_archive_dplus2",
                lag_days=2,
            )
            for name, unit in SERIES.items()
        ]
        return pd.concat(frames, ignore_index=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = []
        for i, result in enumerate(pool.map(month, files.itertuples(index=False)), start=1):
            parts.append(result)
            if i % 12 == 0 or i == len(files):
                print(f"Архивы опционов Si: {i}/{len(files)}", flush=True)
    return pd.concat(parts, ignore_index=True).sort_values(["series", "date"])


def si_option_features(root, dates):
    """Присоединить только известные на 00:00 МСК значения."""
    grid = pd.date_range(dates.min() - pd.Timedelta(days=5), dates.max(), tz="Europe/Moscow")
    columns = {}
    for name in SERIES:
        aligned = asof_series(read_series(root, name), grid, max_age_days=7).reindex(
            columns=["value", "age_days"]
        )
        level = aligned.value
        scaled = np.log(level.where(level > 0)) if name.startswith("si_iv_") else level
        key = f"fx__{name}"
        columns.update(
            {
                f"{key}_level": level,
                f"{key}_age": aligned.age_days.where(level.notna()),
                f"{key}_change_1": scaled.diff(),
                f"{key}_change_5": scaled.diff(5),
            }
        )
    return pd.DataFrame(columns).tz_localize(None).reindex(dates)
