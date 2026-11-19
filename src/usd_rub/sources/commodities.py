"""Месячные цены World Bank Pink Sheet."""

import re
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from usd_rub.sources.tables import workbook
from usd_rub.storage import observations

PAGE = "https://www.worldbank.org/en/research/commodity-markets"


def parse_pink_sheet(payload):
    table = workbook(payload)["Monthly Prices"]
    rows = table.index[table.iloc[:, 0].astype(str).str.fullmatch(r"\d{4}M\d{2}")]
    if not len(rows):
        raise ValueError("Pink Sheet: не найдены месячные наблюдения")
    header = rows.min() - 2
    dates = pd.to_datetime(table.loc[rows].iloc[:, 0], format="%YM%m")
    frames = []
    for label, name, expected_unit, unit in [
        ("Gold", "gold_monthly", "($/troy oz)", "USD/troy oz"),
        ("Liquefied natural gas, Japan", "lng_japan_monthly", "($/mmbtu)", "USD/MMBtu"),
    ]:
        columns = table.columns[table.iloc[header].astype(str).str.strip() == label]
        if len(columns) != 1 or str(table.loc[header + 1, columns[0]]).strip() != expected_unit:
            raise ValueError(f"Pink Sheet: изменились название или единица {label}")
        values = table.loc[rows, columns[0]].replace({"…": None, "...": None})
        frames.append(observations(dates, values, name, unit, payload))
    return pd.concat(frames, ignore_index=True)


def fetch_pink_sheet(client, start, end):
    page = client.get(PAGE)
    links = [
        urljoin(page.url, a["href"])
        for a in BeautifulSoup(page.content, "html.parser").select("a[href]")
        if re.search(r"CMO-Historical-Data-Monthly\.xlsx(?:\?|$)", a["href"], re.I)
    ]
    if len(set(links)) != 1:
        raise ValueError("Pink Sheet: не найдена однозначная текущая месячная таблица")
    return parse_pink_sheet(client.get(links[0]))
