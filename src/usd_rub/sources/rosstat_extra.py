"""Действующие таблицы Росстата по промышленности и недельным ценам."""

import hashlib
import re

import pandas as pd

from usd_rub.sources.events import MONTHS_GENITIVE
from usd_rub.sources.tables import discover_table_links, parse_period_rows, workbook
from usd_rub.storage import number, observations


def fetch_industry(client, start, end):
    page = client.get("https://rosstat.gov.ru/enterprise_industrial")
    links = discover_table_links(page, "Индексы промышленного производства с исключением сезонного")
    if not links:
        raise ValueError("Росстат: не найдена действующая таблица промышленного производства")
    frames, failures = [], []
    for link in links:
        try:
            payload = client.get(link["url"])
            sheets = workbook(payload)
            sheet = next(k for k, t in sheets.items() if t.shape[1] >= 5)
            frame = parse_period_rows(
                payload,
                {
                    "1": {
                        "series": "industrial_production_mom_raw",
                        "unit": "% previous month=100",
                    },
                    "2": {"series": "industrial_production_mom_sa", "unit": "% previous month=100"},
                },
                sheet=sheet,
                year_suffix_pattern=r"2",
            )
            frames.append(frame)
        except (ValueError, RuntimeError, OSError) as exc:
            failures.append({"url": link["url"], "error": str(exc)})
    if not frames:
        raise ValueError(f"Росстат: не удалось получить производство: {failures}")
    frame = pd.concat(frames, ignore_index=True)
    if frame.groupby(["series", "date"])["value"].nunique().gt(1).any():
        raise ValueError(
            "Промпроизводство: архивы содержат разные ревизии; требуется выбор винтажа"
        )
    return frame.drop_duplicates(["series", "date"])


def parse_weekly_items(payload):
    frames = []
    for sheet, table in workbook(payload).items():
        if not re.fullmatch(r"20\d{2}", sheet):
            continue
        dates = []
        for value in table.iloc[3]:
            match = re.search(r"(\d{1,2})\s+([а-я]+)", str(value).lower())
            dates.append(
                pd.Timestamp(int(sheet), MONTHS_GENITIVE[match[2]], int(match[1]))
                if match and match[2] in MONTHS_GENITIVE
                else pd.NaT
            )
        dates = pd.Series(dates)
        valid = dates.notna()
        for _, row in table.iloc[4:].iterrows():
            label = str(row.iloc[0]).strip()
            if label == "nan" or not row[valid].notna().any():
                continue
            values = row[valid]
            try:
                parsed = values.map(number)
            except ValueError:
                continue
            if parsed.notna().sum() == 0:
                continue
            key = hashlib.sha256(label.lower().encode()).hexdigest()[:12]
            frame = observations(
                dates[valid],
                parsed,
                "weekly_item_price_" + key,
                "% previous registration=100",
                payload,
            )
            frame["series_label"] = label
            frames.append(frame)
    if not frames:
        raise ValueError("Росстат: не найдены недельные индексы товаров")
    return pd.concat(frames, ignore_index=True)


def fetch_weekly_items(client, start, end):
    page = client.get("https://rosstat.gov.ru/statistics/price/")
    links = discover_table_links(page, r"Еженедельные индексы потребительских цен")
    if len(links) != 1:
        raise ValueError("Росстат: неоднозначная ссылка недельных индексов")
    return parse_weekly_items(client.get(links[0]["url"]))
