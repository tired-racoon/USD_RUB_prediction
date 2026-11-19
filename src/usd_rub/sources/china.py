"""Месячные PMI Китая из официальных таблиц NBS."""

import calendar
import json
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from usd_rub.sources.reports import save_document
from usd_rub.storage import Client, observations

BASE = "https://www.stats.gov.cn/english/PressRelease/"
ARCHIVE = [
    "201602/t20160201_1314222.html",
    "201702/t20170208_1460251.html",
    "201802/t20180202_1580233.html",
    "201902/t20190202_1648173.html",
    "202002/t20200203_1724922.html",
    "202102/t20210201_1813033.html",
    "202202/t20220207_1827228.html",
    "202302/t20230201_1892629.html",
    "202402/t20240205_1947279.html",
    "202502/t20250210_1958651.html",
    "202602/t20260202_1962478.html",
]


def parse_pmi(payload):
    soup = BeautifulSoup(payload.content, "html.parser")
    candidates = [
        table
        for table in soup.select("table")
        if any(cell.get_text(" ", strip=True) == "PMI" for cell in table.select("td,th"))
    ]
    if len(candidates) != 1:
        raise ValueError("NBS: не найдена однозначная таблица Manufacturing PMI")
    dates, values = [], []
    year = None
    months = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
    for row in candidates[0].select("tr"):
        cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in row.select("td,th")]
        if len(cells) != 7:
            continue
        match = re.fullmatch(r"(?:(\d{4})\s*[-–]\s*)?([A-Za-z]+)", cells[0])
        if not match or match[2].lower() not in months:
            continue
        if match[1]:
            year = int(match[1])
        if year is None:
            raise ValueError("NBS: месяц без начального года")
        date = pd.Timestamp(year, months[match[2].lower()], 1)
        if dates and date <= dates[-1]:
            raise ValueError("NBS: повтор или нарушение порядка месяцев")
        dates.append(date)
        values.append(cells[1:4])
    if not dates:
        raise ValueError("NBS: не найдены значения PMI")
    frames = [
        observations(dates, [v[i] for v in values], name, "diffusion index, neutral=50", payload)
        for i, name in enumerate(
            ["china_manufacturing_pmi", "china_pmi_production", "china_pmi_new_orders"]
        )
    ]
    result = pd.concat(frames, ignore_index=True)
    text = soup.get_text(" ", strip=True)
    publication = re.search(r"(20\d{2}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s+Print", text)
    if publication:
        result["reported_publication_time"] = pd.Timestamp(
            f"{publication[1]} {publication[2]}", tz="Asia/Shanghai"
        ).tz_convert("UTC")
        result["availability_basis"] = "current_document_version_observed_at_ingestion"
        result["publication_timezone"] = "Asia/Shanghai"
        result["vintage_limitation"] = (
            "Текущий текст датированного документа; неизменность прошлой версии не сертифицирована"
        )
    return result


def fetch_pmi(client, start, end):
    index = client.get(BASE)
    latest = [
        urljoin(BASE, a["href"])
        for a in BeautifulSoup(index.content, "html.parser").select("ul.list a[href]")
        if re.search("Purchasing Managers|PMI", a.get("title", a.get_text()), re.I)
    ]
    if not latest:
        raise ValueError("NBS: в текущем индексе нет выпуска PMI")
    urls = list(dict.fromkeys([urljoin(BASE, path) for path in ARCHIVE] + latest))

    def download(url):
        with Client(client.root, refresh=client.refresh) as local:
            payload = local.get(url)
            frame = parse_pmi(payload)
            save_document(local, payload, "nbs_pmi", period=frame.date.max().date())
            return frame

    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = list(pool.map(download, urls))
    folder = client.root / "documents" / "nbs_pmi"
    (folder / "manifest.json").write_text(json.dumps({"urls": urls}, ensure_ascii=False, indent=2))
    return pd.concat(frames, ignore_index=True)
