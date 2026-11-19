"""Официальные месячные цены Urals и документы Минэкономразвития."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from usd_rub.sources.events import MONTHS_GENITIVE
from usd_rub.sources.reports import save_document
from usd_rub.sources.tables import MONTHS
from usd_rub.storage import Client, observations

BASE = "https://economy.gov.ru/material/departments/d12/konyunktura_mirovyh_tovarnyh_rynkov/"


def parse_urals(payload):
    soup = BeautifulSoup(payload.content, "html.parser")
    main = soup.select_one("main") or soup
    text = re.sub(r"\s+", " ", main.get_text(" ", strip=True))
    period = re.search(r"Юралс[»\"]?\s+за\s+([а-я]+)\s+(20\d{2})\s+года", text, re.IGNORECASE)
    price = re.search(
        r"Средняя цена\s+(\d+(?:[,.]\d+)?)\s+долл\.?\s*США\s+за\s+баррель", text, re.IGNORECASE
    )
    if not period or not price or period[1].lower() not in MONTHS:
        raise ValueError("Urals: не найдена однозначная месячная цена")
    date = pd.Timestamp(int(period[2]), MONTHS[period[1].lower()], 1)
    frame = observations([date], [price[1]], "urals_official_monthly", "USD/bbl", payload)
    frame["evidence"] = price[0]
    publication = re.search(
        r"Дата публикации:\s*(\d{1,2})\s+([а-я]+)\s+(20\d{2})", text, re.IGNORECASE
    )
    if publication and publication[2].lower() in MONTHS_GENITIVE:
        frame["reported_publication_date"] = str(
            pd.Timestamp(
                int(publication[3]), MONTHS_GENITIVE[publication[2].lower()], int(publication[1])
            ).date()
        )
        stamp = pd.Timestamp(frame.reported_publication_date.iloc[0], tz="Europe/Moscow")
        frame["available_at"] = (stamp + pd.Timedelta(days=1)).tz_convert("UTC")
        frame["availability_basis"] = "reported_publication_date_end_of_day"
    return frame


def fetch_urals(client, start, end):
    def index_year(year):
        links = []
        with Client(client.root, refresh=client.refresh) as local:
            for page in range(1, 60):
                payload = local.request(
                    "GET",
                    BASE + "submaterials/",
                    params={"type": "", "page": page, "year": year, "month": ""},
                    headers={"X-Requested-With": "XMLHttpRequest"},
                )
                text = payload.content.decode("utf-8").replace("\\'", "'")
                soup = BeautifulSoup(text, "html.parser")
                if soup.title:
                    raise ValueError("Urals: сервер не применил архивный запрос")
                found = [
                    urljoin(BASE, a["href"])
                    for a in soup.select("a[href]")
                    if "o_srednem_urovne_cen_nefti_sorta_yurals" in a["href"]
                ]
                if any(
                    not any(f"_{candidate}_goda" in link for candidate in (year, year - 1))
                    for link in found
                ):
                    raise ValueError("Urals: сервер не применил фильтр года")
                links.extend(found)
                more = re.search(r"submaterailsLoadMore\s*=\s*parseInt\('([01])'\)", text)
                if not more or more[1] == "0":
                    break
            else:
                raise ValueError("Urals: превышен предел страниц")
        return list(dict.fromkeys(links))

    with ThreadPoolExecutor(max_workers=4) as pool:
        indexes = list(
            pool.map(index_year, range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))
        )
    urls = list(dict.fromkeys(url for index in indexes for url in index))

    def download(url):
        with Client(client.root, refresh=client.refresh) as local:
            payload = local.get(url)
            frame = parse_urals(payload)
            save_document(local, payload, "economy_urals", period=frame.date.iloc[0].date())
            return frame

    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = list(pool.map(download, urls))
    if not frames:
        raise ValueError("Urals: не найден архив официальных месячных цен")
    folder = client.root / "documents" / "economy_urals"
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "urls": urls,
                "years": {
                    str(y): len(items)
                    for y, items in zip(
                        range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1),
                        indexes,
                        strict=True,
                    )
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return pd.concat(frames, ignore_index=True)
