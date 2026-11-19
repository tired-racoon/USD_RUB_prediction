"""Календарь ЦБ и импорт документированных событий."""

import re
import shutil
import subprocess
from io import BytesIO

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader

from usd_rub.storage import observations

MONTHS_GENITIVE = {
    name: i
    for i, name in enumerate(
        [
            "января",
            "февраля",
            "марта",
            "апреля",
            "мая",
            "июня",
            "июля",
            "августа",
            "сентября",
            "октября",
            "ноября",
            "декабря",
        ],
        1,
    )
}


def russian_date(text):
    match = re.search(r"(\d{1,2})\s+([а-я]+)\s+(20\d{2})", text.replace("\xa0", " "))
    if match is None:
        raise ValueError(f"Не распознана дата: {text}")
    day, month, year = match.groups()
    return pd.Timestamp(int(year), MONTHS_GENITIVE[month], int(day))


def parse_meetings(payload):
    soup = BeautifulSoup(payload.content, "html.parser")
    dates = []
    for day in soup.select(".main-events_day"):
        title = day.get_text(" ", strip=True).replace("\xa0", " ")
        if "Заседание Совета директоров" not in title:
            continue
        date_text = day.select_one(".date").get_text(" ", strip=True)
        if not re.search(r"20\d{2}", date_text):
            tab = day.find_parent(attrs={"role": "tabpanel"})
            tab_label = soup.find(id=tab.get("aria-labelledby")) if tab else None
            if tab_label is None:
                raise ValueError(f"Календарь ЦБ: не найден год {date_text}")
            date_text += " " + tab_label.get_text(" ", strip=True)
        dates.append(russian_date(date_text))
    for tab in soup.select('[role="tabpanel"]'):
        label = soup.find(id=tab.get("aria-labelledby"))
        if label is None:
            continue
        year = re.search(r"20\d{2}", label.get_text())
        if year is None:
            continue
        for row in tab.select("table tr"):
            cells = row.select("td")
            if len(cells) < 2 or "Заседание Совета директоров" not in row.get_text(" ", strip=True):
                continue
            date_text = cells[0].get_text(" ", strip=True)
            if not re.search(r"20\d{2}", date_text):
                date_text += " " + year.group()
            dates.append(russian_date(date_text))
    if not dates:
        raise ValueError("Календарь ЦБ: не найдены заседания")
    return observations(dates, [1] * len(dates), "cbr_meeting", "event", payload)


def fetch_meetings(client, start, end):
    return parse_meetings(client.get("https://www.cbr.ru/dkp/cal_mp/"))


def extract_document(payload):
    if payload.content.startswith(b"%PDF"):
        executable = shutil.which("pdftotext")
        if executable:
            try:
                result = subprocess.run(
                    [executable, "-layout", "-", "-"],
                    input=payload.content,
                    capture_output=True,
                    check=True,
                    timeout=60,
                )
                pages = result.stdout.decode("utf-8").split("\f")
                if pages and not pages[-1].strip():
                    pages.pop()
                if any(page.strip() for page in pages):
                    return [{"page": i + 1, "text": text} for i, text in enumerate(pages)]
            except (OSError, subprocess.SubprocessError):
                pass
        reader = PdfReader(BytesIO(payload.content))
        return [
            {"page": i + 1, "text": page.extract_text() or ""}
            for i, page in enumerate(reader.pages)
        ]
    soup = BeautifulSoup(payload.content, "html.parser")
    for tag in soup.select("script,style,nav,header,footer"):
        tag.decompose()
    main = soup.select_one("main") or soup
    return [{"page": 1, "text": main.get_text("\n", strip=True)}]


def parse_events_csv(payload):
    events = pd.read_csv(BytesIO(payload.content))
    required = {
        "event_id",
        "event_type",
        "event_time",
        "publication_time",
        "effective_time",
        "source_url",
        "value",
        "unit",
        "evidence",
    }
    if not required.issubset(events.columns):
        raise ValueError(f"Не хватает полей событий: {sorted(required - set(events.columns))}")
    if events[list(required)].isna().any().any() or events["event_id"].duplicated().any():
        raise ValueError("События: пропуски обязательных полей или повтор event_id")
    for column in ("event_time", "publication_time", "effective_time"):
        for value in events[column]:
            if pd.Timestamp(value).tzinfo is None:
                raise ValueError(f"{column}: требуется часовой пояс")
        events[column] = pd.to_datetime(events[column], utc=True)
    events["value"] = pd.to_numeric(events["value"], errors="raise")
    events["ingested_at"] = payload.ingested_at
    events["raw_sha256"] = payload.sha256
    return events
