"""Открытые XML-выпуски налогового календаря ФНС."""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin
from xml.etree import ElementTree as ET

import pandas as pd
from bs4 import BeautifulSoup

from usd_rub.storage import Client, observations

MONTHS = {
    name: i
    for i, name in enumerate(
        "january february march april may june july august september october november december".split(),
        1,
    )
}


def release_date(url):
    match = re.search(r"data-(\d{8})-", url)
    if not match:
        raise ValueError("ФНС: не найдена дата выпуска XML")
    for fmt in ("%Y%m%d", "%d%m%Y"):
        try:
            return pd.to_datetime(match[1], format=fmt)
        except ValueError:
            pass
    raise ValueError("ФНС: не распознана дата выпуска")


def parse_calendar(payload):
    try:
        text = payload.content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = payload.content.decode("windows-1251")
    fragments = re.findall(r"<calendar\b[^>]*>[\s\S]*?</calendar>", text)
    if len(fragments) != 1:
        raise ValueError("ФНС: ожидался один полный XML-календарь")
    root = ET.fromstring(fragments[0])
    release = release_date(payload.url)
    release_id = hashlib.sha256((payload.url + payload.sha256).encode()).hexdigest()
    publication = (release.tz_localize("Europe/Moscow") + pd.Timedelta(days=1)).tz_convert("UTC")
    dates, payments, filings, holidays, events = [], [], [], [], []
    for year in root.findall("year"):
        for month in year.findall("month"):
            month_number = MONTHS[month.attrib["name"].lower()]
            for day in month.findall("day"):
                date = pd.Timestamp(int(year.attrib["index"]), month_number, int(day.attrib["num"]))
                soup = BeautifulSoup(day.text or "", "html.parser")
                payment_count, filing_count = 0, 0
                tax = ""
                seen = set()
                for paragraph in soup.select("p,li"):
                    if paragraph.find(["p", "li"]) is not None:
                        continue
                    evidence = paragraph.get_text(" ", strip=True)
                    strong = paragraph.find(["strong", "b"])
                    if strong:
                        tax = strong.get_text(" ", strip=True).rstrip(":")
                    if re.search(r"уплачива|уплатить|уплата\b", evidence, re.IGNORECASE):
                        kind = "tax_payment_day"
                    elif re.search(
                        r"представля|отчетност|уведомлен|деклараци", evidence, re.IGNORECASE
                    ):
                        kind = "tax_filing_day"
                    else:
                        continue
                    effective = date.tz_localize("Europe/Moscow")
                    logical = hashlib.sha256(
                        (str(date.date()) + kind + tax + re.sub(r"\s+", " ", evidence)).encode()
                    ).hexdigest()
                    if logical in seen:
                        continue
                    seen.add(logical)
                    payment_count += kind == "tax_payment_day"
                    filing_count += kind == "tax_filing_day"
                    events.append(
                        {
                            "event_id": hashlib.sha256((release_id + logical).encode()).hexdigest(),
                            "event_key": logical,
                            "calendar_year": int(date.year),
                            "release_id": release_id,
                            "availability_basis": "assumed_archive_release_date_end_of_day",
                            "event_type": kind,
                            "event_time": str(effective),
                            "effective_time": str(effective),
                            "publication_time": str(publication),
                            "source_url": payload.url,
                            "tax_name": tax,
                            "value": 1,
                            "unit": "event",
                            "evidence": evidence,
                            "raw_sha256": payload.sha256,
                            "ingested_at": payload.ingested_at,
                        }
                    )
                dates.append(date)
                payments.append(payment_count)
                filings.append(filing_count)
                holidays.append(int(day.attrib.get("type") == "holiday"))
    if not dates:
        raise ValueError("ФНС: не найдены календарные даты")
    frames = []
    for name, values in [
        ("fns_tax_payment_count", payments),
        ("fns_tax_filing_count", filings),
        ("ru_holiday_fns", holidays),
    ]:
        frame = observations(
            dates, values, name, "calendar entries" if "count" in name else "flag", payload
        )
        frame["available_at"] = publication
        frame["availability_basis"] = "assumed_archive_release_date_end_of_day"
        frame["release_date"] = str(release.date())
        frame["vintage"] = release_id
        frames.append(frame)
    for year in sorted({date.year for date in dates}):
        key = f"calendar_coverage_{year}"
        effective = pd.Timestamp(year, 1, 1).tz_localize("Europe/Moscow")
        events.append(
            {
                "event_id": hashlib.sha256((release_id + key).encode()).hexdigest(),
                "event_key": key,
                "calendar_year": year,
                "release_id": release_id,
                "event_type": "calendar_coverage",
                "event_time": str(effective),
                "effective_time": str(effective),
                "publication_time": str(publication),
                "availability_basis": "assumed_archive_release_date_end_of_day",
                "source_url": payload.url,
                "tax_name": "calendar coverage",
                "value": 0,
                "unit": "coverage",
                "evidence": "Календарный год присутствует в выпуске XML",
                "raw_sha256": payload.sha256,
                "ingested_at": payload.ingested_at,
            }
        )
    return pd.concat(frames, ignore_index=True), events


def fetch_tax_calendar(client, start, end):
    page = client.get("https://www.nalog.gov.ru/opendata/7707329152-kalendar/")
    soup = BeautifulSoup(page.content, "html.parser")
    urls = list(
        dict.fromkeys(
            urljoin(page.url, a["href"])
            for a in soup.select("a[href]")
            if re.search(r"data-\d{8}-structure-\d{8}\.xml", a["href"])
        )
    )
    if not urls or max(release_date(url) for url in urls).year < pd.Timestamp(end).year:
        raise ValueError(
            f"ФНС: не найден действующий выпуск календаря за {pd.Timestamp(end).year} год"
        )

    def download(url):
        try:
            with Client(client.root, refresh=client.refresh) as local:
                frame, records = parse_calendar(local.get(url))
            return frame, records, None
        except (ValueError, RuntimeError, OSError, ET.ParseError, KeyError) as exc:
            return None, [], {"url": url, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, urls))
    folder = client.root / "calendars"
    folder.mkdir(parents=True, exist_ok=True)
    events = [event for _, records, _ in results for event in records]
    pd.DataFrame(events).drop_duplicates("event_id").to_csv(folder / "fns_events.csv", index=False)
    errors = [error for _, _, error in results if error]
    (folder / "fns_manifest.json").write_text(
        json.dumps(
            {
                "releases": urls,
                "events": len(events),
                "errors": errors,
                "missing_calendar_years": sorted(
                    set(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))
                    - {event["calendar_year"] for event in events}
                ),
                "warning": "Дата выпуска из имени файла является допущением; отсутствующие годы не заполняются",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    frames = [frame for frame, _, _ in results if frame is not None]
    if not frames:
        raise ValueError(f"ФНС: календарь не получен: {errors}")
    return pd.concat(frames, ignore_index=True)
