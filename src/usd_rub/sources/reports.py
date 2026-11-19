"""Архивы официальных обзоров и проверяемые числовые фрагменты."""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from usd_rub.sources.events import extract_document
from usd_rub.storage import Client, observations


def save_document(client, payload, source, title="", period=None, publication_time=None):
    pages = extract_document(payload)
    record = {
        "source": source,
        "source_url": payload.url,
        "raw_sha256": payload.sha256,
        "ingested_at": payload.ingested_at,
        "title": title,
        "period": str(period) if period is not None else None,
        "publication_time": publication_time,
        "pages": pages,
    }
    folder = client.root / "documents" / source
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{payload.sha256}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


def report_links(client, page_url, pattern):
    page = client.get(page_url)
    soup = BeautifulSoup(page.content, "html.parser")
    links = [
        {"url": urljoin(page.url, a["href"]), "title": a.get_text(" ", strip=True)}
        for a in soup.select("a[href]")
        if re.search(pattern, a["href"], re.IGNORECASE)
    ]
    return list({item["url"]: item for item in links}.values())


def orfr_period(url):
    match = re.search(r"ORFR_(20\d{2})-(\d{1,2})(?:\.pdf|/|$)", url, re.IGNORECASE)
    if match and 1 <= int(match[2]) <= 12:
        return pd.Timestamp(int(match[1]), int(match[2]), 1)
    return None


def parse_orfr(payload, period, pages=None):
    pages = pages if pages is not None else extract_document(payload)
    patterns = [
        (
            "nonfinancial_net_fx_sales",
            "USD bn",
            r"(?:[Чч]истые продажи|[Нн]етто-продаж[иа]?)[^.]{0,120}нефинансов\w* компани[^.]{0,200}?(?:составил\w*|до)\s*(\d+(?:,\d+)?)\s*млрд\s*долл",
        ),
        (
            "exporter_net_fx_sales",
            "USD bn",
            r"(?:[Нн]етто-продаж\w*|[Чч]истые продажи)[^.]{0,150}экспортер[^.]{0,180}?(?:составил\w*|до)\s*(\d+(?:,\d+)?)\s*млрд\s*долл",
        ),
        (
            "household_net_fx_purchases",
            "RUB bn",
            r"[Нн]етто-покупки (?:иностранной )?валюты физическими лицами.{0,180}?(?:объем составил|составили)\s*(\d+(?:,\d+)?)\s*млрд\s*руб",
        ),
    ]
    frames = []
    for name, unit, pattern in patterns:
        matches = []
        for page in pages:
            text = re.sub(r"\s+", " ", page["text"])
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append((float(match[1].replace(",", ".")), page["page"], match[0]))
        buckets = {}
        for value, page, evidence in matches:
            cumulative = bool(re.search(r"с начала года|нарастающ|накопленн", evidence, re.I))
            key = name + "_ytd" if cumulative else name
            buckets.setdefault(key, []).append((value, page, evidence))
        for key, selected in buckets.items():
            if len({value for value, _, _ in selected}) == 1:
                value, page, evidence = selected[0]
                frame = observations([period], [value], key, unit, payload)
                frame["evidence"] = evidence
                frame["document_page"] = page
                frame["aggregation"] = "year_to_date" if key.endswith("_ytd") else "monthly"
                frame["extraction_method"] = "qualified_regex_v2"
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_orfr(client, start, end):
    links = report_links(
        client, "https://www.cbr.ru/analytics/finstab/orfr/", r"/ORFR_[^/]+(?:\.pdf|/)$"
    )
    selected = [(item, orfr_period(item["url"])) for item in links]
    selected = [
        (item, date)
        for item, date in selected
        if date is not None and pd.Timestamp(start) <= date <= pd.Timestamp(end)
    ]
    frames, manifest = [], []

    def process(entry):
        item, date = entry
        with Client(client.root, refresh=client.refresh) as local:
            payload = local.get(item["url"])
            if not payload.content.startswith(b"%PDF"):
                soup = BeautifulSoup(payload.content, "html.parser")
                pdf = next(
                    (
                        a
                        for a in soup.select("a[href]")
                        if re.search(r"ORFR_.*\.pdf", a["href"], re.IGNORECASE)
                    ),
                    None,
                )
                if pdf is None:
                    raise ValueError("ORFR: не найден PDF выпуска")
                payload = local.get(urljoin(payload.url, pdf["href"]))
            record = save_document(local, payload, "cbr_orfr", item["title"], date.date())
            frame = parse_orfr(payload, date, record["pages"])
            return record, frame

    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(process, entry): entry for entry in selected}
        for task in as_completed(tasks):
            item, date = tasks[task]
            try:
                record, frame = task.result()
                manifest.append(
                    {
                        "period": str(date.date()),
                        "url": record["source_url"],
                        "raw_sha256": record["raw_sha256"],
                        "extracted_rows": len(frame),
                    }
                )
                if not frame.empty:
                    frames.append(frame)
            except (ValueError, RuntimeError, OSError) as exc:
                manifest.append({"url": item["url"], "error": str(exc)})
    folder = Path(client.root) / "documents" / "cbr_orfr"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not frames:
        raise ValueError("ORFR: документы сохранены, однозначных числовых фрагментов не найдено")
    return pd.concat(frames, ignore_index=True)


def fetch_cbr_decisions(client, start, end):
    from usd_rub.sources.events import russian_date

    page = client.get("https://www.cbr.ru/dkp/mp_dec/")
    soup = BeautifulSoup(page.content, "html.parser")
    items = {}

    def collect(fragment):
        dates = []
        for day in fragment.select(".previews_day"):
            label = day.select_one(".previews_day-date")
            if label is None:
                continue
            date = russian_date(label.get_text(" ", strip=True))
            dates.append(date)
            for item in day.select(".previews_item"):
                link = item.select_one("a[href]")
                if link is None or not re.search(r"/press/pr/", link["href"], re.IGNORECASE):
                    continue
                if pd.Timestamp(start) <= date <= pd.Timestamp(end):
                    url = urljoin(page.url, link["href"])
                    time = item.select_one(".previews_item-time")
                    clock = time.get_text(strip=True) if time else ""
                    publication = (
                        str(
                            date.tz_localize("Europe/Moscow")
                            + pd.Timedelta(hours=int(clock[:2]), minutes=int(clock[3:5]))
                        )
                        if re.fullmatch(r"\d{2}:\d{2}", clock)
                        else None
                    )
                    items[url] = {
                        "url": url,
                        "date": str(date.date()),
                        "title": link.get_text(" ", strip=True),
                        "reported_publication_time": publication,
                    }
        return dates

    collect(soup)
    button = soup.select_one("button[data-cross-ajax-url]")
    next_url = urljoin(page.url, button["data-cross-ajax-url"]) if button else None
    visited = set()
    while next_url:
        if next_url in visited:
            raise ValueError("ЦБ: повтор страницы архива решений")
        visited.add(next_url)
        response = client.get(next_url)
        fragment = BeautifulSoup(response.content, "html.parser")
        dates = collect(fragment)
        if not dates or min(dates) < pd.Timestamp(start):
            break
        pointer = fragment.select_one("[data-cross-ajax-url]")
        next_url = urljoin(response.url, pointer["data-cross-ajax-url"]) if pointer else None

    def process(item):
        try:
            with Client(client.root, refresh=client.refresh) as local:
                payload = local.get(item["url"])
                record = save_document(
                    local,
                    payload,
                    "cbr_decisions",
                    item["title"],
                    item["date"],
                    item["reported_publication_time"],
                )
            return {**item, "raw_sha256": record["raw_sha256"], "status": "downloaded"}
        except (ValueError, RuntimeError, OSError) as exc:
            return {**item, "status": "error", "error": str(exc)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(process, items.values()))
    folder = client.root / "documents" / "cbr_decisions"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    return records
