"""Архив опубликованных действий OFAC с пагинацией."""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import parse_qs, urljoin, urlsplit

import pandas as pd
from bs4 import BeautifulSoup

from usd_rub.sources.events import extract_document
from usd_rub.storage import Client


def classify_action(row):
    """Правила классификации заголовка, без подсчёта сущностей по тексту."""
    title = row.get("title", "")
    relevant = bool(re.search(r"Russia|Ukraine", title, re.I))
    kinds = []
    for name, pattern in [
        ("removal", r"removal|delist"),
        ("license", r"licen[cs]"),
        ("designation", r"designations\b|designation(?!\s+removal)\b"),
        ("enforcement", r"settlement|enforcement|penalt"),
        ("amendment", r"update|correction|amend|change"),
    ]:
        if re.search(pattern, title, re.I):
            kinds.append(name)
    sectors = [
        name
        for name, pattern in [
            ("banking", r"bank|financial"),
            ("energy", r"oil|gas|energy|petroleum"),
            ("shipping", r"vessel|ship|tanker|maritime"),
            ("payments", r"payment|exchange|clearing"),
        ]
        if re.search(pattern, title, re.I)
    ]
    return {
        "russia_relevance": "title_explicit" if relevant else "needs_review",
        "action_types": kinds or ["other"],
        "sectors": sectors,
        "classification_method": "title_rules_v1",
        "classification_needs_review": True,
        "count_unit": "document, not entities",
        "effective_time_known": False,
    }


def action_events(rows, coverage_start=None, coverage_end=None):
    events = []
    seen = set()
    for row in rows:
        if row["source_url"] in seen:
            continue
        seen.add(row["source_url"])
        classification = classify_action(row)
        if classification["russia_relevance"] != "title_explicit":
            continue
        publication = (
            pd.Timestamp(row["date"], tz="America/New_York") + pd.DateOffset(days=1)
        ).tz_convert("UTC")
        key = "ofac_" + hashlib.sha256(row["source_url"].encode()).hexdigest()
        events.append(
            {
                "event_id": key,
                "event_key": key,
                "event_type": "ofac_russia_action_notice",
                "event_time": str(publication),
                "effective_time": str(publication),
                "publication_time": str(publication),
                "source_url": row["source_url"],
                "raw_sha256": row["raw_sha256"],
                "value": 1,
                "unit": "document",
                "evidence": row["title"],
                "availability_basis": "official_index_date_end_of_day_New_York",
                "effective_time_semantics": "announcement available; legal effective date not inferred",
                "action_types": json.dumps(classification["action_types"]),
                "sectors": json.dumps(classification["sectors"]),
                "coverage_start": coverage_start,
                "coverage_end": coverage_end,
            }
        )
    return pd.DataFrame(events)


def parse_actions(payload):
    soup = BeautifulSoup(payload.content, "html.parser")
    results = []
    for row in soup.select(".views-row"):
        links = [a for a in row.select("a[href]") if re.search(r"/recent-actions/\d{8}", a["href"])]
        if not links:
            continue
        link = links[0]
        text = row.get_text(" ", strip=True)
        match = re.search(r"[A-Z][a-z]+ \d{1,2}, 20\d{2}", text)
        if match is None:
            raise ValueError("OFAC: не найдена дата публикации в записи")
        date = pd.to_datetime(match.group(), format="%B %d, %Y")
        results.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "title": link.get_text(" ", strip=True),
                "source_url": urljoin(payload.url, link["href"]),
                "evidence": text,
                "publication_time_precision": "date_only",
                "raw_sha256": payload.sha256,
                "ingested_at": payload.ingested_at,
            }
        )
    next_link = soup.select_one('a[rel="next"]')
    current_page = int(parse_qs(urlsplit(payload.url).query).get("page", ["0"])[0])
    higher_page = any(
        int(parse_qs(urlsplit(link["href"]).query).get("page", ["0"])[0]) > current_page
        for link in soup.select(".usa-pagination a[href]")
    )
    header = soup.select_one(".view-header")
    count = re.search(
        r"Displaying\s+[\d,]+\s*-\s*([\d,]+)\s+of\s+([\d,]+)",
        header.get_text(" ", strip=True) if header else "",
    )
    more_results = bool(count and int(count[1].replace(",", "")) < int(count[2].replace(",", "")))
    return results, next_link is not None or higher_page or more_results


def fetch_actions(client, start, end, documents=False):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end:
        raise ValueError("Начало периода OFAC позже окончания")
    all_rows = []
    seen = set()
    seen_pages = set()
    for page in range(1000):
        payload = client.get(
            "https://ofac.treasury.gov/recent-actions",
            {
                "ra-start-date": start.strftime("%m/%d/%Y"),
                "ra-end-date": end.strftime("%m/%d/%Y"),
                "page": page,
            },
        )
        rows, has_next = parse_actions(payload)
        signature = tuple(row["source_url"] for row in rows)
        if rows and signature in seen_pages:
            raise ValueError("OFAC: сервер повторил страницу при пагинации")
        seen_pages.add(signature)
        if not rows and has_next:
            raise ValueError("OFAC: пустая страница с продолжением; изменился формат")
        for row in rows:
            if not start <= pd.Timestamp(row["date"]) <= end:
                raise ValueError("OFAC: сервер не применил фильтр дат")
            if row["source_url"] in seen:
                original = next(
                    item for item in all_rows if item["source_url"] == row["source_url"]
                )
                original.setdefault("index_occurrences", []).append(
                    {"date": row["date"], "title": row["title"]}
                )
                continue
            seen.add(row["source_url"])
            all_rows.append(row)
        if documents and page % 25 == 0:
            print(f"OFAC: прочитано страниц {page + 1}, записей {len(all_rows)}", flush=True)
        if not has_next:
            break
    else:
        raise ValueError("OFAC: превышен предел страниц")
    if documents:
        folder = client.root / "documents" / "ofac_archive"
        folder.mkdir(parents=True, exist_ok=True)

        def download(row):
            try:
                with Client(client.root, refresh=client.refresh) as local:
                    document = local.get(row["source_url"])
                row["document_raw_sha256"] = document.sha256
                row["pages"] = extract_document(document)
                row["classification"] = classify_action(row)
                (folder / f"{document.sha256}.json").write_text(
                    json.dumps(row, ensure_ascii=False, indent=2)
                )
            except (ValueError, RuntimeError, OSError) as exc:
                row["document_error"] = str(exc)
            return row

        with ThreadPoolExecutor(max_workers=4) as pool:
            tasks = [pool.submit(download, row) for row in all_rows]
            for index, task in enumerate(as_completed(tasks), 1):
                task.result()
                if index % 100 == 0:
                    print(f"OFAC: обработано текстов {index}/{len(all_rows)}", flush=True)
    return all_rows
