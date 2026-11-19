"""Доступность наблюдений по сохранённым официальным публикациям."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from usd_rub.sources.events import russian_date


def restore_document_times(root):
    folder = Path(root) / "documents" / "cbr_decisions"
    manifest = folder / "manifest.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    by_url = {r["url"]: r.get("reported_publication_time") for r in records}
    restored = 0
    for path in folder.glob("*.json"):
        if path.name == "manifest.json":
            continue
        record = json.loads(path.read_text())
        publication = by_url.get(record.get("source_url"))
        if publication:
            record["publication_time"] = publication
            record["publication_time_basis"] = "reported_archive_time"
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            restored += 1
    return restored


def key_rate_releases(root):
    releases = []
    for path in (Path(root) / "documents" / "cbr_decisions").glob("*.json"):
        if path.name == "manifest.json":
            continue
        record = json.loads(path.read_text())
        publication = record.get("publication_time")
        title = record.get("title", "")
        text = re.sub(r"\s+", " ", " ".join(p["text"] for p in record.get("pages", [])))
        rate = re.search(r"(?:на\s+уровне|до)\s*(\d+(?:[,.]\d+)?)\s*%", title, re.I)
        if rate is None and "ключевой ставке" in title.lower():
            rate = re.search(
                r"ключевую ставку.{0,45}?(?:до|на\s+уровне)\s*(\d+(?:[,.]\d+)?)\s*%",
                text[:1000],
                re.I,
            )
        if not publication or not rate:
            continue
        match = re.search(
            r"\bс\s+(\d{1,2}\s+[а-я]+\s+20\d{2})\s*(?:года|г\.)?\s*(?:ключев|сниз|повыс|установ|состав)",
            text,
            re.I,
        )
        effective = russian_date(match[1]) if match else pd.Timestamp(record["period"])
        stamp = pd.Timestamp(publication)
        if match is None:
            candidates = []
            for value in re.findall(r"\b[cс]\s+(\d{2}\.\d{2}\.\d{2,4})(?!\d)", text, re.I):
                candidate = pd.to_datetime(
                    value, format="%d.%m.%y" if len(value) == 8 else "%d.%m.%Y"
                )
                if (
                    pd.Timestamp(record["period"])
                    <= candidate
                    <= pd.Timestamp(record["period"]) + pd.Timedelta(days=7)
                ):
                    candidates.append(candidate)
            if candidates:
                effective = min(candidates)
        if effective < stamp.tz_convert("Europe/Moscow").tz_localize(None).normalize():
            continue
        releases.append(
            {
                "effective": effective,
                "publication": stamp.tz_convert("UTC"),
                "value": float(rate[1].replace(",", ".")),
                "source_url": record["source_url"],
                "raw_sha256": record["raw_sha256"],
            }
        )
    return sorted(releases, key=lambda r: (r["effective"], r["publication"]))


def enrich_availability(frame, root):
    result = frame.copy()
    if "key_rate" not in set(result.series):
        return result
    releases = key_rate_releases(root)
    if not releases:
        return result
    selected = result.series.eq("key_rate")
    for index in result.index[selected]:
        row = result.loc[index]
        applicable = [r for r in releases if r["effective"] <= pd.Timestamp(row["date"])]
        if not applicable:
            continue
        release = applicable[-1]
        if not np.isclose(float(row["value"]), release["value"], atol=1e-10):
            continue
        result.loc[index, "available_at"] = release["publication"]
        result.loc[index, "availability_basis"] = "official_rate_decision"
        result.loc[index, "publication_source_url"] = release["source_url"]
        result.loc[index, "publication_raw_sha256"] = release["raw_sha256"]
    return result
