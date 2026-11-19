"""Исходные ответы и нормализованные наблюдения."""

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
import numpy as np
import pandas as pd


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def public_url(url) -> str:
    parts = urlsplit(str(url))
    query = [(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in {"api_key", "token"}]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


@dataclass
class Payload:
    content: bytes
    url: str
    sha256: str
    ingested_at: str
    headers: dict = field(default_factory=dict)


class DownloadError(RuntimeError):
    pass


class Client:
    def __init__(self, root: str | Path = "data", refresh: bool = False, browser_fallback=False):
        self.root = Path(root)
        self.refresh = refresh
        self.browser_fallback = browser_fallback
        self.http = httpx.Client(
            timeout=60,
            follow_redirects=True,
            headers={"User-Agent": "USD-RUB-research/0.1 (historical data)"},
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.http.close()

    def get(self, url: str, params: dict | None = None) -> Payload:
        return self.request("GET", url, params=params)

    def request(self, method: str, url: str, **kwargs) -> Payload:
        request = self.http.build_request(method, url, **kwargs)
        varying = {
            name: request.headers[name]
            for name in ("accept", "x-requested-with", "range", "authorization")
            if name in kwargs.get("headers", {})
            or any(k.lower() == name for k in kwargs.get("headers", {}))
        }
        variant = json.dumps(varying, sort_keys=True).encode() if varying else b""
        key = hashlib.sha256(
            method.encode() + str(request.url).encode() + request.content + variant
        ).hexdigest()
        index = self.root / "raw" / "requests" / f"{key}.json"
        if index.exists() and not self.refresh:
            metadata = json.loads(index.read_text())
            blob = self.root / "raw" / "objects" / metadata["sha256"]
            content = blob.read_bytes()
            if hashlib.sha256(content).hexdigest() != metadata["sha256"]:
                raise DownloadError(f"Повреждён кэш: {blob}")
            return Payload(
                content,
                metadata["url"],
                metadata["sha256"],
                metadata["ingested_at"],
                metadata.get("headers", {}),
            )
        response = None
        transport_error = None
        for attempt in range(3):
            try:
                response = self.http.send(request)
                if response.status_code not in {429, 500, 502, 503, 504}:
                    break
            except httpx.TransportError as exc:
                transport_error = exc
                if attempt == 2:
                    break
            if attempt < 2:
                time.sleep(2**attempt)
        host = request.url.host
        browser_host = any(
            host == domain or host.endswith("." + domain)
            for domain in ("rosstat.gov.ru", "minfin.gov.ru")
        )
        if (
            self.browser_fallback
            and method == "GET"
            and browser_host
            and (response is None or response.status_code >= 400)
        ):
            from usd_rub.browser import browser_download

            content, final_url = browser_download(str(request.url))
            browser_request = httpx.Request("GET", final_url)
            response = httpx.Response(
                200,
                request=browser_request,
                content=content,
                headers={"X-Research-Transport": "Google Chrome public fetch"},
            )
        if response is None:
            raise DownloadError(f"{url}: {transport_error or 'нет ответа'}")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DownloadError(f"HTTP {response.status_code}: {url}") from exc
        digest = hashlib.sha256(response.content).hexdigest()
        ingested = utc_now()
        metadata = {
            "url": public_url(response.url),
            "sha256": digest,
            "ingested_at": ingested,
            "status": response.status_code,
            "headers": dict(response.headers),
            "method": method,
        }
        objects = self.root / "raw" / "objects"
        objects.mkdir(parents=True, exist_ok=True)
        blob = objects / digest
        if not blob.exists():
            blob.write_bytes(response.content)
        index.parent.mkdir(parents=True, exist_ok=True)
        history = self.root / "raw" / "history.jsonl"
        with history.open("a") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        index.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        return Payload(
            response.content, public_url(response.url), digest, ingested, dict(response.headers)
        )


def number(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().replace("\u00a0", "").replace("\u202f", "").replace(" ", "")
    text = text.replace("−", "-").replace(",", ".").replace("%", "")
    if text in {"", ".", "..", "...", "-", "—", "н/д", "nan"}:
        return np.nan
    try:
        result = float(text)
    except ValueError as exc:
        raise ValueError(f"Не число: {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"Неконечное число: {value!r}")
    return result


def observations(
    dates,
    values,
    series: str,
    unit: str,
    payload: Payload,
    availability: str = "unknown",
    lag_days: int = 0,
) -> pd.DataFrame:
    timestamps = [pd.Timestamp(x) for x in dates]
    if any(pd.isna(x) for x in timestamps):
        raise ValueError(f"Некорректная дата в ряду {series}")
    dates = pd.DatetimeIndex([x.tz_localize(None).normalize() for x in timestamps])
    values = [number(x) for x in values]
    frame = pd.DataFrame({"date": dates, "value": values}).dropna(subset=["value"])
    frame["series"] = series
    frame["unit"] = unit
    frame["source_url"] = payload.url
    frame["raw_sha256"] = payload.sha256
    frame["ingested_at"] = payload.ingested_at
    frame["vintage"] = payload.sha256
    frame["availability_basis"] = availability
    if availability == "unknown":
        frame["available_at"] = pd.Timestamp(payload.ingested_at)
    else:
        local_dates = pd.DatetimeIndex(frame["date"]).tz_localize("Europe/Moscow")
        frame["available_at"] = (local_dates + pd.Timedelta(days=lag_days)).tz_convert("UTC")
    duplicate = frame.duplicated(["date"], keep=False)
    if duplicate.any():
        check = frame.loc[duplicate].groupby("date")["value"].nunique()
        if (check > 1).any():
            raise ValueError(f"Противоречивые значения {series}: {check[check > 1].index.tolist()}")
        frame = frame.drop_duplicates("date")
    return frame.sort_values("date").reset_index(drop=True)


def save_series(frame: pd.DataFrame, root: str | Path):
    from usd_rub.freshness import ensure_allowed

    folder = Path(root) / "normalized"
    folder.mkdir(parents=True, exist_ok=True)
    for name, part in frame.groupby("series"):
        ensure_allowed(name)
        prices = part.unit.isin(["RUB/USD", "RUB/CNY", "RUB/unit", "RUB per 1000 USD"])
        if (prices & part.value.le(0)).any():
            raise ValueError(f"Неположительная валютная котировка: {name}")
        path = folder / f"{name}.csv"
        if path.exists():
            old = pd.read_csv(path, float_precision="round_trip")
            part = pd.concat([old, part.assign(date=part["date"].dt.strftime("%Y-%m-%d"))])
        bounds = part.groupby(["date", "vintage"])["value"].agg(["min", "max"])
        if (~np.isclose(bounds["min"], bounds["max"], rtol=1e-12, atol=1e-12)).any():
            raise ValueError(f"Разные значения при одном винтаже: {name}")
        part = part.sort_values(["date", "ingested_at"]).drop_duplicates(
            ["date", "vintage"], keep="first"
        )
        if name == "key_rate":
            from usd_rub.availability import enrich_availability

            part = enrich_availability(part.reset_index(drop=True), root)
        part.to_csv(path, index=False)


def read_series(root: str | Path, series: str) -> pd.DataFrame:
    path = Path(root) / "normalized" / f"{series}.csv"
    frame = pd.read_csv(path, parse_dates=["date"], float_precision="round_trip")
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, format="mixed")
    return frame


def coverage(frame: pd.DataFrame, start, end) -> dict:
    from usd_rub.freshness import coverage_gaps, freshness

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    unique = frame.sort_values("ingested_at").drop_duplicates(["series", "date"], keep="last")
    first, last = unique["date"].min(), unique["date"].max()
    gaps = coverage_gaps(frame.series.iloc[0], unique.date)
    missing_years = gaps["missing_calendar_years"]
    long_span = bool(first <= last - pd.DateOffset(years=10))
    frequency = unique["date"].sort_values().diff().dt.days.median()
    tolerance = max(15, int(frequency * 3)) if pd.notna(frequency) else 15
    full = bool(
        first <= start + pd.Timedelta(days=tolerance) and last >= end - pd.Timedelta(days=tolerance)
    )
    return {
        "rows": len(unique),
        "first": str(first.date()),
        "last": str(last.date()),
        "requested_start": str(start.date()),
        "requested_end": str(end.date()),
        "covers_requested_bounds": full,
        "history_span_at_least_10y": long_span,
        "history_at_least_10y": long_span and not missing_years and not gaps["missing_periods"],
        "missing_calendar_years": missing_years,
        "coverage_gaps": gaps,
        "continuous_checked_history": not missing_years and not gaps["missing_periods"],
        "freshness": freshness(frame, end),
        "max_gap_days": int(unique["date"].sort_values().diff().dt.days.max())
        if len(unique) > 1
        else 0,
    }


def input_fingerprints(paths):
    """Хеши файлов, от которых зависит производный результат."""
    return [
        {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for path in sorted(set(map(Path, paths)))
    ]
