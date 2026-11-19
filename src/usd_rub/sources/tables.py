"""Исторические XLS/XLSX/HTML-таблицы официальной статистики."""

import hashlib
import re
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from usd_rub.storage import Payload, observations, utc_now

MONTHS = {
    name: i
    for i, name in enumerate(
        [
            "январь",
            "февраль",
            "март",
            "апрель",
            "май",
            "июнь",
            "июль",
            "август",
            "сентябрь",
            "октябрь",
            "ноябрь",
            "декабрь",
        ],
        1,
    )
}


def period_date(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value)
    if not isinstance(value, str):
        return pd.NaT
    text = re.sub(r"\s*\*+\s*$", "", value.strip().lower())
    short = re.fullmatch(r"([а-я]{3,})\.?\s*(\d{2}|\d{4})", text)
    if short:
        months = {name[:3]: month for name, month in MONTHS.items()}
        label, year = short.groups()
        if label[:3] in months:
            year = int(year) + (2000 if len(year) == 2 else 0)
            return pd.Timestamp(year, months[label[:3]], 1)
    return pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=True)


def load_payload(client, path_or_url: str):
    if path_or_url.startswith(("http://", "https://")):
        return client.get(path_or_url)
    path = Path(path_or_url).resolve()
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    objects = client.root / "raw" / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    if not (objects / digest).exists():
        (objects / digest).write_bytes(content)
    return Payload(content, str(path), digest, utc_now())


def workbook(payload):
    if payload.content[:2] == b"PK":
        return pd.read_excel(
            BytesIO(payload.content), sheet_name=None, header=None, engine="openpyxl"
        )
    if payload.content[:4] == b"\xd0\xcf\x11\xe0":
        return pd.read_excel(BytesIO(payload.content), sheet_name=None, header=None, engine="xlrd")
    text = payload.content.decode("utf-8-sig")
    if "<table" in text.lower():
        results = {}
        for i, table in enumerate(
            pd.read_html(StringIO(text), header=None, flavor="lxml", thousands=None, decimal=",")
        ):
            if not isinstance(table.columns, pd.RangeIndex):
                headers = [
                    list(table.columns.get_level_values(level))
                    for level in range(table.columns.nlevels)
                ]
                table = pd.concat(
                    [pd.DataFrame(headers), pd.DataFrame(table.to_numpy())], ignore_index=True
                )
            results[str(i)] = table
        return results
    raise ValueError("Ожидалась таблица XLS, XLSX или HTML")


def parse_row_timeseries(
    payload,
    sheet: str,
    row_pattern: str,
    name: str,
    unit: str,
    date_row: int = 0,
    label_column: int = 0,
):
    sheets = workbook(payload)
    table = sheets[sheet]
    labels = table.iloc[:, label_column].fillna("").astype(str).str.strip()
    selected = labels.str.fullmatch(row_pattern)
    if selected.sum() != 1:
        raise ValueError(f"Ожидалась одна строка {row_pattern!r}, найдено {selected.sum()}")
    dates = pd.to_datetime(table.iloc[date_row].map(period_date))
    valid = dates.notna() & dates.dt.year.between(1990, 2100)
    values = table.loc[selected].iloc[0, :]
    result = observations(dates[valid], values[valid], name, unit, payload)
    labels = dict(zip(dates[valid], table.iloc[date_row][valid].astype(str), strict=True))
    result["source_period_label"] = result["date"].map(labels)
    return result


def fetch_money(client, start, end):
    payload = client.get("https://www.cbr.ru/vfs/statistics/credit_statistics/monetary_agg.xlsx")
    frames = []
    for pattern, name in [
        ("Денежный агрегат М2", "m2"),
        (r"Денежный агрегат M2X\s*\(широкая денежная масса\)", "m2x"),
    ]:
        frames.append(parse_row_timeseries(payload, "Денежные агрегаты", pattern, name, "RUB bn"))
    return pd.concat(frames, ignore_index=True)


def fetch_trade(client, start, end):
    payload = client.get("https://www.cbr.ru/vfs/statistics/credit_statistics/trade/trade.xls")
    table = workbook(payload)["Ежемесячные"]
    labels = [str(v).lower() for v in table.iloc[5]]
    if "всего" not in labels[2] or "всего" not in labels[8]:
        raise ValueError("Изменилась структура таблицы внешней торговли")
    dates, exports, imports = [], [], []
    short_months = {m[:3]: i for m, i in MONTHS.items()}
    for _, row in table.iterrows():
        year, month = str(row.iloc[0]).strip(), str(row.iloc[1]).strip().lower()[:3]
        if re.fullmatch(r"(?:19|20)\d{2}(?:\.0)?", year) and month in short_months:
            dates.append(pd.Timestamp(int(float(year)), short_months[month], 1))
            exports.append(row.iloc[2])
            imports.append(row.iloc[8])
    frames = [
        observations(dates, values, name, "USD mn", payload)
        for name, values in [("exports", exports), ("imports", imports)]
    ]
    balance = frames[0].copy()
    balance["series"] = "trade_balance"
    balance["value"] = frames[0]["value"].to_numpy() - frames[1]["value"].to_numpy()
    return pd.concat([*frames, balance], ignore_index=True)


def discover_table_links(payload, pattern: str):
    soup = BeautifulSoup(payload.content, "html.parser")
    results = []
    for link in soup.select("a[href]"):
        container = link.find_parent(class_="document-list__item")
        context = (container or link).get_text(" ", strip=True)
        if not context:
            context = link.parent.get_text(" ", strip=True)
        href = urljoin(payload.url, link["href"])
        if re.search(pattern, context, re.I) and re.search(r"\.(xlsx?|html?)(?:\?|$)", href, re.I):
            results.append({"url": href, "title": context})
    return list({entry["url"]: entry for entry in results}.values())


def parse_month_matrix(
    payload, name: str, unit: str, sheet: str | None = None, header_pattern: str | None = None
):
    results = []
    for sheet_name, table in workbook(payload).items():
        if sheet is not None and str(sheet_name) != sheet:
            continue
        if header_pattern is not None:
            text = table.astype(str).to_string(index=False, header=False)
            if not re.search(header_pattern, text, re.I):
                continue
        for row_idx in range(len(table) - 1):
            row = table.iloc[row_idx]
            year_cols = {}
            for col_idx, value in enumerate(row):
                if re.fullmatch(r"(?:19|20)\d{2}(?:\.0|\s*г\.?)?", str(value).strip()):
                    year_cols[col_idx] = int(float(re.sub(r"\s*г\.?$", "", str(value))))
            if len(year_cols) < 2:
                continue
            pairs = []
            seen_months = set()
            for _, data in table.iloc[row_idx + 1 : row_idx + 17].iterrows():
                labels = [str(v).strip().lower() for v in data.iloc[: min(year_cols)]]
                month = next((MONTHS[label] for label in labels if label in MONTHS), None)
                if month is None:
                    continue
                if month in seen_months:
                    break
                seen_months.add(month)
                for column, year in year_cols.items():
                    pairs.append((pd.Timestamp(year, month, 1), data.iloc[column]))
            if pairs:
                results.append(
                    observations(
                        [d for d, _ in pairs],
                        [v for _, v in pairs],
                        name,
                        unit,
                        payload,
                    )
                )
        if results:
            break
    if not results:
        raise ValueError("Не найдена матрица: годы в столбцах, месяцы в строках")
    result = pd.concat(results, ignore_index=True)
    conflicts = result.groupby("date")["value"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Найдено несколько разных таблиц: укажите лист и заголовок")
    return result.drop_duplicates("date")


def parse_long_table(payload, config: dict):
    sheets = workbook(payload)
    key = config.get("sheet", next(iter(sheets)))
    table = sheets[key]
    date_col = int(config["date_column"])
    dates = pd.to_datetime(
        table.iloc[:, date_col],
        format=config.get("date_format", "mixed"),
        errors="coerce",
        dayfirst=True,
    )
    dates = dates.where((dates.dt.year >= 1990) & (dates.dt.year <= 2100))
    valid = dates.notna()
    frames = []
    for column, spec in config["columns"].items():
        result = observations(
            dates[valid],
            table.loc[valid].iloc[:, int(column)],
            spec["series"],
            spec["unit"],
            payload,
        )
        result["value"] *= spec.get("scale", 1)
        frames.append(result)
    if not frames or not any(len(f) for f in frames):
        raise ValueError("Нет наблюдений в заданных столбцах")
    return pd.concat(frames, ignore_index=True)


def fetch_rosstat(client, config, start, end):
    page = client.get(config["page"])
    links = discover_table_links(page, config["pattern"])
    if not links:
        raise ValueError(f"Росстат: не найдена историческая таблица {config['pattern']}")
    frames = []
    errors = []
    for link in links:
        try:
            payload = client.get(link["url"])
            frames.append(
                parse_month_matrix(
                    payload,
                    config["series"],
                    config["unit"],
                    sheet=config.get("sheet"),
                    header_pattern=config.get("header_pattern"),
                )
            )
        except (ValueError, KeyError) as exc:
            errors.append(f"{link['url']}: {exc}")
    if not frames:
        raise ValueError("; ".join(errors))
    return pd.concat(frames, ignore_index=True)


def fetch_zenodo(client, start, end):
    import json

    metadata = client.get("https://zenodo.org/api/records/20538624")
    record = json.loads(metadata.content)
    file = next(f for f in record["files"] if f["key"].endswith(".xls"))
    payload = client.get(file["links"]["self"])
    algorithm, digest = file["checksum"].split(":", 1)
    if hashlib.new(algorithm, payload.content).hexdigest() != digest:
        raise ValueError("Zenodo: контрольная сумма не совпала")
    sheets = workbook(payload)
    table = next(iter(sheets.values()))
    dates = pd.to_datetime(table.iloc[1:, 0], errors="coerce")
    valid = dates.notna()
    frames = []
    for i, column in enumerate(table.iloc[0, 1:], 1):
        name = re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")
        frames.append(
            observations(
                dates[valid],
                table.iloc[1:, i].loc[valid],
                f"zenodo_{i}_{name}",
                "unverified",
                payload,
            )
        )
    return pd.concat(frames, ignore_index=True)


def parse_period_rows(
    payload,
    columns: dict,
    sheet: str | None = None,
    label_column: int = 0,
    year_suffix_pattern=None,
):
    table = workbook(payload)
    table = table[sheet] if sheet else next(iter(table.values()))
    year = None
    pairs = []
    for _, row in table.iterrows():
        label = str(row.iloc[label_column]).strip().lower()
        if year_suffix_pattern is not None:
            label = re.sub(r"^(20\d{2})" + year_suffix_pattern + r"$", r"\1", label)
        if re.fullmatch(r"(?:19|20)\d{2}(?:\.0|\s*г\.?)?", label):
            year = int(float(re.sub(r"\s*г\.?$", "", label)))
        elif year and label in MONTHS:
            pairs.append((pd.Timestamp(year, MONTHS[label], 1), row))
        elif isinstance(row.iloc[label_column], (datetime, pd.Timestamp)):
            pairs.append((pd.Timestamp(row.iloc[label_column]), row))
    if not pairs:
        raise ValueError("Не найдены строки год/месяц")
    return pd.concat(
        [
            observations(
                [d for d, _ in pairs],
                [r.iloc[int(col)] for _, r in pairs],
                spec["series"],
                spec["unit"],
                payload,
            )
            for col, spec in columns.items()
        ],
        ignore_index=True,
    )
