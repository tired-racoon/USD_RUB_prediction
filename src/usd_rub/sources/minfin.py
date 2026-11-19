"""Нефтегазовые доходы Минфина из исторических таблиц."""

import re
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from usd_rub.sources.tables import discover_table_links, parse_row_timeseries, period_date, workbook


def ytd_to_monthly(frame):
    pieces = []
    for _, part in frame.groupby("series"):
        part = part.sort_values("date").copy()
        if part["date"].dt.to_period("M").duplicated().any():
            raise ValueError("Накопленные суммы: несколько значений за один месяц")
        month_number = part["date"].dt.year * 12 + part["date"].dt.month
        previous = part["value"].shift()
        valid_previous = month_number.diff() == 1
        monthly = (part["value"] - previous).where(valid_previous)
        january = part["date"].dt.month == 1
        monthly.loc[january] = part.loc[january, "value"]
        part["value"] = monthly
        pieces.append(part.dropna(subset=["value"]))
    return pd.concat(pieces, ignore_index=True)


def parse_oil_gas_revenue(payload, budget=False):
    sheets = workbook(payload)
    results = []
    for sheet, table in sheets.items():
        text = table.head(15).astype(str).to_string(index=False, header=False).lower()
        if not re.search(r"млрд\.?\s*(?:руб|рублей)", text):
            continue
        matching_columns = [
            column
            for column in range(min(4, table.shape[1]))
            if table.iloc[:, column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.fullmatch(r"Нефтегазовые доходы\s*\*?", case=False)
            .any()
        ]
        if len(matching_columns) != 1:
            continue
        for row in range(min(15, len(table))):
            dates = pd.to_datetime(table.iloc[row].map(period_date))
            if (dates.notna() & dates.dt.year.between(1990, 2100)).sum() < 2:
                continue
            specs = (
                [
                    (r"Доходы, всего", "budget_revenue"),
                    (r"Расходы, всего", "budget_expenditure"),
                    (r"Дефицит \(-\)/Профицит \(\+\)", "budget_balance"),
                    (r"Ненефтегазовые доходы", "budget_non_oil_revenue"),
                    (r"НДС \(внутренний\)", "budget_domestic_vat"),
                    (r"Налог на прибыль", "budget_profit_tax"),
                ]
                if budget
                else [(r"Нефтегазовые доходы\s*\*?", "oil_gas_revenue")]
            )
            parsed = pd.concat(
                [
                    parse_row_timeseries(
                        payload,
                        sheet,
                        "(?i)" + label,
                        name,
                        "RUB bn",
                        date_row=row,
                        label_column=matching_columns[0],
                    )
                    for label, name in specs
                ],
                ignore_index=True,
            )
            if re.search(r"накоп|с начала года|нарастающ", text):
                raw = parsed.copy()
                raw["series"] += "_ytd"
                parsed = pd.concat([parsed.pipe(ytd_to_monthly), raw], ignore_index=True)
            results.append(parsed)
            break
    if not results:
        raise ValueError("Минфин: не распознан формат. Используйте import-table с явной схемой.")
    result = pd.concat(results, ignore_index=True)
    if result.groupby(["series", "date"])["value"].nunique().gt(1).any():
        raise ValueError("Минфин: конфликтующие значения; выберите конкретный лист")
    return result.drop_duplicates(["series", "date"])


def fetch_oil_gas_revenue(client, start, end, budget=False):
    page = client.get("https://minfin.gov.ru/ru/statistics/fedbud/")
    links = discover_table_links(page, r"нефтегазов.*доход|доход.*нефтегазов")
    soup = BeautifulSoup(page.content, "html.parser")
    document = next(
        (
            a
            for a in soup.select("a[href]")
            if "Краткая ежемесячная информация" in a.get_text(" ", strip=True)
        ),
        None,
    )
    if document is not None:
        doc = client.get(urljoin(page.url, document["href"]))
        for a in BeautifulSoup(doc.content, "html.parser").select("a[href]"):
            if re.search(r"\.xlsx?(?:\?|$)", a["href"], re.IGNORECASE):
                links.append({"url": urljoin(doc.url, a["href"])})
    links = list({item["url"]: item for item in links}.values())
    if not links:
        raise ValueError("Минфин: не найдены исторические таблицы нефтегазовых доходов")
    results = []
    errors = []
    for link in links:
        try:
            results.append(parse_oil_gas_revenue(client.get(link["url"]), budget=budget))
        except ValueError as exc:
            errors.append(str(exc))
    if not results:
        raise ValueError("; ".join(errors))
    return pd.concat(results, ignore_index=True)


def parse_fnb(payload):
    table = next(iter(workbook(payload).values()))
    dates = pd.to_datetime(table.iloc[2].map(period_date))
    if "млрд руб" not in " ".join(table.iloc[1].astype(str)).lower():
        raise ValueError("ФНБ: неизвестная единица измерения")
    specs = [
        ("Объем на начало периода", "fnb_opening_rub_bn"),
        ("Объем на конец периода", "fnb_closing_rub_bn"),
        ("Поступления", "fnb_inflows_rub_bn"),
        ("Изъятия", "fnb_withdrawals_rub_bn"),
        (
            "Курсовая разница от переоценки активов в иностранной валюте и золота",
            "fnb_revaluation_rub_bn",
        ),
    ]
    frames = []
    labels = table.iloc[:, 1].fillna("").astype(str).str.strip()
    valid = dates.notna()
    from usd_rub.storage import observations

    for label, name in specs:
        selected = table.loc[labels == label]
        if len(selected) != 1:
            raise ValueError(f"ФНБ: не найдена однозначная строка {label}")
        raw_values = selected.iloc[0][valid]
        values = raw_values.map(
            lambda v: re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰*]+$", "", v) if isinstance(v, str) else v
        )
        frame = observations(dates[valid], values, name, "RUB bn", payload)
        frame["source_value_labels"] = frame["date"].map(
            dict(zip(dates[valid], raw_values.astype(str), strict=True))
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def fetch_fnb(client, start, end):
    page = client.get("https://minfin.gov.ru/ru/perfomance/nationalwealthfund/")
    soup = BeautifulSoup(page.content, "html.parser")
    links = [
        urljoin(page.url, a["href"])
        for a in soup.select("a[href]")
        if re.search(r"\.xlsx?(?:\?|$)", a["href"], re.IGNORECASE)
    ]
    if not links:
        raise ValueError("ФНБ: не найдена актуальная таблица движения средств")
    return parse_fnb(client.get(links[0]))


def parse_budget_operations(payload):
    from usd_rub.sources.cbr_extra import find_row, row_values

    table = next(iter(workbook(payload).values()))
    dates = pd.to_datetime(table.iloc[3].map(period_date))
    specs = [
        (r"Налог на добычу полезных ископаемых", "ndpi_revenue_rub_bn"),
        (r"НДД", "ndd_revenue_rub_bn"),
        (r"Базовый месячный объем нефтегазовых доходов", "oil_gas_baseline_revenue_rub_bn"),
        (r"Дополнительные нефтегазовые доходы", "oil_gas_additional_revenue_rub_bn"),
        (
            r"Объем покупки \(продажи\) иностранной валюты и золота\*+",
            "minfin_fx_gold_operations_rub_bn",
        ),
    ]
    return pd.concat(
        [
            row_values(payload, table, find_row(table, label), dates, name, "RUB bn")
            for label, name in specs
        ],
        ignore_index=True,
    )


def fetch_budget_operations(client, start, end):
    page = client.get("https://minfin.gov.ru/ru/statistics/fedbud/")
    soup = BeautifulSoup(page.content, "html.parser")
    link = next(
        (
            a
            for a in soup.select("a[href]")
            if "формировании и использовании дополнительных нефтегазовых доходов"
            in a.get_text(" ", strip=True)
        ),
        None,
    )
    if link is None:
        raise ValueError("Минфин: не найден действующий документ бюджетных операций")
    page = client.get(urljoin(page.url, link["href"]))
    link = BeautifulSoup(page.content, "html.parser").select_one('a[href*=".xlsx"]')
    if link is None:
        raise ValueError("Минфин: не найден XLSX бюджетных операций")
    return parse_budget_operations(client.get(urljoin(page.url, link["href"])))
