"""XML, SOAP и исторические таблицы Банка России."""

from datetime import date
from io import StringIO
from xml.etree import ElementTree as ET

import pandas as pd
from bs4 import BeautifulSoup

from usd_rub.storage import Client, observations

DAILY = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
SECINFO = "https://www.cbr.ru/secinfo/secinfo.asmx"
NS = "http://web.cbr.ru/"


def xml_root(content: bytes):
    root = ET.fromstring(content)
    for element in root.iter():
        element.tag = element.tag.split("}")[-1]
    fault = root.find(".//faultstring")
    if fault is not None:
        raise ValueError(f"SOAP: {fault.text}")
    return root


def soap(client: Client, service: str, operation: str, params: dict):
    envelope = ET.Element("{http://schemas.xmlsoap.org/soap/envelope/}Envelope")
    body = ET.SubElement(envelope, "{http://schemas.xmlsoap.org/soap/envelope/}Body")
    node = ET.SubElement(body, f"{{{NS}}}{operation}")
    for key, value in params.items():
        ET.SubElement(node, f"{{{NS}}}{key}").text = str(value)
    return client.request(
        "POST",
        service,
        content=ET.tostring(envelope, encoding="utf-8"),
        headers={"Content-Type": "text/xml; charset=utf-8", "SOAPAction": f'"{NS}{operation}"'},
    )


def parse_fx(payload, series: str):
    rows = xml_root(payload.content).findall(".//Record")
    return observations(
        [pd.to_datetime(r.attrib["Date"], format="%d.%m.%Y") for r in rows],
        [float(r.findtext("Value").replace(",", ".")) / float(r.findtext("Nominal")) for r in rows],
        series,
        "RUB/unit",
        payload,
        "effective_date_midnight",
    )


def fetch_fx(client, code: str, series: str, start, end):
    payload = client.get(
        "https://www.cbr.ru/scripts/XML_dynamic.asp",
        {
            "date_req1": pd.Timestamp(start).strftime("%d/%m/%Y"),
            "date_req2": pd.Timestamp(end).strftime("%d/%m/%Y"),
            "VAL_NM_RQ": code,
        },
    )
    return parse_fx(payload, series)


def parse_soap(payload, row_tag: str, date_tag: str, columns: dict):
    rows = xml_root(payload.content).findall(f".//{row_tag}")
    rows = [r for r in rows if r.find(date_tag) is not None]
    frames = []
    for tag, (series, unit) in columns.items():
        frames.append(
            observations(
                [r.findtext(date_tag) for r in rows],
                [r.findtext(tag) for r in rows],
                series,
                unit,
                payload,
            )
        )
    return pd.concat(frames, ignore_index=True)


SOAP_TABLES = {
    "repo_claims": (
        DAILY,
        "Repo_debt2XML",
        "fromDate",
        "ToDate",
        "RD",
        "Date",
        {
            "DebtTotal": ("cbr_repo_claims_total", "RUB mn"),
            "DebtTH": ("cbr_repo_claims_fine_tuning_auction", "RUB mn"),
            "DebtOCH": ("cbr_repo_claims_main_auction", "RUB mn"),
        },
    ),
    "correspondent_accounts": (
        DAILY,
        "OstatDynamicXML",
        "fromDate",
        "ToDate",
        "Ostat",
        "DateOst",
        {
            "InRuss": ("bank_correspondent_accounts_ru", "RUB mn"),
            "InMoscow": ("bank_correspondent_accounts_moscow", "RUB mn"),
        },
    ),
    "bank_deposits": (
        DAILY,
        "OstatDepoNewXML",
        "fromDate",
        "ToDate",
        "odn",
        "DT",
        {
            "TOTAL": ("bank_deposits_cbr_total", "RUB mn"),
            "OV_P": ("bank_deposits_cbr_overnight", "RUB mn"),
        },
    ),
    "deposit_rate": (
        DAILY,
        "DepoDynamicXML",
        "fromDate",
        "ToDate",
        "Depo",
        "DateDepo",
        {"Overnight": ("cbr_deposit_overnight_rate", "% p.a.")},
    ),
    "key_rate": (
        DAILY,
        "KeyRate",
        "fromDate",
        "ToDate",
        "KR",
        "DT",
        {
            "Rate": ("key_rate", "% p.a."),
        },
    ),
    "ruonia": (
        DAILY,
        "RuoniaXML",
        "fromDate",
        "ToDate",
        "ro",
        "D0",
        {
            "ruo": ("ruonia", "% p.a."),
            "vol": ("ruonia_volume", "RUB bn"),
        },
    ),
    "ofz": (
        SECINFO,
        "zcyc_params",
        "FromDate",
        "ToDate",
        "ZCYC",
        "D0",
        {f"v_{year}_0": (f"ofz_{year}y", "% p.a.") for year in (1, 2, 5, 10)},
    ),
    "liquidity_operations": (
        DAILY,
        "SaldoXML",
        "fromDate",
        "ToDate",
        "So",
        "Dt",
        {
            "DEADLINEBS": ("liquidity_operations", "RUB bn"),
        },
    ),
    "reserves": (
        DAILY,
        "mrrf7DXML",
        "fromDate",
        "ToDate",
        "mr",
        "D0",
        {
            "val": ("international_reserves", "USD bn"),
        },
    ),
    "otc": (
        SECINFO,
        "VKO",
        "DateFrom",
        "DateTo",
        "VK",
        "D0",
        {
            f"{prefix}_{currency}": (f"otc_{currency.lower()}_{label}", "RUB bn")
            for prefix, label in [("FULL", "total"), ("KO", "interbank")]
            for currency in ("USD", "EUR", "CNY")
        },
    ),
}


def fetch_soap_table(client, name: str, start, end):
    service, op, date_from, date_to, row, dt, columns = SOAP_TABLES[name]
    frames = []
    cursor, end = pd.Timestamp(start), pd.Timestamp(end)
    while cursor <= end:
        till = min(cursor + pd.DateOffset(years=1) - pd.Timedelta(days=1), end)
        payload = soap(
            client,
            service,
            op,
            {
                date_from: cursor.strftime("%Y-%m-%dT00:00:00"),
                date_to: till.strftime("%Y-%m-%dT00:00:00"),
            },
        )
        frames.append(parse_soap(payload, row, dt, columns))
        cursor = till + pd.Timedelta(days=1)
    return pd.concat(frames, ignore_index=True)


def parse_html_table(payload, column_map: dict, date_column: int = 0):
    tables = pd.read_html(
        StringIO(payload.content.decode("utf-8-sig")), flavor="lxml", thousands=None, decimal=","
    )
    frames = []
    for table in tables:
        if len(table.columns) <= max(column_map):
            continue
        dates = pd.to_datetime(table.iloc[:, date_column], format="%d.%m.%Y", errors="coerce")
        valid = dates.notna()
        if not valid.any():
            continue
        for column, (series, unit) in column_map.items():
            frames.append(
                observations(
                    dates[valid],
                    table.loc[valid].iloc[:, column],
                    series,
                    unit,
                    payload,
                )
            )
        break
    if not frames:
        raise ValueError("Не найдена таблица с датами; формат источника изменился")
    return pd.concat(frames, ignore_index=True)


def fetch_liquidity(client, start, end):
    payload = client.get(
        "https://www.cbr.ru/hd_base/bliquidity/",
        {
            "UniDbQuery.Posted": "True",
            "UniDbQuery.From": pd.Timestamp(start).strftime("%d.%m.%Y"),
            "UniDbQuery.To": pd.Timestamp(end).strftime("%d.%m.%Y"),
        },
    )
    soup = BeautifulSoup(payload.content, "html.parser")
    header = " ".join(t.get_text(" ", strip=True) for t in soup.select("table th"))
    tables = pd.read_html(
        StringIO(payload.content.decode("utf-8-sig")), flavor="lxml", thousands=None, decimal=","
    )
    for table in tables:
        labels = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in table.columns]
        matches = [
            i
            for i, label in enumerate(labels)
            if "Дефицит (+)/ профицит (-) ликвидности" in label and "без учета" not in label
        ]
        if matches:
            return parse_html_table(payload, {matches[0]: ("structural_liquidity", "RUB bn")})
    raise ValueError(f"Не найдена колонка структурной ликвидности: {header[:200]}")


def validate_range(start, end):
    start, end = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    if start > end:
        raise ValueError("Дата начала позже даты окончания")
    return start, end


def fetch_fx_operations(client, start, end):
    payload = client.get(
        "https://www.cbr.ru/hd_base/valintbr/valvol_day/",
        {
            "UniDbQuery.Posted": "True",
            "UniDbQuery.From": pd.Timestamp(start).strftime("%d.%m.%Y"),
            "UniDbQuery.To": pd.Timestamp(end).strftime("%d.%m.%Y"),
        },
    )
    return parse_html_table(payload, {1: ("cbr_fx_net_purchases", "USD mn")})
