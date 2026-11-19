"""Дополнительные действующие таблицы Банка России."""

import re
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from usd_rub.sources.cbr import parse_html_table
from usd_rub.sources.tables import period_date, workbook
from usd_rub.storage import observations

BOP_URL = "https://www.cbr.ru/vfs/statistics/credit_statistics/bop/bal_of_payments_ap.xlsx"
BUSINESS_URL = "https://www.cbr.ru/Content/Document/File/135603/mp_survey_data.xlsx"


def row_values(payload, table, row, dates, name, unit):
    valid = dates.notna()
    return observations(dates[valid], table.iloc[row][valid], name, unit, payload)


def find_row(table, label, column=0, occurrence=0):
    labels = table.iloc[:, column].fillna("").astype(str).str.strip()
    matches = labels.loc[labels.str.fullmatch(label, case=False)].index.tolist()
    if len(matches) <= occurrence:
        raise ValueError(f"Не найдена строка {label}, вхождение {occurrence}")
    return matches[occurrence]


def parse_bop(payload):
    table = workbook(payload)["Кварталы"]
    header = next(
        (
            i
            for i in range(10)
            if table.iloc[i].astype(str).str.contains(r"[1-4] квартал 20\d{2}").any()
        ),
        None,
    )
    if header is None:
        raise ValueError("ПБ: не найден квартальный заголовок")

    def quarter(value):
        match = re.search(r"([1-4])\s+квартал\s+(\d{4})", str(value))
        if match:
            return pd.Period(f"{match[2]}Q{match[1]}", freq="Q").end_time.normalize()
        return pd.NaT

    dates = pd.to_datetime(table.iloc[header].map(quarter))
    specs = [
        ("Счет текущих операций", "current_account_quarterly", 0),
        ("Товары", "bop_goods_balance_quarterly", 0),
        ("Экспорт", "bop_goods_exports_quarterly", 0),
        ("Импорт", "bop_goods_imports_quarterly", 0),
        ("Услуги", "bop_services_balance_quarterly", 0),
        ("Экспорт", "bop_services_exports_quarterly", 1),
        ("Импорт", "bop_services_imports_quarterly", 1),
        ("Инвестиционные доходы", "bop_investment_income_quarterly", 0),
        ("Вторичные доходы", "bop_secondary_income_quarterly", 0),
        ("Счет операций с капиталом", "bop_capital_account_quarterly", 0),
        ("Сальдо финансового счета", "bop_financial_account_quarterly", 0),
        ("Прочая задолженность", "bop_other_receivables_assets_quarterly", 0),
        ("Прочая задолженность", "bop_other_receivables_liabilities_quarterly", 1),
        ("Резервные активы", "bop_reserve_assets_flow_quarterly", 0),
    ]
    frames = [
        row_values(payload, table, find_row(table, label, occurrence=occ), dates, name, "USD mn")
        for label, name, occ in specs
    ]
    for i, label in table.iloc[:, 0].items():
        if re.search(
            r"^Сальдо первичных доходов|^Сальдо вторичных доходов|^Финансовый счет|^Чистые ошибки",
            str(label),
            re.IGNORECASE,
        ):
            names = {
                "Сальдо первичных доходов": "bop_primary_income",
                "Сальдо вторичных доходов": "bop_secondary_income",
            }
            name = next(
                (v for k, v in names.items() if str(label).startswith(k)),
                "bop_financial_account"
                if str(label).startswith("Финансовый")
                else "bop_errors_omissions",
            )
            frames.append(row_values(payload, table, i, dates, name + "_quarterly", "USD mn"))
    return pd.concat(frames, ignore_index=True)


def fetch_bop(client, start, end):
    return parse_bop(client.get(BOP_URL))


def parse_bop_estimate(payload):
    from usd_rub.sources.events import russian_date

    table = next(iter(workbook(payload).values()))
    text = table.fillna("").astype(str).to_string(index=False, header=False)
    if not re.search(r"млрд\s+долларов", text, re.I):
        raise ValueError("Оценка ПБ: неизвестные единицы")
    publication = re.search(
        r"Дата последнего обновления:\s*(\d{1,2}\s+[а-я]+\s+20\d{2})", text, re.I
    )
    if not publication:
        raise ValueError("Оценка ПБ: отсутствует дата обновления")
    stamp = (
        russian_date(publication[1]).tz_localize("Europe/Moscow") + pd.Timedelta(days=1)
    ).tz_convert("UTC")
    header = next(
        i
        for i in range(min(10, len(table)))
        if table.iloc[i].astype(str).str.contains(r"I\s+квартал", regex=True).any()
    )
    dates, frequencies, columns = [], [], []
    for column in range(1, table.shape[1]):
        label = re.sub(r"\s+", " ", str(table.iloc[header, column]))
        if "Справочно" in label:
            break
        match = re.search(r"\b(IV|III|II|I)\s+квартал\s+(20\d{2})", label)
        if match:
            dates.append(
                pd.Period(
                    f"{match[2]}Q{['I', 'II', 'III', 'IV'].index(match[1]) + 1}", freq="Q"
                ).end_time.normalize()
            )
            frequencies.append("quarterly")
            columns.append(column)
        elif not re.search(r"[-–]", label):
            month = re.match(r"([а-я]+)\s+(20\d{2})", label, re.I)
            date = period_date(month[0]) if month else pd.NaT
            if pd.notna(date):
                dates.append(date)
                frequencies.append("monthly")
                columns.append(column)
    frames = []
    for label, name, occurrence in [
        ("Счет текущих операций", "current_account", 0),
        ("Товары", "goods_balance", 0),
        ("Экспорт", "goods_exports", 0),
        ("Импорт", "goods_imports", 0),
        ("Услуги", "services_balance", 0),
        ("Экспорт", "services_exports", 1),
        ("Импорт", "services_imports", 1),
    ]:
        row = find_row(table, label, occurrence=occurrence)
        for frequency in ["monthly", "quarterly"]:
            chosen = [i for i, value in enumerate(frequencies) if value == frequency]
            if not chosen:
                continue
            part = observations(
                [dates[i] for i in chosen],
                [table.iloc[row, columns[i]] for i in chosen],
                f"bop_estimate_{name}_{frequency}",
                "USD bn",
                payload,
            )
            part["available_at"] = stamp
            part["availability_basis"] = "reported_source_update_date_end_of_day"
            part["estimate_status"] = "separate_early_estimate_current_vintage"
            frames.append(part)
    if not frames:
        raise ValueError("Оценка ПБ: нет распознанных периодов")
    return pd.concat(frames, ignore_index=True)


def fetch_bop_estimate(client, start, end):
    return parse_bop_estimate(
        client.get("https://www.cbr.ru/vfs/statistics/credit_statistics/bop/bop_est.xlsx")
    )


def parse_business(payload):
    table = workbook(payload)["Экономика всего"]
    dates = pd.to_datetime(table.iloc[:, 0].map(period_date))
    valid = dates.notna()
    headers = table.iloc[3].fillna("").astype(str)
    frames = []
    patterns = [
        (r"^Индикатор бизнес-климата Банка России$", "business_climate"),
        (r"^Ценовые ожидания предприятий", "corporate_price_expectations"),
        (r"Как изменились.*отпускные.*цены|Как изменились.*цены", "corporate_prices"),
        (r"Как изменится.*объем производства", "corporate_output_expectations"),
        (r"Как изменится.*спрос", "corporate_demand_expectations"),
    ]
    for pattern, name in patterns:
        matches = [i for i, text in headers.items() if re.search(pattern, text, re.IGNORECASE)]
        for occurrence, column in enumerate(matches):
            suffix = "_sa" if occurrence else "_raw"
            frames.append(
                observations(
                    dates[valid],
                    table.loc[valid].iloc[:, column],
                    name + suffix,
                    "balance points",
                    payload,
                )
            )
    if not frames:
        raise ValueError("Мониторинг предприятий: изменились заголовки")
    return pd.concat(frames, ignore_index=True)


def fetch_business(client, start, end):
    return parse_business(client.get(BUSINESS_URL))


def parse_expectations(payload):
    table = workbook(payload)["Данные за все годы"]
    dates = pd.to_datetime(table.iloc[1].map(period_date))
    frames = []
    for label, name, occurrence in [
        (r"наблюдаемая инфляция \(в %\)", "household_inflation_observed", 0),
        (r"ожидаемая инфляция \(в %\)", "household_inflation_expected_1y", 0),
        (r"ожидаемая инфляция \(в %\)", "household_inflation_expected_5y", 1),
    ]:
        frames.append(
            row_values(
                payload, table, find_row(table, label, occurrence=occurrence), dates, name, "% p.a."
            )
        )
    return pd.concat(frames, ignore_index=True)


def fetch_expectations(client, start, end):
    page = client.get("https://www.cbr.ru/statistics/ddkp/inflationary_expectations/")
    soup = BeautifulSoup(page.content, "html.parser")
    links = [
        urljoin(page.url, a["href"])
        for a in soup.select("a[href]")
        if re.search(r"Infl_exp_\d{2}-\d{2}\.xlsx", a["href"], re.IGNORECASE)
    ]
    if not links:
        raise ValueError("Не найден актуальный XLSX инфляционных ожиданий")
    return parse_expectations(client.get(links[0]))


def fetch_cny_funding(client, start, end):
    payload = client.get(
        "https://www.cbr.ru/hd_base/ImpliedRateCNY/",
        {
            "UniDbQuery.Posted": "True",
            "UniDbQuery.From": pd.Timestamp(start).strftime("%d.%m.%Y"),
            "UniDbQuery.To": pd.Timestamp(end).strftime("%d.%m.%Y"),
        },
    )
    return parse_html_table(payload, {1: ("cny_implied_overnight_rate", "% p.a.")})


def parse_settlements(payload):
    sheets = workbook(payload)
    frames = []
    for sheet, prefix in [("ЭКСПОРТ месяцы", "export"), ("ИМПОРТ месяцы", "import")]:
        table = sheets[sheet]
        dates = pd.to_datetime(table.iloc[1].map(period_date))
        if str(table.iloc[2, 0]).strip() != "Всего":
            raise ValueError("Изменилась структура валютных расчётов")
        for row, suffix in [(2, "rub"), (3, "listed_states"), (4, "other")]:
            frames.append(
                row_values(payload, table, row, dates, f"{prefix}_settlement_share_{suffix}", "%")
            )
    return pd.concat(frames, ignore_index=True)


def fetch_settlements(client, start, end):
    return parse_settlements(
        client.get("https://www.cbr.ru/vfs/statistics/credit_statistics/cur_str_new.xlsx")
    )
