"""Форматы действующих источников и защита от неверной истории."""

import hashlib
from io import BytesIO

import httpx
import pandas as pd
import pytest

from usd_rub.audit import audit_data
from usd_rub.catalog import SOURCES, fetch
from usd_rub.features import build_features
from usd_rub.freshness import require_current
from usd_rub.sources.cbr_extra import parse_bop, parse_expectations
from usd_rub.sources.china import parse_pmi
from usd_rub.sources.commodities import parse_pink_sheet
from usd_rub.sources.futures import contract_info, front_windows
from usd_rub.sources.minfin import parse_fnb, parse_oil_gas_revenue
from usd_rub.sources.ofac import fetch_actions
from usd_rub.sources.rosstat_extra import parse_weekly_items
from usd_rub.sources.sanctions import parse_bis, parse_eu, parse_ofac_list, parse_uk
from usd_rub.sources.tables import parse_period_rows
from usd_rub.sources.tax_calendar import parse_calendar, release_date
from usd_rub.sources.urals import parse_urals
from usd_rub.storage import Client, Payload, coverage, observations, save_series


def payload(content, url="https://example.org/data", headers=None):
    content = content.encode() if isinstance(content, str) else content
    return Payload(
        content, url, hashlib.sha256(content).hexdigest(), "2026-11-18T10:00:00Z", headers or {}
    )


def xlsx(rows, sheet="Data"):
    buffer = BytesIO()
    pd.DataFrame(rows).to_excel(buffer, index=False, header=False, sheet_name=sheet)
    return payload(buffer.getvalue())


@pytest.mark.parametrize(
    "name", ["gdelt_ru_event_count", "otc_usd_total", "liquidity_operations", "zenodo_x"]
)
def test_excluded_series_cannot_enter_saved_data_or_features(tmp_path, name):
    frame = observations(["2026-01-01"], [1], name, "unit", payload("x"))
    with pytest.raises(ValueError, match="исключён"):
        save_series(frame, tmp_path)
    with pytest.raises(ValueError, match="исключён"):
        build_features({name: frame}, "2026-01-01", "2026-01-02")


def test_current_policy_rejects_stale_subseries_and_disabled_source():
    old = observations(["2025-01-01"], [1], "old", "unit", payload("x"))
    new = observations(["2026-01-01"], [1], "new", "unit", payload("x"))
    require_current(new, as_of="2026-01-01")
    with pytest.raises(ValueError, match="old"):
        require_current(pd.concat([old, new]), as_of="2026-01-01")
    with pytest.raises(ValueError, match="исключён"):
        fetch(SOURCES["cbr_otc"], None, "2015-01-01", "2026-11-18")


def test_ten_year_span_does_not_hide_missing_calendar_year(tmp_path):
    dates = [f"{year}-01-01" for year in range(2015, 2027) if year != 2017]
    frame = observations(dates, [1] * len(dates), "calendar", "event", payload("x"))
    result = coverage(frame, "2015-01-01", "2026-01-01")
    assert result["history_span_at_least_10y"]
    assert not result["history_at_least_10y"]
    assert result["missing_calendar_years"] == [2017]
    save_series(frame, tmp_path)
    raw = tmp_path / "raw" / "objects"
    raw.mkdir(parents=True)
    (raw / payload("x").sha256).write_bytes(b"x")
    assert (
        audit_data(tmp_path, as_of="2026-01-01")["series"][0]["status"]
        == "active_long_history_with_gaps"
    )


def test_ajax_and_html_have_separate_cache_and_preserve_headers(tmp_path):
    calls = []

    def respond(request):
        calls.append(request.headers.get("x-requested-with", "html"))
        return httpx.Response(
            200, text=calls[-1], headers={"Last-Modified": "Wed, 18 Nov 2026 08:00:00 GMT"}
        )

    with Client(tmp_path) as client:
        client.http.close()
        client.http = httpx.Client(transport=httpx.MockTransport(respond))
        html = client.get("https://example.org/data")
        ajax = client.request("GET", html.url, headers={"X-Requested-With": "XMLHttpRequest"})
        cached = client.request("GET", html.url, headers={"x-requested-with": "XMLHttpRequest"})
        assert client.get(html.url).content == b"html"
    assert calls == ["html", "XMLHttpRequest"]
    assert ajax.content == cached.content == b"XMLHttpRequest"
    assert cached.headers["last-modified"].startswith("Wed, 18 Nov 2026")


def test_bop_uses_quarter_end_and_keeps_services_separate_from_goods():
    labels = [
        "Счет текущих операций",
        "Товары",
        "Экспорт",
        "Импорт",
        "Услуги",
        "Экспорт",
        "Импорт",
        "Инвестиционные доходы",
        "Вторичные доходы",
        "Счет операций с капиталом",
        "Сальдо финансового счета",
        "Прочая задолженность",
        "Прочая задолженность",
        "Резервные активы",
    ]
    p = xlsx(
        [[None, "1 квартал 2026 г.", "2 квартал 2026 г."]]
        + [[label, i, i + 1] for i, label in enumerate(labels)],
        "Кварталы",
    )
    frame = parse_bop(p)
    assert set(frame.date) == {pd.Timestamp("2026-03-31"), pd.Timestamp("2026-06-30")}
    assert frame.loc[frame.series == "bop_goods_exports_quarterly", "value"].tolist() == [2, 3]
    assert frame.loc[frame.series == "bop_services_exports_quarterly", "value"].tolist() == [5, 6]


def test_expectations_do_not_merge_one_and_five_year_horizons():
    p = xlsx(
        [
            [None, None, None],
            [None, "янв.26", "фев.26"],
            ["наблюдаемая инфляция (в %)", 14, 15],
            ["ожидаемая инфляция (в %)", 13, 12],
            ["ожидаемая инфляция (в %)", 10, 9],
        ],
        "Данные за все годы",
    )
    frame = parse_expectations(p)
    assert frame.loc[frame.series == "household_inflation_expected_1y", "value"].tolist() == [
        13,
        12,
    ]
    assert frame.loc[frame.series == "household_inflation_expected_5y", "value"].tolist() == [10, 9]


def test_fnb_superscript_is_not_a_decimal_digit():
    labels = [
        "Объем на начало периода",
        "Объем на конец периода",
        "Поступления",
        "Изъятия",
        "Курсовая разница от переоценки активов в иностранной валюте и золота",
    ]
    p = xlsx(
        [[None] * 4, [None, "млрд руб.", None, None], [None, None, "янв.26", "фев.26"]]
        + [[None, label, "1 234,5¹", 1200] for label in labels]
    )
    frame = parse_fnb(p)
    assert frame.value.iloc[0] == 1234.5
    assert frame.source_value_labels.iloc[0] == "1 234,5¹"
    assert set(frame.unit) == {"RUB bn"}


def test_budget_balance_sign_and_monthly_differences():
    labels = [
        "Нефтегазовые доходы",
        "Доходы, всего",
        "Расходы, всего",
        "Дефицит (-)/Профицит (+)",
        "Ненефтегазовые доходы",
        "НДС (внутренний)",
        "Налог на прибыль",
    ]
    rows = [
        ["млрд руб., накопленным итогом с начала года", None, None],
        ["Показатель", "янв.26", "фев.26"],
    ] + [[label, 10, 25] for label in labels]
    rows[labels.index("Дефицит (-)/Профицит (+)") + 2][1:] = [-10, -25]
    frame = parse_oil_gas_revenue(xlsx(rows), budget=True)
    assert frame.loc[frame.series == "budget_balance", "value"].tolist() == [-10, -15]
    assert frame.loc[frame.series == "budget_balance_ytd", "value"].tolist() == [-10, -25]


def test_futures_roll_by_expiry_and_reject_wrong_decade(tmp_path):
    contracts = [
        {"secid": "new", "expiry": pd.Timestamp("2026-06-18")},
        {"secid": "old", "expiry": pd.Timestamp("2026-03-19")},
    ]
    windows = front_windows(contracts, "2026-03-13", "2026-03-16")
    assert [(c["secid"], str(a.date()), str(b.date())) for c, a, b in windows] == [
        ("old", "2026-03-13", "2026-03-14"),
        ("new", "2026-03-15", "2026-03-16"),
    ]

    def respond(request):
        return httpx.Response(
            200,
            json={
                "description": {"columns": ["name", "value"], "data": [["LSTTRADE", "2025-03-20"]]}
            },
        )

    with Client(tmp_path) as client:
        client.http.close()
        client.http = httpx.Client(transport=httpx.MockTransport(respond))
        assert contract_info(client, "SiH5", 2015) is None


def test_rosstat_year_footnote_does_not_repeat_previous_year():
    p = xlsx([["20222", None], ["январь", 100], ["20232", None], ["январь", 101]])
    frame = parse_period_rows(p, {"1": {"series": "x", "unit": "%"}}, year_suffix_pattern="2")
    assert frame.date.tolist() == [pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01")]


def test_weekly_prices_keep_product_and_reference_period():
    p = xlsx(
        [[None, None], [None, None], [None, None], [None, "на 10 января **"], ["Яблоки", "100,5"]],
        "2026",
    )
    frame = parse_weekly_items(p)
    assert frame.series.iloc[0].startswith("weekly_item_price_")
    assert frame.series_label.iloc[0] == "Яблоки"
    assert frame.date.iloc[0] == pd.Timestamp("2026-01-10")
    assert frame.value.iloc[0] == 100.5


def test_urals_uses_observation_month_not_article_date():
    p = payload(
        "<main>О среднем уровне цен нефти сорта «Юралс» за октябрь 2026 года. Средняя цена 101,46 долл. США за баррель. Дата публикации: 5 ноября 2026</main>"
    )
    frame = parse_urals(p)
    assert frame.date.iloc[0] == pd.Timestamp("2026-10-01")
    assert frame.value.iloc[0] == 101.46
    assert frame.reported_publication_date.iloc[0] == "2026-11-05"
    assert frame.available_at.iloc[0] == pd.Timestamp("2026-11-05T21:00:00Z")


def test_sanction_snapshots_use_release_dates_not_last_designation():
    p = payload(
        "source,name,start_date\nEntity List,Example Company,2025-01-01\n",
        headers={"last-modified": "Tue, 17 Nov 2026 14:30:32 GMT"},
    )
    frame, stamp = parse_bis(p)
    assert len(frame) == 1 and stamp.year == 2026
    assert frame.start_date.iloc[0] == "2025-01-01"
    _, unknown = parse_bis(payload(p.content))
    assert pd.isna(unknown)
    eu, stamp = parse_eu(
        payload(
            '<export generationDate="2026-10-23T10:00:00Z"><sanctionEntity euReferenceNumber="EU.1"><nameAlias wholeName="Example"/><regulation publicationDate="2020-01-01"/></sanctionEntity></export>'
        )
    )
    assert stamp.year == 2026 and eu.publication_date.iloc[0] == "2020-01-01"
    uk = "Report Date: 17-Nov-2026\nUnique ID,Regime Name,Name 1,Name 2,Name 3,Name 4,Name 5,Name 6\n1,Russia,Example,,,,,\n"
    frame, stamp = parse_uk(payload(uk))
    assert stamp == pd.Timestamp("2026-11-17") and frame.name.iloc[0] == "Example"


def test_ofac_list_validates_record_count_and_preserves_programmes():
    xml = '<sdnList xmlns="https://example.org"><publshInformation><Publish_Date>11/13/2026</Publish_Date><Record_Count>1</Record_Count></publshInformation><sdnEntry><uid>42</uid><lastName>Example Ship</lastName><sdnType>Vessel</sdnType><programList><program>RUSSIA-EO14024</program></programList><vesselInfo><vesselType>Oil Tanker</vesselType></vesselInfo></sdnEntry></sdnList>'
    frame, stamp = parse_ofac_list(payload(xml))
    assert stamp == pd.Timestamp("2026-11-13")
    assert frame.vessel_type.iloc[0] == "Oil Tanker"
    assert frame.programmes.iloc[0] == '["RUSSIA-EO14024"]'
    with pytest.raises(ValueError, match="число записей"):
        parse_ofac_list(payload(xml.replace("<Record_Count>1", "<Record_Count>2")))


def test_fns_keeps_publication_and_payment_distinct_from_filing():
    text = '<?xml version="1.0" encoding="windows-1251"?><calendar><year index="2026"><month name="january"><day num="28"><![CDATA[<p><b>НДПИ</b></p><p>Уплачивают налог за декабрь.</p><p>Представляют декларацию.</p>]]></day><day num="1" type="holiday"/></month></year></calendar>trailing garbage'
    p = payload(text, "https://data.nalog.ru/data-20260122-structure-20140228.xml")
    frame, events = parse_calendar(p)
    assert any(row["event_type"] == "calendar_coverage" for row in events)
    events = [row for row in events if row["event_type"] != "calendar_coverage"]
    assert {row["event_type"] for row in events} == {"tax_payment_day", "tax_filing_day"}
    assert {row["tax_name"] for row in events} == {"НДПИ"}
    assert set(frame.available_at) == {pd.Timestamp("2026-01-22T21:00:00Z")}
    assert frame.loc[frame.series == "ru_holiday_fns", "value"].tolist() == [1, 0]
    assert release_date("data-30122020-structure.xml") == pd.Timestamp("2020-12-30")


def test_pink_sheet_validates_units_and_does_not_fill_missing_prices():
    rows = [
        ["Title", None, None],
        [None, "Gold", "Liquefied natural gas, Japan"],
        [None, "($/troy oz)", "($/mmbtu)"],
        ["2026M09", 4000, "…"],
        ["2026M10", 4105, 12],
    ]
    frame = parse_pink_sheet(xlsx(rows, "Monthly Prices"))
    assert frame.loc[frame.series == "lng_japan_monthly", "value"].tolist() == [12]
    rows[2][1] = "($/gram)"
    with pytest.raises(ValueError, match="единица"):
        parse_pink_sheet(xlsx(rows, "Monthly Prices"))


def test_ofac_deduplicates_overlap_but_detects_repeated_pages(tmp_path):
    def respond(request):
        page = int(request.url.params["page"])
        dates = [1, 2] if page == 0 else [2, 3]
        text = "".join(
            f'<div class="views-row"><a href="/recent-actions/2015010{day}">Action</a>January 0{day}, 2015</div>'
            for day in dates
        )
        if page == 0:
            text += '<a rel="next" href="?page=1">Next</a>'
        return httpx.Response(200, text=text)

    with Client(tmp_path) as client:
        client.http.close()
        client.http = httpx.Client(transport=httpx.MockTransport(respond))
        rows = fetch_actions(client, "2015-01-01", "2015-01-03")
    assert len(rows) == 3
    assert rows[1]["index_occurrences"][0]["date"] == "2015-01-02"

    def repeated(request):
        return httpx.Response(
            200,
            text='<div class="views-row"><a href="/recent-actions/20150101">Action</a>January 01, 2015</div><a rel="next" href="?page=1">Next</a>',
        )

    with Client(tmp_path / "repeated") as client:
        client.http.close()
        client.http = httpx.Client(transport=httpx.MockTransport(repeated))
        with pytest.raises(ValueError, match="повторил страницу"):
            fetch_actions(client, "2015-01-01", "2015-01-03")


def test_nbs_months_inherit_explicit_year_and_reject_source_year_typo():
    header = "<tr><td></td><td>PMI</td><td>Production Index</td><td>New Order Index</td><td>Inventory</td><td>Employment</td><td>Delivery</td></tr>"

    def row(label, value):
        return f"<tr><td>{label}</td><td>{value}</td><td>51</td><td>50</td><td>49</td><td>48</td><td>50</td></tr>"

    html = (
        "<table>"
        + header
        + row("2025-November", 49)
        + row("December", 50)
        + row("2026-January", 51)
        + "</table>"
    )
    frame = parse_pmi(payload(html))
    assert frame.loc[frame.series == "china_manufacturing_pmi", "value"].tolist() == [49, 50, 51]
    assert frame.date.max() == pd.Timestamp("2026-01-01")
    with pytest.raises(ValueError, match="нарушение порядка"):
        parse_pmi(payload(html.replace("2026-January", "2025-January")))
