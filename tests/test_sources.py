"""Парсеры, пагинация, архивирование и качество наблюдений."""

import hashlib
from io import BytesIO

import httpx
import numpy as np
import pandas as pd
import pytest

from usd_rub.sources.cbr import parse_fx, parse_html_table, parse_soap
from usd_rub.sources.events import parse_events_csv, parse_meetings
from usd_rub.sources.markets import fetch_moex, parse_cboe, parse_fred
from usd_rub.sources.tables import (
    discover_table_links,
    parse_long_table,
    parse_month_matrix,
    parse_row_timeseries,
)
from usd_rub.storage import (
    Client,
    DownloadError,
    Payload,
    number,
    observations,
    read_series,
    save_series,
)


def payload(content):
    if isinstance(content, str):
        content = content.encode()
    return Payload(
        content,
        "https://example.org/data",
        hashlib.sha256(content).hexdigest(),
        "2026-11-17T10:00:00+00:00",
    )


def xlsx(data):
    buffer = BytesIO()
    pd.DataFrame(data).to_excel(buffer, index=False, header=False, sheet_name="Data")
    return payload(buffer.getvalue())


def test_cbr_fx_divides_by_nominal():
    p = payload(
        '<ValCurs><Record Date="01.01.2015"><Nominal>10</Nominal><Value>105,21</Value></Record></ValCurs>'
    )
    result = parse_fx(p, "cny_rub")
    assert result.value.iloc[0] == pytest.approx(10.521)
    assert result.available_at.iloc[0] == pd.Timestamp("2014-12-31T21:00:00Z")


def test_cbr_soap_namespaces_and_missing_values():
    p = payload(
        '<Envelope xmlns="urn:soap"><Body><Ruonia xmlns=""><ro><D0>2015-01-12T00:00:00+03:00</D0><ruo>17.55</ruo><vol>234.6</vol></ro><ro><D0>2015-01-13</D0><ruo>17.19</ruo></ro></Ruonia></Body></Envelope>'
    )
    result = parse_soap(p, "ro", "D0", {"ruo": ("ruonia", "%"), "vol": ("volume", "RUB bn")})
    assert len(result) == 3
    assert result.loc[result.series == "volume", "value"].iloc[0] == 234.6
    assert set(result.availability_basis) == {"unknown"}


def test_fred_dots_and_old_date_column():
    result = parse_fred(
        payload("DATE,DEXCHUS\n2015-01-01,.\n2015-01-02,6.2\n"), "DEXCHUS", "usd_cny", "CNY/USD"
    )
    assert len(result) == 1
    assert result.value.iloc[0] == 6.2
    assert result.available_at.iloc[0] == pd.Timestamp("2026-11-17T10:00:00Z")


def test_cboe_us_dates():
    result = parse_cboe(payload("DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,17,18,16,17.24\n"))
    assert result.date.iloc[0] == pd.Timestamp("1990-01-02")


def test_iss_follows_cursor_until_all_pages(tmp_path):
    calls = []

    def respond(request):
        start = int(request.url.params["start"])
        calls.append(start)
        count = 2 if start == 0 else 1
        data = [[f"2015-01-0{i + 1}", 10 + i] for i in range(start, start + count)]
        return httpx.Response(
            200,
            json={
                "history": {"columns": ["TRADEDATE", "CLOSE"], "data": data},
                "history.cursor": {
                    "columns": ["INDEX", "TOTAL", "PAGESIZE"],
                    "data": [[start, 3, 2]],
                },
            },
        )

    with Client(tmp_path) as client:
        client.http.close()
        client.http = httpx.Client(transport=httpx.MockTransport(respond))
        result = fetch_moex(
            client,
            {"path": "test", "columns": {"CLOSE": ("test", "RUB")}},
            "2015-01-01",
            "2015-01-03",
        )
    assert calls == [0, 2]
    assert len(result) == 3


def test_raw_cache_and_refresh_preserve_versions_and_hide_key(tmp_path):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, content=f"answer-{len(calls)}".encode())

    with Client(tmp_path) as client:
        client.http.close()
        client.http = httpx.Client(transport=httpx.MockTransport(respond))
        one = client.get("https://example.org/data", {"api_key": "secret"})
        again = client.get("https://example.org/data", {"api_key": "secret"})
        assert one.sha256 == again.sha256 and len(calls) == 1
        client.refresh = True
        two = client.get("https://example.org/data", {"api_key": "secret"})
        assert one.sha256 != two.sha256
    assert len(list((tmp_path / "raw/objects").iterdir())) == 2
    assert "secret" not in (tmp_path / "raw/history.jsonl").read_text()


def test_corrupt_cache_rejected(tmp_path):
    with Client(tmp_path) as client:
        client.http.close()
        client.http = httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"abc"))
        )
        p = client.get("https://example.org/data")
        (tmp_path / "raw/objects" / p.sha256).write_bytes(b"def")
        with pytest.raises(DownloadError):
            client.get("https://example.org/data")


def test_month_matrix_excludes_totals_and_missing_future():
    p = xlsx(
        [
            ["Month", 2015, 2016],
            ["Январь", 100.1, 100.2],
            ["Февраль", 100.3, np.nan],
            ["Год", 105, 106],
        ]
    )
    result = parse_month_matrix(p, "cpi", "%")
    assert len(result) == 3
    assert list(result.date) == list(pd.to_datetime(["2015-01-01", "2015-02-01", "2016-01-01"]))


def test_row_timeseries_and_schema_failure():
    p = xlsx([["Name", pd.Timestamp("2015-01-01"), pd.Timestamp("2015-02-01")], ["M2", 100, 101]])
    result = parse_row_timeseries(p, "Data", "M2", "m2", "RUB bn")
    assert result.value.tolist() == [100, 101]
    with pytest.raises(ValueError):
        parse_row_timeseries(p, "Data", "missing", "m2", "RUB bn")


def test_long_table_ignores_headers_and_scales_units():
    p = xlsx([["Дата", "Значение"], ["01.01.2015", "1 234,5"], ["01.02.2015", "1 235,5"]])
    result = parse_long_table(
        p,
        {
            "date_column": 0,
            "date_format": "%d.%m.%Y",
            "columns": {"1": {"series": "x", "unit": "RUB bn", "scale": 0.001}},
        },
    )
    np.testing.assert_allclose(result.value, [1.2345, 1.2355])


def test_save_keeps_revisions_and_is_idempotent(tmp_path):
    first = observations(["2015-01-01"], [10], "x", "unit", payload(b"one"))
    second = observations(["2015-01-01"], [11], "x", "unit", payload(b"two"))
    save_series(first, tmp_path)
    save_series(first, tmp_path)
    save_series(second, tmp_path)
    result = read_series(tmp_path, "x")
    assert len(result) == 2
    assert set(result.value) == {10, 11}


def test_meeting_calendar_ignores_summary():
    html = '<div class="main-events_day"><div class="date">12 февраля 2027 года</div>Заседание Совета директоров Банка России</div><div class="main-events_day"><div class="date">25 февраля 2027 года</div>Резюме обсуждения</div>'
    assert len(parse_meetings(payload(html))) == 1


def test_event_requires_evidence_and_timezone():
    with pytest.raises(ValueError):
        parse_events_csv(payload("date,value\n2015-01-01,1\n"))


def test_conflicting_duplicates_fail():
    with pytest.raises(ValueError):
        observations(["2015-01-01", "2015-01-01"], [1, 2], "x", "unit", payload(b"x"))


def test_invalid_observation_date_has_clear_error():
    with pytest.raises(ValueError, match="Некорректная дата"):
        observations([pd.NaT], [1], "x", "unit", payload(b"x"))


def test_number_does_not_silently_drop_invalid_text():
    assert number("1\xa0234,5") == 1234.5
    assert np.isnan(number("—"))
    with pytest.raises(ValueError):
        number("unexpected")


def test_html_decimal_comma_is_not_a_thousands_separator():
    p = payload(
        "<table><tr><th>Дата</th><th>Ставка</th></tr><tr><td>01.01.2015</td><td>17,55</td></tr></table>"
    )
    result = parse_html_table(p, {1: ("rate", "%")})
    assert result.value.iloc[0] == 17.55


def test_html_year_headers_are_kept_for_month_matrix():
    p = payload(
        "<table><tr><th>Месяц</th><th>2015</th><th>2016</th></tr><tr><td>Январь</td><td>100,1</td><td>100,2</td></tr></table>"
    )
    result = parse_month_matrix(p, "cpi", "%")
    assert result.value.tolist() == [100.1, 100.2]


def test_month_matrix_does_not_mix_annual_december_with_monthly_december():
    months = [
        "Январь",
        "Февраль",
        "Март",
        "Апрель",
        "Май",
        "Июнь",
        "Июль",
        "Август",
        "Сентябрь",
        "Октябрь",
        "Ноябрь",
        "Декабрь",
    ]
    p = xlsx(
        [["Month", 2015, 2016]]
        + [[m, 100.1, 100.2] for m in months]
        + [["к декабрю предыдущего года", None, None], ["Декабрь", 110, 120]]
    )
    result = parse_month_matrix(p, "cpi", "%")
    assert len(result) == 24
    assert result.value.max() == 100.2


def test_russian_period_labels_with_footnotes():
    p = xlsx([["Name", "янв.23 ***", "фев.23 ***", "01.03.2023"], ["M2", 100, 101, 102]])
    result = parse_row_timeseries(p, "Data", "M2", "m2", "RUB bn")
    assert result.date.tolist() == list(pd.date_range("2023-01-01", periods=3, freq="MS"))
    assert "***" in result.source_period_label.iloc[0]


def test_rosstat_link_uses_title_from_sibling_container():
    p = payload(
        '<div class="document-list__item"><div><a href="/cpi.xlsx">XLSX</a></div><div>ИПЦ с 1991 г.</div></div>'
    )
    result = discover_table_links(p, "ИПЦ")
    assert result[0]["url"] == "https://example.org/cpi.xlsx"


def test_browser_fallback_requires_explicit_opt_in_and_marks_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr("usd_rub.browser.browser_download", lambda url: (b"public-file", url))
    with Client(tmp_path, browser_fallback=True) as client:
        client.http.close()
        client.http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(403)))
        result = client.get("https://rosstat.gov.ru/test.xlsx")
    assert result.content == b"public-file"
    assert "Google Chrome public fetch" in (tmp_path / "raw/history.jsonl").read_text()
