"""Архивные события, накопленные суммы и сохранение винтажей."""

import hashlib
from io import BytesIO
from zipfile import ZipFile

import httpx
import pandas as pd
import pytest

from usd_rub.features import event_features
from usd_rub.sources.events import parse_meetings
from usd_rub.sources.gdelt import parse_gdelt
from usd_rub.sources.minfin import parse_oil_gas_revenue, ytd_to_monthly
from usd_rub.sources.ofac import fetch_actions, parse_actions
from usd_rub.storage import Client, Payload, observations, read_series, save_series


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


def test_same_vintage_refresh_preserves_first_availability(tmp_path):
    first = observations(["2015-01-01"], [10], "x", "unit", payload(b"one"))
    later = first.copy()
    later["ingested_at"] = "2026-11-18T10:00:00Z"
    later["available_at"] = pd.Timestamp("2026-11-18T10:00:00Z")
    save_series(first, tmp_path)
    save_series(later, tmp_path)
    result = read_series(tmp_path, "x")
    assert result.available_at.iloc[0] == pd.Timestamp("2026-11-17T10:00:00Z")


def test_ytd_differences_do_not_cross_gaps_or_year_boundaries():
    data = observations(
        ["2020-01-01", "2020-02-01", "2020-04-01", "2021-01-01"],
        [10, 25, 60, 12],
        "oil_gas",
        "RUB bn",
        payload(b"x"),
    )
    result = ytd_to_monthly(data)
    assert result.value.tolist() == [10, 15, 12]
    assert pd.Timestamp("2020-04-01") not in result.date.to_list()


def test_minfin_requires_units_and_specific_row():
    valid = xlsx(
        [
            ["млрд руб.", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")],
            ["Нефтегазовые доходы", 10, 11],
        ]
    )
    assert parse_oil_gas_revenue(valid).value.tolist() == [10, 11]
    invalid = xlsx(
        [
            ["проценты", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")],
            ["Нефтегазовые доходы", 10, 11],
        ]
    )
    with pytest.raises(ValueError):
        parse_oil_gas_revenue(invalid)


def test_minfin_follows_numbered_layout_and_separates_ytd():
    p = xlsx(
        [
            [None, "млрд руб., накопленным итогом с начала года", None, None],
            ["№", "Показатель", "янв.23 ***", "фев.23 ***"],
            ["1.1", "Нефтегазовые доходы", 10, 25],
        ]
    )
    result = parse_oil_gas_revenue(p)
    assert result.loc[result.series == "oil_gas_revenue", "value"].tolist() == [10, 15]
    assert result.loc[result.series == "oil_gas_revenue_ytd", "value"].tolist() == [10, 25]


def test_old_cbr_meeting_table_recovers_year_from_tab():
    html = '<a id="year">2015 год</a><div role="tabpanel" aria-labelledby="year"><table><tr><td>30 января</td><td>Заседание Совета директоров Банка России</td></tr></table></div>'
    result = parse_meetings(payload(html))
    assert result.date.iloc[0] == pd.Timestamp("2015-01-30")


def test_ofac_dates_are_publication_dates_and_follow_pagination(tmp_path):
    calls = []

    def respond(request):
        page = int(request.url.params["page"])
        calls.append(page)
        text = f'<div class="views-row"><a href="/recent-actions/2015010{page + 1}">Action</a>January 0{page + 1}, 2015</div>'
        if page == 0:
            text += '<a rel="next" href="?page=1">Next</a>'
        return httpx.Response(200, text=text)

    with Client(tmp_path) as client:
        client.http.close()
        client.http = httpx.Client(transport=httpx.MockTransport(respond))
        result = fetch_actions(client, "2015-01-01", "2015-01-02")
    assert calls == [0, 1]
    assert len(result) == 2
    assert result[0]["publication_time_precision"] == "date_only"


def test_gdelt_filters_country_and_uses_archive_day():
    rows = []
    for i, country in enumerate(["RUS", "USA"]):
        row = [""] * 58
        row[0] = str(i)
        row[1] = "20100101"
        row[7] = country
        row[30] = "-2"
        row[34] = "-3"
        rows.append("\t".join(row))
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("20150101.export.CSV", "\n".join(rows))
    result = parse_gdelt(payload(buffer.getvalue()), "2015-01-01").set_index("series")
    assert result.loc["gdelt_ru_event_count", "value"] == 1
    assert result.loc["gdelt_ru_avg_tone", "value"] == -3
    assert result.loc["gdelt_ru_event_count", "date"] == pd.Timestamp("2015-01-01")


def test_future_event_only_known_after_announcement():
    events = pd.DataFrame(
        {
            "event_type": ["tax"],
            "publication_time": ["2020-01-02T12:00:00Z"],
            "effective_time": ["2020-01-05T00:00:00+03:00"],
            "value": [1],
        }
    )
    cutoffs = pd.to_datetime(
        ["2020-01-01T12:00:00Z", "2020-01-03T12:00:00Z", "2020-01-05T12:00:00Z"]
    )
    result = event_features(events, cutoffs)
    assert pd.isna(result.days_to_tax.iloc[0])
    assert result.days_to_tax.iloc[1] == 2
    assert result.event_tax.iloc[2] == 1


def test_ofac_two_page_pager_does_not_require_next_link():
    p = payload(
        '<div class="view-header">Displaying 1 - 10 of 20 results.</div><nav class="usa-pagination"><a href="?page=1">2</a></nav>'
    )
    _, has_next = parse_actions(p)
    assert has_next
