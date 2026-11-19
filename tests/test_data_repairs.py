"""Регрессии ошибок нормализации, времени публикации и версий календаря."""

import json

import pandas as pd
import pytest
from test_current_sources import payload

from usd_rub.availability import enrich_availability, restore_document_times
from usd_rub.features import asof_series, event_features
from usd_rub.freshness import coverage_gaps, ensure_allowed, freshness, require_current
from usd_rub.sources.ofac import action_events, classify_action
from usd_rub.sources.reports import parse_orfr
from usd_rub.sources.sanctions import parse_uk
from usd_rub.sources.tax_calendar import parse_calendar
from usd_rub.storage import observations


def test_ytd_and_monthly_household_purchases_never_share_a_series():
    p = payload("pdf")
    pages = [
        {
            "page": 2,
            "text": "Нетто-покупки иностранной валюты физическими лицами с начала года составили 584 млрд руб. Нетто-покупки валюты физическими лицами за месяц составили 103 млрд руб.",
        }
    ]
    data = parse_orfr(p, pd.Timestamp("2025-07-01"), pages).set_index("series")
    assert data.loc["household_net_fx_purchases_ytd", "value"] == 584
    assert data.loc["household_net_fx_purchases", "value"] == 103
    assert data.loc["household_net_fx_purchases_ytd", "aggregation"] == "year_to_date"


def test_freshness_cannot_accept_january_just_because_year_matches():
    data = observations(["2026-01-01"], [1], "usd_rub", "RUB/USD", payload("x"))
    with pytest.raises(ValueError, match="возраст"):
        require_current(data, as_of="2026-11-18")
    require_current(data, as_of="2026-01-02")


def test_monthly_age_counts_from_period_end_and_known_cpi_absence_is_preserved():
    data = observations(
        ["2026-08-01", "2026-09-01", "2026-10-01"],
        [1, 2, 3],
        "cpi_ru_mom_index",
        "index",
        payload("x"),
    )
    assert freshness(data, "2026-11-18")["age_days_after_period_end"] == 18
    stock = data.assign(series="m2")
    assert freshness(stock, "2026-11-18")["age_days_after_period_end"] == 48
    gaps = coverage_gaps("cpi_us", pd.to_datetime(["2025-09-01", "2025-11-01", "2025-12-01"]))
    assert gaps["missing_periods"] == []
    assert gaps["documented_absences"][0]["period"] == "2025-10"


def test_fns_missing_year_is_not_filled_with_previous_year_count():
    data = observations(
        ["2016-12-31", "2018-01-01"], [5, 7], "fns_tax_payment_count", "count", payload("x")
    )
    data["availability_basis"] = "recorded_publication"
    data["available_at"] = pd.Timestamp("2016-01-01T00:00:00Z")
    result = asof_series(data, pd.to_datetime(["2017-01-01T12:00:00Z", "2018-01-01T12:00:00Z"]))
    assert pd.isna(result.value.iloc[0])
    assert result.value.iloc[1] == 7


def test_calendar_latest_release_replaces_removed_events_without_early_knowledge():
    text = '<calendar><year index="2026"><month name="january"><day num="28"><![CDATA[<p><b>НДПИ</b></p><p>Уплачивают налог.</p><p>Уплачивают налог.</p>]]></day></month></year></calendar>'
    first = payload(text, "https://example.org/data-20260101-structure-20140228.xml")
    later = payload(
        text.replace("<p>Уплачивают налог.</p>", ""),
        "https://example.org/data-20260120-structure-20140228.xml",
    )
    _, old = parse_calendar(first)
    _, new = parse_calendar(later)
    assert len([r for r in old if r["event_type"] == "tax_payment_day"]) == 1
    events = pd.DataFrame(old + new)
    cutoffs = pd.to_datetime(["2026-01-10T12:00:00Z", "2026-01-28T12:00:00Z"])
    result = event_features(events, cutoffs, allow_assumptions=True)
    assert result.days_to_tax_payment_day.iloc[0] == 18
    assert result.event_tax_payment_day.iloc[1] == 0
    assert pd.isna(result.days_to_tax_payment_day.iloc[1])
    assert "event_tax_payment_day" not in event_features(events, cutoffs).columns


def test_key_rate_release_is_not_used_before_effective_day_or_for_different_rate(tmp_path):
    folder = tmp_path / "documents" / "cbr_decisions"
    folder.mkdir(parents=True)
    record = {
        "source_url": "https://example.org/decision",
        "period": "2026-01-23",
        "title": "Банк России снизил ключевую ставку до 14,00% годовых",
        "pages": [{"text": "С 26 января 2026 года ключевая ставка составляет 14 процентов."}],
        "raw_sha256": "a" * 64,
        "publication_time": None,
    }
    (folder / "one.json").write_text(json.dumps(record))
    (folder / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "url": record["source_url"],
                    "reported_publication_time": "2026-01-23T13:30:00+03:00",
                }
            ]
        )
    )
    assert restore_document_times(tmp_path) == 1
    data = observations(
        ["2026-01-23", "2026-01-26", "2026-01-27"], [14, 14, 15], "key_rate", "%", payload("x")
    )
    enriched = enrich_availability(data, tmp_path)
    assert enriched.availability_basis.tolist() == ["unknown", "official_rate_decision", "unknown"]
    assert enriched.available_at.iloc[1] == pd.Timestamp("2026-01-23T10:30:00Z")


def test_uk_aliases_do_not_inflate_entity_count():
    p = payload(
        "Report Date: 17-Nov-2026\nUnique ID,Regime Name,Name 1,Name 2,Name 3,Name 4,Name 5,Name 6\n1,Russia,Main,,,,,\n1,Russia,Alias,,,,,\n"
    )
    table, _ = parse_uk(p)
    assert len(table) == 1
    assert json.loads(table.aliases.iloc[0]) == ["Alias"]


def test_ofac_country_and_action_direction_are_not_assumed_for_all_documents():
    first = {
        "date": "2026-11-13",
        "title": "Russia-related Designation Removal; General License",
        "source_url": "https://example.org/action",
        "raw_sha256": "b" * 64,
    }
    second = dict(
        first, title="Counter Narcotics Designations", source_url="https://example.org/other"
    )
    info = classify_action(first)
    assert "removal" in info["action_types"] and "license" in info["action_types"]
    assert "designation" not in info["action_types"]
    events = action_events([first, second, first])
    assert len(events) == 1
    assert pd.Timestamp(events.publication_time.iloc[0]) == pd.Timestamp("2026-11-14T05:00:00Z")


def test_unreconciled_fnb_flows_cannot_enter_model_features():
    for name in ["fnb_inflows_rub_bn", "fnb_withdrawals_rub_bn", "fnb_revaluation_rub_bn"]:
        with pytest.raises(ValueError, match="ФНБ"):
            ensure_allowed(name)


def test_bop_estimate_keeps_current_months_separate_from_ytd_and_prior_year():
    from test_current_sources import xlsx

    from usd_rub.sources.cbr_extra import parse_bop_estimate

    rows = [
        ["млрд долларов США", "Дата последнего обновления: 16 ноября 2026 года", None, None, None],
        [
            "Показатель",
            "январь - сентябрь 2026 г.",
            "III квартал 2026 г. (оценка)",
            "июль 2026 г. (оценка)",
            "Справочно:",
        ],
    ] + [
        [label, 99, 10, 1.3, 7]
        for label in [
            "Счет текущих операций",
            "Товары",
            "Экспорт",
            "Импорт",
            "Услуги",
            "Экспорт",
            "Импорт",
        ]
    ]
    result = parse_bop_estimate(xlsx(rows))
    row = result.loc[result.series.eq("bop_estimate_current_account_monthly")].iloc[0]
    assert row.value == 1.3 and row.date == pd.Timestamp("2026-07-01")
    assert row.available_at == pd.Timestamp("2026-11-16T21:00:00Z")
    assert not result.value.eq(99).any()
    assert set(result.unit) == {"USD bn"}


def test_feature_build_records_input_and_event_dependencies(tmp_path):
    from usd_rub.cli import main
    from usd_rub.storage import save_series

    data = observations(
        ["2026-11-17"], [82], "usd_rub", "RUB/USD", payload("x"), "effective_date_midnight"
    )
    save_series(data, tmp_path)
    output = tmp_path / "features" / "daily.csv"
    assert (
        main(
            [
                "--data-dir",
                str(tmp_path),
                "features",
                "--series",
                "usd_rub",
                "--start",
                "2026-11-17",
                "--end",
                "2026-11-18",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    meta = json.loads(output.with_suffix(".json").read_text())
    assert any(item["path"].endswith("usd_rub.csv") for item in meta["inputs"])
    assert all(len(item["sha256"]) == 64 for item in meta["inputs"])
    assert meta["all_null_columns"] == []


def test_missing_event_archive_is_unknown_rather_than_zero():
    events = pd.DataFrame(
        {
            "event_type": ["uk_update"],
            "publication_time": ["2026-01-05T12:00:00Z"],
            "effective_time": ["2026-01-05T12:00:00Z"],
            "value": [1],
            "coverage_start": ["2026-01-01T00:00:00Z"],
            "coverage_end": ["2026-01-10T12:00:00Z"],
        }
    )
    cutoffs = pd.to_datetime(
        ["2025-12-31T12:00:00Z", "2026-01-04T12:00:00Z", "2026-01-11T12:00:00Z"]
    )
    result = event_features(events, cutoffs)
    assert pd.isna(result.event_uk_update.iloc[0])
    assert result.event_uk_update.iloc[1] == 0
    assert pd.isna(result.event_uk_update.iloc[2])


def test_identical_calendar_body_in_different_releases_preserves_both_publications():
    text = '<calendar><year index="2026"><month name="january"><day num="28"><![CDATA[<p>Уплачивают налог.</p>]]></day></month></year></calendar>'
    first, old = parse_calendar(
        payload(text, "https://example.org/data-20260101-structure-20140228.xml")
    )
    second, new = parse_calendar(
        payload(text, "https://example.org/data-20260120-structure-20140228.xml")
    )
    assert set(first.vintage).isdisjoint(set(second.vintage))
    assert {r["event_id"] for r in old}.isdisjoint({r["event_id"] for r in new})


def test_current_pmi_article_date_does_not_certify_historical_vintage():
    from usd_rub.sources.china import parse_pmi

    html = "<p>2026-11-01 09:30 Print</p><table><tr><td>PMI</td></tr><tr><td>2026 - October</td><td>49.3</td><td>50</td><td>49</td><td>1</td><td>1</td><td>1</td></tr></table>"
    p = payload(html)
    data = parse_pmi(p)
    assert set(data.available_at) == {pd.Timestamp(p.ingested_at)}
    assert data.reported_publication_time.iloc[0] == pd.Timestamp("2026-11-01T01:30:00Z")
    aligned = asof_series(
        data.loc[data.series.eq("china_manufacturing_pmi")],
        pd.to_datetime(["2026-11-02T12:00:00Z"]),
    )
    assert pd.isna(aligned.value.iloc[0])
