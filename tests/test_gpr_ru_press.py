"""Расчёт и границы архивного индекса для России."""

import hashlib
import json
from datetime import date

import pandas as pd
import pytest

from usd_rub.features import build_features
from usd_rub.sources import gpr_ru_press as press
from usd_rub.storage import Payload


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("ЕС ввел санкции против российских банков", True),
        ("ПВО сбила беспилотники под Курском", True),
        ("Путин объявил о начале специальной военной операции", True),
        ("Ядерные угрозы в адрес Москвы", True),
        ("Израиль объявил о специальной военной операции в Ливане", False),
        ("В Москве провели полицейскую спецоперацию", False),
        ("Москва и Киев договорились о прекращении огня", True),
        ("Атака дронов на порт в Израиле", False),
        ("Террористическая угроза в Индии", False),
        ("В Москве началась выставка", False),
        ("Футболисты из России отразили атаку соперника", False),
        ("В Москве пройдет выставка о Великой Отечественной войне", False),
        ("В России выросли продажи беспилотников", False),
        ("Суд Москвы арестовал предпринимателя с санкции прокурора", False),
    ],
)
def test_risk_headline(title, expected):
    assert press.risk_headline(title) is expected


def test_both_archives_and_pagination():
    day = date(2020, 2, 1)
    base = "/news/2020/02/01/"
    mk_url = press.MK + base
    mk = f'<link rel="canonical" href="{mk_url}">' + "".join(
        f'<a class="news-listing__item-link" href="{url}"><h3>{title}</h3></a>'
        for url, title in [
            (press.MK + "/a", "Санкции против РФ"),
            (press.MK + "/a", "Санкции против РФ"),
            (press.MK + "/b", "Открытие выставки"),
            ("https://www.sportmk.ru/c", "Новости спорта"),
        ]
    )

    def page(active, items):
        links = "".join(
            f'<a class="active">{i}</a>' if i == active else f'<a href="{base}page_{i}">{i}</a>'
            for i in (1, 2)
        )
        cards = "".join(f'<div data-id="{i}"><a><h3>{title}</h3></a></div>' for i, title in items)
        return (
            f'<script>var data_date="{day}";</script><div class="an">{cards}</div>'
            f'<div class="pages">{links}</div>'
        )

    pages = {
        mk_url: mk,
        press.INTERFAX + base: page(1, [(1, "Военные учения в России")]),
        press.INTERFAX + base + "page_2": page(
            2, [(1, "Военные учения в России"), (2, "Новости театра"), (3, "Сбор урожая")]
        ),
    }

    class Client:
        def get(self, url):
            body = pages[url].encode()
            return Payload(body, url, hashlib.sha256(body).hexdigest(), "2026-01-01T00:00:00Z")

    counts, inputs = press.collect_day(Client(), day)
    assert list(counts.values()) == [3, 1, 2, 1]
    assert set(inputs) == set(pages)
    with pytest.raises(ValueError, match="архивная дата"):
        press.parse_mk(mk.encode(), date(2020, 2, 2))
    with pytest.raises(ValueError, match="страница"):
        press.parse_interfax(pages[press.INTERFAX + base].encode(), day, 2)
    with pytest.raises(ValueError, match="пустая лента"):
        press.parse_mk(f'<link rel="canonical" href="{mk_url}">'.encode(), day)


def test_full_month_counts_and_historical_availability(monkeypatch, tmp_path):
    def count(client, day):
        values = [10, 2, 100, 0] if day.day == 1 else [1, 0, 1, 0]
        return dict(zip(press.COUNTS, values, strict=True)), {"url": "sha"}

    monkeypatch.setattr(press, "collect_day", count)
    client = type("Client", (), {"root": tmp_path})()
    rows = press.fetch_ru_press(client, "2020-01-31", "2020-03-01")
    monthly = rows.loc[rows.series.eq(press.MONTHLY)].squeeze()
    assert monthly.date == pd.Timestamp("2020-02-01")
    assert monthly.value == pytest.approx(50 * 2 / 38)
    inputs = json.loads((tmp_path / monthly.source_url).read_text())["inputs"]
    assert len(inputs) == 29
    daily = rows.loc[rows.series.eq(press.DAILY)]
    strict = build_features({press.DAILY: daily}, "2020-02-01", "2020-02-02")
    assert strict[press.DAILY].isna().all()
    research = build_features(
        {press.DAILY: daily}, "2020-02-01", "2020-02-02", assumed_lags={press.DAILY: 1}
    )
    assert research[press.DAILY].tolist() == [0, 10]
    repeated = press.fetch_ru_press(client, "2020-02-01", "2020-02-01").iloc[0]
    assert repeated.raw_sha256 == daily.loc[daily.date.eq("2020-02-01"), "raw_sha256"].iloc[0]
