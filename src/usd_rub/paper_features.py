"""Признаки кросс-курса, свопов и макроэкономического опроса."""

import re
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from usd_rub.features import asof_series
from usd_rub.sources.tables import workbook
from usd_rub.storage import number, observations, read_series
from usd_rub.yuan_experiment import ECB_ARCHIVE, ecb_usd_cny

CBR = "https://www.cbr.ru"
SWAP_START = "01.01.2023"
SWAP_PATHS = {
    "limits": "/hd_base/swap_info/sell/",
    "volume": "/hd_base/swap_info/swapinfosellvol/",
    "claims": "/hd_base/claim-fx-swap/",
}
SURVEY_XLSX = f"{CBR}/Content/Document/File/144490/full.xlsx"
SURVEY_CALENDAR = f"{CBR}/calendar/"
MARKET_SERIES = ("cny_tom_close", "cny_tom_vwap")


def dated_rows(payload, required_header):
    soup = BeautifulSoup(payload.content, "html.parser")
    table = next(
        (
            tag
            for tag in soup.select("table")
            if required_header in " ".join(tag.stripped_strings).replace("\xa0", " ")
        ),
        None,
    )
    if table is None:
        raise ValueError(f"Выгрузка ЦБ без ожидаемого столбца: {required_header}")
    rows = [
        [cell.get_text(" ", strip=True) for cell in row.select("td")] for row in table.select("tr")
    ]
    result = pd.DataFrame(
        [row for row in rows if row and re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", row[0])]
    )
    if result.empty:
        raise ValueError(f"Пустая выгрузка ЦБ: {payload.url}")
    return result.set_index(
        pd.to_datetime(result.pop(0), format="%d.%m.%Y"), verify_integrity=True
    ).sort_index()


def end_of_day(dates):
    return (
        (pd.DatetimeIndex(dates) + pd.Timedelta(days=1)).tz_localize("Europe/Moscow")
    ).tz_convert("UTC")


def swap_history(payloads):
    limits = dated_rows(payloads["limits"], "Максимальный объем продажи инвалюты")
    volume = dated_rows(payloads["volume"], "млн ед. инвалюты")[1].map(number)
    claims = dated_rows(payloads["claims"], "млн юаней")[2].map(number)
    settlement = pd.to_datetime(limits[2], format="%d.%m.%Y")
    amount = settlement.map(claims)
    limit = limits[7].map(number) * 1000
    comparable = volume.reindex(amount.index)
    active = comparable.notna()
    if (
        not limit.gt(0).all()
        or not pd.to_datetime(limits[1], format="%d.%m.%Y").eq(limits.index).all()
        or not settlement.gt(limits.index).all()
        or (amount < 0).any()
        or not volume.index.isin(amount.index).all()
        or not np.allclose(amount[active], comparable[active], atol=0.11, rtol=0)
        or (amount.gt(0) & ~active).any()
    ):
        raise ValueError("Объёмы, лимиты и требования по свопам ЦБ не согласуются")
    return pd.DataFrame(
        {
            "settlement": settlement,
            "limit_mn": limit,
            "used_mn": amount,
            "reported_volume_mn": comparable,
        }
    ).rename_axis("trade_date")


def swap_features(history, payloads, cutoffs):
    result = {}
    for name, values, unit, release_dates, source in (
        ("limit_mn", history.limit_mn, "CNY mn", history.index, "limits"),
        ("used_mn", history.used_mn, "CNY mn", history.settlement, "claims"),
        (
            "usage_fraction",
            history.used_mn / history.limit_mn,
            "ratio",
            history.settlement,
            "claims",
        ),
    ):
        frame = observations(
            history.index,
            values,
            f"cbr_cny_swap_{name}",
            unit,
            payloads[source],
            availability="cbr_report_date_end_of_day",
        )
        frame["available_at"] = end_of_day(
            pd.Series(release_dates, index=history.index).reindex(frame.date)
        )
        aligned = asof_series(frame, cutoffs, max_age_days=15)
        result[f"extended__rates__cbr_cny_swap_{name}"] = aligned.value.to_numpy()
        if name == "used_mn":
            result["extended__rates__cbr_cny_swap_trade_age"] = aligned.age_days.where(
                aligned.value.notna()
            ).to_numpy()
    return pd.DataFrame(result, index=cutoffs.tz_localize(None))


def survey_release_dates(calendar):
    soup = BeautifulSoup(calendar.content, "html.parser")
    dates = {}
    for row in soup.select("tr"):
        cells = [
            " ".join(cell.get_text(" ", strip=True).lower().split()) for cell in row.select("td")
        ]
        if "макроэкономический опрос банка россии" not in cells:
            continue
        release = pd.to_datetime(cells[-1], format="%d.%m.%Y")
        period = release.to_period("M")
        if period in dates and dates[period] != release:
            raise ValueError(f"Несколько дат публикации опроса за {period}")
        dates[period] = release
    if not dates:
        raise ValueError("В календаре не найдены даты публикации опроса")
    return dates


def survey_features(payload, calendar, cutoffs):
    sheets = workbook(payload)
    release_dates = survey_release_dates(calendar)
    dates = cutoffs.tz_localize(None)
    features, releases = {}, {}
    for sheet, label, name in (
        ("1", "CPI", "cpi_dec_yoy"),
        ("3", "Key rate", "key_rate"),
        ("11", "USD / RUB rate", "usd_rub"),
    ):
        table = sheets[sheet]
        if not str(table.iat[4, 1]).startswith(label) or table.iat[7, 2] != "Median":
            raise ValueError(f"Изменилась таблица медиан опроса ЦБ: {name}")
        median_end = table.index[table[2].eq("Average")][0]
        median_rows = table.iloc[7:median_end]
        medians = median_rows.iloc[:, 4:].map(number)
        medians.index = pd.to_datetime(median_rows[3]).dt.year.to_numpy()
        medians.columns = pd.PeriodIndex(pd.to_datetime(table.iloc[5, 4:]), freq="M")
        previous = medians.diff(axis=1)
        reported_dates = []
        for month in medians.columns:
            release = release_dates.get(month, month.end_time.normalize())
            reported_dates.append(release)
            releases[month] = (release, month in release_dates)
        available = pd.DatetimeIndex(reported_dates) + pd.Timedelta(days=1)
        for field, forecasts in (("median", medians), ("revision", previous)):
            aligned = forecasts.T.set_axis(available).reindex(
                dates, method="ffill", tolerance=pd.Timedelta(days=92)
            )
            for year_offset, year_label in ((0, "current_year"), (1, "next_year")):
                years = dates.year + year_offset
                by_year = aligned.reindex(columns=years.unique())
                year_pos = by_year.columns.get_indexer(years)
                key = f"extended__macro__cbr_survey_{name}_{year_label}_{field}"
                features[key] = by_year.to_numpy()[np.arange(len(dates)), year_pos]
    releases = pd.DataFrame.from_dict(releases, orient="index", columns=["date", "calendar_match"])
    return pd.DataFrame(features, index=dates), releases.rename_axis("survey_month")


def market_features(payload, local, target):
    cutoffs = target.index.tz_localize("Europe/Moscow")
    with ZipFile(BytesIO(payload.content)) as archive:
        with archive.open("eurofxref-hist.csv") as file:
            ecb = pd.read_csv(file, usecols=["Date", "USD", "JPY"]).dropna()
    yen = observations(
        ecb.Date,
        ecb.JPY / ecb.USD,
        "ecb_usd_jpy",
        "JPY/USD",
        payload,
        availability="ecb_daily_next_moscow_midnight",
        lag_days=1,
    )
    aligned = {
        name: asof_series(frame, cutoffs, max_age_days=15).set_axis(target.index)
        for name, frame in {**local, "cny": ecb_usd_cny(payload), "jpy": yen}.items()
    }
    cny, jpy, close, vwap = (aligned[name] for name in ("cny", "jpy", *MARKET_SERIES))
    features = {}
    for name, data in (("cny", cny), ("jpy", jpy)):
        log_rate = np.log(data.value)
        for days in (1, 5):
            features[f"global__ecb_usd_{name}_return_{days}d"] = log_rate.diff(days)
    features["global__ecb_usd_cny_age"] = cny.age_days.where(cny.value.notna())
    for name, data in (("close", close), ("vwap", vwap)):
        features[f"fx__ecb_cross_{name}_basis"] = np.log(target / (cny.value * data.value))
    date_gap = (vwap.observation_date - cny.observation_date).dt.days
    features["fx__ecb_cross_vwap_date_gap"] = date_gap.where((cny.value * vwap.value).notna())
    return pd.DataFrame(features, index=target.index)


def build_paper_features(client, root, target):
    cutoffs = target.index.tz_localize("Europe/Moscow")
    end = target.index.max().strftime("%d.%m.%Y")
    args = {"UniDbQuery.Posted": "True", "UniDbQuery.From": SWAP_START, "UniDbQuery.To": end}
    cny_args = {**args, "UniDbQuery.Cur": "2", "UniDbQuery.P1": "0"}
    swaps = {
        name: client.get(CBR + path, args if name == "claims" else cny_args)
        for name, path in SWAP_PATHS.items()
    }
    ecb = client.get(ECB_ARCHIVE)
    survey = client.get(SURVEY_XLSX)
    calendar = client.get(SURVEY_CALENDAR)
    history = swap_history(swaps)
    macro, releases = survey_features(survey, calendar, cutoffs)
    local = {name: read_series(root, name) for name in MARKET_SERIES}
    features = pd.concat(
        [market_features(ecb, local, target), swap_features(history, swaps, cutoffs), macro], axis=1
    ).rename_axis(target.index.name)
    return (
        features,
        history,
        releases,
        {**swaps, "ecb": ecb, "survey": survey, "calendar": calendar},
    )
