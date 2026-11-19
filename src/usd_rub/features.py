"""Сборка доступных к моменту прогноза наблюдений."""

from bisect import bisect_right, insort

import numpy as np
import pandas as pd

from usd_rub.freshness import ensure_allowed, freshness
from usd_rub.meeting_features import cbr_meeting_features


def asof_series(frame, cutoffs, allow_assumptions=False, max_age_days=None):
    frame = frame.copy()
    cutoffs = pd.DatetimeIndex(cutoffs)
    if cutoffs.tz is None:
        raise ValueError("Для cutoff требуется часовой пояс")
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, format="mixed")
    frame["date"] = pd.to_datetime(frame["date"])
    if not allow_assumptions:
        frame = frame.loc[~frame["availability_basis"].str.startswith("assumed")]
    ordered = frame.sort_values(["available_at", "date", "ingested_at"])
    pending = list(ordered.itertuples(index=False))
    pointer = 0
    known = {}
    known_dates = []
    rows = []
    for cutoff in sorted(cutoffs):
        while pointer < len(pending) and pending[pointer].available_at <= cutoff:
            row = pending[pointer]
            if row.date not in known:
                insort(known_dates, row.date)
            known[row.date] = row
            pointer += 1
        local_cutoff = cutoff.tz_convert("Europe/Moscow").tz_localize(None)
        position = bisect_right(known_dates, local_cutoff)
        if position == 0:
            rows.append({"cutoff": cutoff, "value": np.nan, "observation_date": pd.NaT})
            continue
        row = known[known_dates[position - 1]]
        age = (local_cutoff.normalize() - row.date).days
        stale = max_age_days is not None and age > max_age_days
        name = getattr(row, "series", "")
        if name.startswith("fns_") or name == "ru_holiday_fns":
            stale = stale or row.date != local_cutoff.normalize()
        rows.append(
            {
                "cutoff": cutoff,
                "value": np.nan if stale else row.value,
                "observation_date": row.date,
                "age_days": age,
                "available_at": row.available_at,
                "availability_basis": row.availability_basis,
            }
        )
    return pd.DataFrame(rows).set_index("cutoff").reindex(cutoffs)


def build_features(frames, start, end, cutoff="15:00", assumed_lags=None, events=None):
    hour, minute = map(int, cutoff.split(":"))
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("Некорректный cutoff")
    dates = pd.date_range(start, end, freq="D", tz="Europe/Moscow")
    cutoffs = dates + pd.Timedelta(hours=hour, minutes=minute)
    aligned_columns = {}
    for name, frame in frames.items():
        ensure_allowed(name)
        frame = frame.copy()
        if assumed_lags and name in assumed_lags:
            lag = int(assumed_lags[name])
            if lag < 0:
                raise ValueError("Лаг публикации не может быть отрицательным")
            unknown = frame["availability_basis"] == "unknown"
            local = pd.DatetimeIndex(frame.loc[unknown, "date"]).tz_localize("Europe/Moscow")
            frame.loc[unknown, "available_at"] = (local + pd.Timedelta(days=lag)).tz_convert("UTC")
            frame.loc[unknown, "availability_basis"] = f"assumed_lag_{lag}d_latest_vintage"
        policy_frame = frame if "series" in frame else frame.assign(series=name)
        policy = freshness(policy_frame, end)
        limit = policy.get("max_age_days")
        if limit is not None and policy.get("frequency") == "monthly":
            limit += 31
        aligned = asof_series(
            frame, cutoffs, allow_assumptions=bool(assumed_lags), max_age_days=limit
        )
        aligned_columns[name] = aligned["value"]
        aligned_columns[f"{name}__age_days"] = aligned.get("age_days", np.nan)
    result = pd.DataFrame(aligned_columns, index=cutoffs)
    pairs = {
        "ruonia_key_spread": ("ruonia", "key_rate"),
        "ofz_10y_2y_spread": ("ofz_10y", "ofz_2y"),
        "ust_slope": ("ust_10y", "ust_2y"),
        "rate_differential": ("key_rate", "fed_funds"),
    }
    for name, (a, b) in pairs.items():
        if a in result and b in result:
            result[name] = result[a] - result[b]
    if {"cny_tom_close", "usd_cny"}.issubset(result):
        result["usd_rub_cross"] = result["cny_tom_close"].where(
            result["cny_tom_close"] > 0
        ) * result["usd_cny"].where(result["usd_cny"] > 0)
        if "usd_rub" in result:
            result["cross_spread"] = result["usd_rub"] - result["usd_rub_cross"]
            result["cross_spread_pct"] = 100 * result["cross_spread"] / result["usd_rub_cross"]
            result["cross_log_basis"] = np.log(
                result["usd_rub"].where(result["usd_rub"] > 0)
            ) - np.log(result["usd_rub_cross"])
    if "brent" in result:
        result["brent_lag_30d"] = result["brent"].shift(30)
        result["brent_lag_60d"] = result["brent"].shift(60)
    result["weekday"] = result.index.dayofweek
    result["day_of_month"] = result.index.day
    result["month"] = result.index.month
    result["days_to_month_end"] = result.index.days_in_month - result.index.day
    result["days_to_25_nominal"] = [
        (d.replace(day=25) - d).days
        if d.day <= 25
        else ((d + pd.offsets.MonthBegin(1)).replace(day=25) - d).days
        for d in result.index
    ]
    result["days_to_28_nominal"] = [
        (d.replace(day=28) - d).days
        if d.day <= 28
        else ((d + pd.offsets.MonthBegin(1)).replace(day=28) - d).days
        for d in result.index
    ]
    result["market_regime"] = np.select(
        [
            result.index < pd.Timestamp("2022-02-24", tz="Europe/Moscow"),
            result.index < pd.Timestamp("2024-06-13", tz="Europe/Moscow"),
        ],
        ["pre_2022", "2022_to_moex_stop"],
        default="otc_since_2024",
    )
    if events is not None and not events.empty:
        result = result.join(
            event_features(events, result.index, allow_assumptions=bool(assumed_lags))
        )
    result = result.join(cbr_meeting_features(result.index))
    result.index.name = "cutoff"
    return result


def event_features(events, cutoffs, allow_assumptions=False):
    events = events.copy()
    if not allow_assumptions and "availability_basis" in events:
        events = events.loc[~events.availability_basis.fillna("unknown").str.startswith("assumed")]
    for name in ("publication_time", "effective_time"):
        events[name] = pd.to_datetime(events[name], utc=True)
    rows = []
    categories = sorted(set(events["event_type"]) - {"calendar_coverage"})
    versioned = "release_id" in events and "calendar_year" in events
    bounds = {}
    for category in categories:
        subset = events.loc[events.event_type.eq(category)]
        start = (
            pd.to_datetime(subset.coverage_start.dropna(), utc=True).min()
            if "coverage_start" in subset
            else pd.NaT
        )
        end = (
            pd.to_datetime(subset.coverage_end.dropna(), utc=True).max()
            if "coverage_end" in subset
            else pd.NaT
        )
        bounds[category] = start, end
    for cutoff in cutoffs:
        date = cutoff.tz_convert("Europe/Moscow").normalize()
        known = events.loc[events["publication_time"] <= cutoff]
        known_years = set()
        if versioned:
            calendar = known.loc[known.release_id.notna() & known.calendar_year.notna()]
            versions = calendar.sort_values(["publication_time", "release_id"]).drop_duplicates(
                "calendar_year", keep="last"
            )
            known_years = set(versions.calendar_year.astype(int))
            selected = versions.set_index("calendar_year").release_id.to_dict()
            keep = known.release_id.isna() | known.calendar_year.map(selected).eq(known.release_id)
            known = known.loc[keep]
        if "event_key" in known:
            has_key = known.event_key.notna()
            known = pd.concat(
                [
                    known.loc[~has_key],
                    known.loc[has_key]
                    .sort_values("publication_time")
                    .drop_duplicates("event_key", keep="last"),
                ]
            )
        row = {"cutoff": cutoff}
        for category in categories:
            subset = known.loc[known["event_type"] == category]
            effective_dates = subset["effective_time"].dt.tz_convert("Europe/Moscow").dt.normalize()
            today = subset.loc[effective_dates == date]
            row[f"event_{category}"] = float(today["value"].sum())
            lower, upper = bounds[category]
            outside_coverage = (
                pd.notna(lower) and cutoff < lower or pd.notna(upper) and cutoff > upper
            )
            if outside_coverage:
                row[f"event_{category}"] = np.nan
            if category.startswith("tax_") and versioned and date.year not in known_years:
                row[f"event_{category}"] = np.nan
            next_dates = effective_dates.loc[effective_dates >= date]
            row[f"days_to_{category}"] = (
                (next_dates.min() - date).days if len(next_dates) else np.nan
            )
            if outside_coverage:
                row[f"days_to_{category}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("cutoff")
