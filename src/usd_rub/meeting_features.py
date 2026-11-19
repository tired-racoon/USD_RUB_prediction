"""Признаки заседаний ЦБ по датированным версиям расписания и объявлениям."""

import json
from pathlib import Path

import pandas as pd

CALENDAR_PATH = Path(__file__).with_name("cbr_meetings.json")
MOSCOW = "Europe/Moscow"


def publication_time(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("Время публикации календаря ЦБ должно содержать часовой пояс")
    return stamp


def cbr_meeting_features(cutoffs, calendar=None):
    """Ноль дней означает день заседания по московскому календарю."""
    if calendar is None:
        calendar = json.loads(CALENDAR_PATH.read_text(encoding="utf-8"))
    cutoffs = pd.DatetimeIndex(cutoffs)
    if cutoffs.tz is None:
        raise ValueError("Для cutoff требуется часовой пояс")
    versions = sorted(
        (publication_time(item["known_at"]), item["year"], pd.DatetimeIndex(item["dates"]))
        for item in calendar["schedules"]
    )
    extraordinary = [
        (
            pd.Timestamp(item["date"]),
            publication_time(item["known_at"]),
            publication_time(item["decision_at"]),
        )
        for item in calendar["unscheduled"]
    ]
    covered_from, covered_to = map(pd.Timestamp, calendar["unscheduled_coverage"])
    rows = []
    for cutoff in cutoffs:
        today = cutoff.tz_convert(MOSCOW).tz_localize(None).normalize()
        schedules = {}
        for known, year, dates in versions:
            if known > cutoff:
                break
            schedules[year] = dates
        candidates = [dates[dates >= today].min() for dates in schedules.values()]
        announced = [date for date, known, _ in extraordinary if known <= cutoff]
        candidates.extend(date for date in announced if date >= today)
        upcoming = min((date for date in candidates if pd.notna(date)), default=pd.NaT)
        decided = [date for date, _, decision in extraordinary if decision <= cutoff]
        last_decision = max(decided, default=pd.NaT)
        rows.append(
            {
                "days_to_cbr_meeting": (upcoming - today).days if pd.notna(upcoming) else None,
                "cbr_unscheduled_meeting_flag": int(today in announced)
                if covered_from <= today <= covered_to
                else None,
                "days_since_cbr_unscheduled_meeting": (today - last_decision).days
                if pd.notna(last_decision)
                else None,
            }
        )
    return pd.DataFrame(rows, index=cutoffs).astype("Int64")
