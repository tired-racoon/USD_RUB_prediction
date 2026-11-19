"""Срок до заседания и доступность внеочередных решений."""

import pandas as pd

from usd_rub.meeting_features import cbr_meeting_features


def test_unscheduled_meetings_do_not_appear_before_public_information():
    cutoffs = pd.DatetimeIndex(
        [
            "2022-02-28 00:00",
            "2022-02-28 09:45",
            "2022-03-01 00:00",
            "2022-05-25 10:00",
            "2022-05-26 00:00",
        ],
        tz="Europe/Moscow",
    )
    frame = cbr_meeting_features(cutoffs)
    assert frame.days_to_cbr_meeting.tolist() == [18, 0, 17, 1, 0]
    assert frame.cbr_unscheduled_meeting_flag.tolist() == [0, 1, 0, 0, 1]
    assert pd.isna(frame.days_since_cbr_unscheduled_meeting.iloc[0])
    assert frame.days_since_cbr_unscheduled_meeting.iloc[1:3].tolist() == [0, 1]
    assert frame.days_since_cbr_unscheduled_meeting.iloc[-1] > 0


def test_schedule_revision_replaces_old_dates_only_when_published():
    cutoffs = pd.DatetimeIndex(
        ["2019-10-29 23:59", "2019-10-30 00:00", "2026-12-19 00:00"], tz="Europe/Moscow"
    )
    calendar = {
        "schedules": [
            {"year": 2020, "known_at": "2019-08-09T00:00:00+03:00", "dates": ["2020-01-10"]},
            {"year": 2020, "known_at": "2019-10-30T00:00:00+03:00", "dates": ["2020-01-17"]},
        ],
        "unscheduled": [],
        "unscheduled_coverage": ["2015-01-01", "2026-11-18"],
    }
    result = cbr_meeting_features(cutoffs, calendar)
    assert result.days_to_cbr_meeting.iloc[:2].tolist() == [73, 79]
    assert pd.isna(result.days_to_cbr_meeting.iloc[-1])
    assert pd.isna(result.cbr_unscheduled_meeting_flag.iloc[-1])
