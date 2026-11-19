import pandas as pd

from usd_rub import paper_features
from usd_rub.storage import Payload


def test_survey_uses_prediction_year_at_cutoff_and_keeps_revision_same_year(monkeypatch):
    sheets = {}
    for sheet, name in (("1", "CPI"), ("3", "Key rate"), ("11", "USD / RUB rate")):
        table = pd.DataFrame(None, index=range(11), columns=range(6), dtype=object)
        table.iat[4, 1] = name
        table.iloc[5, 4:] = pd.to_datetime(["2024-11-01", "2024-12-01"])
        table.iat[7, 2] = "Median"
        table.iat[10, 2] = "Average"
        table.iloc[7:10, 3] = pd.to_datetime(["2024-12-31", "2025-12-31", "2026-12-31"])
        table.iloc[7:10, 4:] = [[90, 95], [100, 110], [120, 130]]
        sheets[sheet] = table
    monkeypatch.setattr(paper_features, "workbook", lambda _: sheets)
    html = """<table><tr><td>1</td><td>Макроэкономический опрос Банка России</td>
    <td>11.12.2024</td></tr></table>"""
    calendar = Payload(html.encode(), "calendar", "hash", "2024-12-11T12:00:00Z")
    cutoffs = pd.DatetimeIndex(["2024-12-11", "2024-12-12", "2025-01-01"], tz="Europe/Moscow")

    features, _ = paper_features.survey_features(None, calendar, cutoffs)

    assert features.extended__macro__cbr_survey_usd_rub_current_year_median.tolist() == [
        90,
        95,
        110,
    ]
    assert features.extended__macro__cbr_survey_usd_rub_next_year_median.tolist() == [100, 110, 130]
    assert features.extended__macro__cbr_survey_usd_rub_current_year_revision.iloc[1:].tolist() == [
        5,
        10,
    ]
